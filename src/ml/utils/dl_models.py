"""Fábrica única de modelos Deep Learning (NeuralForecast).

Un solo lugar define arquitectura, exógenas, tune y train para los 4 modelos
globales (LSTM, NHITS, TFT, Informer). Las carpetas src/ml/{modelo}/ quedan
como wrappers delgados que preservan la interfaz del orquestador.

Reglas encapsuladas aquí (antes duplicadas en 8 archivos):
- MQLoss(level=[90]) -> bandas lo-90/hi-90 (90% CI).
- Split de validación RELATIVO a cada planta (evita 'missing combinations').
- Higiene GPU (del + gc + empty_cache) tras cada fit/trial (RTX 2060, 6GB).
- Subsetting por estrategia (toy: 2 plantas; half: 1 año; total: 2 años).
- Cada modelo mapea 'hidden_size' a SU parámetro real de arquitectura
  (bug histórico: LSTM/NHITS no aceptan hidden_size en nf 3.x y caían
  silenciosamente en un fallback sin tunear).
"""
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import LSTM, NHITS, TFT, Informer

STAT_EXOG = ['macrozona_idx', 'potencia_neta_mw']
HIST_EXOG = ['humedad-relativa', 'radiacion-global-instantanea', 'temp-aire-seco']
FUTR_EXOG = ['sin_hour', 'cos_hour', 'sin_month', 'cos_month',
             'sin_season', 'cos_season', 'estacion_idx']
# Informer solo soporta exógenas futuras: el clima entra como futr
# (el pronóstico meteorológico day-ahead es información futura legítima).
FUTR_ONLY_EXOG = HIST_EXOG + FUTR_EXOG

MODEL_SPECS = {
    "lstm": {
        "cls": LSTM, "label": "LSTM",
        "arch": lambda h: {"encoder_hidden_size": h},
        "exog": "full", "hidden_choices": [32, 64, 128],
    },
    "nhits": {
        "cls": NHITS, "label": "NHITS",
        "arch": lambda h: {"mlp_units": 3 * [[h, h]]},
        "exog": "full", "hidden_choices": [32, 64, 128],
    },
    "tft": {
        "cls": TFT, "label": "TFT",
        "arch": lambda h: {"hidden_size": h},
        "exog": "full", "hidden_choices": [16, 32, 64],
    },
    "informer": {
        "cls": Informer, "label": "Informer",
        "arch": lambda h: {"hidden_size": h},
        "exog": "futr_only", "hidden_choices": [32, 64, 128],
    },
}


def _exog_kwargs(spec) -> dict:
    if spec["exog"] == "futr_only":
        return {"futr_exog_list": FUTR_ONLY_EXOG}
    return {"stat_exog_list": STAT_EXOG, "hist_exog_list": HIST_EXOG,
            "futr_exog_list": FUTR_EXOG}


def _static_df(df: pd.DataFrame, spec):
    if spec["exog"] == "futr_only":
        return None
    return df[['unique_id'] + STAT_EXOG].drop_duplicates('unique_id')


def _build(spec, hidden: int, lr: float, max_steps: int, batch_size: int,
           windows_batch_size: int, acc_grad: int = 1, input_size: int = 168):
    return spec["cls"](
        h=24,
        input_size=input_size,
        learning_rate=lr,
        max_steps=max_steps,
        scaler_type='robust',
        batch_size=batch_size,
        windows_batch_size=windows_batch_size,
        accumulate_grad_batches=acc_grad,
        loss=MQLoss(level=[90]),
        **spec["arch"](hidden),
        **_exog_kwargs(spec),
    )


def _cleanup(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _strategy_subset(train_df: pd.DataFrame, strategy: str):
    """Subsetting por estrategia para no reventar los 6GB de VRAM."""
    if strategy == 'toy':
        plantas = train_df['unique_id'].unique().tolist()[:2]
        return train_df[train_df['unique_id'].isin(plantas)].copy(), 8, 1, 1
    if strategy == 'half':
        min_date = train_df['ds'].max() - pd.DateOffset(years=1)
        return train_df[train_df['ds'] >= min_date].copy(), 16, 8, 5
    min_date = train_df['ds'].max() - pd.DateOffset(years=2)
    return train_df[train_df['ds'] >= min_date].copy(), 16, 8, 10


def run_dl_train(model_name: str, train_df: pd.DataFrame, strategy: str = "toy"):
    spec = MODEL_SPECS[model_name]
    label = spec["label"]
    print(f"[{label}] Iniciando TRAIN global probabilistico con variables exogenas...")

    models_dir = Path(f"models/{strategy}/{model_name}")
    params_path = models_dir / "best_params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}

    train_df, batch_size, acc_grad, epochs = _strategy_subset(train_df, strategy)
    if strategy == 'toy':
        max_steps = 10
    else:
        # Steps dinámicos: pasadas estadísticamente significativas sobre el dataset
        max_steps = max(10, int((len(train_df) / batch_size) * epochs))

    model_obj = _build(spec,
                       hidden=params.get('hidden_size', 64),
                       lr=params.get('learning_rate', 1e-3),
                       max_steps=max_steps, batch_size=batch_size,
                       windows_batch_size=256, acc_grad=acc_grad,
                       input_size=params.get('input_size', 168))
    nf = NeuralForecast(models=[model_obj], freq='h')
    static_df = _static_df(train_df, spec)

    try:
        nf.fit(df=train_df, static_df=static_df)
    except Exception as e:
        print(f"[{label}] ERROR de memoria o fitting: {e}")
        # Fallback agresivo para 6GB VRAM
        _cleanup(nf, model_obj)
        model_obj = _build(spec, hidden=32, lr=1e-3, max_steps=100,
                           batch_size=8, windows_batch_size=32, acc_grad=16)
        nf = NeuralForecast(models=[model_obj], freq='h')
        nf.fit(df=train_df, static_df=static_df)

    nf.save(path=str(models_dir / "nf_models"), overwrite=True)
    print(f"[{label}] Entrenamiento completado y guardado.")
    _cleanup(nf, model_obj)


def run_dl_tune(model_name: str, train_df: pd.DataFrame, strategy: str = "toy"):
    import optuna
    spec = MODEL_SPECS[model_name]
    label = spec["label"]
    print(f"[{label}] Iniciando TUNE global dinamico con variables exogenas...")

    out_dir = Path(f"models/{strategy}/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split de validación RELATIVO a cada planta (regla anti 'missing combinations')
    max_dates = train_df.groupby('unique_id')['ds'].max().rename('max_ds').reset_index()
    train_df = train_df.merge(max_dates, on='unique_id')
    train_subset = train_df[train_df['ds'] <= train_df['max_ds'] - pd.Timedelta(days=7)].drop(columns=['max_ds'])
    val_subset = train_df[train_df['ds'] > train_df['max_ds'] - pd.Timedelta(days=7)].drop(columns=['max_ds'])

    plantas_tune = train_subset['unique_id'].unique().tolist()[:3]
    train_subset = train_subset[train_subset['unique_id'].isin(plantas_tune)].copy()
    val_subset = val_subset[val_subset['unique_id'].isin(plantas_tune)].copy()

    n_trials = 2 if strategy == "toy" else 5
    batch_size = 8
    max_steps = 10 if strategy == 'toy' else max(10, int((len(train_subset) / batch_size) * 2))
    static_df = _static_df(train_subset, spec)

    def objective(trial):
        hidden = trial.suggest_categorical('hidden_size', spec["hidden_choices"])
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        nf = model_obj = None
        try:
            model_obj = _build(spec, hidden=hidden, lr=lr, max_steps=max_steps,
                               batch_size=batch_size, windows_batch_size=32)
            nf = NeuralForecast(models=[model_obj], freq='h')
            nf.fit(df=train_subset, static_df=static_df)
            val_preds = nf.predict(futr_df=val_subset)

            merged = val_preds.reset_index().merge(
                val_subset[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='inner')
            col = f"{label}-median"
            if col not in merged.columns or merged.empty:
                return 9999.0
            return float(((merged['y'] - merged[col]) ** 2).mean() ** 0.5)
        except Exception as e:
            print(f"[{label}] Trial fallido: {e}")
            return 9999.0
        finally:
            _cleanup(nf, model_obj)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = {
        'input_size': 168,
        'hidden_size': study.best_params.get('hidden_size', 64),
        'learning_rate': study.best_params.get('learning_rate', 1e-3),
    }
    (out_dir / "best_params.json").write_text(json.dumps(best_params))
    print(f"[{label}] TUNE completado: {best_params}")

    _cleanup(study, train_subset, val_subset)
    return best_params
