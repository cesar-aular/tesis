"""Baselines Deep Learning ESTRICTAMENTE LOCALES (LSTM / N-HiTS por planta).

Contraparte del paradigma global Cross-Site: cada modelo se entrena SOLO con la
historia de la planta objetivo (como xgb_local), representando "lo que se
obtiene tras esperar a recolectar historia propia" — el enfoque tradicional
que el anteproyecto contrasta contra el modelo global Cold-Start.

Reglas de rigor:
- ANTI train-on-test: TODAS las ventanas de evaluación (raw, operacional y
  estacionales) se excluyen del entrenamiento local.
- El target NO se imputa para entrenar (filas sin y se descartan) ni para
  métricas (solo observaciones reales). Únicamente el CONTEXTO de inferencia
  se rellena (interpolación) para que el modelo pueda procesarlo — el paradigma
  local dispone de sus sensores y de su pasado real.
- Mismos horizontes y ventanas que los modelos globales -> comparación 1:1.
"""
import gc
import json
from pathlib import Path

from src.ml.utils.quiet import silence_noise
silence_noise()

import pandas as pd
import torch
from neuralforecast import NeuralForecast

from src.ml.utils.dl_models import (MODEL_SPECS, _build, _compute_max_steps,
                                    normalize_target)
from src.ml.utils.dl_test import (HIST_HOURS, ROLLOUT_DAYS, make_hourly_grid,
                                  _apply_physics)
from src.ml.utils.eval_window import eval_window_variants, WINDOW_HOURS
from src.ml.utils.metrics import calculate_metrics
from src.ml.utils.idempotency import mark_run_completed
from src.ml.visualize import plot_forecast_rollout

LOCAL_MAX_STEPS_CAP = 300  # serie única: converge rápido; early stopping decide


def get_local_train_df(grid: pd.DataFrame, variants: dict) -> pd.DataFrame:
    """Serie de entrenamiento local: excluye TODAS las ventanas de evaluación
    y descarta huecos de generación (el target jamás se imputa para entrenar)."""
    excluded = pd.Series(False, index=grid.index)
    for _, start in variants.items():
        excluded.iloc[start:start + WINDOW_HOURS] = True
    train = grid[~excluded].dropna(subset=['y'])
    return train


def run_local_dl(model_name: str, test_df: pd.DataFrame, results_dir: Path,
                 macrozona: str, estacion: str, planta: str, strategy: str = "toy"):
    """Entrena y evalúa un modelo DL local (una planta) sobre las mismas
    ventanas/horizontes que los modelos globales."""
    spec = MODEL_SPECS[model_name]
    label = spec["label"]
    result_name = f"{model_name}_local"
    print(f"[{label} Local] Planta {planta}: entrenamiento estrictamente local...")

    grid = make_hourly_grid(test_df, planta)
    needed = HIST_HOURS + ROLLOUT_DAYS * 24
    if len(grid) < needed:
        print(f"[{label} Local] {planta} sin datos suficientes. Saltando.")
        return

    capacidad = max(float(grid['potencia_neta_mw'].iloc[0]), 1e-6)
    variants = eval_window_variants(grid['y'].reset_index(drop=True), capacidad,
                                    dates=grid['ds'])

    train_df = get_local_train_df(grid, variants)
    if len(train_df) < HIST_HOURS * 4:
        print(f"[{label} Local] {planta}: historia local insuficiente tras excluir "
              f"ventanas de evaluación ({len(train_df)} filas). Saltando.")
        return
    # Mismo espacio normalizado que los modelos globales (comparación 1:1)
    train_df = normalize_target(train_df)

    # Hiperparámetros: reutiliza los del modelo global de la estrategia
    # (misma familia arquitectónica; tunear 2x47 modelos locales es prohibitivo)
    params_path = Path(f"models/{strategy}/{model_name}/best_params.json")
    params = json.loads(params_path.read_text()) if params_path.exists() else {}

    batch_size, windows_batch = 8, 256  # serie única: batches moderados
    max_steps = 10 if strategy == 'toy' else _compute_max_steps(
        len(train_df), batch_size, windows_batch, epochs=8, cap=LOCAL_MAX_STEPS_CAP)
    patience = -1 if strategy == 'toy' else 5
    val_size = 0 if patience < 0 else 168

    model_obj = _build(spec,
                       hidden=params.get('hidden_size', 64),
                       lr=params.get('learning_rate', 1e-3),
                       max_steps=max_steps, batch_size=batch_size,
                       windows_batch_size=windows_batch,
                       input_size=params.get('input_size', 168),
                       early_stop_patience=patience)
    nf = NeuralForecast(models=[model_obj], freq='h')

    static_df = None
    if spec["exog"] != "futr_only":
        static_df = train_df[['unique_id', 'macrozona_idx', 'potencia_neta_mw']].drop_duplicates('unique_id')

    nf.fit(df=train_df, static_df=static_df, val_size=val_size)

    pred_cols = [f'{label}-median', f'{label}-lo-90', f'{label}-hi-90']
    rename_map = {f'{label}-median': 'y_pred',
                  f'{label}-lo-90': 'y_pred_lo_90',
                  f'{label}-hi-90': 'y_pred_hi_90'}
    real_y = test_df[['unique_id', 'ds', 'y']].copy()

    def _predict(current_hist, futr):
        """Contexto normalizado -> predicción des-normalizada a MWh."""
        futr_input = futr.drop(columns=['y'])
        preds = nf.predict(df=current_hist, futr_df=futr_input,
                           static_df=static_df).reset_index()
        if preds[f'{label}-median'].isna().any():
            raise RuntimeError(f"{label} Local produjo NaN para {planta}.")
        for col in pred_cols:
            if col in preds.columns:
                preds[col] = preds[col] * capacidad
        return _apply_physics(preds, futr, pred_cols)

    def _save(preds, horizon):
        merged = real_y.merge(preds, on=['unique_id', 'ds'], how='inner')
        out = merged.rename(columns=rename_map)
        out_dir = results_dir / result_name / macrozona / estacion / planta / horizon
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_dir / "preds.parquet")
        metrics = calculate_metrics(out['y'], out['y_pred'],
                                    y_lo=out.get('y_pred_lo_90'), y_hi=out.get('y_pred_hi_90'))
        mark_run_completed(out_dir, f"{result_name}_{horizon}", planta, estacion, metrics)
        plot_forecast_rollout(df_real=out, df_pred=out,
                              output_path=out_dir / "forecast_plot.png",
                              model_name=f"{label} Local ({horizon})", planta=planta)

    for suffix, start in variants.items():
        if start + needed > len(grid):
            continue
        window = grid.iloc[start:].reset_index(drop=True)

        # CONTEXTO REAL: el paradigma local observa su propio pasado. Solo se
        # interpola el hueco puntual para que el modelo procese la ventana
        # ("se rellena de forma que el modelo lo entienda"); ni el train ni
        # las métricas usan estos valores. Normalizado como el train.
        hist_df = window.iloc[:HIST_HOURS].copy()
        hist_df['y'] = (hist_df['y'].interpolate(limit_direction='both')
                        .fillna(0.0) / capacidad)

        _save(_predict(hist_df, window.iloc[HIST_HOURS:HIST_HOURS + 24].copy()),
              f"day1{suffix}")

        current_hist = hist_df.copy()
        all_preds = []
        for day in range(ROLLOUT_DAYS):
            s = HIST_HOURS + day * 24
            futr_day = window.iloc[s:s + 24].copy()
            preds = _predict(current_hist, futr_day)
            all_preds.append(preds)
            new_tail = futr_day.copy()
            # autorregresivo en espacio normalizado
            new_tail['y'] = preds[f'{label}-median'].values / capacidad
            current_hist = pd.concat([current_hist.iloc[24:], new_tail],
                                     ignore_index=True)
        _save(pd.concat(all_preds, ignore_index=True), f"rollout7d{suffix}")

    del nf, model_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[{label} Local] Completado para {planta}.")
