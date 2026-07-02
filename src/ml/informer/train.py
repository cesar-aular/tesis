import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import Informer
from neuralforecast.losses.pytorch import MQLoss
from pathlib import Path
import json
import gc
import torch

# Informer (nf 3.x) solo soporta exogenas FUTURAS (EXOGENOUS_FUTR=True).
# El clima se declara como futr_exog: en produccion el pronostico meteorologico
# day-ahead esta disponible, por lo que es informacion legitimamente "futura conocida".
FUTR_EXOG = ['humedad-relativa', 'radiacion-global-instantanea', 'temp-aire-seco',
             'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
             'sin_season', 'cos_season', 'estacion_idx']


def run_train(train_df: pd.DataFrame, strategy: str = "toy"):
    print(f"[Informer] Iniciando TRAIN global probabilistico con variables exogenas...")

    models_dir = Path(f"models/{strategy}/informer")
    params_path = models_dir / "best_params.json"
    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)
    else:
        params = {'input_size': 168}

    if strategy == 'toy':
        batch_size = 8
        acc_grad = 1
        EPOCHS = 1
        plantas_train = train_df['unique_id'].unique().tolist()[:2]
        train_df = train_df[train_df['unique_id'].isin(plantas_train)].copy()
    elif strategy == 'half':
        batch_size = 16
        acc_grad = 8
        EPOCHS = 5
        min_date = train_df['ds'].max() - pd.DateOffset(years=1)
        train_df = train_df[train_df['ds'] >= min_date].copy()
    else:
        batch_size = 16
        acc_grad = 8
        EPOCHS = 10
        min_date = train_df['ds'].max() - pd.DateOffset(years=2)
        train_df = train_df[train_df['ds'] >= min_date].copy()

    num_windows = len(train_df)
    if strategy == 'toy':
        max_steps = 10
    else:
        max_steps = max(10, int((num_windows / batch_size) * EPOCHS))

    model_obj = Informer(
        h=24,
        input_size=params.get('input_size', 168),
        hidden_size=params.get('hidden_size', 64),
        learning_rate=params.get('learning_rate', 1e-3),
        max_steps=max_steps,
        scaler_type='robust',
        batch_size=batch_size,
        windows_batch_size=256,
        accumulate_grad_batches=acc_grad,
        loss=MQLoss(level=[90]),
        futr_exog_list=FUTR_EXOG,
    )

    nf = NeuralForecast(models=[model_obj], freq='h')

    try:
        nf.fit(df=train_df)
    except Exception as e:
        print(f"[Informer] ERROR de memoria o fitting: {e}")
        # Fallback agresivo para 6GB VRAM
        model_obj = Informer(h=24, input_size=168, hidden_size=32, max_steps=100,
                             batch_size=8, windows_batch_size=32, accumulate_grad_batches=16,
                             loss=MQLoss(level=[90]), futr_exog_list=FUTR_EXOG)
        nf = NeuralForecast(models=[model_obj], freq='h')
        nf.fit(df=train_df)

    nf.save(path=str(models_dir / "nf_models"), overwrite=True)
    print(f"[Informer] Entrenamiento completado y guardado.")

    try:
        del nf
        del model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
