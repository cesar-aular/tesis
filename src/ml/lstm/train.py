import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import MQLoss
from pathlib import Path
import json
import gc
import torch

def run_train(train_df: pd.DataFrame, strategy: str = "toy"):
    print(f"[LSTM] Iniciando TRAIN global probabilistico con variables exogenas...")
    
    models_dir = Path(f"models/{strategy}/lstm")
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
        # Only use last 1 year of data for half to save memory
        min_date = train_df['ds'].max() - pd.DateOffset(years=1)
        train_df = train_df[train_df['ds'] >= min_date].copy()
    else:
        batch_size = 16
        acc_grad = 8
        EPOCHS = 10
        # Use last 2 years for total to prevent OOM
        min_date = train_df['ds'].max() - pd.DateOffset(years=2)
        train_df = train_df[train_df['ds'] >= min_date].copy()
        
    num_windows = len(train_df)
    if strategy == 'toy':
        max_steps = 10
    else:
        max_steps = max(10, int((num_windows / batch_size) * EPOCHS))
    
    stat_exog = ['macrozona_idx', 'potencia_neta_mw']
    hist_exog = ['humedad-relativa', 'radiacion-global-instantanea', 'temp-aire-seco']
    futr_exog = ['sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'sin_season', 'cos_season', 'estacion_idx']
    model_obj = LSTM(
        h=24,
        input_size=params.get('input_size', 168),
        # nf 3.x: el parametro es encoder_hidden_size ('hidden_size' caia en
        # trainer_kwargs y rompia el fit, forzando siempre el fallback sin tunear)
        encoder_hidden_size=params.get('hidden_size', 64),
        learning_rate=params.get('learning_rate', 1e-3),
        max_steps=max_steps,
        scaler_type='robust',
        batch_size=batch_size,
        windows_batch_size=256,
        accumulate_grad_batches=acc_grad,
        loss=MQLoss(level=[90]),
        stat_exog_list=stat_exog,
        hist_exog_list=hist_exog,
        futr_exog_list=futr_exog
    )
    
    nf = NeuralForecast(models=[model_obj], freq='h')
    
    try:
        static_df = train_df[['unique_id', 'macrozona_idx', 'potencia_neta_mw']].drop_duplicates()
        nf.fit(df=train_df, static_df=static_df)
    except Exception as e:
        print(f"[LSTM] ERROR de memoria o fitting: {e}")
        # Fallback agresivo
        model_obj = LSTM(h=24, input_size=168, max_steps=100, batch_size=8, windows_batch_size=32, accumulate_grad_batches=16,
                          loss=MQLoss(level=[90]), stat_exog_list=stat_exog, hist_exog_list=hist_exog, futr_exog_list=futr_exog)
        nf = NeuralForecast(models=[model_obj], freq='h')
        nf.fit(df=train_df, static_df=static_df)
    
    nf.save(path=str(models_dir / "nf_models"), overwrite=True)
    print(f"[LSTM] Entrenamiento completado y guardado.")

    try:
        del nf
        del model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
