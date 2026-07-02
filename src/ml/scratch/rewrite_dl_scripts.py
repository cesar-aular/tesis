import os
from pathlib import Path

models = ["lstm", "nhits", "tft"]
classes = ["LSTM", "NHITS", "TFT"]

for mod, cls in zip(models, classes):
    
    # --- TUNE.PY ---
    tune_content = f"""import pandas as pd
import optuna
import json
import gc
import torch
from pathlib import Path
from neuralforecast import NeuralForecast
from neuralforecast.models import {cls}
from neuralforecast.losses.pytorch import MQLoss

def run_tune(train_df: pd.DataFrame, strategy: str = "toy"):
    print(f"[{cls}] Iniciando TUNE global dinamico con variables exogenas...")
    
    out_dir = Path(f"models/{{strategy}}/{mod}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    max_dates = train_df.groupby('unique_id')['ds'].max().reset_index()
    max_dates.rename(columns={{'ds': 'max_ds'}}, inplace=True)
    train_df = train_df.merge(max_dates, on='unique_id')
    
    train_subset = train_df[train_df['ds'] <= train_df['max_ds'] - pd.Timedelta(days=7)].drop(columns=['max_ds']).copy()
    val_subset = train_df[train_df['ds'] > train_df['max_ds'] - pd.Timedelta(days=7)].drop(columns=['max_ds']).copy()
    
    plantas_tune = train_subset['unique_id'].unique().tolist()[:3] 
    train_subset = train_subset[train_subset['unique_id'].isin(plantas_tune)]
    val_subset = val_subset[val_subset['unique_id'].isin(plantas_tune)]
    
    n_trials = 2 if strategy == "toy" else 5
    EPOCHS = 2
    batch_size = 8
    num_windows = len(train_subset)
    if strategy == 'toy':
        max_steps = 10
    else:
        max_steps = max(10, int((num_windows / batch_size) * EPOCHS))
    
    stat_exog = ['macrozona_idx', 'potencia_neta_mw']
    hist_exog = ['humedad-relativa', 'radiacion-global-instantanea', 'temp-aire-seco']
    futr_exog = ['sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'sin_season', 'cos_season', 'estacion_idx']
    
    def objective(trial):
        hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64]) if '{cls}' == 'TFT' else trial.suggest_categorical('hidden_size', [32, 64, 128])
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        model_obj = {cls}(
            h=24,
            input_size=168,
            max_steps=max_steps,
            learning_rate=lr,
            batch_size=batch_size,
            windows_batch_size=32,
            scaler_type='robust',
            loss=MQLoss(level=[90]),
            stat_exog_list=stat_exog,
            hist_exog_list=hist_exog,
            futr_exog_list=futr_exog
        )
        
        nf = NeuralForecast(models=[model_obj], freq='h')
        
        try:
            static_df = train_subset[['unique_id', 'macrozona_idx', 'potencia_neta_mw']].drop_duplicates()
            nf.fit(df=train_subset, static_df=static_df)
            val_preds = nf.predict(futr_df=val_subset)
            
            merged = val_preds.reset_index().merge(val_subset[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='inner')
            col_target = f"{cls}-median"
            if col_target not in merged.columns or merged.empty:
                return 9999.0
            
            rmse = ((merged['y'] - merged[col_target])**2).mean()**0.5
            return rmse
        except Exception as e:
            print(f"[{cls}] Trial fallido: {{e}}")
            return 9999.0
        finally:
            if 'nf' in locals():
                del nf
            if 'model_obj' in locals():
                del model_obj
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = {{
        'input_size': 168,
        'hidden_size': study.best_params.get('hidden_size', 64),
        'learning_rate': study.best_params.get('learning_rate', 1e-3)
    }}
    
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_params, f)
        
    print(f"[{cls}] TUNE completado: {{best_params}}")
    
    try:
        del study
        del train_subset
        del val_subset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
        
    return best_params
"""
    
    # --- TRAIN.PY ---
    train_content = f"""import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import {cls}
from neuralforecast.losses.pytorch import MQLoss
from pathlib import Path
import json
import gc
import torch

def run_train(train_df: pd.DataFrame, strategy: str = "toy"):
    print(f"[{cls}] Iniciando TRAIN global probabilistico con variables exogenas...")
    
    models_dir = Path(f"models/{{strategy}}/{mod}")
    params_path = models_dir / "best_params.json"
    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)
    else:
        params = {{'input_size': 168}}
        
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
    model_obj = {cls}(
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
        stat_exog_list=stat_exog,
        hist_exog_list=hist_exog,
        futr_exog_list=futr_exog
    )
    
    nf = NeuralForecast(models=[model_obj], freq='h')
    
    try:
        static_df = train_df[['unique_id', 'macrozona_idx', 'potencia_neta_mw']].drop_duplicates()
        nf.fit(df=train_df, static_df=static_df)
    except Exception as e:
        print(f"[{cls}] ERROR de memoria o fitting: {{e}}")
        # Fallback agresivo
        model_obj = {cls}(h=24, input_size=168, max_steps=100, batch_size=8, windows_batch_size=32, accumulate_grad_batches=16,
                          loss=MQLoss(level=[90]), stat_exog_list=stat_exog, hist_exog_list=hist_exog, futr_exog_list=futr_exog)
        nf = NeuralForecast(models=[model_obj], freq='h')
        nf.fit(df=train_df, static_df=static_df)
    
    nf.save(path=str(models_dir / "nf_models"), overwrite=True)
    print(f"[{cls}] Entrenamiento completado y guardado.")

    try:
        del nf
        del model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
"""

    out_mod = Path(f"src/ml/{mod}")
    with open(out_mod / "tune.py", "w") as f:
        f.write(tune_content)
    with open(out_mod / "train.py", "w") as f:
        f.write(train_content)

print("Updated tune.py and train.py for all models.")
