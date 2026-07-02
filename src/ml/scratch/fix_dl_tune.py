import os
from pathlib import Path

dl_models = ["lstm", "nhits", "tft"]

tune_template = """import pandas as pd
import optuna
import json
import gc
import torch
from pathlib import Path
from neuralforecast import NeuralForecast
from neuralforecast.models import {ModelClass}

def run_tune(train_df: pd.DataFrame, strategy: str = "toy"):
    print("[{ModelName}] Iniciando TUNE global dinámico...")
    
    out_dir = Path("models/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Validacion con ultimos 7 dias
    val_cutoff = train_df['ds'].max() - pd.Timedelta(days=7)
    train_subset = train_df[train_df['ds'] <= val_cutoff].copy()
    val_subset = train_df[train_df['ds'] > val_cutoff].copy()
    
    # Submuestreo extremo para Tune para no demorar dias
    plantas_tune = train_subset['unique_id'].unique().tolist()[:3] 
    train_subset = train_subset[train_subset['unique_id'].isin(plantas_tune)]
    val_subset = val_subset[val_subset['unique_id'].isin(plantas_tune)]
    
    n_trials = 2 if strategy == "toy" else 5
    max_steps = 20 if strategy == "toy" else 100
    
    def objective(trial):
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        model_obj = {ModelClass}(
            h=24,
            input_size=168,
            max_steps=max_steps,
            learning_rate=lr,
            batch_size=32,
            windows_batch_size=128,
            scaler_type='robust'
            # No accumulate_grad_batches needed for tiny tune sample
        )
        
        nf = NeuralForecast(models=[model_obj], freq='h')
        
        try:
            nf.fit(df=train_subset[['unique_id', 'ds', 'y']])
            
            # Predict the next h=24 for the validation split
            # NeuralForecast cross validation predict directly:
            val_preds = nf.predict()
            
            # Merge with real to calculate RMSE
            merged = val_preds.reset_index().merge(val_subset[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='inner')
            if '{ModelClass}' not in merged.columns or merged.empty:
                return 9999.0
            
            rmse = ((merged['y'] - merged['{ModelClass}'])**2).mean()**0.5
            return rmse
        except Exception as e:
            print(f"[{ModelName}] Trial fallido: {{e}}")
            return 9999.0
            
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = {{
        'input_size': 168,
        'hidden_size': study.best_params.get('hidden_size', 64),
        'max_steps': 1000 if strategy != 'toy' else 100,
        'learning_rate': study.best_params.get('learning_rate', 1e-3)
    }}
    
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_params, f)
        
    print(f"[{ModelName}] TUNE completado: {{best_params}}")
    
    # MEMORY CLEANUP VITAL
    del study
    del train_subset
    del val_subset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return best_params
"""

for mod in dl_models:
    mod_path = Path("src/ml") / mod
    m_name = mod.upper()
    c_name = "LSTM" if mod == "lstm" else "NHITS" if mod == "nhits" else "TFT"
    
    with open(mod_path / "tune.py", "w") as f:
        # Some special considerations for TFT
        content = tune_template.format(model_name=mod, ModelName=m_name, ModelClass=c_name)
        if c_name == "TFT":
            content = content.replace("hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])", "hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64])")
        f.write(content)

print("Updated DL tune.py with dynamic Optuna tuning and aggressive memory cleanup.")
