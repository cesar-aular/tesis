import pandas as pd
import optuna
import json
import gc
import torch
from pathlib import Path
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import MQLoss

def run_tune(train_df: pd.DataFrame, strategy: str = "toy"):
    print(f"[TFT] Iniciando TUNE global dinamico con variables exogenas...")
    
    out_dir = Path(f"models/{strategy}/tft")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    max_dates = train_df.groupby('unique_id')['ds'].max().reset_index()
    max_dates.rename(columns={'ds': 'max_ds'}, inplace=True)
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
        hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64])
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        model_obj = TFT(
            h=24,
            input_size=168,
            hidden_size=hidden_size,  # antes el trial no usaba hidden_size
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
            col_target = f"TFT-median"
            if col_target not in merged.columns or merged.empty:
                return 9999.0
            
            rmse = ((merged['y'] - merged[col_target])**2).mean()**0.5
            return rmse
        except Exception as e:
            print(f"[TFT] Trial fallido: {e}")
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
    
    best_params = {
        'input_size': 168,
        'hidden_size': study.best_params.get('hidden_size', 64),
        'learning_rate': study.best_params.get('learning_rate', 1e-3)
    }
    
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_params, f)
        
    print(f"[TFT] TUNE completado: {best_params}")
    
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
