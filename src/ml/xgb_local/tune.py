import pandas as pd
import xgboost as xgb
import optuna
from pathlib import Path
import json
from joblib import Parallel, delayed

from src.ml.utils.features import generate_lags

def tune_plant(planta, df_planta, out_dir):
    """Sintoniza XGBoost para una sola planta (Local)."""
    cutoff = df_planta['ds'].max() - pd.Timedelta(days=7)
    train_data = df_planta[df_planta['ds'] <= cutoff]
    val_data = df_planta[df_planta['ds'] > cutoff]
    
    if len(train_data) < 24 or len(val_data) < 24:
        return planta, None
        
    X_train = train_data.drop(columns=['y', 'ds', 'unique_id'])
    y_train = train_data['y']
    X_val = val_data.drop(columns=['y', 'ds', 'unique_id'])
    y_val = val_data['y']
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'enable_categorical': True,
            'tree_method': 'hist',
            'n_jobs': 1
        }
        
        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        
        rmse = ((y_val - preds)**2).mean()**0.5
        return rmse
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    
    best_params = study.best_params
    with open(out_dir / f"best_params_{planta}.json", "w") as f:
        json.dump(best_params, f)
        
    return planta, best_params

def run_tune(silver_df: pd.DataFrame, planta: str, strategy: str = "toy"):
    print(f"[XGB Local] Generando lags para TUNE de {planta}...")
    train_df = silver_df[silver_df['unique_id'] == planta].copy()
    train_df = generate_lags(train_df, lags=[1, 24, 168])
    train_df = train_df.dropna()
    
    cat_cols = [c for c in train_df.select_dtypes(include=['object', 'string']).columns if c not in ['unique_id', 'ds']]
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        
    out_dir = Path("models/xgb_local")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[XGB Local] Tuning {planta} con Optuna...")
    tune_plant(planta, train_df, out_dir)
    print(f"[XGB Local] Tuning completado para {planta}.")
