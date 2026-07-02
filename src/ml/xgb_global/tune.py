import pandas as pd
import xgboost as xgb
import optuna
import json
from pathlib import Path

def run_tune(train_df: pd.DataFrame, strategy: str = "toy"):
    print("[XGB Global] Iniciando TUNE global...")
    
    # Cast categories
    cat_cols = [c for c in train_df.select_dtypes(include=['object', 'string']).columns if c not in ['unique_id', 'ds']]
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        
    out_dir = Path("models/xgb_global")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Validation split (last 7 days of training data globally)
    cutoff = train_df['ds'].max() - pd.Timedelta(days=7)
    train_data = train_df[train_df['ds'] <= cutoff]
    val_data = train_df[train_df['ds'] > cutoff]
    
    if strategy == "toy":
        pass # No downsampling on data, just limited plants in orchestrator
        
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
            'n_jobs': -1
        }
        
        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        
        rmse = ((y_val - preds)**2).mean()**0.5
        return rmse
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5 if strategy == "toy" else 20)
    
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f)
        
    print("[XGB Global] TUNE completado.")
    return study.best_params
