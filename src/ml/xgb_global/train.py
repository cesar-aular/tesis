import pandas as pd
import xgboost as xgb
import joblib
import json
from pathlib import Path

def run_train(train_df: pd.DataFrame, strategy: str = "toy"):
    print("[XGB Global] Iniciando TRAIN global...")
            
    models_dir = Path("models/xgb_global")
    models_dir.mkdir(parents=True, exist_ok=True)

    params_path = models_dir / "best_params.json"
    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)
    else:
        params = {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5}
        
    params['enable_categorical'] = True
    params['tree_method'] = 'hist'
    params['n_jobs'] = -1
    
    if strategy == "toy":
        pass # No downsampling on data, just limited plants in orchestrator
        
    X_train = train_df.drop(columns=['y', 'ds', 'unique_id'])
    y_train = train_df['y']
    
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, models_dir / "xgb_global_model.joblib")
    print("[XGB Global] Entrenamiento completado y guardado.")
