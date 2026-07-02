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

    X_train = train_df.drop(columns=['y', 'ds', 'unique_id'])
    y_train = train_df['y']

    # GPU (xgboost 3.x): ~5-10x mas rapido sobre los 4.3M de filas N-1
    try:
        model = xgb.XGBRegressor(**params, device='cuda', random_state=42)
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"[XGB Global] GPU no disponible ({e}); usando CPU.")
        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
    
    joblib.dump(model, models_dir / "xgb_global_model.joblib")
    print("[XGB Global] Entrenamiento completado y guardado.")
