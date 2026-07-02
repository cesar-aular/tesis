import pandas as pd
import xgboost as xgb
import joblib
import json
from pathlib import Path
from joblib import Parallel, delayed

from src.ml.utils.features import generate_lags

def train_plant(planta, df_planta, models_dir):
    """Entrena XGBoost Local para una sola planta (Multithreading)."""
    X_train = df_planta.drop(columns=['y', 'ds', 'unique_id'])
    y_train = df_planta['y']
    
    # Cargar hiperparametros
    params_path = models_dir / f"best_params_{planta}.json"
    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)
    else:
        # Fallback
        params = {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5}
        
    params['enable_categorical'] = True
    params['tree_method'] = 'hist'
    params['n_jobs'] = 1
    
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, models_dir / f"xgb_local_{planta}.joblib")
    return planta

def run_train(silver_df: pd.DataFrame, planta: str, strategy: str = "toy"):
    print(f"[XGB Local] Iniciando TRAIN local para {planta}...")
    
    # XGB Local es un benchmark que SÍ usa la historia de la planta objetivo
    train_df = silver_df[silver_df['unique_id'] == planta].copy()
            
    models_dir = Path("models/xgb_local")
    
    plantas = train_df['unique_id'].unique().tolist()
    if strategy == "toy":
        plantas = plantas[:2]
        
    print(f"[XGB Local] Entrenando modelo local para {planta}...")
    train_plant(planta, train_df, models_dir)
    print(f"[XGB Local] Entrenamiento finalizado para {planta}.")
