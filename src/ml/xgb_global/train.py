import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
from pathlib import Path

# Regresión cuantílica multi-salida: P5 / P50 / P95 en un solo modelo ->
# intervalos de 90% comparables 1:1 con MQLoss(level=[90]) de los DL.
QUANTILES = np.array([0.05, 0.5, 0.95])


def run_train(train_df: pd.DataFrame, strategy: str = "toy"):
    print("[XGB Global] Iniciando TRAIN global (cuantilico P5/P50/P95)...")

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
    params['objective'] = 'reg:quantileerror'
    params['quantile_alpha'] = QUANTILES

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

    # La inferencia corre sobre DataFrames en RAM: alinear el device evita el
    # warning "Falling back to prediction using DMatrix" (mismatch GPU/CPU)
    model.set_params(device='cpu')
    joblib.dump(model, models_dir / "xgb_global_model.joblib")
    print("[XGB Global] Entrenamiento completado y guardado.")
