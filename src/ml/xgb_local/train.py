import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
from pathlib import Path

from src.ml.utils.features import generate_lags
from src.ml.utils.eval_window import eval_window_variants, WINDOW_HOURS

LAGS = [24, 168]
# Regresión cuantílica multi-salida (xgboost >= 2.0): P5 / P50 / P95 en un
# solo modelo -> intervalos de 90% comparables 1:1 con MQLoss(level=[90]) DL.
QUANTILES = np.array([0.05, 0.5, 0.95])


def run_train(silver_df: pd.DataFrame, planta: str, strategy: str = "toy"):
    """Baseline LOCAL: entrena SOLO con la historia de la planta objetivo.

    Rigor del benchmark:
    - ANTI-LEAKAGE (train-on-test): se EXCLUYEN del entrenamiento TODAS las
      ventanas de evaluacion (raw Y operacional, cada una = 168h contexto +
      7 dias de evaluacion). El modelo local representa "lo que obtendrias tras
      esperar a recolectar historia propia".
    - Lags autorregresivos [24h, 168h]: el paradigma local SI dispone de la
      serie real de la planta, a diferencia del modelo global Cold-Start.
    """
    print(f"[XGB Local] Iniciando TRAIN local para {planta}...")

    plant_df = (silver_df[silver_df['unique_id'] == planta]
                .sort_values('ds')
                .reset_index(drop=True)
                .copy())

    # Excluir TODAS las ventanas de evaluacion (los targets evaluados no se entrenan)
    capacidad = float(plant_df['potencia_neta_mw'].iloc[0])
    excluded = pd.Series(False, index=plant_df.index)
    for _, start in eval_window_variants(plant_df['y'], capacidad, dates=plant_df['ds']).items():
        excluded.iloc[start:start + WINDOW_HOURS] = True

    train_df = plant_df[~excluded].copy()
    if train_df.empty:
        print(f"[XGB Local] {planta} sin historia suficiente tras excluir ventanas de test.")
        return

    # Lags calculados DENTRO del slice de entrenamiento (sin tocar la ventana de test)
    train_df = generate_lags(train_df, LAGS)
    train_df = train_df.dropna(subset=[f'lag_{lag}' for lag in LAGS])

    models_dir = Path("models/xgb_local")
    models_dir.mkdir(parents=True, exist_ok=True)

    params_path = models_dir / f"best_params_{planta}.json"
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

    # GPU (xgboost 3.x) con fallback a CPU
    try:
        model = xgb.XGBRegressor(**params, device='cuda', random_state=42)
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"[XGB Local] GPU no disponible ({e}); usando CPU.")
        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)

    # Inferencia sobre DataFrames en RAM: alinear device evita el warning
    # "Falling back to prediction using DMatrix" (mismatch GPU/CPU)
    model.set_params(device='cpu')
    joblib.dump(model, models_dir / f"xgb_local_{planta}.joblib")
    print(f"[XGB Local] Entrenamiento finalizado para {planta} ({len(train_df)} filas, lags={LAGS}).")
