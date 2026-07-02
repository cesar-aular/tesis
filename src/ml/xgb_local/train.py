import pandas as pd
import xgboost as xgb
import joblib
import json
from pathlib import Path

from src.ml.utils.features import generate_lags

# Ventana LOPO reservada para evaluacion Cold-Start: 168h de contexto + 7 dias
TEST_WINDOW_HOURS = 168 + 7 * 24
LAGS = [24, 168]


def run_train(silver_df: pd.DataFrame, planta: str, strategy: str = "toy"):
    """Baseline LOCAL: entrena SOLO con la historia de la planta objetivo.

    Rigor del benchmark:
    - ANTI-LEAKAGE (train-on-test): las primeras TEST_WINDOW_HOURS horas de la
      serie (contexto + ventana de evaluacion) se EXCLUYEN del entrenamiento.
      El modelo local representa "lo que obtendrias tras esperar a recolectar
      historia propia": entrena con los datos posteriores a la ventana evaluada.
    - Lags autorregresivos [24h, 168h]: el paradigma local SI dispone de la
      serie real de la planta, a diferencia del modelo global Cold-Start.
    """
    print(f"[XGB Local] Iniciando TRAIN local para {planta}...")

    plant_df = (silver_df[silver_df['unique_id'] == planta]
                .sort_values('ds')
                .copy())

    # Excluir la ventana de test (los targets de evaluacion no se entrenan)
    train_df = plant_df.iloc[TEST_WINDOW_HOURS:].copy()
    if train_df.empty:
        print(f"[XGB Local] {planta} sin historia suficiente tras excluir ventana de test.")
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

    X_train = train_df.drop(columns=['y', 'ds', 'unique_id'])
    y_train = train_df['y']

    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, models_dir / f"xgb_local_{planta}.joblib")
    print(f"[XGB Local] Entrenamiento finalizado para {planta} ({len(train_df)} filas, lags={LAGS}).")
