import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
from pathlib import Path

from src.ml.utils.features import generate_lags
from src.ml.utils.eval_window import eval_window_variants, WINDOW_HOURS
from src.ml.utils.grid import make_hourly_grid

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

    # GRILLA CANONICA: las ventanas se definen sobre las mismas horas
    # calendario que en test (XGB y DL comparten grilla; ver utils/grid.py)
    grid = make_hourly_grid(plant_df, planta)
    capacidad = float(grid['potencia_neta_mw'].iloc[0])
    excluded = pd.Series(False, index=grid.index)
    for _, start in eval_window_variants(grid['y'], capacidad, dates=grid['ds']).items():
        excluded.iloc[start:start + WINDOW_HOURS] = True

    # ANTI-LEAKAGE de lags: enmascarar la y de TODAS las ventanas de evaluacion
    # ANTES de calcular lags — un lag calendario-verdadero de una fila de train
    # cercana a una ventana apuntaria DENTRO de ella (y evaluada como feature).
    grid_lags = grid.copy()
    grid_lags.loc[excluded, 'y'] = float('nan')
    grid_lags = generate_lags(grid_lags, LAGS)
    # Restaurar la y real de las filas de train (solo los LAGS quedan ciegos
    # a las ventanas; el target de entrenamiento es la observacion real)
    grid_lags['y'] = grid['y']

    train_df = grid_lags[~excluded].dropna(subset=['y'])
    if train_df.empty:
        print(f"[XGB Local] {planta} sin historia suficiente tras excluir ventanas de test.")
        return
    # Los lags NaN (hueco real o referencia a ventana enmascarada) se conservan:
    # XGBoost enruta valores faltantes de forma nativa, igual que en test.

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
