"""Tests de la grilla canónica de evaluación (alineación XGB/DL) y de la
idempotencia por completitud de planta.

Contexto: el 21% de las ventanas de evaluación tenía huecos internos; el slice
posicional de XGB sobre la serie cruda evaluaba horas calendario distintas a
las del DL (grilla continua). Ahora TODOS los modelos seleccionan ventanas
sobre la grilla de utils/grid.py.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.utils.grid import make_hourly_grid
from src.ml.utils.eval_window import eval_window_variants, WINDOW_HOURS
from src.ml.utils.features import generate_lags
from src.ml.utils.idempotency import (check_plant_model_done,
                                      mark_plant_model_complete,
                                      mark_run_completed)


def _serie_con_hueco(n_dias=40, capacidad=10.0, hueco=(200, 230)):
    """Serie horaria con producción diurna y un hueco de sensor DENTRO de la
    primera ventana de evaluación (posiciones 200-230 < WINDOW_HOURS=336)."""
    ds = pd.date_range("2021-01-01", periods=n_dias * 24, freq="h")
    y = np.array([capacidad * 0.6 if 8 <= t.hour <= 18 else 0.0 for t in ds])
    df = pd.DataFrame({"unique_id": "p", "ds": ds, "y": y,
                       "potencia_neta_mw": capacidad,
                       "radiacion-global-instantanea": np.where(
                           (ds.hour >= 8) & (ds.hour <= 18), 500.0, 0.0)})
    return df.drop(index=range(*hueco)).reset_index(drop=True)  # hueco real


def test_ventana_sobre_grilla_es_contigua_en_calendario():
    """Con un hueco dentro de la ventana, el slice posicional sobre la serie
    cruda abarca más calendario que 336h; sobre la grilla es exacto."""
    raw = _serie_con_hueco()
    grid = make_hourly_grid(raw, "p")

    span_raw = raw['ds'].iloc[WINDOW_HOURS - 1] - raw['ds'].iloc[0]
    span_grid = grid['ds'].iloc[WINDOW_HOURS - 1] - grid['ds'].iloc[0]
    assert span_raw > pd.Timedelta(hours=WINDOW_HOURS - 1), \
        "el mock debe tener un hueco dentro de la ventana"
    assert span_grid == pd.Timedelta(hours=WINDOW_HOURS - 1), \
        "la grilla debe ser horaria contigua"
    # y de la grilla: NaN en el hueco (jamás imputada)
    assert grid['y'].isna().sum() == 30


def test_grilla_propaga_columnas_de_xgb():
    """Las columnas extra de silver_unified (categóricas, dummies, prior)
    sobreviven a la reindexación; las dummies marcan 1 en filas nuevas."""
    raw = _serie_con_hueco()
    raw['macrozona'] = pd.Categorical(['Norte Chico'] * len(raw))
    raw['pr_regional'] = 0.42
    raw['temp-aire-seco_is_imputed'] = 0.0
    grid = make_hourly_grid(raw, "p")

    assert grid['macrozona'].isna().sum() == 0
    assert (grid['pr_regional'] == 0.42).all()
    nuevas = grid['y'].isna()
    assert (grid.loc[nuevas, 'temp-aire-seco_is_imputed'] == 1.0).all(), \
        "las filas creadas por la grilla tienen clima interpolado -> dummy=1"


def test_lags_de_train_ciegos_a_las_ventanas_de_evaluacion():
    """ANTI-LEAKAGE: con la y de las ventanas enmascarada antes de los lags,
    ninguna fila de train recibe como lag un valor de y evaluado."""
    raw = _serie_con_hueco(hueco=(900, 901))  # hueco fuera de la ventana
    grid = make_hourly_grid(raw, "p")
    capacidad = 10.0
    variants = eval_window_variants(grid['y'], capacidad, dates=grid['ds'])

    excluded = pd.Series(False, index=grid.index)
    for _, start in variants.items():
        excluded.iloc[start:start + WINDOW_HOURS] = True

    masked = grid.copy()
    masked.loc[excluded, 'y'] = np.nan
    lagged = generate_lags(masked, [24, 168])

    # Filas de train hasta 24h despues de una ventana: lag_24 apunta adentro
    # de la ventana -> debe ser NaN (la mascara lo garantiza)
    fin_ventanas = [s + WINDOW_HOURS for s in variants.values()]
    for fin in fin_ventanas:
        zona = lagged.iloc[fin:fin + 24]
        zona_train = zona[~excluded.iloc[fin:fin + 24].values]
        diurna = zona_train[zona_train['ds'].dt.hour.between(8, 18)]
        if len(diurna):
            assert diurna['lag_24'].isna().all(), \
                "lag_24 de una fila de train contiene y de la ventana evaluada"


def test_idempotencia_requiere_completitud_total(tmp_path):
    """Un marker de rollout raw NO basta: la planta está completa solo con el
    marker de completitud (evita perder ventanas estacionales tras una falla)."""
    plant_dir = tmp_path / "lstm" / "Norte" / "Verano" / "p1"
    roll_dir = plant_dir / "rollout7d"
    roll_dir.mkdir(parents=True)
    mark_run_completed(roll_dir, "lstm_rollout7d", "p1", "Verano", {"RMSE": 1.0})

    assert not check_plant_model_done(tmp_path, "lstm", "p1"), \
        "el marker de rollout raw no debe bastar (ventanas estacionales pendientes)"

    mark_plant_model_complete(plant_dir, "lstm", "p1", "Verano")
    assert check_plant_model_done(tmp_path, "lstm", "p1")
