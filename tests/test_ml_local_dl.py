import numpy as np
import pandas as pd
import sys
from pathlib import Path
import pytest
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.ml.utils.dl_local import get_local_train_df
    from src.ml.utils.eval_window import eval_window_variants, WINDOW_HOURS
except ImportError:
    get_local_train_df = None


def _grid(n_dias: int = 400, capacidad: float = 10.0):
    """Grilla horaria sintetica con produccion diurna y algunos huecos de y."""
    n = n_dias * 24
    ds = pd.date_range("2021-01-01", periods=n, freq="h")
    y = np.array([capacidad * 0.6 if 8 <= t.hour <= 18 else 0.0 for t in ds])
    y[1000:1010] = np.nan  # hueco real de sensor
    return pd.DataFrame({"unique_id": "p", "ds": ds, "y": y,
                         "potencia_neta_mw": capacidad})


def test_train_local_excluye_todas_las_ventanas_de_evaluacion():
    """ANTI train-on-test: ninguna fila de las ventanas (raw/operacional/
    estacionales) puede aparecer en el set de entrenamiento local."""
    if get_local_train_df is None:
        pytest.fail("get_local_train_df no implementado aun.")

    grid = _grid()
    variants = eval_window_variants(grid['y'].reset_index(drop=True), 10.0,
                                    dates=grid['ds'])
    assert len(variants) >= 4, "el mock deberia producir varias ventanas"

    train = get_local_train_df(grid, variants)

    ds_train = set(train['ds'])
    for suffix, start in variants.items():
        window_ds = set(grid['ds'].iloc[start:start + WINDOW_HOURS])
        overlap = ds_train & window_ds
        assert not overlap, (f"LEAKAGE train-on-test: {len(overlap)} filas de la "
                             f"ventana '{suffix or 'raw'}' estan en el train local.")


def test_train_local_no_imputa_target():
    """Las filas con y faltante se DESCARTAN (jamas se imputan para entrenar)."""
    if get_local_train_df is None:
        pytest.fail("get_local_train_df no implementado aun.")

    grid = _grid()
    variants = eval_window_variants(grid['y'].reset_index(drop=True), 10.0,
                                    dates=grid['ds'])
    train = get_local_train_df(grid, variants)
    assert not train['y'].isna().any(), "el train local contiene y imputado/NaN"
