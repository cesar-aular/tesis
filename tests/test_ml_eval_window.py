import pandas as pd
import numpy as np
import sys
from pathlib import Path
import pytest
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.ml.utils.eval_window import find_operational_start
except ImportError:
    find_operational_start = None


def _serie(horas_cero: int, dias_produccion: int = 10, capacidad: float = 10.0):
    """Serie sintetica: rampa de puesta en marcha (ceros) + produccion diurna normal."""
    n = horas_cero + dias_produccion * 24
    ds = pd.date_range("2021-01-01", periods=n, freq="h")
    y = np.zeros(n)
    for i in range(horas_cero, n):
        hora = ds[i].hour
        if 8 <= hora <= 18:  # produccion solar diurna
            y[i] = capacidad * 0.6
    return pd.Series(y, index=ds)


def test_operational_start_salta_la_rampa():
    """La ventana operacional debe comenzar DESPUES de la rampa de ceros."""
    if find_operational_start is None:
        pytest.fail("find_operational_start no implementado aun.")

    horas_cero = 336  # dos semanas de comisionamiento en cero
    y = _serie(horas_cero)
    start = find_operational_start(y, capacity=10.0)

    assert start is not None
    assert start >= horas_cero, f"start={start} cae dentro de la rampa de ceros (<{horas_cero})"
    # y no absurdamente tarde: dentro de las primeras 48h de produccion real
    assert start <= horas_cero + 48


def test_operational_start_none_si_nunca_produce():
    """Si la planta jamas produce de forma sostenida, no hay ventana operacional."""
    if find_operational_start is None:
        pytest.fail("find_operational_start no implementado aun.")

    y = pd.Series(np.zeros(1000), index=pd.date_range("2021-01-01", periods=1000, freq="h"))
    assert find_operational_start(y, capacity=10.0) is None


def test_operational_start_desde_cero_si_no_hay_rampa():
    """Planta que produce desde el inicio: la ventana operacional parte al comienzo."""
    if find_operational_start is None:
        pytest.fail("find_operational_start no implementado aun.")

    y = _serie(horas_cero=0)
    start = find_operational_start(y, capacity=10.0)
    assert start is not None and start <= 24
