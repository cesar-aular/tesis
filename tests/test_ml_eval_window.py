import pandas as pd
import numpy as np
import sys
from pathlib import Path
import pytest
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.ml.utils.eval_window import find_operational_start, eval_window_variants
except ImportError:
    find_operational_start = None
    eval_window_variants = None


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


def test_variantes_estacionales_cubren_las_cuatro_estaciones():
    """Una serie de 2 años debe producir ventanas de evaluacion en las 4 estaciones."""
    if eval_window_variants is None:
        pytest.fail("eval_window_variants no implementado aun.")

    # 2 años de produccion diurna continua desde el 1 de enero (Verano en Chile)
    n = 2 * 365 * 24
    ds = pd.date_range("2021-01-01", periods=n, freq="h")
    y = pd.Series([10.0 * 0.6 if 8 <= t.hour <= 18 else 0.0 for t in ds], index=ds)

    variants = eval_window_variants(y, capacity=10.0, dates=ds)

    for estacion in ["_verano", "_otono", "_invierno", "_primavera"]:
        assert estacion in variants, f"Falta la ventana estacional {estacion}"
        start = variants[estacion]
        # la ventana debe caber y comenzar en la estacion correcta
        assert start + (168 + 7 * 24) <= n
    # el mes de inicio corresponde a la estacion
    assert ds[variants["_invierno"]].month in (6, 7, 8)
    assert ds[variants["_primavera"]].month in (9, 10, 11)
    assert ds[variants["_verano"]].month in (12, 1, 2)
    assert ds[variants["_otono"]].month in (3, 4, 5)
