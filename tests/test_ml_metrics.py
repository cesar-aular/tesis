"""Tests de las métricas de pronóstico (coherencia con series solares)."""
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.utils.metrics import calculate_metrics


def test_metricas_puntuales_valores_conocidos():
    y = np.array([0.0, 0.0, 10.0, 20.0])   # dos horas nocturnas + dos productivas
    p = np.array([0.0, 2.0, 8.0, 24.0])

    m = calculate_metrics(y, p)

    # MAE = (0 + 2 + 2 + 4)/4 = 2 ; MBE = (0+2-2+4)/4 = 1 (sobrepronóstico)
    assert abs(m["MAE"] - 2.0) < 1e-9
    assert abs(m["MBE"] - 1.0) < 1e-9
    # MAPE solo horas productivas: (2/10 + 4/20)/2 = 20%
    assert abs(m["MAPE"] - 20.0) < 1e-9
    # WMAPE = (0+2+2+4)/(0+0+10+20) = 8/30 = 26.67%
    assert abs(m["WMAPE"] - 100 * 8 / 30) < 1e-6
    # R² = 1 - SS_res/SS_tot
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    assert abs(m["R2"] - (1 - ss_res / ss_tot)) < 1e-9


def test_mape_indefinida_en_ventana_toda_cero():
    """Una rampa 100% en cero no puede fabricar porcentajes: MAPE/WMAPE = NaN,
    R² = NaN (sin varianza), y sMAPE definida en 0 para el caso 0/0."""
    y = np.zeros(24)
    p = np.zeros(24)
    m = calculate_metrics(y, p)
    assert np.isnan(m["MAPE"])
    assert np.isnan(m["WMAPE"])
    assert np.isnan(m["R2"])
    assert np.isnan(m["rRMSE"]), "media cero -> rRMSE indefinido, jamas 0.0 'perfecto'"
    assert m["sMAPE"] == 0.0
    assert m["RMSE"] == 0.0


def test_cobertura_y_pinball_con_bandas():
    y = np.array([5.0, 10.0, 15.0, 50.0])
    p = y.copy()
    lo = np.array([0.0, 8.0, 10.0, 60.0])   # la última banda NO contiene y
    hi = np.array([10.0, 12.0, 20.0, 70.0])
    m = calculate_metrics(y, p, y_lo=lo, y_hi=hi)
    assert abs(m["Coverage_90"] - 0.75) < 1e-9
    assert m["Pinball_P05"] >= 0 and m["Pinball_P95"] >= 0
