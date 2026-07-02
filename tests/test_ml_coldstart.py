"""Tests del contexto sintético Cold-Start y la física de post-proceso.

Cubre la semilla REAL de producción (src/ml/utils/dl_test.synthetic_context):
- normalizada por capacidad (y_norm = PR × perfil, en [0, 1])
- perfil guiado por la radiación del contexto (exógena legítima)
- fallback a campana horaria cuando la radiación no aporta señal
- JAMÁS usa la generación real de la planta (anti-leakage)
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.utils.dl_test import synthetic_context, _apply_physics
from src.ml.utils.dl_models import normalize_target


def _hist(n=48, ghi=None):
    ds = pd.date_range("2021-06-01", periods=n, freq="h")
    df = pd.DataFrame({
        "unique_id": "planta_nueva", "ds": ds,
        # y real presente a propósito: la semilla DEBE ignorarla/sobrescribirla
        "y": np.random.RandomState(0).uniform(0, 40, n),
        "potencia_neta_mw": 50.0,
    })
    if ghi is not None:
        df["radiacion-global-instantanea"] = ghi
    return df


def test_semilla_sigue_la_radiacion_y_es_normalizada():
    """Con señal solar real, el perfil es GHI/GHI_max y el techo es PR."""
    horas = np.arange(48) % 24
    ghi = np.where((horas >= 8) & (horas <= 18), 800 * np.sin((horas - 8) / 10 * np.pi), 0.0)
    hist = _hist(ghi=ghi)
    pr = 0.6

    out = synthetic_context(hist, pr)

    esperado = pr * np.clip(ghi / ghi.max(), 0, 1)
    np.testing.assert_allclose(out["y"].to_numpy(), esperado, atol=1e-9)
    assert out["y"].max() <= pr + 1e-9, "y_norm no puede superar el PR regional"
    assert (out["y"] >= 0).all()


def test_semilla_no_usa_generacion_real():
    """LEAKAGE: ningún valor de la y real puede sobrevivir en la semilla."""
    ghi = np.tile(np.concatenate([np.zeros(8), np.linspace(0, 700, 8), np.zeros(8)]), 2)
    hist = _hist(ghi=ghi)
    y_real = hist["y"].copy()
    out = synthetic_context(hist, 0.5)
    assert not np.allclose(out["y"].to_numpy(), y_real.to_numpy()), \
        "la semilla contiene la generación real (leakage)"


def test_semilla_fallback_campana_sin_radiacion():
    """Sin señal de radiación (hueco imputado a ~0), usa la campana horaria."""
    hist = _hist(ghi=np.zeros(48))  # gmax <= 50 -> sin señal
    pr = 0.4
    out = synthetic_context(hist, pr)
    y = out.set_index(out["ds"].dt.hour)["y"]
    assert y.loc[12].max() > y.loc[0].max(), "la campana debe tener pico al mediodía"
    assert out["y"].max() <= pr + 1e-9


def test_normalize_target_solo_usa_capacidad_estatica():
    """y_norm = y / capacidad; el DataFrame original no se muta."""
    df = pd.DataFrame({"unique_id": ["a", "a", "b"],
                       "y": [10.0, 20.0, 3.0],
                       "potencia_neta_mw": [50.0, 50.0, 3.0]})
    out = normalize_target(df)
    np.testing.assert_allclose(out["y"].to_numpy(), [0.2, 0.4, 1.0])
    assert df["y"].tolist() == [10.0, 20.0, 3.0], "normalize_target mutó el original"


def test_mascara_nocturna_astronomica():
    """A las 02:00 la predicción se fuerza a 0 aunque la radiación venga
    imputada alta (huecos largos interpolados no deben crear producción)."""
    ds = pd.date_range("2021-06-01 00:00", periods=6, freq="h")  # horas 0..5
    futr = pd.DataFrame({"ds": ds, "radiacion-global-instantanea": 500.0})
    preds = pd.DataFrame({"m-median": [5.0] * 6})
    out = _apply_physics(preds, futr, ["m-median"])
    assert (out.loc[:4, "m-median"] == 0).all(), \
        "horas 0-4 son noche astronómica: predicción debe ser 0"
    assert out.loc[5, "m-median"] == 5.0, "05:00+ se rige por la radiación"
