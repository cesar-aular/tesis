import pytest
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.ml.utils.data_loader import load_lopo_split
except ImportError:
    load_lopo_split = None

try:
    from src.ml.utils.regional_prior import compute_regional_pr, lookup_regional_pr
except ImportError:
    compute_regional_pr = None
    lookup_regional_pr = None

def test_lopo_leakage_no_target_in_train():
    """
    Verifica que al cargar la data para el ciclo LOPO,
    la planta objetivo no aparezca jamás en el set de entrenamiento.
    """
    if load_lopo_split is None:
        pytest.fail("load_lopo_split no implementado aún.")
        
    # Mock data to simulate silver_unified.parquet
    silver_mock = pd.DataFrame({
        "unique_id": ["planta_a", "planta_a", "planta_b", "planta_b", "planta_c", "planta_c"],
        "ds": pd.date_range("2021-01-01", periods=6),
        "y": [10, 11, 20, 21, 30, 31],
        "macrozona": ["norte", "norte", "norte", "norte", "sur", "sur"]
    })
    
    # Mock test file path from Gold
    test_planta = "planta_c"
    
    train_df, test_df = load_lopo_split(silver_mock, test_planta)
    
    # Check 1: Target plant is completely absent from train_df
    assert test_planta not in train_df["unique_id"].unique(), f"Leakage detectado: {test_planta} sigue en train_df."
    
    # Check 2: train_df contains other plants
    assert "planta_a" in train_df["unique_id"].unique()
    assert "planta_b" in train_df["unique_id"].unique()
    
    # Check 3: test_df contains only the target plant
    assert (test_df["unique_id"] == test_planta).all()


def test_regional_pr_is_leakage_free():
    """
    El prior regional de eficiencia (PR) debe calcularse EXCLUSIVAMENTE con las
    plantas de entrenamiento: el y de la planta objetivo no puede influir en el
    valor que recibe como feature. (Lección: PR = y/capacidad fue el target
    disfrazado, corr=1.0 -> benchmark inválido.)
    """
    if compute_regional_pr is None:
        pytest.fail("compute_regional_pr no implementado aún.")

    silver_mock = pd.DataFrame({
        "unique_id": ["a"] * 4 + ["b"] * 4 + ["target"] * 4,
        "ds": list(pd.date_range("2021-01-01", periods=4, freq="h")) * 3,
        "y": [5, 5, 5, 5, 10, 10, 10, 10, 999, 999, 999, 999],
        "potencia_neta_mw": [10.0] * 4 + [20.0] * 4 + [10.0] * 4,
        "macrozona": ["norte"] * 12,
        "estacion_año": ["Verano"] * 12,
    })

    train_df, test_df = load_lopo_split(silver_mock, "target")
    table = compute_regional_pr(train_df)

    # eficiencia por planta: a = 5/10 = 0.5 ; b = 10/20 = 0.5 -> prior = 0.5
    pr = lookup_regional_pr(table, "norte", "Verano")
    assert abs(pr - 0.5) < 1e-9, f"Prior esperado 0.5, obtenido {pr}"

    # El y=999 (absurdo) de la planta objetivo NO debe mover el prior:
    # si lo hiciera, el valor crudo de la tabla seria >> 0.5 (999/10 = 99.9 en la media).
    table_with_leak = compute_regional_pr(silver_mock)  # calculo INCORRECTO a proposito
    raw_leaky = float(table_with_leak.loc[
        (table_with_leak['macrozona'] == 'norte') &
        (table_with_leak['estacion_año'] == 'Verano'), 'pr_regional'].iloc[0])
    assert raw_leaky > 30, "sanity: el mock deberia disparar el prior si hubiera leakage"

    # Fallback razonable cuando no existe la combinacion macrozona/estacion
    fallback = lookup_regional_pr(table, "zona_inexistente", "Invierno")
    assert 0.0 < fallback <= 1.0
