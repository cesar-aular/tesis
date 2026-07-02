import pytest
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.ml.utils.data_loader import load_lopo_split
except ImportError:
    load_lopo_split = None

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
