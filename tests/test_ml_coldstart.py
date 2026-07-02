import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.ml.utils.coldstart import generate_synthetic_context
except ImportError:
    generate_synthetic_context = None

def test_synthetic_context_generation():
    """
    Verifica que el contexto generado para Deep Learning se base puramente
    en la heurística (Capacidad * PR * Perfil) y no contenga data real (Leakage).
    """
    if generate_synthetic_context is None:
        pytest.fail("generate_synthetic_context no está implementado aún.")
        
    # Test plant that is missing history
    test_plant_id = "planta_nueva"
    capacidad_mw = 50.0
    pr_regional = 0.8
    
    # 24 hours of data
    dates = pd.date_range("2021-01-01", periods=24, freq="h")
    
    # Simulated regional profile (normalized 0 to 1)
    perfil_regional = np.sin(np.linspace(0, np.pi, 24))
    
    # Context should be purely heuristic
    synthetic_df = generate_synthetic_context(
        unique_id=test_plant_id,
        dates=dates,
        capacity_mw=capacidad_mw,
        pr=pr_regional,
        profile=perfil_regional
    )
    
    assert len(synthetic_df) == 24
    assert synthetic_df["unique_id"].iloc[0] == test_plant_id
    
    # Check that max generation doesn't exceed Capacity * PR
    max_gen = synthetic_df["y"].max()
    assert max_gen <= (capacidad_mw * pr_regional) + 0.1 # small tolerance
