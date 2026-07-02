import pytest
import os
import sys
from pathlib import Path
import tempfile
import json
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.ml.utils.idempotency import check_run_completed, mark_run_completed
except ImportError:
    check_run_completed = None
    mark_run_completed = None

def test_ml_idempotency_markers():
    """
    Verifica que la lógica de idempotencia del ML pipeline
    detecte correctamente si un fold/planta/estacion ya fue procesado.
    """
    if check_run_completed is None:
        pytest.fail("check_run_completed no está implementado aún.")
        
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)
        
        planta = "planta_a"
        estacion = "verano"
        modelo = "xgb_local"
        
        # Check initially false
        assert not check_run_completed(results_dir, modelo, planta, estacion)
        
        # Mark completed
        mark_run_completed(results_dir, modelo, planta, estacion, metrics={"rmse": 1.0})
        
        # Check true
        assert check_run_completed(results_dir, modelo, planta, estacion)
        
        # Verify JSON was written
        marker_file = results_dir / f"{modelo}_{planta}_{estacion}_done.json"
        assert marker_file.exists()
        
        with open(marker_file, 'r') as f:
            data = json.load(f)
            assert data["metrics"]["rmse"] == 1.0
