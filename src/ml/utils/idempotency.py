import json
from pathlib import Path
from typing import Dict, Any

def get_marker_path(results_dir: Path | str, model_name: str, planta: str, estacion: str) -> Path:
    results_dir = Path(results_dir)
    return results_dir / f"{model_name}_{planta}_{estacion}_done.json"

def check_run_completed(results_dir: Path | str, model_name: str, planta: str, estacion: str) -> bool:
    """
    Revisa si un modelo ya fue evaluado para una planta y estación específica.
    """
    marker = get_marker_path(results_dir, model_name, planta, estacion)
    return marker.exists()

def mark_run_completed(results_dir: Path | str, model_name: str, planta: str, estacion: str, metrics: Dict[str, Any]):
    """
    Crea un JSON de completitud indicando que la iteración LOPO finalizó con éxito.
    Contiene las métricas como respaldo.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    marker = get_marker_path(results_dir, model_name, planta, estacion)
    
    data = {
        "model": model_name,
        "planta": planta,
        "estacion": estacion,
        "metrics": metrics
    }
    
    with open(marker, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
