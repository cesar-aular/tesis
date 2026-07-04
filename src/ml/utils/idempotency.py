import json
from pathlib import Path
from typing import Dict, Any


def check_plant_model_done(results_dir: Path | str, model_name: str, planta: str) -> bool:
    """
    Idempotencia real del pipeline LOPO: un modelo/planta está completo SOLO
    si existe su marker de completitud de planta, escrito al terminar TODAS
    las ventanas (raw + operacional + estacionales). Antes se usaba el marker
    del rollout raw: una falla a mitad de las ventanas estacionales dejaba a
    la planta "completa" con estacionales faltantes para siempre.

    Estructura: results/{strategy}/{model}/{macrozona}/{estacion}/{planta}/
                {model}_complete_{planta}_*_done.json
    """
    base = Path(results_dir) / model_name
    if not base.exists():
        return False
    pattern = f"*/*/{planta}/{model_name}_complete_{planta}_*_done.json"
    return any(base.glob(pattern))


def mark_plant_model_complete(plant_dir: Path | str, model_name: str,
                              planta: str, estacion: str):
    """Marca que TODAS las ventanas del modelo para la planta terminaron."""
    mark_run_completed(plant_dir, f"{model_name}_complete", planta, estacion,
                       {"status": "all_windows_done"})

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
