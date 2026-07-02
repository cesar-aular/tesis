import pandas as pd
import numpy as np

def generate_synthetic_context(unique_id: str, dates: pd.DatetimeIndex, 
                               capacity_mw: float, pr: float, 
                               profile: np.ndarray) -> pd.DataFrame:
    """
    Genera un contexto sintético basado en heurísticas regionales.
    Este contexto se usa para "inicializar" la ventana de los modelos Deep Learning 
    en un escenario Cold-Start estricto, sin incurrir en Data Leakage.
    
    unique_id: ID de la planta
    dates: Fechas (horas) que conformarán la ventana de contexto
    capacity_mw: Capacidad instalada de la planta
    pr: Performance Ratio de la macrozona (0 a 1)
    profile: Perfil horario normalizado de generación (array del mismo largo que dates)
    """
    
    # y = capacity * PR * profile_factor
    # Evitar valores negativos
    y_synth = np.clip(capacity_mw * pr * profile, a_min=0, a_max=None)
    
    df = pd.DataFrame({
        "unique_id": unique_id,
        "ds": dates,
        "y": y_synth
    })
    
    return df
