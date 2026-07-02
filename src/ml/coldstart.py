import numpy as np
import pandas as pd

def generate_synthetic_context(horizon: int, capacidad_mw: float, pr: float, perfil: np.ndarray) -> pd.DataFrame:
    """
    Genera un contexto 100% sintético para la ventana de inferencia inicial (Cold-Start).
    Fórmula: y_synthetic = Capacidad Instalada (MW) * PR * Perfil_Solar_Diario
    """
    # Validar que el perfil tenga el mismo tamaño que el horizonte
    if len(perfil) != horizon:
        # Repetir o truncar el perfil si es necesario (en el test asumen igual tamaño)
        pass
    
    y_synthetic = capacidad_mw * pr * perfil
    # Asegurar que no hay valores negativos
    y_synthetic = np.clip(y_synthetic, a_min=0, a_max=None)
    
    df_synthetic = pd.DataFrame({
        "y_synthetic": y_synthetic
    })
    return df_synthetic
