import pandas as pd
from typing import Tuple

def load_lopo_split(silver_df: pd.DataFrame, test_planta: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dada la base Silver completa y el identificador de la planta objetivo,
    devuelve el train_df (todas las plantas excepto la objetivo) y
    un test_df (solo la planta objetivo).
    
    Esta función garantiza que no haya Leakage de la planta objetivo 
    hacia el set de entrenamiento.
    """
    train_df = silver_df[silver_df["unique_id"] != test_planta].copy()
    test_df = silver_df[silver_df["unique_id"] == test_planta].copy()
    
    return train_df, test_df
