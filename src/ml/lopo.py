import pandas as pd

def prepare_lopo_splits(df: pd.DataFrame, target_plant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara los conjuntos de entrenamiento y prueba para Leave-One-Plant-Out (LOPO).
    El conjunto de prueba (test_df) contendrá SOLO los datos de la planta objetivo.
    El conjunto de entrenamiento (train_df) contendrá los datos de TODAS LAS DEMÁS plantas (N-1),
    asegurando que haya cero filtración (leakage) de la planta objetivo al entrenamiento.
    """
    test_df = df[df["unique_id"] == target_plant].copy()
    train_df = df[df["unique_id"] != target_plant].copy()
    
    return train_df, test_df
