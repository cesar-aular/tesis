import pandas as pd

def generate_lags(df: pd.DataFrame, lags: list[int], target_col: str = 'y') -> pd.DataFrame:
    """
    Genera lags just-in-time de forma vectorizada agrupada por unique_id.
    Esto permite a XGBoost entrenar sin contaminar el dataset principal.
    """
    df_lags = df.copy()
    
    # Sort to ensure chronological order
    df_lags = df_lags.sort_values(by=['unique_id', 'ds'])
    
    for lag in lags:
        df_lags[f'lag_{lag}'] = df_lags.groupby('unique_id')[target_col].shift(lag)
        
    return df_lags
