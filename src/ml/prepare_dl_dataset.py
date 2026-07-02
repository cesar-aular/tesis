import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

def prepare_dl_dataset():
    print("[DL Dataset] Generando dataset optimizado para Deep Learning...")
    silver_path = Path("data/silver/silver_unified.parquet")
    if not silver_path.exists():
        print("[DL Dataset] ERROR: No se encontrA3 silver_unified.parquet")
        return
        
    df = pd.read_parquet(silver_path)
    
    # Manejar nulos si existieran para evitar errores en PyTorch
    df = df.fillna(0)
    
    le_macrozona = LabelEncoder()
    le_estacion = LabelEncoder()
    
    df['macrozona_idx'] = le_macrozona.fit_transform(df['macrozona'].astype(str))
    df['estacion_idx'] = le_estacion.fit_transform(df['estacion_a\u00f1o'].astype(str))
    
    # Save encoders to models/utils so they can be reused at inference
    out_dir = Path("models/utils")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(le_macrozona, out_dir / "le_macrozona.pkl")
    joblib.dump(le_estacion, out_dir / "le_estacion.pkl")
    
    # Cast to lower precision for memory
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    
    # Reordenar para limpieza (botar columnas de texto irrelevantes para DL)
    dl_columns = [
        'unique_id', 'ds', 'y', 
        'macrozona_idx', 'estacion_idx', 'potencia_neta_mw',
        'humedad-relativa', 'radiacion-global-instantanea', 'temp-aire-seco',
        'sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'sin_season', 'cos_season'
    ]
    df_dl = df[dl_columns].copy()
    
    out_path = Path("data/silver/silver_dl.parquet")
    df_dl.to_parquet(out_path)
    print(f"[DL Dataset] Guardado existosamente en {out_path} con shape {df_dl.shape}")

if __name__ == "__main__":
    prepare_dl_dataset()
