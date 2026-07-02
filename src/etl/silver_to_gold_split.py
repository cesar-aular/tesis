import os
import pandas as pd
from pathlib import Path

def process_gold_split(silver_dir: str | Path, gold_dir: str | Path):
    silver_dir = Path(silver_dir)
    gold_dir = Path(gold_dir)
    
    gold_dir.mkdir(parents=True, exist_ok=True)
    
    unified_path = silver_dir / "silver_unified.parquet"
    if not unified_path.exists():
        print("[ETL Gold] No se encontro silver_unified.parquet")
        return
        
    print("[ETL Gold] Cargando Silver Data para Particion...")
    df = pd.read_parquet(unified_path)
    
    if df.empty:
        return
        
    # Particion por Macrozona y Estacion del aAo (LOPO preparation)
    # La Capa Gold es EXCLUSIVAMENTE para los Test Sets (Leave-One-Plant-Out).
    # Guardamos cada planta como un archivo individual de test.
    # Estructura: data/gold/{macrozona}/{estacion_año}/{unique_id}_test.parquet
    
    estacion_col = next((c for c in df.columns if 'estacion' in c and 'a' in c), 'estacion_año')
    
    for macrozona, group_mz in df.groupby('macrozona'):
        mz_safe = macrozona.replace(" ", "_").lower()
        
        for estacion, group_es in group_mz.groupby(estacion_col):
            es_safe = estacion.replace(" ", "_").lower()
            
            out_dir = gold_dir / mz_safe / es_safe
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Iterar por cada planta para generar su archivo de validación Hold-Out (LOPO)
            for unique_id, group_planta in group_es.groupby('unique_id'):
                out_file = out_dir / f"{unique_id}_test.parquet"
                group_planta.to_parquet(out_file, index=False)
            
    print("[ETL Gold] Particionamiento LOPO finalizado exitosamente!")
    
if __name__ == "__main__":
    process_gold_split("data/silver", "data/gold")
