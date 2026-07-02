import sys
sys.path.insert(0, 'c:/Users/cesar/Desktop/code/tesis-final')
import pandas as pd
from pathlib import Path
from src.etl.landing_to_bronze_generation import parse_raw_csv_gen

df = pd.read_parquet('data/bronze/maestro_generacion.parquet')
valid_plants_lower = set(str(n).lower().strip() for n in df['nombre'].unique() if pd.notna(n))

try:
    res = parse_raw_csv_gen(Path('data/landing/generacion_raw/202001GxSEN.csv'), valid_plants_lower)
    print("Dict keys length:", len(res))
    if list(res.keys()):
        print("First df shape:", res[list(res.keys())[0]].shape)
except Exception as e:
    print("ERROR:", e)
