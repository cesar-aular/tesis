import pandas as pd
from pathlib import Path

silver_dir = Path("data/silver")
unified_path = silver_dir / "silver_unified.parquet"

if not unified_path.exists():
    print(f"El dataset {unified_path} no existe aun.")
else:
    df = pd.read_parquet(unified_path)
    
    print("=================== MINI-INFORME CAPA SILVER ===================")
    print(f"Total de Puntos de Datos (Filas): {len(df):,}")
    print(f"Columnas disponibles: {df.columns.tolist()}")
    
    plantas_totales = df['unique_id'].nunique()
    print(f"\nTotal de Plantas Solares Procesadas: {plantas_totales}")
    
    print("\nDesglose de Plantas por Macrozona:")
    plantas_mz = df[['unique_id', 'macrozona']].drop_duplicates()
    print(plantas_mz['macrozona'].value_counts().to_string())
    
    print("\nEstadísticas del Performance Ratio (PR):")
    if 'PR' in df.columns:
        print(df['PR'].describe().to_string())
    else:
        print("La columna PR no se encontró.")
        
    print("\nEjemplo de las Primeras 5 filas (Snippet):")
    pd.set_option('display.max_columns', None)
    print(df.head(5))
    print("================================================================")
