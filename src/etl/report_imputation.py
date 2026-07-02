import pandas as pd
from pathlib import Path

def generate_imputation_report():
    silver_path = Path("data/silver/silver_unified.parquet")
    if not silver_path.exists():
        print("Error: No se encontró silver_unified.parquet")
        return
        
    print("Cargando Silver Data...")
    df = pd.read_parquet(silver_path)
    df['year'] = df['ds'].dt.year
    
    total_records = len(df)
    records_2024 = (df['year'] == 2024).sum()
    
    print(f"\\n{'='*50}")
    print(f" RESUMEN DE INTEGRIDAD DE DATOS (CAPA SILVER)")
    print(f"{'='*50}")
    print(f"Total de Registros (Horas-Planta): {total_records:,}")
    print(f"Registros en 2024: {records_2024:,}")
    
    print(f"\\n{'='*50}")
    print(f" DETALLE DE IMPUTACIÓN POR VARIABLE (FFILL/BFILL)")
    print(f"{'='*50}")
    
    imputed_cols = [c for c in df.columns if c.endswith('_is_imputed')]
    
    for imp_col in imputed_cols:
        var_name = imp_col.replace('_is_imputed', '')
        
        # Overall imputation
        total_imputed = df[imp_col].sum()
        pct_imputed = (total_imputed / total_records) * 100
        
        # 2024 specific imputation
        df_2024 = df[df['year'] == 2024]
        imputed_2024 = df_2024[imp_col].sum()
        pct_2024 = (imputed_2024 / records_2024) * 100 if records_2024 > 0 else 0
        
        print(f"\\nVariable: {var_name.upper()}")
        print(f"  -> Total Imputado (Histórico): {int(total_imputed):,} registros ({pct_imputed:.2f}%)")
        print(f"  -> Imputado Específicamente en 2024: {int(imputed_2024):,} registros ({pct_2024:.2f}%)")
        
    print(f"\\n{'='*50}")
    print(f" ARQUITECTURA GOLD (TEST LOPO)")
    print(f"{'='*50}")
    
    gold_dir = Path("data/gold")
    if gold_dir.exists():
        total_test_files = len(list(gold_dir.rglob("*_test.parquet")))
        print(f"Total de archivos de validación LOPO generados: {total_test_files}")
    
if __name__ == "__main__":
    generate_imputation_report()
