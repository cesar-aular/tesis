import pandas as pd
import sys

def main():
    try:
        df = pd.read_parquet('data/silver/silver_unified.parquet')
        
        print("\n=== REPORTE FINAL CAPA SILVER ===")
        print(f"Total de registros: {len(df):,}")
        print(f"Numero total de plantas solares: {df['unique_id'].nunique()}")
        print("\nPlantas por Macrozona:")
        plantas_mz = df.groupby('macrozona')['unique_id'].nunique().sort_values(ascending=False)
        for mz, count in plantas_mz.items():
            print(f"  - {mz}: {count} plantas")
            
        print("\nPeriodo temporal:")
        print(f"  Desde: {df['ds'].min()}")
        print(f"  Hasta: {df['ds'].max()}")
        
        print("\nColumnas finales en Silver:")
        for col in df.columns:
            print(f"  - {col} ({df[col].dtype})")
            
        print("\nSample de datos (head):")
        pd.set_option('display.max_columns', None)
        print(df.head(3))
        print("=================================\n")
    except Exception as e:
        print(f"Error reading silver unified: {e}")

if __name__ == '__main__':
    main()
