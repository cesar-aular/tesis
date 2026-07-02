import pandas as pd
df = pd.read_parquet('data/bronze/maestro_generacion.parquet')
print(df['nombre'].head(10).tolist())
