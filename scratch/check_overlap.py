import pandas as pd
df = pd.read_parquet('data/bronze/maestro_generacion.parquet')
valid_plants_lower = set(str(n).lower().strip() for n in df['nombre'].unique() if pd.notna(n))

df_raw = pd.read_csv('data/landing/generacion_raw/202001GxSEN.csv', sep=';', low_memory=False)
raw_plants = set(str(n).lower().strip() for n in df_raw['Central'].unique() if pd.notna(n))

intersection = valid_plants_lower.intersection(raw_plants)
print("Valid plants in maestro:", len(valid_plants_lower))
print("Raw plants in 202001:", len(raw_plants))
print("Intersection:", len(intersection))
if len(intersection) > 0:
    print(list(intersection)[:5])
