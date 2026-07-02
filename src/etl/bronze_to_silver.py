import os
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_macrozona_by_region(region):
    if pd.isna(region): return "Desconocida"
    region = str(region).lower()
    if any(x in region for x in ["arica", "tarapaca", "antofagasta"]): return "Norte Grande"
    if any(x in region for x in ["atacama", "coquimbo"]): return "Norte Chico"
    if any(x in region for x in ["valparaiso", "metropolitana", "o'higgins", "ohiggins", "maule"]): return "Zona Central"
    if any(x in region for x in ["nuble", "ñuble", "biobio", "araucania", "rios", "lagos"]): return "Zona Sur"
    if any(x in region for x in ["aysen", "magallanes"]): return "Zona Austral"
    return "Desconocida"

def get_macrozona_by_latitud(lat):
    if pd.isna(lat): return "Desconocida"
    if lat > -26.0: return "Norte Grande"
    if -32.1 <= lat <= -26.0: return "Norte Chico"
    if -36.2 <= lat < -32.1: return "Zona Central"
    if -44.0 <= lat < -36.2: return "Zona Sur"
    if lat < -44.0: return "Zona Austral"
    return "Desconocida"

def get_season(month):
    if month in [12, 1, 2]: return 'Verano'
    if month in [3, 4, 5]: return 'Otoño'
    if month in [6, 7, 8]: return 'Invierno'
    if month in [9, 10, 11]: return 'Primavera'
    return 'Desconocida'

def add_cyclical_features(df, date_col='ds'):
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Hora del dia (24)
    hours_in_day = 24
    df['sin_hour'] = np.sin(2 * np.pi * df[date_col].dt.hour / hours_in_day)
    df['cos_hour'] = np.cos(2 * np.pi * df[date_col].dt.hour / hours_in_day)
    
    # Dia del mes (31)
    days_in_month = df[date_col].dt.days_in_month
    df['sin_day_month'] = np.sin(2 * np.pi * df[date_col].dt.day / days_in_month)
    df['cos_day_month'] = np.cos(2 * np.pi * df[date_col].dt.day / days_in_month)
    
    # Dia del año (365)
    days_in_year = 365.25 # Leap year approx
    df['sin_day_year'] = np.sin(2 * np.pi * df[date_col].dt.dayofyear / days_in_year)
    df['cos_day_year'] = np.cos(2 * np.pi * df[date_col].dt.dayofyear / days_in_year)
    
    # Mes del año (12)
    months_in_year = 12
    df['sin_month'] = np.sin(2 * np.pi * df[date_col].dt.month / months_in_year)
    df['cos_month'] = np.cos(2 * np.pi * df[date_col].dt.month / months_in_year)
    
    # Estacion (4)
    df['estacion_año'] = df[date_col].dt.month.apply(get_season)
    season_map = {'Verano': 1, 'Otoño': 2, 'Invierno': 3, 'Primavera': 4, 'Desconocida': 0}
    df['season_num'] = df['estacion_año'].map(season_map)
    df['sin_season'] = np.sin(2 * np.pi * df['season_num'] / 4)
    df['cos_season'] = np.cos(2 * np.pi * df['season_num'] / 4)
    df = df.drop(columns=['season_num'])
    
    return df

def fast_parse_dates(series):
    # Ya no es necesaria, los datos vienen limpios de Bronze.
    return pd.to_datetime(series)

def process_silver(bronze_dir: str | Path, silver_dir: str | Path, write_csv: bool = True):
    bronze_dir = Path(bronze_dir)
    silver_dir = Path(silver_dir)
    silver_dir.mkdir(parents=True, exist_ok=True)
    
    import unicodedata
    import re
    def make_safe(s):
        s = ''.join(c for c in unicodedata.normalize('NFD', str(s).lower().strip()) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[<>:"/\\|?*\']', '', s).replace(' ', '_')
        
    print("[ETL Silver] Cargando Maestros...")
    # 1. Cargar Maestros
    df_maestro_gen = pd.read_parquet(bronze_dir / "maestro_generacion.parquet")
    df_maestro_gen['macrozona'] = df_maestro_gen['region'].apply(get_macrozona_by_region)
    df_maestro_gen['safe_name'] = df_maestro_gen['nombre'].apply(make_safe)
    plant_to_macrozona = dict(zip(df_maestro_gen['safe_name'], df_maestro_gen['macrozona']))
    plant_to_cap = dict(zip(df_maestro_gen['safe_name'], df_maestro_gen['potencia_neta_mw']))
    
    df_maestro_exo = pd.read_parquet(bronze_dir / "maestro_exogenas.parquet")
    df_maestro_exo['macrozona'] = df_maestro_exo['latitud'].apply(get_macrozona_by_latitud)
    col_name_exo = 'nombre_mapped' if 'nombre_mapped' in df_maestro_exo.columns else 'nombre'
    df_maestro_exo['safe_name'] = df_maestro_exo[col_name_exo].apply(make_safe)
    st_to_macrozona = dict(zip(df_maestro_exo['safe_name'], df_maestro_exo['macrozona']))

    # 2. Agrupar Variables Exogenas por Macrozona
    print("[ETL Silver] Consolidando Variables Exogenas por Macrozona...")
    exo_dfs = []
    exo_dir = bronze_dir / "exogenas"
    if exo_dir.exists():
        for var_dir in exo_dir.iterdir():
            if not var_dir.is_dir(): continue
            var_name = var_dir.name
            
            var_dfs = []
            for csv_file in var_dir.glob("*.csv"):
                df_st = pd.read_csv(csv_file, low_memory=False)
                if df_st.empty or 'unique_id' not in df_st.columns: continue

                st_name = df_st['unique_id'].iloc[0]
                macrozona = st_to_macrozona.get(st_name, "Desconocida")
                if macrozona == "Desconocida": continue

                # Dedup defensivo por estacion: duplicados desiguales sesgarian
                # el promedio por macrozona (bug historico de append en Bronze)
                df_st = df_st.drop_duplicates(subset=['unique_id', 'ds'], keep='last')
                df_st['macrozona'] = macrozona
                var_dfs.append(df_st)
                
            if var_dfs:
                # Concatenate all stations for this variable
                df_var = pd.concat(var_dfs, ignore_index=True)
                df_var['ds'] = pd.to_datetime(df_var['ds'], format='ISO8601', errors='coerce')
                # Promedio por Macrozona y Fecha (Hora)
                df_var_mz = df_var.groupby(['macrozona', 'ds'])[var_name].mean().reset_index()
                exo_dfs.append(df_var_mz)

    # Merge all exogenous variables into one dataframe per Macrozona-Date
    if exo_dfs:
        df_exo_consolidated = exo_dfs[0]
        for df_ in exo_dfs[1:]:
            df_exo_consolidated = pd.merge(df_exo_consolidated, df_, on=['macrozona', 'ds'], how='outer')
    else:
        df_exo_consolidated = pd.DataFrame(columns=['macrozona', 'ds'])

    # 3. Procesar Generacion (Plantas Solares)
    print("[ETL Silver] Consolidando Generacion...")
    gen_dfs = []
    gen_dir = bronze_dir / "generacion"
    if gen_dir.exists():
        for csv_file in gen_dir.glob("*.csv"):
            df_gen = pd.read_csv(csv_file, low_memory=False)
            if df_gen.empty or 'unique_id' not in df_gen.columns: continue
            
            plant_name = df_gen['unique_id'].iloc[0]
            macrozona = plant_to_macrozona.get(plant_name, "Desconocida")
            if macrozona == "Desconocida": continue
            
            cap = plant_to_cap.get(plant_name, np.nan)
            
            df_gen = df_gen.rename(columns={'generacion_mwh': 'y'})
            df_gen['ds'] = pd.to_datetime(df_gen['ds'], format='ISO8601', errors='coerce')
            df_gen['macrozona'] = macrozona
            df_gen['potencia_neta_mw'] = cap

            gen_dfs.append(df_gen)

    df_unified = pd.concat(gen_dfs, ignore_index=True) if gen_dfs else pd.DataFrame()

    # Dedup defensivo: una (planta, hora) = UNA fila. El bug de append en Bronze
    # triplico el dataset historicamente (14.4M filas vs 4.4M reales).
    if not df_unified.empty:
        antes = len(df_unified)
        df_unified = df_unified.drop_duplicates(subset=['unique_id', 'ds'], keep='last')
        if antes != len(df_unified):
            print(f"[ETL Silver] WARNING: {antes - len(df_unified)} filas duplicadas "
                  f"(unique_id, ds) eliminadas de Generacion (bronze contaminado).")
    
    # 4. Cruzar Generacion con Variables Exogenas por Macrozona y Fecha
    if not df_unified.empty and not df_exo_consolidated.empty:
        print("[ETL Silver] Cruzando Variables Exogenas...")
        df_unified = pd.merge(df_unified, df_exo_consolidated, on=['macrozona', 'ds'], how='left')

    # 5. Agregar Variables Cíclicas Temporales
    if not df_unified.empty:
        print("[ETL Silver] Generando Features Cíclicos...")
        df_unified = add_cyclical_features(df_unified, date_col='ds')

        # 6. ANTI-LEAKAGE: NO calcular PR = y/capacidad aqui.
        # Esa columna es el target normalizado (corr(PR, y) = 1.0) y alimentarla
        # como feature invalida los benchmarks. El prior regional leakage-free
        # se calcula en la capa ML (src/ml/utils/regional_prior.py) usando
        # exclusivamente plantas de entrenamiento dentro de cada split LOPO.

        # 7. Imputacion de Outliers y NaNs en Exogenas
        print("[ETL Silver] Imputando valores perdidos en exogenas y guardando log...")
        exo_columns = [c for c in df_exo_consolidated.columns if c not in ['macrozona', 'ds']]
        imp_logs = []
        
        # Ordenamos antes de imputar para que ffill tenga sentido cronológico
        df_unified = df_unified.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        
        for col in exo_columns:
            if col in df_unified.columns:
                dummy_col = f"{col}_is_imputed"
                df_unified[dummy_col] = df_unified[col].isna().astype(int)
                
                missing_mask = df_unified[dummy_col] == 1
                if missing_mask.any():
                    imp_df = df_unified[missing_mask][['unique_id', 'ds', 'macrozona']].copy()
                    imp_df['variable_imputada'] = col
                    imp_logs.append(imp_df)
                    
                # Rellenar arrastrando el último valor (ffill) y luego bfill para evitar sesgos a 0.0 en clima
                df_unified[col] = df_unified.groupby('macrozona')[col].ffill()
                df_unified[col] = df_unified.groupby('macrozona')[col].bfill()
                df_unified[col] = df_unified[col].fillna(0.0) # Fallback
                
        if imp_logs:
            df_log = pd.concat(imp_logs, ignore_index=True)
            df_log.to_csv(silver_dir / "imputation_log.csv", index=False)
            print(f"[ETL Silver] Se detectaron {len(df_log)} registros exogenos imputados (ffill/bfill).")
        
        # Guardar en Parquet (CSV opcional: duplica ~4.4GB y varios minutos)
        print(f"[ETL Silver] Escribiendo Salidas Unified ({len(df_unified)} registros)...")
        df_unified.to_parquet(silver_dir / "silver_unified.parquet", index=False)
        if write_csv:
            df_unified.to_csv(silver_dir / "silver_unified.csv", index=False)

    print("[ETL Silver] ¡Capa Silver Finalizada!")
    return True

if __name__ == "__main__":
    process_silver("data/bronze", "data/silver")
