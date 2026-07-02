import os
import pandas as pd
from pathlib import Path
import unicodedata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

def clean_column_names(df):
    def clean_str(s):
        s = str(s).lower().strip()
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        s = s.replace(' ', '_').replace('[', '').replace(']', '')
        return s
    df.columns = [clean_str(c) for c in df.columns]
    return df

def parse_raw_csv_exo(csv_file, valid_stations_names, valid_stations_codes, code_to_name, var_name):
    import pandas as pd
    import re
    
    def clean_str_local(s):
        import unicodedata
        s = str(s).lower().strip()
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        s = s.replace(' ', '_').replace('[', '').replace(']', '')
        return s
        
    try:
        df_raw = pd.read_csv(csv_file, sep=',', encoding='utf-8', on_bad_lines='skip', low_memory=False)
    except Exception:
        df_raw = pd.read_csv(csv_file, sep=',', encoding='latin1', on_bad_lines='skip', low_memory=False)
    
    df_raw.columns = [clean_str_local(c) for c in df_raw.columns]
    
    station_col = None
    for col in ['estacion', 'nombre', 'central', 'estacion/ubicacion', 'codigo_estacion']:
        if col in df_raw.columns:
            station_col = col
            break
    
    if not station_col:
        station_col = df_raw.columns[0]
        
    mask_names = df_raw[station_col].astype(str).str.lower().str.strip().isin(valid_stations_names)
    mask_codes = df_raw[station_col].astype(str).str.strip().isin(valid_stations_codes)
    
    if mask_codes.sum() > mask_names.sum():
        df_raw['nombre_mapped'] = df_raw[station_col].astype(str).map(code_to_name)
        station_col = 'nombre_mapped'
        mask = mask_codes
    else:
        mask = mask_names

    df_filtered = df_raw[mask].copy()
    
    if df_filtered.empty:
        return {}
    
    # Create ISO datetime 'ds'
    date_col = next((col for col in ['date', 'momento_medicion', 'fecha'] if col in df_filtered.columns), None)
    if date_col:
        # CR2 dates are often YYYY-MM-DD-HH-MM, fallback to format='mixed'
        df_filtered['ds'] = pd.to_datetime(df_filtered[date_col], format='mixed', errors='coerce', utc=True)
        df_filtered['ds'] = df_filtered['ds'].dt.tz_localize(None)
        
        # Truncate to 2014-2024
        df_filtered = df_filtered[df_filtered['ds'].dt.year.between(2014, 2024)]
    
    if df_filtered.empty or 'ds' not in df_filtered.columns:
        return {}
        
    # Ensure float for target variable
    meta_cols_raw = ['estacion', 'nombre', 'central', 'estacion/ubicacion', 'codigo_estacion', 'date', 'momento_medicion', 'fecha', 'ds', 'nombre_mapped', 'codigo']
    val_cols = [c for c in df_filtered.columns if c not in meta_cols_raw]
    
    if val_cols:
        val_col = val_cols[0]
        df_filtered[val_col] = pd.to_numeric(df_filtered[val_col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # CRITICAL: Filter out missing/outlier indicator -9999 and NaNs
        df_filtered = df_filtered.dropna(subset=[val_col])
        df_filtered = df_filtered[df_filtered[val_col] != -9999.0]
        df_filtered = df_filtered[df_filtered[val_col] != -9999]
        
        # Rename to variable name
        df_filtered = df_filtered.rename(columns={val_col: var_name})
    
    result = {}
    import unicodedata
    for st_name, group in df_filtered.groupby(station_col):
        safe_name = ''.join(c for c in unicodedata.normalize('NFD', str(st_name).lower().strip()) if unicodedata.category(c) != 'Mn')
        safe_name = re.sub(r'[<>:"/\\|?*\']', '', safe_name).replace(' ', '_')
        
        group['unique_id'] = safe_name
        
        # Select final columns
        cols_to_keep = ['unique_id', 'ds', var_name]
        final_cols = [c for c in cols_to_keep if c in group.columns]
        
        group = group.dropna(subset=['ds']).sort_values('ds')
        result[safe_name] = group[final_cols]
        
    return result

def plot_timeline_exo(csv_file, visuales_dir, var_name):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df_estacion = pd.read_csv(csv_file, low_memory=False)
    
    if 'ds' in df_estacion.columns and var_name in df_estacion.columns:
        df_estacion['ds'] = pd.to_datetime(df_estacion['ds'])
        
        if not df_estacion.empty:
            vis_var_dir = visuales_dir / "exogenas-raw" / var_name
            vis_var_dir.mkdir(parents=True, exist_ok=True)
            
            plt.figure(figsize=(12, 4))
            plt.plot(df_estacion['ds'], df_estacion[var_name], linewidth=1)
            plt.title(f"{var_name} - {csv_file.stem}")
            plt.xlabel("Fecha")
            plt.ylabel(var_name)
            plt.tight_layout()
            plt.savefig(vis_var_dir / f"{csv_file.stem}_timeline.png")
            plt.close()

def process_exogenous(landing_dir: str | Path, bronze_dir: str | Path, visuales_dir: str | Path, force: bool = False):
    landing_dir = Path(landing_dir)
    bronze_dir = Path(bronze_dir)
    visuales_dir = Path(visuales_dir)
    
    data_exists = (bronze_dir / "exogenas").exists() and any((bronze_dir / "exogenas").iterdir())
    vis_exists = (visuales_dir / "exogenas-raw").exists() and any((visuales_dir / "exogenas-raw").iterdir())
    
    if data_exists and vis_exists and not force:
        print("[ETL Exogenas] Omitiendo fase: ya existen datos y visuales. Usa --force para reescribir.")
        return True
    
    bronze_dir.mkdir(parents=True, exist_ok=True)
    (bronze_dir / "exogenas").mkdir(parents=True, exist_ok=True)
    (visuales_dir / "exogenas-raw").mkdir(parents=True, exist_ok=True)
    
    maestro_path = landing_dir / "datos-centrales-exogenas.xlsx"
    if not maestro_path.exists():
        return
        
    df_maestro = pd.read_excel(maestro_path)
    df_maestro = clean_column_names(df_maestro)
    
    df_maestro_valido = df_maestro[
        (df_maestro['nombre'].notna()) & 
        (df_maestro['latitud'].notna()) & 
        (df_maestro['longitud'].notna())
    ].copy()
    
    for col in df_maestro_valido.columns:
        if df_maestro_valido[col].dtype == 'object':
            df_maestro_valido[col] = df_maestro_valido[col].astype(str)
            
    df_maestro_valido.to_parquet(bronze_dir / "maestro_exogenas.parquet", index=False)
    
    valid_stations_names = df_maestro_valido['nombre'].astype(str).str.lower().str.strip().tolist()
    valid_stations_codes = df_maestro_valido['codigo'].astype(str).tolist()
    code_to_name = dict(zip(df_maestro_valido['codigo'].astype(str), df_maestro_valido['nombre']))
    
    raw_csv_dir = landing_dir / "variables_externas"
    
    if not data_exists or force:
        if raw_csv_dir.exists():
            for var_dir in raw_csv_dir.iterdir():
                if not var_dir.is_dir(): continue
                var_name = var_dir.name
                csv_files = list(var_dir.glob("*.csv"))
                if not csv_files: continue
                
                print(f"[ETL Exogenas] Procesando {var_name} ({len(csv_files)} archivos) en paralelo...")
                max_w = max(1, (os.cpu_count() or 2) - 1)
                with ProcessPoolExecutor(max_workers=max_w) as executor:
                    futures = [executor.submit(parse_raw_csv_exo, f, valid_stations_names, valid_stations_codes, code_to_name, var_name) for f in csv_files]
                    for future in as_completed(futures):
                        res_dict = future.result()
                        for safe_name, group in res_dict.items():
                            var_out_dir = bronze_dir / "exogenas" / var_name
                            var_out_dir.mkdir(parents=True, exist_ok=True)
                            out_file = var_out_dir / f"{safe_name}.csv"
                            header = not out_file.exists()
                            group.to_csv(out_file, mode='a', header=header, index=False)

    if not vis_exists or force:
        print("[ETL Exogenas] Generando visuales agregadas (Mensual, Anual, Macrozona)...")
        # Map latitud to macrozona
        def get_mz(lat):
            try:
                lat = float(lat)
                if lat > -26.0: return "Norte Grande"
                if -32.1 <= lat <= -26.0: return "Norte Chico"
                if -36.2 <= lat < -32.1: return "Zona Central"
                if -44.0 <= lat < -36.2: return "Zona Sur"
                if lat < -44.0: return "Zona Austral"
            except:
                pass
            return "Desconocida"
            
        st_to_mz = {}
        if 'nombre_mapped' in df_maestro_valido.columns:
            st_col = 'nombre_mapped'
        else:
            st_col = 'nombre'
            
        import unicodedata
        for _, row in df_maestro_valido.iterrows():
            sn = ''.join(c for c in unicodedata.normalize('NFD', str(row[st_col]).lower().strip()) if unicodedata.category(c) != 'Mn').replace(' ', '_')
            st_to_mz[sn] = get_mz(row.get('latitud', None))
            
        exo_dir = bronze_dir / "exogenas"
        vis_out = visuales_dir / "exogenas-raw"
        
        if exo_dir.exists():
            for var_dir in exo_dir.iterdir():
                if not var_dir.is_dir(): continue
                var_name = var_dir.name
                dfs = []
                for f in var_dir.glob("*.csv"):
                    df = pd.read_csv(f, low_memory=False)
                    if not df.empty and 'unique_id' in df.columns:
                        mz = st_to_mz.get(df['unique_id'].iloc[0], 'Desconocida')
                        if mz != 'Desconocida':
                            df['macrozona'] = mz
                            dfs.append(df)
                            
                if dfs:
                    df_all = pd.concat(dfs, ignore_index=True)
                    df_all['ds'] = pd.to_datetime(df_all['ds'], format='ISO8601', errors='coerce')
                    df_all = df_all.dropna(subset=['ds'])
                    df_all['year'] = df_all['ds'].dt.year
                    df_all['month'] = df_all['ds'].dt.month
                    
                    import matplotlib.pyplot as plt
                    # Mensual
                    plt.figure(figsize=(10, 4))
                    df_all.groupby('month')[var_name].mean().plot(kind='bar', color='skyblue')
                    plt.title(f"Promedio Mensual - {var_name}")
                    plt.tight_layout()
                    plt.savefig(vis_out / f"{var_name}_mensual.png")
                    plt.close()
                    
                    # Anual
                    plt.figure(figsize=(10, 4))
                    df_all.groupby('year')[var_name].mean().plot(kind='line', marker='o', color='orange')
                    plt.title(f"Evolucion Anual - {var_name}")
                    plt.tight_layout()
                    plt.savefig(vis_out / f"{var_name}_anual.png")
                    plt.close()
                    
                    # Macrozona
                    plt.figure(figsize=(10, 4))
                    df_all.groupby('macrozona')[var_name].mean().plot(kind='bar', color='lightgreen')
                    plt.title(f"Promedio por Macrozona - {var_name}")
                    plt.tight_layout()
                    plt.savefig(vis_out / f"{var_name}_macrozona.png")
                    plt.close()
                    
    return True

if __name__ == "__main__":
    process_exogenous("data/landing", "data/bronze", "visuales")
