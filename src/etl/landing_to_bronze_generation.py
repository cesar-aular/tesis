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

def clean_string_values(s):
    if pd.isna(s): return s
    s = str(s).lower().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def parse_raw_csv_gen(csv_file, valid_plants_lower):
    import pandas as pd
    import re
    from datetime import timedelta
    
    df_raw = pd.read_csv(csv_file, sep=';', low_memory=False)
    
    # 1. Standardize columns
    def clean_str_local(s):
        import unicodedata
        s = str(s).lower().strip()
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        s = s.replace(' ', '_').replace('[', '').replace(']', '')
        return s
    df_raw.columns = [clean_str_local(c) for c in df_raw.columns]
    
    plant_col = next((col for col in ['planta', 'nombre', 'central'] if col in df_raw.columns), None)
    if not plant_col:
        return {}
    
    mask = df_raw[plant_col].astype(str).str.lower().str.strip().isin(valid_plants_lower)
    df_filtered = df_raw[mask].copy()
    
    if df_filtered.empty:
        return {}
        
    # 2. Filter by Year 2014-2024
    if 'year' in df_filtered.columns:
        valid_years = tuple(str(y) for y in range(2014, 2025))
        df_filtered = df_filtered[df_filtered['year'].astype(str).isin(valid_years)]
    
    if df_filtered.empty:
        return {}
        
    # 3. Create ISO datetime 'ds'
    if 'date' in df_filtered.columns and 'hour' in df_filtered.columns:
        # Some dates might be DD/MM/YYYY, format='mixed' handles it.
        # But to be safe, dayfirst=True is needed for DD/MM/YYYY
        df_filtered['date_obj'] = pd.to_datetime(df_filtered['date'], format='mixed', dayfirst=True, errors='coerce')
        
        # Handle hour 24 -> next day 00:00
        # First cast hour to int
        df_filtered['hour'] = pd.to_numeric(df_filtered['hour'], errors='coerce').fillna(0).astype(int)
        
        # Create timedelta for hours
        # If hour is 24, timedelta(hours=24) will automatically roll over to next day!
        df_filtered['ds'] = df_filtered['date_obj'] + pd.to_timedelta(df_filtered['hour'], unit='h')
        
        # Drop temporary columns
        df_filtered = df_filtered.drop(columns=['date', 'hour', 'date_obj', 'year', 'month', 'day'], errors='ignore')
    
    # 4. Cast Target Variable to float
    target_col = next((col for col in ['generacion_mwh', 'generacion'] if col in df_filtered.columns), None)
    if target_col:
        df_filtered['generacion_mwh'] = pd.to_numeric(df_filtered[target_col].astype(str).str.replace(',', '.'), errors='coerce')
        if target_col != 'generacion_mwh':
            df_filtered = df_filtered.drop(columns=[target_col])
            
    # 5. Add unique_id and Keep only essential columns
    result = {}
    import unicodedata
    for plant_name, group in df_filtered.groupby(plant_col):
        safe_name = ''.join(c for c in unicodedata.normalize('NFD', str(plant_name).lower().strip()) if unicodedata.category(c) != 'Mn')
        safe_name = re.sub(r'[<>:"/\\|?*\']', '', safe_name).replace(' ', '_')
        
        group['unique_id'] = safe_name
        
        # Select final columns
        cols_to_keep = ['unique_id', 'ds', 'generacion_mwh']
        final_cols = [c for c in cols_to_keep if c in group.columns]
        
        # Sort and dropna
        group = group.dropna(subset=['ds', 'generacion_mwh']).sort_values('ds')
        
        result[safe_name] = group[final_cols]
        
    return result

def plot_timeline_gen(csv_file, visuales_dir):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df_planta = pd.read_csv(csv_file, low_memory=False)
    if 'ds' in df_planta.columns and 'generacion_mwh' in df_planta.columns:
        df_planta['ds'] = pd.to_datetime(df_planta['ds'])
        
        if not df_planta.empty:
            plt.figure(figsize=(12, 4))
            plt.plot(df_planta['ds'], df_planta['generacion_mwh'], linewidth=1)
            plt.title(f"ProducciA3n HistA3rica - {csv_file.stem}")
            plt.xlabel("Fecha")
            plt.ylabel("GeneraciA3n (MWh)")
            plt.tight_layout()
            plt.savefig(visuales_dir / "generacion-raw" / f"{csv_file.stem}_timeline.png")
            plt.close()

def process_generation(landing_dir: str | Path, bronze_dir: str | Path, visuales_dir: str | Path, force: bool = False):
    landing_dir = Path(landing_dir)
    bronze_dir = Path(bronze_dir)
    visuales_dir = Path(visuales_dir)
    
    data_exists = (bronze_dir / "generacion").exists() and any((bronze_dir / "generacion").iterdir())
    vis_exists = (visuales_dir / "generacion-raw").exists() and any((visuales_dir / "generacion-raw").iterdir())
    
    if data_exists and vis_exists and not force:
        print("[ETL Generacion] Omitiendo fase: ya existen datos y visuales. Usa --force para reescribir.")
        return True
    
    bronze_dir.mkdir(parents=True, exist_ok=True)
    (bronze_dir / "generacion").mkdir(parents=True, exist_ok=True)
    (visuales_dir / "generacion-raw").mkdir(parents=True, exist_ok=True)
    
    maestro_path = landing_dir / "datos-centrales-generacion.xlsx"
    if not maestro_path.exists():
        return
        
    df_maestro = pd.read_excel(maestro_path)
    
    if not vis_exists or force:
        plt.figure(figsize=(10, 6))
        col_tec = 'Tecnología' if 'Tecnología' in df_maestro.columns else ('Tecnologa' if 'Tecnologa' in df_maestro.columns else None)
        if col_tec:
            df_maestro[col_tec].value_counts().head(10).plot(kind='bar')
            plt.title("Top 10 Tecnologías en Raw")
            plt.tight_layout()
            plt.savefig(visuales_dir / "generacion-raw" / "top_tecnologias.png")
        plt.close()
    
    df_maestro = clean_column_names(df_maestro)
    
    if 'potencia_neta_mw' in df_maestro.columns:
        df_maestro['potencia_neta_mw'] = pd.to_numeric(df_maestro['potencia_neta_mw'].astype(str).str.replace(',', '.'), errors='coerce')

    if 'tecnologia' in df_maestro.columns:
        df_maestro['tecnologia_clean'] = df_maestro['tecnologia'].apply(clean_string_values)
    
    df_maestro_solar = df_maestro[
        (df_maestro.get('tecnologia_clean', '') == 'solar fotovoltaica') &
        (df_maestro['nombre'].notna()) &
        (df_maestro['region'].notna())
    ].copy()
    
    if 'tecnologia_clean' in df_maestro_solar.columns:
        df_maestro_solar.drop(columns=['tecnologia_clean'], inplace=True)
        
    for col in df_maestro_solar.columns:
        if df_maestro_solar[col].dtype == 'object':
            df_maestro_solar[col] = df_maestro_solar[col].astype(str)
            
    df_maestro_solar.to_parquet(bronze_dir / "maestro_generacion.parquet", index=False)
    valid_plants_lower = set(str(n).lower().strip() for n in df_maestro_solar['nombre'].unique() if pd.notna(n))
    
    raw_csv_dir = landing_dir / "generacion_raw"
    
    if not data_exists or force:
        # CRITICO: limpiar salidas previas. La escritura por planta usa mode='a'
        # (necesario para ensamblar multiples archivos crudos dentro de UNA corrida);
        # sin esta limpieza, cada re-ejecucion con --force APENDIZA el dataset
        # completo otra vez (bug historico: bronze quedo triplicado, ratio ~3.17x).
        for old_csv in (bronze_dir / "generacion").glob("*.csv"):
            old_csv.unlink()
        if raw_csv_dir.exists():
            csv_files = list(raw_csv_dir.glob("*.csv"))
            if csv_files:
                print(f"[ETL Generacion] Procesando {len(csv_files)} archivos crudos en paralelo...")
                max_w = max(1, (os.cpu_count() or 2) - 1)
                with ProcessPoolExecutor(max_workers=max_w) as executor:
                    futures = [executor.submit(parse_raw_csv_gen, f, valid_plants_lower) for f in csv_files]
                    for future in as_completed(futures):
                        res_dict = future.result()
                        for safe_name, group in res_dict.items():
                            out_file = bronze_dir / "generacion" / f"{safe_name}.csv"
                            header = not out_file.exists()
                            group.to_csv(out_file, mode='a', header=header, index=False)
                            
        # Post-processing: discard plants with too little data (< 720 hours)
        gen_files = list((bronze_dir / "generacion").glob("*.csv"))
        for f in gen_files:
            try:
                df_temp = pd.read_csv(f, usecols=['generacion_mwh'])
                if len(df_temp) < 720:
                    f.unlink()
            except Exception:
                pass

    if not vis_exists or force:
        gen_files = list((bronze_dir / "generacion").glob("*.csv"))
        if gen_files:
            print(f"[ETL Generacion] Calculando metricas para visuales (5 mejores y 5 peores)...")
            stats = []
            for f in gen_files:
                try:
                    df_temp = pd.read_csv(f, usecols=['generacion_mwh'])
                    stats.append({
                        'file': f,
                        'count': len(df_temp),
                        'std': df_temp['generacion_mwh'].std()
                    })
                except Exception:
                    pass
                    
            stats_df = pd.DataFrame(stats).dropna()
            stats_df = stats_df.sort_values(by=['count', 'std'], ascending=[False, False])
            
            best_files = stats_df.head(5)['file'].tolist()
            worst_files = stats_df.tail(5)['file'].tolist()
            target_files = best_files + worst_files
            
            print(f"[ETL Generacion] Generando 10 visuales en paralelo...")
            max_w = max(1, (os.cpu_count() or 2) - 1)
            with ProcessPoolExecutor(max_workers=max_w) as executor:
                futures = [executor.submit(plot_timeline_gen, f, visuales_dir) for f in target_files]
                for future in as_completed(futures):
                    future.result()
                    
    return True

if __name__ == "__main__":
    process_generation("data/landing", "data/bronze", "visuales")
