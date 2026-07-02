import os
import pandas as pd

base_dir = 'data/landing/variables_externas'
variables = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for var in variables:
    files = [f for f in os.listdir(os.path.join(base_dir, var)) if f.endswith('.csv') and f != 'empty_files.csv']
    stats = {'tot': 0, 'comp': 0, 'part': 0, 'none': 0}
    
    def clean_str_local(s):
        import unicodedata
        s = str(s).lower().strip()
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        s = s.replace(' ', '_').replace('[', '').replace(']', '')
        return s

    for f in files:
        try:
            df = pd.read_csv(os.path.join(base_dir, var, f), encoding='latin1', low_memory=False, sep=',', on_bad_lines='skip')
            df.columns = [clean_str_local(c) for c in df.columns]
            date_col = next((col for col in ['date', 'momento_medicion', 'fecha'] if col in df.columns), None)
            
            if not date_col:
                df = pd.read_csv(os.path.join(base_dir, var, f), encoding='latin1', low_memory=False, sep=';', on_bad_lines='skip')
                df.columns = [clean_str_local(c) for c in df.columns]
                date_col = next((col for col in ['date', 'momento_medicion', 'fecha'] if col in df.columns), None)
                
            if date_col:
                # Use robust pd.to_datetime!
                df['ds'] = pd.to_datetime(df[date_col], format='mixed', errors='coerce', utc=True)
                df['year'] = df['ds'].dt.year
                c_2024 = (df['year'] == 2024).sum()
                
                stats['tot'] += 1
                if c_2024 > 8700: stats['comp'] += 1
                elif c_2024 > 0: stats['part'] += 1
                else: stats['none'] += 1
        except Exception as e:
            pass
            
    print(f"\\n--- 2024 DATA FOR: {var.upper()} ---")
    print(f"Total Stations: {stats['tot']} | Complete 2024 (>8700h): {stats['comp']} | Partial 2024: {stats['part']} | No 2024: {stats['none']}")
