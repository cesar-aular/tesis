import os
import pandas as pd

try:
    maestro = pd.read_excel('data/landing/datos-centrales-exogenas.xlsx')
    maestro['codigo'] = maestro.iloc[:, 0].astype(str)
    code_to_owner = dict(zip(maestro['codigo'], maestro['propietario']))
    
    base_dir = 'data/landing/variables_externas'
    variables = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for var in variables:
        files = [f for f in os.listdir(os.path.join(base_dir, var)) if f.endswith('.csv') and f != 'empty_files.csv']
        owner_stats = {}
        for f in files:
            try:
                df = pd.read_csv(os.path.join(base_dir, var, f), encoding='latin1', low_memory=False, sep=',', on_bad_lines='skip')
                
                # Check column headers for date
                def clean_str_local(s):
                    import unicodedata
                    s = str(s).lower().strip()
                    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
                    s = s.replace(' ', '_').replace('[', '').replace(']', '')
                    return s
                df.columns = [clean_str_local(c) for c in df.columns]
                
                date_col = next((col for col in ['date', 'momento_medicion', 'fecha'] if col in df.columns), None)
                if not date_col:
                    df = pd.read_csv(os.path.join(base_dir, var, f), encoding='latin1', low_memory=False, sep=';', on_bad_lines='skip')
                    df.columns = [clean_str_local(c) for c in df.columns]
                    date_col = next((col for col in ['date', 'momento_medicion', 'fecha'] if col in df.columns), None)
                    
                if date_col:
                    df['year'] = df[date_col].astype(str).str[:4]
                    c_2024 = (df['year'] == '2024').sum()
                    code = f.split('_')[1].split('.')[0] if '_' in f else 'unknown'
                    owner = code_to_owner.get(code, 'Unknown')
                    if owner not in owner_stats: 
                        owner_stats[owner] = {'tot': 0, 'comp': 0, 'part': 0, 'none': 0}
                    owner_stats[owner]['tot'] += 1
                    if c_2024 > 8700: owner_stats[owner]['comp'] += 1
                    elif c_2024 > 0: owner_stats[owner]['part'] += 1
                    else: owner_stats[owner]['none'] += 1
            except Exception as e:
                pass
        
        print(f"\\n--- 2024 DATA AVAILABILITY FOR: {var.upper()} ---")
        for k, v in owner_stats.items(): 
            print(f"{k} -> Total Stations: {v['tot']} | Complete 2024 (>8700h): {v['comp']} | Partial 2024: {v['part']} | No 2024: {v['none']}")
            
except Exception as e:
    print(f"Error: {e}")
