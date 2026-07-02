import os
from pathlib import Path

models = ["lstm", "nhits", "tft"]

for mod in models:
    # --- TUNE.PY ---
    tune_path = Path(f"src/ml/{mod}/tune.py")
    if tune_path.exists():
        with open(tune_path, "r") as f:
            content = f.read()
        
        # Add static_df extraction
        new_content = content.replace("nf.fit(df=train_subset)", "static_df = train_subset[['unique_id', 'macrozona_idx', 'potencia_neta_mw']].drop_duplicates()\n            nf.fit(df=train_subset, static_df=static_df)")
        
        with open(tune_path, "w") as f:
            f.write(new_content)
            
    # --- TRAIN.PY ---
    train_path = Path(f"src/ml/{mod}/train.py")
    if train_path.exists():
        with open(train_path, "r") as f:
            content = f.read()
            
        new_content = content.replace("nf.fit(df=train_df)", "static_df = train_df[['unique_id', 'macrozona_idx', 'potencia_neta_mw']].drop_duplicates()\n        nf.fit(df=train_df, static_df=static_df)")
        
        with open(train_path, "w") as f:
            f.write(new_content)
            
print("Added static_df to fit calls.")
