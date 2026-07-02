import os
from pathlib import Path

models = ["lstm", "nhits", "tft"]

for mod in models:
    tune_path = Path(f"src/ml/{mod}/tune.py")
    if tune_path.exists():
        with open(tune_path, "r") as f:
            content = f.read()
            
        content = content.replace("val_preds = nf.predict()", "val_preds = nf.predict(futr_df=val_subset)")
        
        with open(tune_path, "w") as f:
            f.write(content)
            
print("Added futr_df to tune predicts.")
