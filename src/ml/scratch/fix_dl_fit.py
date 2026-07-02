import os
from pathlib import Path

dl_models = ["lstm", "nhits", "tft"]

for mod in dl_models:
    train_path = Path("src/ml") / mod / "train.py"
    with open(train_path, "r") as f:
        content = f.read()
    
    # Replace the nf.fit call
    content = content.replace("nf.fit(df=train_df)", "nf.fit(df=train_df[['unique_id', 'ds', 'y']])")
    
    with open(train_path, "w") as f:
        f.write(content)
        
print("Updated nf.fit calls to only use unique_id, ds, y")
