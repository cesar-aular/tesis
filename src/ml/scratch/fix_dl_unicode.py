import os
from pathlib import Path

dl_models = ["lstm", "nhits", "tft"]

for mod in dl_models:
    train_path = Path("src/ml") / mod / "tune.py"
    with open(train_path, "r", encoding="cp1252") as f:
        content = f.read()
    
    content = content.replace("dinámico", "dinamico")
    
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(content)
        
print("Fixed encoding issue in tune.py")
