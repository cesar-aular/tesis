import os
from pathlib import Path

dl_models = ["lstm", "nhits", "tft"]

for mod in dl_models:
    train_path = Path("src/ml") / mod / "train.py"
    with open(train_path, "r") as f:
        content = f.read()
    
    if "import gc" not in content:
        content = content.replace("import json", "import json\nimport gc\nimport torch")
    
    if "gc.collect()" not in content:
        cleanup = """
    # MEMORY CLEANUP VITAL
    del nf
    del model_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
"""
        content = content.replace('print(f"[{ModelName}] Entrenamiento completado y guardado.")', 'print(f"[{ModelName}] Entrenamiento completado y guardado.")\n' + cleanup.replace("{ModelName}", mod.upper()))
        
    with open(train_path, "w") as f:
        f.write(content)
        
print("Added explicit memory cleanup to DL train scripts")
