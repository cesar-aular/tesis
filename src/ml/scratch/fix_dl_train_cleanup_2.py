import os
from pathlib import Path

dl_models = ["lstm", "nhits", "tft"]

cleanup = """
    # MEMORY CLEANUP VITAL
    try:
        del nf
        del model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
"""

for mod in dl_models:
    train_path = Path("src/ml") / mod / "train.py"
    with open(train_path, "r") as f:
        content = f.read()
    
    if "MEMORY CLEANUP VITAL" not in content:
        content = content.replace('print("[{ModelName}] Entrenamiento completado y guardado.")'.replace("{ModelName}", mod.upper()), 
                                  'print("[{ModelName}] Entrenamiento completado y guardado.")\n'.replace("{ModelName}", mod.upper()) + cleanup)
        
    with open(train_path, "w") as f:
        f.write(content)
        
print("Cleanup appended correctly")
