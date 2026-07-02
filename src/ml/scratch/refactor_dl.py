import os
from pathlib import Path

models = ["lstm", "nhits", "tft"]

for mod in models:
    tune_path = Path(f"src/ml/{mod}/tune.py")
    train_path = Path(f"src/ml/{mod}/train.py")
    test_path = Path(f"src/ml/{mod}/test.py")
    
    # 1. TUNE
    if tune_path.exists():
        with open(tune_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Strategy path
        content = content.replace(f'out_dir = Path(f"models/{mod}")', f'out_dir = Path(f"models/{{strategy}}/{mod}")')
        # Fix batch sizes
        content = content.replace('batch_size = 32', 'batch_size = 8')
        content = content.replace('windows_batch_size=128', 'windows_batch_size=32')
        content = content.replace('windows_batch_size=256', 'windows_batch_size=32')
        with open(tune_path, "w", encoding="utf-8") as f:
            f.write(content)
            
    # 2. TRAIN
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Strategy path
        content = content.replace(f'models_dir = Path(f"models/{mod}")', f'models_dir = Path(f"models/{{strategy}}/{mod}")')
        # Fix batch sizes
        content = content.replace('batch_size = 16', 'batch_size = 16')
        content = content.replace('batch_size = 32', 'batch_size = 8')
        content = content.replace('acc_grad = 8', 'acc_grad = 4')
        content = content.replace('windows_batch_size=128', 'windows_batch_size=32')
        content = content.replace('windows_batch_size=256', 'windows_batch_size=32')
        with open(train_path, "w", encoding="utf-8") as f:
            f.write(content)
            
    # 3. TEST
    if test_path.exists():
        with open(test_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Signature
        content = content.replace(
            f'def run_test(test_df: pd.DataFrame, silver_dl: pd.DataFrame, results_dir: Path, macrozona: str, estacion: str, planta: str):',
            f'def run_test(test_df: pd.DataFrame, silver_dl: pd.DataFrame, results_dir: Path, macrozona: str, estacion: str, planta: str, strategy: str = "toy"):'
        )
        # Strategy path
        content = content.replace(f'models_dir = Path(f"models/{mod}/nf_models")', f'models_dir = Path(f"models/{{strategy}}/{mod}/nf_models")')
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(content)

print("Refactorizado DL files con strategy y batch limits reducidos.")
