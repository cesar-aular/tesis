import os
from pathlib import Path

dl_models = ["lstm", "nhits", "tft"]

for mod in dl_models:
    train_path = Path("src/ml") / mod / "train.py"
    with open(train_path, "r") as f:
        content = f.read()
    
    # Remove trainer_kwargs definition
    content = content.replace("trainer_kwargs = {\n        'accumulate_grad_batches': acc_grad\n    }", "")
    
    # Replace trainer_kwargs in the model init
    content = content.replace("trainer_kwargs=trainer_kwargs", "accumulate_grad_batches=acc_grad")
    
    # Replace the fallback trainer_kwargs
    content = content.replace("trainer_kwargs={'accumulate_grad_batches': 16}", "accumulate_grad_batches=16")
    
    with open(train_path, "w") as f:
        f.write(content)
        
print("Fixed trainer kwargs unpacking")
