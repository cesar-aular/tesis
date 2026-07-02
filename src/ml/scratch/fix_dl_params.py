import os
from pathlib import Path

# Fix LSTM
for f in ["tune.py", "train.py"]:
    p = Path(f"src/ml/lstm/{f}")
    if p.exists():
        with open(p, "r") as file:
            content = file.read()
        
        # Replace QuantileLoss with MQLoss
        content = content.replace("QuantileLoss([0.1, 0.5, 0.9])", "MQLoss(quantiles=[0.1, 0.5, 0.9])")
        content = content.replace("from neuralforecast.losses.pytorch import QuantileLoss", "from neuralforecast.losses.pytorch import MQLoss")
        
        # Replace hidden_size with encoder_hidden_size in the LSTM call
        content = content.replace("hidden_size=params.get('hidden_size', 64),", "encoder_hidden_size=params.get('hidden_size', 64),")
        
        with open(p, "w") as file:
            file.write(content)

# Fix NHITS
for f in ["tune.py", "train.py"]:
    p = Path(f"src/ml/nhits/{f}")
    if p.exists():
        with open(p, "r") as file:
            content = file.read()
            
        content = content.replace("QuantileLoss([0.1, 0.5, 0.9])", "MQLoss(quantiles=[0.1, 0.5, 0.9])")
        content = content.replace("from neuralforecast.losses.pytorch import QuantileLoss", "from neuralforecast.losses.pytorch import MQLoss")
        
        # Remove hidden_size from NHITS call
        content = content.replace("hidden_size=params.get('hidden_size', 64),", "")
        
        with open(p, "w") as file:
            file.write(content)
            
# Fix TFT
for f in ["tune.py", "train.py"]:
    p = Path(f"src/ml/tft/{f}")
    if p.exists():
        with open(p, "r") as file:
            content = file.read()
            
        content = content.replace("QuantileLoss([0.1, 0.5, 0.9])", "MQLoss(quantiles=[0.1, 0.5, 0.9])")
        content = content.replace("from neuralforecast.losses.pytorch import QuantileLoss", "from neuralforecast.losses.pytorch import MQLoss")
        
        with open(p, "w") as file:
            file.write(content)
            
print("Fixes applied.")
