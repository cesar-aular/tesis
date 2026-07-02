import os
from pathlib import Path

dl_models = ["lstm", "nhits", "tft"]

train_template = """import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import {ModelClass}
from pathlib import Path
import json

def run_train(train_df: pd.DataFrame, strategy: str = "toy"):
    print("[{ModelName}] Iniciando TRAIN global...")
    
    models_dir = Path("models/{model_name}")
    
    params_path = models_dir / "best_params.json"
    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)
    else:
        params = {{'max_steps': 1000, 'input_size': 168}}
        
    # Memoria y Gradient Accumulation según estrategia (Regla GEMINI.md)
    if strategy in ['half', 'total']:
        batch_size = 16
        acc_grad = 8
    else:
        batch_size = 32
        acc_grad = 1
        
    trainer_kwargs = {{
        'accumulate_grad_batches': acc_grad
    }}
        
    model_obj = {ModelClass}(
        h=24, 
        input_size=params.get('input_size', 168),
        max_steps=params.get('max_steps', 1000),
        scaler_type='robust',
        batch_size=batch_size,
        windows_batch_size=256, # Optimizacion de dataloader
        trainer_kwargs=trainer_kwargs
    )
    
    nf = NeuralForecast(models=[model_obj], freq='h')
    
    try:
        nf.fit(df=train_df)
    except Exception as e:
        print(f"[{ModelName}] ERROR de memoria o fitting: {{e}}")
        # Fallback ultra agresivo
        model_obj = {ModelClass}(h=24, input_size=168, max_steps=500, batch_size=8, windows_batch_size=128, trainer_kwargs={{'accumulate_grad_batches': 16}})
        nf = NeuralForecast(models=[model_obj], freq='h')
        nf.fit(df=train_df)
    
    nf.save(path=str(models_dir / "nf_models"), overwrite=True)
    print("[{ModelName}] Entrenamiento completado y guardado.")
"""

for mod in dl_models:
    mod_path = Path("src/ml") / mod
    m_name = mod.upper()
    c_name = "LSTM" if mod == "lstm" else "NHITS" if mod == "nhits" else "TFT"
    
    with open(mod_path / "train.py", "w") as f:
        f.write(train_template.format(model_name=mod, ModelName=m_name, ModelClass=c_name))

print("Modified all DL train files to include Gradient Accumulation and memory bounding.")
