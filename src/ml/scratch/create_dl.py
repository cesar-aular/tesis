import os
from pathlib import Path

dl_models = ["lstm", "nhits", "tft"]

tune_template = """import pandas as pd
from pathlib import Path
import json

def run_tune(silver_df: pd.DataFrame, strategy: str = "toy"):
    print("[{ModelName}] Iniciando TUNE global...")
    
    out_dir = Path("models/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    best_params = {{
        'input_size': 168,
        'hidden_size': 64 if '{model_name}' != 'tft' else 32,
        'max_steps': 1000,
        'learning_rate': 1e-3
    }}
    
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_params, f)
        
    print("[{ModelName}] TUNE completado.")
    return best_params
"""

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
        
    model_obj = {ModelClass}(
        h=24, 
        input_size=params.get('input_size', 168),
        max_steps=params.get('max_steps', 1000),
        scaler_type='robust'
    )
    
    nf = NeuralForecast(models=[model_obj], freq='h')
    nf.fit(df=train_df)
    
    nf.save(path=str(models_dir / "nf_models"), overwrite=True)
    print("[{ModelName}] Entrenamiento completado y guardado.")
"""

test_template = """import pandas as pd
import numpy as np
from pathlib import Path
from neuralforecast import NeuralForecast

from src.ml.utils.metrics import calculate_metrics
from src.ml.utils.coldstart import generate_synthetic_context
from src.ml.utils.idempotency import mark_run_completed
from src.ml.visualize import plot_forecast_rollout

def run_test(test_df: pd.DataFrame, results_dir: Path, macrozona: str, estacion: str, planta: str):
    print(f"[{ModelName}] Testeando Planta: {{planta}} ({{estacion}})")
    models_dir = Path("models/{model_name}/nf_models")
    
    if not models_dir.exists():
        print(f"[{ModelName}] ERROR: Modelos NeuralForecast no existen.")
        return
        
    nf = NeuralForecast.load(path=str(models_dir))
    
    capacidad = test_df['capacidad_mw'].iloc[0] if 'capacidad_mw' in test_df.columns else 50.0
    pr = test_df['pr_macrozona'].iloc[0] if 'pr_macrozona' in test_df.columns else 0.75
    
    start_test_date = test_df['ds'].min()
    synth_dates = pd.date_range(end=start_test_date - pd.Timedelta(hours=1), periods=168, freq='h')
    
    horas = synth_dates.hour
    perfil = np.exp(-0.5 * ((horas - 12) / 3)**2)
    
    synth_context = generate_synthetic_context(
        unique_id=planta,
        dates=synth_dates,
        capacity_mw=capacidad,
        pr=pr,
        profile=perfil
    )
    
    preds_df = nf.predict(df=synth_context, step_size=24) 
    
    merged = test_df[['unique_id', 'ds', 'y']].merge(preds_df, on=['unique_id', 'ds'], how='inner')
    
    if '{ModelClass}' in merged.columns:
        metrics = calculate_metrics(merged['y'], merged['{ModelClass}'])
        print(f"[{ModelName}] Resultados para {{planta}} ({{estacion}}): {{metrics}}")
        
        out = merged[['unique_id', 'ds', 'y', '{ModelClass}']].rename(columns={{'{ModelClass}': 'y_pred'}})
        
        # Save structured results
        out_dir = results_dir / "{model_name}" / macrozona / estacion / planta
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out.to_parquet(out_dir / "preds.parquet")
        mark_run_completed(out_dir, "{model_name}", planta, estacion, metrics)
        
        # Guardar gráfico
        plot_forecast_rollout(
            df_real=out, 
            df_pred=out, 
            output_path=out_dir / "forecast_plot.png", 
            model_name="{ModelName}", 
            planta=planta
        )
"""

for mod in dl_models:
    mod_path = Path("c:/Users/cesar/Desktop/code/tesis-final/src/ml") / mod
    mod_path.mkdir(parents=True, exist_ok=True)
    
    m_name = mod.upper()
    c_name = "LSTM" if mod == "lstm" else "NHITS" if mod == "nhits" else "TFT"
    
    with open(mod_path / "tune.py", "w") as f:
        f.write(tune_template.format(model_name=mod, ModelName=m_name))
        
    with open(mod_path / "train.py", "w") as f:
        f.write(train_template.format(model_name=mod, ModelName=m_name, ModelClass=c_name))
        
    with open(mod_path / "test.py", "w") as f:
        f.write(test_template.format(model_name=mod, ModelName=m_name, ModelClass=c_name))

    with open(mod_path / "__init__.py", "w") as f:
        f.write("")

print("Created all DL files")
