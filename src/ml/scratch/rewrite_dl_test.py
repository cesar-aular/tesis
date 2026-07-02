import os
from pathlib import Path

models = ["lstm", "nhits", "tft"]
classes = ["LSTM", "NHITS", "TFT"]

for mod, cls in zip(models, classes):
    test_content = f"""import pandas as pd
import numpy as np
from pathlib import Path
from neuralforecast import NeuralForecast

from src.ml.utils.metrics import calculate_metrics
from src.ml.utils.idempotency import mark_run_completed
from src.ml.visualize import plot_forecast_rollout

def run_test(test_df: pd.DataFrame, silver_dl: pd.DataFrame, results_dir: Path, macrozona: str, estacion: str, planta: str):
    print(f"[{cls}] Testeando Planta: {{planta}} ({{estacion}}) con Roll-Out a 7 dias")
    models_dir = Path(f"models/{mod}/nf_models")
    
    if not models_dir.exists():
        print(f"[{cls}] ERROR: Modelos NeuralForecast no existen.")
        return
        
    nf = NeuralForecast.load(path=str(models_dir))
    
    # Check minimum data
    if len(test_df) < 168 + (7*24):
        print(f"[{cls}] Planta {{planta}} no tiene suficientes datos (168h + 7 dias). Skipiando.")
        return
    
    capacidad = test_df['potencia_neta_mw'].iloc[0]
    pr = test_df['radiacion-global-instantanea'].mean() / 1000.0  # Heuristica simple o usar PR real si existiera.
    pr = 0.75 # Hardcodeado a un PR razonable ya que eliminamos pr_macrozona en prepare_dl_dataset para simplificar. Wait, let's use 0.75
    
    # 1. CONSTRUIR CONTEXTO HISTORICO (168h)
    hist_df = test_df.iloc[:168].copy()
    horas = hist_df['ds'].dt.hour
    perfil = np.exp(-0.5 * ((horas - 12) / 3)**2)
    # Heuristica pura: cero leakage de generacion real
    hist_df['y'] = capacidad * pr * perfil
    
    # 2. PREDICCION DE 1 DIA (24h)
    futr_df_day1 = test_df.iloc[168:168+24].copy()
    futr_df_day1_input = futr_df_day1.drop(columns=['y'])
    
    preds_day1 = nf.predict(df=hist_df, futr_df=futr_df_day1_input)
    preds_day1 = preds_day1.reset_index()
    
    # Enforzar consistencia fisica: 0 de noche (radiacion < 5)
    night_mask = futr_df_day1['radiacion-global-instantanea'].values < 5
    for col in ['{cls}-median', '{cls}-lo-90', '{cls}-hi-90']:
        if col in preds_day1.columns:
            preds_day1.loc[night_mask, col] = 0.0
            preds_day1[col] = preds_day1[col].clip(lower=0.0)
    
    # Guardar Dia 1
    merged_day1 = futr_df_day1[['unique_id', 'ds', 'y']].merge(preds_day1, on=['unique_id', 'ds'], how='inner')
    out_dir_day1 = results_dir / "{mod}" / macrozona / estacion / planta / "day1"
    out_dir_day1.mkdir(parents=True, exist_ok=True)
    out_day1 = merged_day1.rename(columns={{'{cls}-median': 'y_pred', '{cls}-lo-90': 'y_pred_lo_90', '{cls}-hi-90': 'y_pred_hi_90'}})
    out_day1.to_parquet(out_dir_day1 / "preds.parquet")
    metrics_day1 = calculate_metrics(out_day1['y'], out_day1['y_pred'])
    mark_run_completed(out_dir_day1, "{mod}_day1", planta, estacion, metrics_day1)
    plot_forecast_rollout(df_real=out_day1, df_pred=out_day1, output_path=out_dir_day1 / "forecast_plot.png", model_name="{cls} (1 Day)", planta=planta)
    
    # 3. PREDICCION ROLL-OUT (7 DIAS / 168h incremental)
    current_hist = hist_df.copy()
    all_preds = []
    
    for day in range(7):
        start_idx = 168 + (day * 24)
        end_idx = start_idx + 24
        futr_day = test_df.iloc[start_idx:end_idx].copy()
        futr_day_input = futr_day.drop(columns=['y'])
        
        preds = nf.predict(df=current_hist, futr_df=futr_day_input).reset_index()
        
        # Fisica
        n_mask = futr_day['radiacion-global-instantanea'].values < 5
        for col in ['{cls}-median', '{cls}-lo-90', '{cls}-hi-90']:
            if col in preds.columns:
                preds.loc[n_mask, col] = 0.0
                preds[col] = preds[col].clip(lower=0.0)
                
        all_preds.append(preds)
        
        # Desplazar ventana
        new_hist_tail = futr_day.copy()
        new_hist_tail['y'] = preds['{cls}-median'].values # AUTORREGRESIVO (usamos la prediccion, NO LA REALIDAD)
        
        current_hist = pd.concat([current_hist.iloc[24:], new_hist_tail], ignore_index=True)
        
    final_preds_7d = pd.concat(all_preds, ignore_index=True)
    
    # Guardar Rollout 7D
    real_7d = test_df.iloc[168:168+(7*24)].copy()
    merged_7d = real_7d[['unique_id', 'ds', 'y']].merge(final_preds_7d, on=['unique_id', 'ds'], how='inner')
    
    out_dir_7d = results_dir / "{mod}" / macrozona / estacion / planta / "rollout7d"
    out_dir_7d.mkdir(parents=True, exist_ok=True)
    out_7d = merged_7d.rename(columns={{'{cls}-median': 'y_pred', '{cls}-lo-90': 'y_pred_lo_90', '{cls}-hi-90': 'y_pred_hi_90'}})
    out_7d.to_parquet(out_dir_7d / "preds.parquet")
    
    metrics_7d = calculate_metrics(out_7d['y'], out_7d['y_pred'])
    mark_run_completed(out_dir_7d, "{mod}_rollout7d", planta, estacion, metrics_7d)
    
    # Graficar con bandas de incertidumbre
    plot_forecast_rollout(df_real=out_7d, df_pred=out_7d, output_path=out_dir_7d / "forecast_plot.png", model_name="{cls} (7 Day Roll-Out)", planta=planta)

    print(f"[{cls}] Roll-Out completado para {{planta}}.")
"""

    out_mod = Path(f"src/ml/{mod}")
    with open(out_mod / "test.py", "w") as f:
        f.write(test_content)

print("Updated test.py for all models with Day 1 and 7-Day Roll-out logic (Autoregressive).")
