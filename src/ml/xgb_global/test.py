import pandas as pd
from pathlib import Path
import joblib

from src.ml.utils.metrics import calculate_metrics
from src.ml.utils.idempotency import mark_run_completed
from src.ml.visualize import plot_forecast_rollout

def run_test(test_df: pd.DataFrame, results_dir: Path, macrozona: str, estacion: str, planta: str):
    print(f"[xgb_global] Testeando Planta: {planta} ({estacion}) con Day1 y 7-Day Rollout")
    model_path = Path("models/xgb_global/xgb_global_model.joblib")
    if not model_path.exists():
        print("[xgb_global] ERROR: Modelo no existe.")
        return
        
    model = joblib.load(model_path)
    
    if len(test_df) < 168 + (7*24):
        print(f"[xgb_global] Planta {planta} no tiene suficientes datos. Skipiando.")
        return
        
    # DAY 1
    test_day1 = test_df.iloc[168:168+24].copy()
        
    X_day1 = test_day1.drop(columns=['unique_id', 'ds', 'y'])
    y_day1 = test_day1['y']
    preds_day1 = model.predict(X_day1)
    
    mask_day1 = test_day1['radiacion-global-instantanea'].values < 5
    preds_day1[mask_day1] = 0.0
    preds_day1 = preds_day1.clip(min=0)
    
    out_day1 = test_day1[['unique_id', 'ds', 'y']].copy()
    out_day1['y_pred'] = preds_day1
    
    out_dir_day1 = results_dir / "xgb_global" / macrozona / estacion / planta / "day1"
    out_dir_day1.mkdir(parents=True, exist_ok=True)
    out_day1.to_parquet(out_dir_day1 / "preds.parquet")
    
    metrics_day1 = calculate_metrics(out_day1['y'], out_day1['y_pred'])
    mark_run_completed(out_dir_day1, "xgb_global_day1", planta, estacion, metrics_day1)
    plot_forecast_rollout(df_real=out_day1, df_pred=out_day1, output_path=out_dir_day1 / "forecast_plot.png", model_name="xgb_global (1 Day)", planta=planta)
    
    # 7 DAY ROLLOUT
    test_7d = test_df.iloc[168:168+(7*24)].copy()
    X_7d = test_7d.drop(columns=['unique_id', 'ds', 'y'])
    preds_7d = model.predict(X_7d)
    
    mask_7d = test_7d['radiacion-global-instantanea'].values < 5
    preds_7d[mask_7d] = 0.0
    preds_7d = preds_7d.clip(min=0)
    
    out_7d = test_7d[['unique_id', 'ds', 'y']].copy()
    out_7d['y_pred'] = preds_7d
    
    out_dir_7d = results_dir / "xgb_global" / macrozona / estacion / planta / "rollout7d"
    out_dir_7d.mkdir(parents=True, exist_ok=True)
    out_7d.to_parquet(out_dir_7d / "preds.parquet")
    
    metrics_7d = calculate_metrics(out_7d['y'], out_7d['y_pred'])
    mark_run_completed(out_dir_7d, "xgb_global_rollout7d", planta, estacion, metrics_7d)
    plot_forecast_rollout(df_real=out_7d, df_pred=out_7d, output_path=out_dir_7d / "forecast_plot.png", model_name="xgb_global (7 Day)", planta=planta)

    print(f"[xgb_global] Completado.")
