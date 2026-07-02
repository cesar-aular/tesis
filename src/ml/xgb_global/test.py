import pandas as pd
from pathlib import Path
import joblib

from src.ml.utils.metrics import calculate_metrics
from src.ml.utils.idempotency import mark_run_completed
from src.ml.utils.eval_window import eval_window_variants, WINDOW_HOURS
from src.ml.visualize import plot_forecast_rollout


def run_test(test_df: pd.DataFrame, results_dir: Path, macrozona: str, estacion: str, planta: str):
    print(f"[xgb_global] Testeando Planta: {planta} ({estacion}) con Day1 y 7-Day Rollout")
    model_path = Path("models/xgb_global/xgb_global_model.joblib")
    if not model_path.exists():
        print("[xgb_global] ERROR: Modelo no existe.")
        return

    model = joblib.load(model_path)

    if len(test_df) < WINDOW_HOURS:
        print(f"[xgb_global] Planta {planta} no tiene suficientes datos. Saltando.")
        return

    test_df = test_df.sort_values('ds').reset_index(drop=True)
    capacidad = float(test_df['potencia_neta_mw'].iloc[0])

    def _evaluate(window: pd.DataFrame, horizon: str):
        X = window.drop(columns=['unique_id', 'ds', 'y'])
        preds = model.predict(X)

        night = window['radiacion-global-instantanea'].values < 5
        preds[night] = 0.0
        preds = preds.clip(min=0)

        out = window[['unique_id', 'ds', 'y']].copy()
        out['y_pred'] = preds

        out_dir = results_dir / "xgb_global" / macrozona / estacion / planta / horizon
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_dir / "preds.parquet")

        metrics = calculate_metrics(out['y'], out['y_pred'])
        mark_run_completed(out_dir, f"xgb_global_{horizon}", planta, estacion, metrics)
        plot_forecast_rollout(df_real=out, df_pred=out,
                              output_path=out_dir / "forecast_plot.png",
                              model_name=f"xgb_global ({horizon})", planta=planta)

    # Sensibilidad: raw, operacional y las 4 estaciones del año
    for suffix, start in eval_window_variants(test_df['y'], capacidad, dates=test_df['ds']).items():
        if start + WINDOW_HOURS > len(test_df):
            continue
        _evaluate(test_df.iloc[start + 168:start + 168 + 24].copy(), f"day1{suffix}")
        _evaluate(test_df.iloc[start + 168:start + 168 + (7 * 24)].copy(), f"rollout7d{suffix}")

    print(f"[xgb_global] Completado.")
