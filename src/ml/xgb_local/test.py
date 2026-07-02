import pandas as pd
from pathlib import Path
import joblib

from src.ml.utils.metrics import calculate_metrics
from src.ml.utils.idempotency import mark_run_completed
from src.ml.utils.features import generate_lags
from src.ml.visualize import plot_forecast_rollout

LAGS = [24, 168]


def run_test(test_df: pd.DataFrame, results_dir: Path, macrozona: str, estacion: str, planta: str):
    """Evalua el baseline local sobre la ventana Cold-Start (day1 + rollout7d).

    Los lags se calculan sobre la serie real de la planta: el paradigma local
    observa sus propios datos en produccion (lag_24/lag_168 son pasado real al
    momento de predecir cada dia). El modelo NUNCA entreno con estas filas.
    """
    print(f"[xgb_local] Testeando Planta: {planta} ({estacion}) con Day1 y 7-Day Rollout")
    model_path = Path(f"models/xgb_local/xgb_local_{planta}.joblib")
    if not model_path.exists():
        print("[xgb_local] ERROR: Modelo no existe.")
        return

    model = joblib.load(model_path)

    if len(test_df) < 168 + (7 * 24):
        print(f"[xgb_local] Planta {planta} no tiene suficientes datos. Saltando.")
        return

    # Lags sobre la serie completa, luego se recortan las ventanas de evaluacion
    test_df = generate_lags(test_df.sort_values('ds'), LAGS)

    def _evaluate(window: pd.DataFrame, horizon: str):
        X = window.drop(columns=['unique_id', 'ds', 'y'])
        preds = model.predict(X)

        night = window['radiacion-global-instantanea'].values < 5
        preds[night] = 0.0
        preds = preds.clip(min=0)

        out = window[['unique_id', 'ds', 'y']].copy()
        out['y_pred'] = preds

        out_dir = results_dir / "xgb_local" / macrozona / estacion / planta / horizon
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_dir / "preds.parquet")

        metrics = calculate_metrics(out['y'], out['y_pred'])
        mark_run_completed(out_dir, f"xgb_local_{horizon}", planta, estacion, metrics)
        plot_forecast_rollout(df_real=out, df_pred=out,
                              output_path=out_dir / "forecast_plot.png",
                              model_name=f"xgb_local ({horizon})", planta=planta)

    _evaluate(test_df.iloc[168:168 + 24].copy(), "day1")
    _evaluate(test_df.iloc[168:168 + (7 * 24)].copy(), "rollout7d")

    print(f"[xgb_local] Completado.")
