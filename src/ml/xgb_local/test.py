"""Evaluación del baseline XGB local sobre las ventanas Cold-Start.

Rigor del benchmark (comparación 1:1 con los modelos DL):
- day1: los lags apuntan al CONTEXTO real (168h previas a la evaluación) —
  el paradigma local observa sus propios sensores hasta el momento del
  pronóstico. Legítimo.
- rollout7d AUTORREGRESIVO: a partir del día 2, lag_24 apunta a horas de la
  propia ventana de evaluación; usar la generación real ahí sería teacher
  forcing (siete pronósticos de 24h con realimentación perfecta, no un
  pronóstico de 7 días). Se realimenta la MEDIANA PREDICHA (P50), exactamente
  como el roll-out DL realimenta su mediana. lag_168 siempre cae en el
  contexto real (legítimo).
- Intervalos: regresión cuantílica (P5/P50/P95) -> Coverage_90 y Pinball.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.ml.utils.metrics import calculate_metrics
from src.ml.utils.idempotency import mark_run_completed, mark_plant_model_complete
from src.ml.utils.eval_window import eval_window_variants, WINDOW_HOURS
from src.ml.utils.features import generate_lags
from src.ml.utils.grid import make_hourly_grid
from src.ml.utils.physics import night_mask
from src.ml.visualize import plot_forecast_rollout

LAGS = [24, 168]
HIST_HOURS = 168
EVAL_DAYS = 7


def _predict_quantiles(model, X):
    """Predicción multi-cuantil (n, 3) -> (lo, med, hi) con monotonía forzada."""
    q = model.predict(X)
    med = q[:, 1].copy()
    # El cruce de cuantiles es posible en quantile regression: se repara
    # (numpy directo: Series.to_numpy() bajo CoW devuelve vistas de solo lectura)
    lo = np.minimum(q[:, 0], med)
    hi = np.maximum(q[:, 2], med)
    return lo, med, hi


def _apply_night(values, night):
    values = values.copy()
    values[night] = 0.0
    return values.clip(min=0)


def run_test(test_df: pd.DataFrame, results_dir: Path, macrozona: str, estacion: str, planta: str):
    print(f"[xgb_local] Testeando Planta: {planta} ({estacion}) con Day1 y 7-Day Rollout")
    model_path = Path(f"models/xgb_local/xgb_local_{planta}.joblib")
    if not model_path.exists():
        print("[xgb_local] ERROR: Modelo no existe.")
        return

    model = joblib.load(model_path)

    # Grilla horaria canónica: mismas horas calendario que los DL
    grid = make_hourly_grid(test_df, planta)
    if len(grid) < WINDOW_HOURS:
        print(f"[xgb_local] Planta {planta} no tiene suficientes datos. Saltando.")
        return
    capacidad = float(grid['potencia_neta_mw'].iloc[0])

    def _save(out: pd.DataFrame, horizon: str):
        # Métricas sólo sobre observaciones reales (huecos de la grilla fuera)
        out = out.dropna(subset=['y'])
        if out.empty:
            return
        out_dir = results_dir / "xgb_local" / macrozona / estacion / planta / horizon
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_dir / "preds.parquet")
        metrics = calculate_metrics(out['y'], out['y_pred'],
                                    y_lo=out['y_pred_lo_90'], y_hi=out['y_pred_hi_90'])
        mark_run_completed(out_dir, f"xgb_local_{horizon}", planta, estacion, metrics)
        plot_forecast_rollout(df_real=out, df_pred=out,
                              output_path=out_dir / "forecast_plot.png",
                              model_name=f"xgb_local ({horizon})", planta=planta)

    # Sensibilidad: raw, operacional y las 4 estaciones del año
    for suffix, start in eval_window_variants(grid['y'], capacidad, dates=grid['ds']).items():
        if start + WINDOW_HOURS > len(grid):
            continue
        window = grid.iloc[start:start + WINDOW_HOURS].reset_index(drop=True)
        y_real = window['y'].copy()  # ground truth intocable para métricas

        # 'work' es la serie que ven los lags: contexto real + P50 realimentada
        work = window.copy()
        day_frames = []
        for d in range(EVAL_DAYS):
            lagged = generate_lags(work, LAGS)
            lo_idx = HIST_HOURS + d * 24
            seg = lagged.iloc[lo_idx:lo_idx + 24]
            X = seg.drop(columns=['unique_id', 'ds', 'y'])
            lo, med, hi = _predict_quantiles(model, X)

            night = night_mask(seg)
            lo, med, hi = (_apply_night(v, night) for v in (lo, med, hi))

            out = window.iloc[lo_idx:lo_idx + 24][['unique_id', 'ds']].copy()
            out['y'] = y_real.iloc[lo_idx:lo_idx + 24].values
            out['y_pred'] = med
            out['y_pred_lo_90'] = lo
            out['y_pred_hi_90'] = hi
            day_frames.append(out)

            # Realimentación autorregresiva: los lags del día siguiente ven P50
            work.iloc[lo_idx:lo_idx + 24, work.columns.get_loc('y')] = med

        _save(day_frames[0].copy(), f"day1{suffix}")
        _save(pd.concat(day_frames, ignore_index=True), f"rollout7d{suffix}")

    # Marker de completitud por planta: TODAS las ventanas terminaron
    mark_plant_model_complete(results_dir / "xgb_local" / macrozona / estacion / planta,
                              "xgb_local", planta, estacion)
    print(f"[xgb_local] Completado.")
