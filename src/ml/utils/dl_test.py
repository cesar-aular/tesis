"""Evaluación LOPO compartida para modelos Deep Learning (NeuralForecast).

Un único roll-out Cold-Start para LSTM / NHITS / TFT / Informer:
- Reindexa la serie objetivo a una grilla horaria CONTINUA (NeuralForecast exige
  combinaciones id/tiempo completas; las series PV reales tienen huecos).
- Siembra la ventana histórica con contexto 100% sintético (cero leakage de y real).
- Predice day1 y roll-out autorregresivo de 7 días (realimenta la mediana predicha).
- Aplica consistencia física (0 nocturno, clip >= 0) y guarda métricas/plots.
"""
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from neuralforecast import NeuralForecast

from src.ml.utils.metrics import calculate_metrics
from src.ml.utils.idempotency import mark_run_completed
from src.ml.visualize import plot_forecast_rollout

HIST_HOURS = 168          # ventana de contexto (7 días)
ROLLOUT_DAYS = 7
SEASON_MAP = {12: 'Verano', 1: 'Verano', 2: 'Verano',
              3: 'Otoño', 4: 'Otoño', 5: 'Otoño',
              6: 'Invierno', 7: 'Invierno', 8: 'Invierno',
              9: 'Primavera', 10: 'Primavera', 11: 'Primavera'}
SEASON_NUM = {'Verano': 1, 'Otoño': 2, 'Invierno': 3, 'Primavera': 4}

WEATHER_COLS = ['humedad-relativa', 'radiacion-global-instantanea', 'temp-aire-seco']
STATIC_COLS = ['macrozona_idx', 'potencia_neta_mw']


def _recompute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features temporales deterministas recalculadas desde ds (válidas en huecos)."""
    ds = df['ds']
    df['sin_hour'] = np.sin(2 * np.pi * ds.dt.hour / 24)
    df['cos_hour'] = np.cos(2 * np.pi * ds.dt.hour / 24)
    df['sin_month'] = np.sin(2 * np.pi * ds.dt.month / 12)
    df['cos_month'] = np.cos(2 * np.pi * ds.dt.month / 12)
    season_num = ds.dt.month.map(SEASON_MAP).map(SEASON_NUM)
    df['sin_season'] = np.sin(2 * np.pi * season_num / 4)
    df['cos_season'] = np.cos(2 * np.pi * season_num / 4)
    return df


def make_hourly_grid(test_df: pd.DataFrame, planta: str) -> pd.DataFrame:
    """Reindexa la serie de la planta a una grilla horaria continua.

    - y queda NaN en huecos (las métricas se calculan sólo contra observaciones reales).
    - Clima se interpola/ffill (conocido u obtenible de pronósticos, sin leakage de y).
    - Features temporales y estacion_idx se recalculan desde ds.
    - Estáticas se propagan (constantes por planta).
    """
    test_df = test_df.sort_values('ds')
    full_range = pd.date_range(test_df['ds'].min(), test_df['ds'].max(), freq='h')
    grid = (test_df.set_index('ds')
            .reindex(full_range)
            .rename_axis('ds')
            .reset_index())
    grid['unique_id'] = planta

    for col in WEATHER_COLS:
        if col in grid.columns:
            grid[col] = grid[col].interpolate(limit_direction='both').ffill().bfill()
    for col in STATIC_COLS:
        if col in grid.columns:
            grid[col] = grid[col].ffill().bfill()

    grid = _recompute_time_features(grid)
    if 'estacion_idx' in grid.columns:
        # estacion_idx es determinista del mes; propagar por si el encoder difiere
        grid['estacion_idx'] = grid['estacion_idx'].ffill().bfill()
    return grid


def synthetic_context(grid_hist: pd.DataFrame, capacidad: float, regional_pr: float) -> pd.DataFrame:
    """Contexto Cold-Start 100% sintético: y = capacidad * PR_regional * perfil_solar."""
    hist = grid_hist.copy()
    horas = hist['ds'].dt.hour
    perfil = np.exp(-0.5 * ((horas - 12) / 3) ** 2)  # campana centrada al mediodía
    hist['y'] = np.clip(capacidad * regional_pr * perfil, 0, None)
    return hist


def _apply_physics(preds: pd.DataFrame, futr: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    night = (futr['radiacion-global-instantanea'].values < 5)
    for col in cols:
        if col in preds.columns:
            preds.loc[night, col] = 0.0
            preds[col] = preds[col].clip(lower=0.0)
    return preds


def run_dl_test(model_label: str, test_df: pd.DataFrame, results_dir: Path,
                macrozona: str, estacion: str, planta: str,
                strategy: str = "toy", regional_pr: float = 0.75):
    """Evaluación LOPO Cold-Start (day1 + rollout 7d) para un modelo NeuralForecast.

    model_label: nombre del modelo en NeuralForecast ("LSTM", "NHITS", "TFT", "Informer").
    regional_pr: Performance Ratio a priori de la macrozona+estación, calculado
                 EXCLUSIVAMENTE con plantas de entrenamiento (cero leakage).
    """
    model_dir_name = model_label.lower()
    print(f"[{model_label}] Testeando Planta: {planta} ({estacion}) con Roll-Out a {ROLLOUT_DAYS} dias")
    models_dir = Path(f"models/{strategy}/{model_dir_name}/nf_models")
    if not models_dir.exists():
        print(f"[{model_label}] ERROR: Modelos NeuralForecast no existen en {models_dir}.")
        return

    nf = NeuralForecast.load(path=str(models_dir))
    pred_cols = [f'{model_label}-median', f'{model_label}-lo-90', f'{model_label}-hi-90']
    rename_map = {f'{model_label}-median': 'y_pred',
                  f'{model_label}-lo-90': 'y_pred_lo_90',
                  f'{model_label}-hi-90': 'y_pred_hi_90'}

    # Grilla horaria continua (fix: NeuralForecast exige ventanas contiguas)
    grid = make_hourly_grid(test_df, planta)
    needed = HIST_HOURS + ROLLOUT_DAYS * 24
    if len(grid) < needed:
        print(f"[{model_label}] Planta {planta} sin datos suficientes ({len(grid)} < {needed}h). Saltando.")
        return

    capacidad = float(grid['potencia_neta_mw'].iloc[0])
    static_df = grid[['unique_id'] + STATIC_COLS].drop_duplicates('unique_id')

    # Ground truth real (solo observaciones reales, para métricas honestas)
    real_y = test_df[['unique_id', 'ds', 'y']].copy()

    # 1. Contexto histórico sintético (Cold-Start estricto)
    hist_df = synthetic_context(grid.iloc[:HIST_HOURS], capacidad, regional_pr)

    def _predict_window(current_hist: pd.DataFrame, futr: pd.DataFrame) -> pd.DataFrame:
        futr_input = futr.drop(columns=['y'])
        preds = nf.predict(df=current_hist, futr_df=futr_input, static_df=static_df)
        preds = preds.reset_index()
        return _apply_physics(preds, futr, pred_cols)

    def _save(preds: pd.DataFrame, horizon: str):
        merged = real_y.merge(preds, on=['unique_id', 'ds'], how='inner')
        out = merged.rename(columns=rename_map)
        out_dir = results_dir / model_dir_name / macrozona / estacion / planta / horizon
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_dir / "preds.parquet")
        metrics = calculate_metrics(out['y'], out['y_pred'],
                                    y_lo=out.get('y_pred_lo_90'), y_hi=out.get('y_pred_hi_90'))
        mark_run_completed(out_dir, f"{model_dir_name}_{horizon}", planta, estacion, metrics)
        plot_forecast_rollout(df_real=out, df_pred=out,
                              output_path=out_dir / "forecast_plot.png",
                              model_name=f"{model_label} ({horizon})", planta=planta)

    # 2. Day 1 (24h)
    futr_day1 = grid.iloc[HIST_HOURS:HIST_HOURS + 24].copy()
    _save(_predict_window(hist_df, futr_day1), "day1")

    # 3. Roll-out autorregresivo 7 días (realimenta la predicción, NUNCA la realidad)
    current_hist = hist_df.copy()
    all_preds = []
    for day in range(ROLLOUT_DAYS):
        start = HIST_HOURS + day * 24
        futr_day = grid.iloc[start:start + 24].copy()
        preds = _predict_window(current_hist, futr_day)
        all_preds.append(preds)

        new_tail = futr_day.copy()
        new_tail['y'] = preds[f'{model_label}-median'].values
        current_hist = pd.concat([current_hist.iloc[24:], new_tail], ignore_index=True)

    _save(pd.concat(all_preds, ignore_index=True), "rollout7d")

    del nf
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    print(f"[{model_label}] Roll-Out completado para {planta}.")
