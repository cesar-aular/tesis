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
from src.ml.utils.eval_window import eval_window_variants
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
    # Dedup defensivo: reindex exige indice unico (una fila por hora)
    test_df = test_df.sort_values('ds').drop_duplicates(subset=['ds'], keep='last')
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


def synthetic_context(grid_hist: pd.DataFrame, regional_pr: float) -> pd.DataFrame:
    """Contexto Cold-Start 100% sintético en espacio NORMALIZADO por capacidad.

    Perfil guiado por la radiación del propio contexto (exógena legítima:
    observada en la estación asociada u obtenible de pronósticos) — captura
    duración real del día, estacionalidad y nubosidad, a diferencia de la
    campana gaussiana fija que comprimía la varianza del escalador (causa de
    la subcobertura medida: TFT 0.00 vs 0.90 nominal). Fallback horario si
    la radiación no aporta señal (hueco largo imputado).

    y_norm = PR_regional × perfil ∈ [0, 1]. JAMÁS usa la generación real.
    """
    hist = grid_hist.copy()
    perfil = None
    if 'radiacion-global-instantanea' in hist.columns:
        ghi = hist['radiacion-global-instantanea'].to_numpy(dtype=float)
        gmax = np.nanmax(ghi) if len(ghi) else 0.0
        if np.isfinite(gmax) and gmax > 50:  # hay señal solar real en el contexto
            perfil = np.clip(np.nan_to_num(ghi) / gmax, 0.0, 1.0)
    if perfil is None:
        horas = hist['ds'].dt.hour.to_numpy()
        perfil = np.exp(-0.5 * ((horas - 12) / 3) ** 2)  # campana de respaldo
    hist['y'] = np.clip(regional_pr * perfil, 0.0, 1.0)
    return hist


# Noche astronómica profunda en Chile continental (UTC-3/-4): sin producción
# solar posible en todo el año. Complementa el umbral de radiación, que puede
# no activarse cuando la radiación viene interpolada sobre huecos largos
# (piso nocturno residual observado en la corrida half pre-normalización).
DEEP_NIGHT_HOURS = {23, 0, 1, 2, 3, 4}


def _apply_physics(preds: pd.DataFrame, futr: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    night = (futr['radiacion-global-instantanea'].values < 5)
    night = night | futr['ds'].dt.hour.isin(DEEP_NIGHT_HOURS).to_numpy()
    for col in cols:
        if col in preds.columns:
            preds.loc[night, col] = 0.0
            preds[col] = preds[col].clip(lower=0.0)
    return preds


def run_dl_test(model_label: str, test_df: pd.DataFrame, results_dir: Path,
                macrozona: str, estacion: str, planta: str,
                strategy: str = "toy", regional_pr: float | dict = 0.75):
    """Evaluación LOPO Cold-Start (day1 + rollout 7d) para un modelo NeuralForecast.

    model_label: nombre del modelo en NeuralForecast ("LSTM", "NHITS", "TFT", "Informer").
    regional_pr: Performance Ratio a priori de la macrozona+estación, calculado
                 EXCLUSIVAMENTE con plantas de entrenamiento (cero leakage).
                 Puede ser un dict {sufijo_ventana: pr} para que cada ventana
                 estacional se siembre con el PR de SU estación (una ventana
                 _invierno usa el PR invernal de la macrozona, no el de la
                 estación de puesta en marcha).
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

    capacidad = max(float(grid['potencia_neta_mw'].iloc[0]), 1e-6)
    static_df = grid[['unique_id'] + STATIC_COLS].drop_duplicates('unique_id')

    # Ground truth real (solo observaciones reales, para métricas honestas)
    real_y = test_df[['unique_id', 'ds', 'y']].copy()

    def _predict_window(current_hist: pd.DataFrame, futr: pd.DataFrame) -> pd.DataFrame:
        """Contexto en espacio normalizado -> predicción des-normalizada a MWh."""
        futr_input = futr.drop(columns=['y'])
        preds = nf.predict(df=current_hist, futr_df=futr_input, static_df=static_df)
        preds = preds.reset_index()
        # Guard: un modelo divergente (NaN) no debe matar una corrida de horas;
        # se aborta SOLO este modelo/planta (sin marker -> idempotencia reintenta)
        if preds[f'{model_label}-median'].isna().any():
            raise RuntimeError(
                f"{model_label} produjo predicciones NaN (modelo divergente). "
                "Revisar precision/lr; este modelo/planta se omite.")
        # Des-normalizar: el modelo opera en y/capacidad, las métricas en MWh
        for col in pred_cols:
            if col in preds.columns:
                preds[col] = preds[col] * capacidad
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

    # ANALISIS DE SENSIBILIDAD: dos definiciones de ventana Cold-Start.
    # '' (raw): desde la primera hora registrada (incluye rampa de puesta en marcha).
    # '_operational': desde la primera produccion sostenida (planta ya operando).
    variants = eval_window_variants(grid['y'].reset_index(drop=True), capacidad,
                                    dates=grid['ds'])
    for suffix, start_idx in variants.items():
        if start_idx + needed > len(grid):
            continue
        window = grid.iloc[start_idx:].reset_index(drop=True)

        # PR de la estación de ESTA ventana (dict) o escalar (compatibilidad)
        if isinstance(regional_pr, dict):
            pr_window = regional_pr.get(suffix, regional_pr.get("", 0.25))
        else:
            pr_window = regional_pr

        # 1. Contexto histórico sintético NORMALIZADO (Cold-Start estricto)
        hist_df = synthetic_context(window.iloc[:HIST_HOURS], pr_window)

        # 2. Day 1 (24h)
        futr_day1 = window.iloc[HIST_HOURS:HIST_HOURS + 24].copy()
        _save(_predict_window(hist_df, futr_day1), f"day1{suffix}")

        # 3. Roll-out autorregresivo 7 días (realimenta la predicción, NUNCA la realidad)
        current_hist = hist_df.copy()
        all_preds = []
        for day in range(ROLLOUT_DAYS):
            start = HIST_HOURS + day * 24
            futr_day = window.iloc[start:start + 24].copy()
            preds = _predict_window(current_hist, futr_day)
            all_preds.append(preds)

            new_tail = futr_day.copy()
            # El contexto autorregresivo vive en espacio normalizado
            new_tail['y'] = preds[f'{model_label}-median'].values / capacidad
            current_hist = pd.concat([current_hist.iloc[24:], new_tail], ignore_index=True)

        _save(pd.concat(all_preds, ignore_index=True), f"rollout7d{suffix}")

    del nf
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    print(f"[{model_label}] Roll-Out completado para {planta}.")
