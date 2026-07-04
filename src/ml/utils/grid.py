"""Grilla horaria continua por planta — REPRESENTACIÓN CANÓNICA de evaluación.

Todos los modelos (XGB y DL) seleccionan y evalúan sus ventanas Cold-Start
sobre esta grilla: mismo inicio, mismas horas calendario, mismos huecos de
generación (y = NaN). Sin esto, un slice posicional sobre la serie cruda con
huecos evalúa fechas distintas que la grilla (auditado: 103/499 ventanas con
huecos internos, 74/94 plantas) y rompe la comparación 1:1.

- y queda NaN en huecos (las métricas puntúan sólo observaciones reales).
- Clima se interpola/ffill (conocido u obtenible de pronósticos, sin leakage).
- Features temporales y estacion_idx se recalculan desde ds.
- Estáticas y categóricas se propagan (constantes por planta).
- Las dummies *_is_imputed se fuerzan a 1 en las filas creadas por la grilla
  (su clima ES interpolado).
"""
import numpy as np
import pandas as pd

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
    if 'sin_hour' in df.columns:
        df['sin_hour'] = np.sin(2 * np.pi * ds.dt.hour / 24)
        df['cos_hour'] = np.cos(2 * np.pi * ds.dt.hour / 24)
    if 'sin_month' in df.columns:
        df['sin_month'] = np.sin(2 * np.pi * ds.dt.month / 12)
        df['cos_month'] = np.cos(2 * np.pi * ds.dt.month / 12)
    if 'sin_season' in df.columns:
        season_num = ds.dt.month.map(SEASON_MAP).map(SEASON_NUM)
        df['sin_season'] = np.sin(2 * np.pi * season_num / 4)
        df['cos_season'] = np.cos(2 * np.pi * season_num / 4)
    # MISMAS fórmulas que Silver (bronze_to_silver.generate_cyclical_features):
    # divisor days_in_month real y 365.25 — un divisor distinto desplaza la
    # feature entre train (Silver) y evaluación (grilla) para xgb_global
    if 'sin_day_month' in df.columns:
        days_in_month = ds.dt.days_in_month
        df['sin_day_month'] = np.sin(2 * np.pi * ds.dt.day / days_in_month)
        df['cos_day_month'] = np.cos(2 * np.pi * ds.dt.day / days_in_month)
    if 'sin_day_year' in df.columns:
        df['sin_day_year'] = np.sin(2 * np.pi * ds.dt.dayofyear / 365.25)
        df['cos_day_year'] = np.cos(2 * np.pi * ds.dt.dayofyear / 365.25)
    return df


def make_hourly_grid(test_df: pd.DataFrame, planta: str) -> pd.DataFrame:
    """Reindexa la serie de la planta a una grilla horaria continua."""
    # Dedup defensivo: reindex exige indice unico (una fila por hora)
    test_df = test_df.sort_values('ds').drop_duplicates(subset=['ds'], keep='last')
    original_ds = set(test_df['ds'])
    full_range = pd.date_range(test_df['ds'].min(), test_df['ds'].max(), freq='h')
    grid = (test_df.set_index('ds')
            .reindex(full_range)
            .rename_axis('ds')
            .reset_index())
    grid['unique_id'] = planta
    new_rows = ~grid['ds'].isin(original_ds)

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

    # Resto de columnas (categoricas de silver_unified, pr_regional, dummies):
    # propagar; las dummies de imputacion se fuerzan a 1 en filas nuevas
    handled = {'unique_id', 'ds', 'y', 'estacion_idx'} | set(WEATHER_COLS) | set(STATIC_COLS)
    for col in grid.columns:
        if col in handled or col.startswith(('sin_', 'cos_')):
            continue
        grid[col] = grid[col].ffill().bfill()
        if col.endswith('_is_imputed'):
            grid.loc[new_rows, col] = 1.0
    return grid
