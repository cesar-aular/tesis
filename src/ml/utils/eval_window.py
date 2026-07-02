"""Definición de ventanas de evaluación Cold-Start (análisis de sensibilidad).

Las plantas recién conectadas suelen registrar una "rampa de puesta en marcha":
días o semanas con generación cero (pruebas, conexión parcial) antes de la
operación comercial sostenida (ej. alto_solar: sus primeras 336h son 100% ceros).

La tesis evalúa DOS definiciones de ventana LOPO:
- ``raw``:         desde la primera hora registrada de la planta (fiel al registro).
- ``operational``: desde la primera producción sostenida (mide el pronóstico de
                   una planta recién OPERANDO, con métricas interpretables).

REGLA: la generación no se toca — esta utilidad solo ELIGE dónde empieza la
ventana; jamás modifica y.
"""
import pandas as pd

# Ventana LOPO: 168h de contexto sintético + 7 días de evaluación
WINDOW_HOURS = 168 + 7 * 24

# Umbrales de "producción sostenida"
PROD_THRESHOLD = 0.05   # >5% de la capacidad cuenta como hora productiva
MIN_PROD_HOURS = 10     # horas productivas requeridas...
LOOKAHEAD_HOURS = 72    # ...dentro de las próximas 72h (solar: ~10-12h/día)


def find_operational_start(y: pd.Series, capacity: float,
                           threshold: float = PROD_THRESHOLD,
                           min_hours: int = MIN_PROD_HOURS,
                           lookahead: int = LOOKAHEAD_HOURS):
    """Índice posicional de la primera hora con producción sostenida, o None.

    Definición: primera posición t tal que en [t, t+lookahead) la planta produce
    más de threshold*capacity en al menos min_hours horas.
    """
    if capacity <= 0 or y.empty:
        return None

    productive = (y.fillna(0) > threshold * capacity).astype(int)
    # suma de horas productivas hacia ADELANTE (rolling sobre la serie invertida)
    fwd_count = productive[::-1].rolling(lookahead, min_periods=1).sum()[::-1]
    # el inicio debe ser una hora PRODUCTIVA que inaugura el periodo sostenido
    # (sin esto, el lookahead dispara hasta 72h antes de la produccion real)
    sustained = (productive == 1) & (fwd_count >= min_hours)
    if not sustained.any():
        return None
    return int(sustained.to_numpy().argmax())


# Estaciones del año (hemisferio sur) -> sufijo de ventana ASCII-safe
SEASON_MONTHS = {
    "_verano": (12, 1, 2),
    "_otono": (3, 4, 5),
    "_invierno": (6, 7, 8),
    "_primavera": (9, 10, 11),
}


def eval_window_variants(y: pd.Series, capacity: float, dates: pd.Series = None) -> dict:
    """Variantes de ventana para el análisis de sensibilidad.

    Devuelve {sufijo: índice_inicio}:
    - ``''`` (raw): inicio 0, siempre presente.
    - ``'_operational'``: primera producción sostenida (si difiere >24h del raw).
    - ``'_verano'/'_otono'/'_invierno'/'_primavera'``: primera ventana sostenida
      cuyo inicio cae en esa estación (requiere ``dates``; posterior al inicio
      operacional para no evaluar sobre la rampa). Permite comparar el Cold-Start
      en las 4 estaciones del año por planta.
    """
    variants = {"": 0}  # raw (sin sufijo: compatibilidad con resultados previos)
    start_op = find_operational_start(y, capacity)
    if (start_op is not None and start_op > 24
            and start_op + WINDOW_HOURS <= len(y)):
        variants["_operational"] = start_op

    if dates is None or start_op is None:
        return variants

    # Máscara de "inicio sostenido válido" (misma definición que operational)
    productive = (y.fillna(0).reset_index(drop=True) > PROD_THRESHOLD * capacity).astype(int)
    fwd_count = productive[::-1].rolling(LOOKAHEAD_HOURS, min_periods=1).sum()[::-1]
    sustained = ((productive == 1) & (fwd_count >= MIN_PROD_HOURS)).to_numpy()

    months = pd.Series(dates).dt.month.to_numpy()
    n = len(y)
    for suffix, season_months in SEASON_MONTHS.items():
        in_season = pd.Series(months).isin(season_months).to_numpy()
        candidates = sustained & in_season
        candidates[:start_op] = False                     # nunca sobre la rampa
        candidates[max(0, n - WINDOW_HOURS) + 1:] = False  # debe caber completa
        if candidates.any():
            variants[suffix] = int(candidates.argmax())
    return variants
