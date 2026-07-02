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


def eval_window_variants(y: pd.Series, capacity: float) -> dict:
    """Variantes de ventana para el análisis de sensibilidad.

    Devuelve {sufijo: índice_inicio}. ``raw`` siempre existe (inicio 0);
    ``operational`` solo si hay producción sostenida, difiere del raw en más de
    un día y deja largo suficiente para contexto + evaluación.
    """
    variants = {"": 0}  # raw (sin sufijo: compatibilidad con resultados previos)
    start_op = find_operational_start(y, capacity)
    if (start_op is not None and start_op > 24
            and start_op + WINDOW_HOURS <= len(y)):
        variants["_operational"] = start_op
    return variants
