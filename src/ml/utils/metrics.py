"""Métricas de evaluación de pronóstico fotovoltaico.

Coherencia con la física solar (series con ~50% de horas nocturnas en cero):
- MAPE se calcula SOLO sobre horas productivas (y > 0): en horas de cero real
  el porcentaje es indefinido; incluirlas con un epsilon fabrica errores
  infinitos que no miden nada.
- WMAPE (suma de errores / suma de reales) es la variante robusta recomendada
  para series intermitentes: pondera por energía, insensible a los ceros.
- rRMSE normaliza por la generación media de la ventana (incluye noches):
  útil para comparar entre plantas, esperar valores >100% en horario horario.
- R² usa la definición estándar 1 - SS_res/SS_tot (NaN si la ventana no tiene
  varianza, p. ej. rampa completamente en cero).
- MBE (sesgo medio) conserva el signo: >0 sobrepronóstico, <0 subpronóstico.
"""
import numpy as np


def pinball_loss(y_true, y_pred_q, q: float) -> float:
    """Pérdida pinball (quantile loss) para el cuantil q."""
    diff = np.asarray(y_true) - np.asarray(y_pred_q)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def calculate_metrics(y_true, y_pred, y_lo=None, y_hi=None):
    """
    Métricas puntuales: RMSE, MAE, MBE, sMAPE, MAPE (horas productivas),
    WMAPE, rRMSE y R². Si hay bandas P5-P95 (MQLoss level=90 o XGB cuantílico):
    Coverage_90 (ideal ~0.90) y Pinball P05/P95.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mbe = float(np.mean(y_pred - y_true))  # >0: sobrepronóstico

    # sMAPE: el 0/0 nocturno se define como 0 vía np.where (sin warning)
    numerator = np.abs(err)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        smape = float(np.mean(np.where(denominator == 0, 0.0,
                                       numerator / denominator)) * 100)

    # MAPE SOLO sobre horas productivas (y>0); NaN si la ventana es toda cero
    productive = y_true > 0
    if productive.any():
        mape = float(np.mean(np.abs(err[productive] / y_true[productive])) * 100)
    else:
        mape = float("nan")

    # WMAPE: robusta a intermitencia (pondera por energía)
    sum_abs_y = float(np.sum(np.abs(y_true)))
    wmape = float(np.sum(np.abs(err)) / sum_abs_y * 100) if sum_abs_y > 0 else float("nan")

    mean_y = float(np.mean(y_true))
    rrmse = (rmse / mean_y) * 100 if mean_y != 0 else 0.0

    # R²: NaN si la ventana no tiene varianza (SS_tot = 0)
    ss_tot = float(np.sum((y_true - mean_y) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan")

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MBE": mbe,
        "sMAPE": smape,
        "MAPE": mape,
        "WMAPE": wmape,
        "rRMSE": float(rrmse),
        "R2": r2,
    }

    # Métricas probabilísticas (bandas P5-P95, nivel 90%)
    if y_lo is not None and y_hi is not None:
        y_lo = np.asarray(y_lo, dtype=float)
        y_hi = np.asarray(y_hi, dtype=float)
        if len(y_lo) == len(y_true) and len(y_hi) == len(y_true):
            inside = (y_true >= y_lo) & (y_true <= y_hi)
            metrics["Coverage_90"] = float(np.mean(inside))
            metrics["Pinball_P05"] = pinball_loss(y_true, y_lo, 0.05)
            metrics["Pinball_P95"] = pinball_loss(y_true, y_hi, 0.95)

    return metrics
