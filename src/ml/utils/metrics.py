import numpy as np


def pinball_loss(y_true, y_pred_q, q: float) -> float:
    """Pérdida pinball (quantile loss) para el cuantil q."""
    diff = np.asarray(y_true) - np.asarray(y_pred_q)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def calculate_metrics(y_true, y_pred, y_lo=None, y_hi=None):
    """
    Calcula RMSE, MAE, sMAPE y rRMSE (punto) y, si hay bandas P5-P95 (level=90),
    métricas probabilísticas: Coverage_90 (ideal ~0.90) y Pinball P05/P95.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    # sMAPE
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.mean(np.where(denominator == 0, 0, numerator / denominator)) * 100

    # rRMSE
    mean_y = np.mean(y_true)
    rrmse = (rmse / mean_y) * 100 if mean_y != 0 else 0.0

    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "sMAPE": float(smape),
        "rRMSE": float(rrmse),
    }

    # Métricas probabilísticas (MQLoss level=[90] -> lo-90 = P5, hi-90 = P95)
    if y_lo is not None and y_hi is not None:
        y_lo = np.asarray(y_lo, dtype=float)
        y_hi = np.asarray(y_hi, dtype=float)
        if len(y_lo) == len(y_true) and len(y_hi) == len(y_true):
            inside = (y_true >= y_lo) & (y_true <= y_hi)
            metrics["Coverage_90"] = float(np.mean(inside))
            metrics["Pinball_P05"] = pinball_loss(y_true, y_lo, 0.05)
            metrics["Pinball_P95"] = pinball_loss(y_true, y_hi, 0.95)

    return metrics
