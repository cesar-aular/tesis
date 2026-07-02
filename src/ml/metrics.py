import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid division by zero
    diff = np.abs(y_true - y_pred)
    return np.mean(np.where(denominator == 0, 0, diff / denominator)) * 100

def rrmse(y_true, y_pred):
    mean_y = np.mean(y_true)
    if mean_y == 0:
        return np.nan
    return (rmse(y_true, y_pred) / mean_y) * 100

def compute_all_metrics(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "rRMSE": rrmse(y_true, y_pred)
    }
