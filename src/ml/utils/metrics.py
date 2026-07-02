import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calcula RMSE, MAE, sMAPE y rRMSE.
    """
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # sMAPE
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.mean(np.where(denominator == 0, 0, numerator / denominator)) * 100
    
    # rRMSE
    mean_y = np.mean(y_true)
    rrmse = (rmse / mean_y) * 100 if mean_y != 0 else 0.0
    
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "sMAPE": float(smape),
        "rRMSE": float(rrmse)
    }
