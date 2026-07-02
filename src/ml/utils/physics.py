"""Consistencia física de las predicciones fotovoltaicas (compartida XGB/DL).

Una hora es "noche" (producción imposible, predicción forzada a 0) si:
- la radiación observada/pronosticada es < 5 W/m², O
- es noche astronómica profunda en Chile continental (23:00-04:59), robusta
  frente a radiación interpolada sobre huecos largos que nunca baja de 5.
"""
import numpy as np
import pandas as pd

DEEP_NIGHT_HOURS = {23, 0, 1, 2, 3, 4}
RADIATION_NIGHT_THRESHOLD = 5.0


def night_mask(window: pd.DataFrame) -> np.ndarray:
    """Máscara booleana de horas nocturnas para un DataFrame con ds y radiación."""
    night = window['radiacion-global-instantanea'].to_numpy() < RADIATION_NIGHT_THRESHOLD
    return night | window['ds'].dt.hour.isin(DEEP_NIGHT_HOURS).to_numpy()
