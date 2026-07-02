# ML Phase: TDD & Architecture Tasks

## 1. Testing (TDD)
- [x] `tests/test_ml_coldstart.py`: Verificará que el contexto inyectado es 100% sintético.
- [x] `tests/test_ml_leakage.py`: Asegurará que la planta objetivo no existe en el set de entrenamiento LOPO.
- [x] `tests/test_ml_idempotency.py`: Validará que los scripts no sobrescriban outputs previos sin confirmación.

## 2. Core ML Utilities
- [x] `src/ml/metrics.py`: Funciones estandarizadas de error (RMSE, MAE, sMAPE, rRMSE).
- [x] `src/ml/visualize.py`: Gráficos con generación real, forecast start, predicción (roll-out) y bandas de confianza (P10-P90).

## 3. Modeling
- [x] `src/ml/tune/`, `src/ml/train/`, `src/ml/test/`: Configurar arquitectura de carpetas.
- [ ] `xgb_local`: Baseline Univariado.
- [ ] Global Models (`xgb_global`, `tft`, `nhits`, `lstm`): Entrenamiento N-1, evaluación LOPO.
- [ ] Implementar Inferencia Roll-out con Cold-Start Heurístico.
