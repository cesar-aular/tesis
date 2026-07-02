# ML Phase: TDD & Architecture Tasks

## 1. Testing (TDD)
- [x] `tests/test_ml_coldstart.py`: Verificará que el contexto inyectado es 100% sintético.
- [x] `tests/test_ml_leakage.py`: Asegurará que la planta objetivo no existe en el set de entrenamiento LOPO.
- [x] `tests/test_ml_leakage.py::test_regional_pr_is_leakage_free`: El prior regional se calcula SOLO con plantas de entrenamiento.
- [x] `tests/test_bronze_to_silver.py`: Assertion anti-leakage — `PR` no debe existir en Silver.
- [x] `tests/test_ml_idempotency.py`: Validará que los scripts no sobrescriban outputs previos sin confirmación.

## 2. Core ML Utilities
- [x] `src/ml/utils/metrics.py`: RMSE, MAE, sMAPE, rRMSE + Coverage_90 y Pinball P05/P95 (probabilísticas).
- [x] `src/ml/visualize.py`: Gráficos con generación real, forecast start, predicción (roll-out) y bandas de confianza.
- [x] `src/ml/utils/dl_test.py`: Roll-out Cold-Start compartido (grilla horaria continua, contexto sintético, física solar).
- [x] `src/ml/utils/regional_prior.py`: Prior PR regional leakage-free (macrozona+estación, train-only).

## 3. Modeling
- [x] Arquitectura de carpetas `tune/train/test` por modelo.
- [x] `xgb_local`: Baseline local con lags [24,168], SIN train-on-test (excluye ventana de evaluación).
- [x] Global Models (`xgb_global`, `tft`, `nhits`, `lstm`, `informer`): Entrenamiento N-1, evaluación LOPO.
- [x] Inferencia Roll-out 7d con Cold-Start heurístico (PR regional, no hardcodeado).
- [ ] PatchTST: descartado (sin soporte exógenas en neuralforecast 3.1.9). Reevaluar si se actualiza la librería.

## 4. Ejecución (pendiente)
- [x] Estrategia `toy` end-to-end (smoke test, 2 plantas × 6 modelos).
- [ ] Estrategia `half` (regenerar con datos sin leakage).
- [ ] Estrategia `total` + reporte comparativo final para la tesis.
