# Changelog

## [2026-07-02] - Optimizaciones de rendimiento (rama `perf-optimizations`)
- **CAUSA RAÍZ de los tiempos absurdos en `half`**: la fórmula heredada de steps
  (`filas/batch × épocas`) ignoraba que cada step de NeuralForecast procesa
  `batch × windows_batch` VENTANAS (16×256=4,096), no 16 filas → programaba
  ~250,000 steps por modelo por fold (≈500 épocas). Corregida:
  `ceil(ventanas/(batch×windows_batch)) × épocas`, con tope (half: 900, total: 1500).
- **Early stopping**: `patience=5` sobre validación (`val_size=168h` por serie),
  chequeo cada 50 steps — si converge antes, para.
- **Precisión mixta fp16** (`16-mixed`): tensor cores de la RTX 2060 (~1.5-2x);
  AMP acumula la pérdida en fp32 — sin riesgo numérico práctico.
- **Batches grandes**: 32×512 sin acumulación de gradiente (VRAM liberada por fp16).
- **Tune cacheado**: hiperparámetros se eligen 1 vez por modelo/estrategia y se
  reutilizan en los 47/94 folds (antes: re-tuneo por fold, 47× el costo). Caveat
  de fuga a nivel de hiperparámetros documentado como despreciable.
- **Sin lightning_logs ni progress bar**: `logger=False`, menos IO.
- Trials de tuning: 3 (antes 5), cortos (~200 steps): rankean configs, no convergen.

## [2026-07-01] - Análisis de Sensibilidad: Ventana Cold-Start Raw vs Operacional
- **Contexto**: la ventana LOPO comenzaba en la primera hora registrada de cada planta,
  que suele caer en la *rampa de puesta en marcha* (ej. alto_solar: primeras 336h = 100%
  ceros; producción sostenida recién un mes después). Métricas degeneradas (sMAPE 200%,
  rRMSE indefinido) al evaluar sobre una planta que aún no opera.
- **Decisión (César)**: evaluar AMBAS definiciones como análisis de sensibilidad.
- **Nuevo** `src/ml/utils/eval_window.py`: `find_operational_start` (primera hora
  productiva que inaugura >10h productivas en las siguientes 72h, umbral 5% capacidad)
  + `eval_window_variants` (raw siempre; operacional solo si difiere >24h y hay largo
  suficiente). TDD: `tests/test_ml_eval_window.py` (3 tests).
- Los 6 modelos evalúan ahora `day1`/`rollout7d` (raw) y `day1_operational`/
  `rollout7d_operational` cuando la planta tiene rampa.
- `xgb_local/train.py` excluye AMBAS ventanas del entrenamiento (anti train-on-test).
- Reporte con columna `Window` (Raw/Operational) y barplots RMSE separados por ventana.
- La generación jamás se modifica: la utilidad solo ELIGE dónde empieza la ventana.

## [2026-07-01] - Auditoría Claude: Leakage crítico, crash Roll-Out, Informer (rama `claude-focused`)
- **🔴 LEAKAGE ELIMINADO**: `PR = y/capacidad` era el target disfrazado (`corr(PR,y)=1.0`);
  alimentado a XGBoost producía rRMSE falso de 0.83% (real: ~56%). Eliminado de Silver.
  Nuevo `src/ml/utils/regional_prior.py`: prior de eficiencia por macrozona+estación
  calculado SOLO con plantas de entrenamiento dentro de cada split LOPO (feature
  `pr_regional` + semilla PR del contexto sintético Cold-Start, antes hardcodeada a 0.75).
- **🔴 LEAKAGE ELIMINADO (xgb_local)**: el baseline local entrenaba con TODA su serie,
  incluida la ventana donde luego era evaluado (train-on-test). Ahora excluye las primeras
  336h (contexto + evaluación).
- **CRASH ARREGLADO**: el Roll-Out DL fallaba con "missing combinations of ids and times"
  en plantas con huecos horarios. Nuevo `src/ml/utils/dl_test.py` compartido reindexa a
  grilla horaria continua (features temporales recalculadas, clima interpolado);
  lstm/nhits/tft ahora son wrappers delgados (~300 líneas duplicadas eliminadas).
- **Idempotencia REAL**: `check_plant_model_done` (glob) reemplaza el check roto que
  nunca coincidía con los markers escritos → los re-runs ya no repiten todo.
- **Nuevo modelo**: Informer (tune/train/test) según anteproyecto; clima como `futr_exog`
  (pronóstico day-ahead legítimamente disponible). PatchTST descartado: sin soporte de
  exógenas en neuralforecast 3.1.9.
- **XGB Local mejorado**: lags autorregresivos [24h, 168h] (`generate_lags` por fin
  conectado).
- **Métricas probabilísticas**: Coverage_90 y Pinball P05/P95; el reporte leía una llave
  inexistente (`rMAE`, siempre 0) — ahora exporta sMAPE/rRMSE/Coverage/Pinball reales.
- **Rendimiento**: silver se carga UNA vez por corrida (antes 5 relecturas de 147MB por
  planta); `toy` submuestrea xgb_global (10 plantas × 1 año); CSV de Silver (4.4GB
  duplicado del Parquet) ahora es opt-in (`write_csv`).
- **Limpieza**: eliminados módulos muertos/duplicados (`ml/metrics.py`, `ml/coldstart.py`,
  `ml/lopo.py`, `ml/utils.py`), `src/ml/scratch/`, `scratch/`, `check_*.py` raíz, binarios
  XGB contaminados y resultados toy obsoletos.
- **Infra**: repo git inicializado (baseline en `master`, trabajo Claude en
  `claude-focused` sin `.gemini/`); `CLAUDE.md` + `.claude/{agents,skills,knowledge}`;
  `prepare_dl_dataset` integrado al orquestador raíz; TDD ampliado con tests
  anti-leakage (7 tests, verdes).

## [2026-06-30] - Orquestador Central e Idempotencia
- Creación de `.gemini/rules/orchestration-mandates.md` para asentar el rol del orquestador en el sistema.
- Implementación de `src/orchestrator.py` que integra validación TDD automática vía `pytest.main()`.
- Incorporación del control de idempotencia y flag `--force` en los procesos de ETL de Generación y Exógenas.
- Resolución de problemas de codificación (delimitadores de CSV chilenos `;`) y corrección de backend de Matplotlib para orquestación continua en consolas sin GUI.

## [2026-06-30] - ETL Fase 1 (Landing a Bronze)
- Refactorización de directrices para remover prefijos numéricos globalmente (`.gemini/`).
- Desarrollo de `src/etl/landing_to_bronze_generation.py` y `src/etl/landing_to_bronze_exogenous.py` de forma separada utilizando la metodología estricta de TDD.
- Desglose de series temporales en formato CSV por planta/estación e ingesta de metadatos maestros en formato Parquet.
- Generación automatizada de gráficos exploratorios para datos crudos en la carpeta `visuales/`.

## [2026-06-30] Agentic Configuration & Documentation Standardization
- **Refactor**: Reestructurado `GEMINI.md` como índice central limpio del proyecto.
- **Added**: Regla de documentación estricta en `.gemini/rules/03-documentation-standards.md` que obliga a planificar en `docs/plan.md`, seguir tareas en `docs/tasks.md` y documentar en `docs/changelog.md`.
- **Added**: Flujo de Change Management documentado en `.gemini/workflows/01-change-management.md`.
- **Added**: Mandatos de Entrenamiento y Modelado (TFT, Idempotencia, STRATEGY=toy) aislados en `.gemini/workflows/02-model-training-workflow.md`.
- **Added**: Contexto Científico y del problema (Cold-Start, LOPO, Cross-Site Learning) extraído hacia `.gemini/knowledge/01-project-context.md`.
- **Config**: Preparado `docs/plan.md`, `docs/tasks.md` y `docs/changelog.md` para iniciar el nuevo flujo de trabajo disciplinado.
