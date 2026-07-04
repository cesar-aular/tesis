# Changelog

## [2026-07-04] - Auditoría profunda pre-corrida (pedido de César) + fixes
Auditoría en 3 frentes antes de la corrida desatendida: skill lopo-leakage-audit,
agente eval_scientist independiente, y revisión línea a línea. Correlaciones
feature-y limpias (<0.6 en ambos silver); split LOPO, semilla sintética,
rollouts autorregresivos y prior train-only verificados correctos.

**Hallazgos corregidos:**
- 🔴 **Ventanas XGB vs DL desalineadas (21% de las ventanas, 74/94 plantas)**:
  XGB seleccionaba/rebanaba ventanas posicionalmente sobre la serie con huecos;
  DL sobre la grilla continua → evaluaban horas calendario distintas. Fix:
  grilla canónica compartida (`utils/grid.py`); XGB train/test operan sobre la
  grilla, métricas solo sobre observaciones reales (y=NaN de huecos excluido).
- 🔴 **Leakage latente de lags calendario**: la y de TODAS las ventanas se
  enmascara antes de calcular lags del train local (un lag de una fila vecina
  apuntaría dentro de la ventana evaluada). Test lo verifica.
- 🟡 **Idempotencia**: marker de completitud POR PLANTA (antes: rollout raw
  bastaba → una falla en ventanas estacionales las perdía para siempre).
- 🟡 **Skew de features cíclicas (M1, eval_scientist)**: grid recalculaba
  sin/cos_day_month con /31 fijo y day_year /366; Silver usa days_in_month
  reales y 365.25 → desplazamiento train/test para xgb_global. Igualadas.
- 🟡 **Semilla sintética ~3.5x deprimida (M2)**: el prior regional promediaba
  TODAS las horas (noches en 0) pero se usaba como amplitud PICO del perfil.
  Ahora: mean(y | y>0)/cap (eficiencia diurna) — amplitud correcta.
- 🟡 **rRMSE=0.0 centinela (M3)**: media cero ahora reporta NaN (0.0 hacía
  parecer perfecta una rampa todo-cero mal predicha).
- 🟡 **Cobertura nocturna regalada (M5)**: nueva Coverage_90_diurna (de noche
  lo=hi=y=0 da ~50% de cobertura sin mérito).
- Guards de ventana vacía en _save DL; módulos tune XGB muertos eliminados
  (tuneaban incluyendo las ventanas de evaluación si alguien los conectaba).
- Sondas medidas (fp32, base completa, 4x512 stride 6): TFT 0.4s/step pico
  4.94GB; LSTM 0.1 / NHITS 0.1 / Informer 0.2 → half ~12-18h, total ~2 días.

**Deuda documentada (no bloquea):** baseline local entrena con historia
posterior a la ventana raw (diseño declarado, favorece al local — explicitar
en tesis); estacion_año ffilled en filas de grilla (no puntúan métricas);
Bronze mode='a' + keep='last' no determinista entre corridas de ETL
(reproducibilidad ETL, Silver actual no afectado). Suite: 24 tests verdes.

## [2026-07-02] - Rigor del benchmark: XGB cuantílico, rollout justo, fp32, métricas ampliadas
- **XGB cuantílico (local y global)**: `reg:quantileerror` con alpha=[.05,.5,.95]
  -> P5/P50/P95 en un solo modelo, reparación de cruces de cuantiles, bandas
  `lo-90/hi-90` idénticas a MQLoss(level=[90]) -> **Coverage_90 y Pinball ahora
  existen para los 8 modelos** (pedido de César).
- **🔴 Teacher forcing eliminado (xgb_local rollout)**: los lags de los días
  2-7 usaban la generación REAL de la propia ventana de evaluación (siete
  day-ahead con realimentación perfecta, no un pronóstico semanal). Ahora es
  autorregresivo: realimenta la P50 predicha, igual que el roll-out DL.
  Medido en toy: day1 32.7% -> rollout 70.7% (la degradación ahora es visible
  y honesta). lag_168 sigue apuntando al contexto real (legítimo local).
- **fp32 completo** (decisión: corrección > velocidad): eliminado `16-mixed`,
  los flags por spec y el retry-NaN del orquestador (parche del síntoma).
  Presupuesto reajustado: batch 16x512, épocas 10 (half) / 12 (total),
  patience 7, topes 1200/2000. Corrida half estimada ~10-16h.
- **Warnings resueltos EN ORIGEN, no ocultados** (política nueva): quiet.py ya
  no tiene ningún filterwarnings (solo orden DLL anti-segfault + niveles de
  logging INFO). Corregidos: device GPU/CPU alineado post-fit en XGB
  ("Falling back to prediction"), `val_check_steps` acotado al presupuesto de
  steps, futr_df de tuning exacto (24h/planta; "Dropped N unused rows").
  Toy completo: 0 warnings.
- **Métricas ampliadas** (`calculate_metrics`): MBE (sesgo con signo), MAPE
  solo sobre horas productivas (y>0; en cero real el porcentaje es indefinido),
  WMAPE (pondera por energía, robusta a intermitencia), R² (NaN sin varianza).
  Consolidado con las columnas nuevas. Tests con valores conocidos (20 verdes).
- **Complejidad vs error**: tab_params.tex + fig_complejidad.png — DL: pesos
  entrenables (torch); XGB: nodos totales del ensamble (splits+hojas).
- **Física compartida**: `utils/physics.py` (umbral radiación + noche
  astronómica 23-04h) usada por XGB y DL por igual.

## [2026-07-02] - Optimización del entrenamiento DL (decisiones de César)
- **Decisiones**: mantener LOPO por planta (máxima base estadística; se descartó
  el LOPO agrupado 5-fold); estructurar la evaluación como zero-shot POR
  ESTACIÓN con todas las plantas (las ventanas estacionales pasan de análisis
  de sensibilidad a eje principal del reporte). Foundation models descartados
  (se sigue entrenando desde cero). ⚠️ "Entrenar con TODA la base y luego
  zero-shot" se descartó por leakage (la planta objetivo quedaría en el train).
- **Normalización por capacidad (DL)**: `normalize_target` — y_norm = y /
  potencia_neta_mw (metadata estática, cero fuga). Todas las plantas en [0,1]:
  el modelo global aprende forma, no escala. Aplica a train/tune (global y
  local); las predicciones se des-normalizan a MWh en test (cuantiles MQLoss
  equivariantes a escala). XGB queda en MWh (baseline intacto).
- **Semilla sintética guiada por radiación**: y_norm = PR_regional ×
  (GHI/GHI_max del contexto) — captura duración del día, estacionalidad y
  nubosidad; fallback a campana horaria si el hueco imputado dejó la radiación
  sin señal (gmax <= 50). Ataca la subcobertura (varianza del scaler realista).
- **Máscara nocturna astronómica**: horas 23-04 forzadas a 0 además del umbral
  de radiación (elimina el piso nocturno residual con radiación interpolada).
- **Tune**: 5 trials (antes 3) sobre datos normalizados; hiperparámetros y
  modelos previos archivados (representación del target cambió).
- **Baseline preservado**: results/half -> results/half_prenorm y models/half
  -> models/half_prenorm (comparación antes/después = material del capítulo V).
- Reporte estacional: tabla por horizonte (day1 + rollout7d) con fila de n.
- `src/ml/utils/coldstart.py` (muerto, campana legacy) eliminado; su test
  reescrito contra la semilla real de producción. Suite: 17 tests verdes.

## [2026-07-02] - Corrida half completa + tesis con resultados reales
- **Corrida half completada** (47 folds LOPO, 8 modelos, EXIT 0 en dos pasadas:
  globales + locales incrementales por idempotencia). Hallazgos de la corrida:
  - NHITS_LOCAL divergió a NaN en 12/47 plantas (fp16 sobre serie única) y
    NHITS/TFT globales en 2 plantas → **fix**: locales DL siempre fp32 +
    reintento automático en fp32 ante NaN en globales (commit 7ad6171).
    Pendiente re-ejecutar los folds faltantes (decisión: otro momento).
  - Resultados (rRMSE mediano, rollout7d operacional): XGB local 68.7% ·
    XGB global 86.2% · LSTM 115.8% · LSTM local 132.5% · TFT 150.5% ·
    Informer 153.2% · NHITS 163.7%. Invierno es la estación más difícil para
    todos. Cobertura de intervalos 90% severamente subcalibrada en globales
    (TFT 0.00, NHITS 0.01 vs ideal 0.90) — el contexto sintético comprime la
    varianza del escalador local.
- **PR regional por ventana estacional**: la semilla sintética de cada ventana
  usa el PR de SU estación (antes: siempre el de puesta en marcha).
- **Reporte clarificado**: columna `Estacion_Conexion` (estación del primer
  registro) vs `Window` (estación donde inicia la ventana evaluada) — una
  planta conectada en Verano también se evalúa con ventana de Invierno
  (sensibilidad por diseño). Nuevo agregado global `window_sensitivity_global.png`.
- **Nuevo `src/ml/report_tesis.py`**: genera 11 figuras + 8 tablas .tex desde
  los resultados hacia `latex-tesis/{figuras,tablas}`.
- **Tesis LaTeX completada con datos reales**: 9 placeholders reemplazados por
  figuras generadas (mapa muestra, TFT, cross-site, solución, ETL, LOPO,
  protocolo de evaluación, medallion, resultados); Gantt pgfgantt real en chp1
  (el duplicado de chp4 se sustituyó por referencia cruzada — figs 4.8/4.9
  ordenadas); chp5 Fase 3 completa (6 tablas + 3 figuras + análisis + contraste
  de hipótesis matizado); chp6 conclusiones/trabajo futuro; anexos con métricas
  por planta e imputación; factibilidades redactadas; resumen/abstract con
  resultados; bibliografía sin warnings APA (@online→@misc, volume/number).
  Compila 0 errores / 0 refs indefinidas (52 págs).
- **Limitación documentada**: el forzamiento nocturno a cero depende de la
  radiación CR2 (interpolada en huecos largos puede quedar ≥5 W/m² de noche)
  → piso nocturno residual en rollouts DL; máscara astronómica propuesta como
  trabajo futuro. Afecta a todos los modelos por igual (comparación justa).

## [2026-07-02] - Baselines DL estrictamente locales (lstm_local, nhits_local)
- Resuelve la discrepancia con el anteproyecto (LSTM debía ser línea base LOCAL):
  nuevo `src/ml/utils/dl_local.py` entrena LSTM y N-HiTS por planta usando SOLO su
  historia propia, con TODAS las ventanas de evaluación excluidas (anti train-on-test)
  y contexto de inferencia real (el paradigma local dispone de sus sensores; huecos
  puntuales del contexto se interpolan solo como input del modelo, nunca para train
  ni métricas).
- Hiperparámetros reutilizados del modelo global de la estrategia (tunear 2×47
  modelos locales es prohibitivo); tope 300 steps + early stopping.
- Integrado al orquestador con idempotencia propia (`{modelo}_local`): una segunda
  pasada tras la corrida half agrega solo los locales sin repetir lo completado.
- TDD: exclusión de ventanas y no-imputación del target (13 tests verdes).
- Ruido de ejecución silenciado (`quiet.py`) + fix segfault pyarrow/lightning (DLL
  Windows) que mataba la corrida half al cargar Silver.

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
- **Estabilidad numérica medida a escala half**: LSTM (recurrente) e Informer
  (atención prob-sparse) divergen a NaN en fp16 → fijados a fp32 vía `MODEL_SPECS`;
  TFT y NHITS mantienen fp16 (estables verificados). `gradient_clip_val=1.0` global.
  Guards: trial NaN→9999; todos-fallan→defaults; predicción NaN aborta solo ese
  modelo/planta; try/except por modelo en el orquestador (un fallo no mata 8h de corrida).
- **XGBoost en GPU** (`device='cuda'`, fallback CPU): fit global de 4.3M filas en ~4s.
- **Benchmark medido (RTX 2060, escala half, por fold)**: TFT 186s · LSTM ~105s ·
  Informer 44s · NHITS 15s · XGB ~10s · overhead ~45s ≈ **~6.7 min/fold**.
  Proyección: `half` (47 folds) ≈ **5.5-6.5h** (antes: días); `total` (94 folds,
  2 años, tope 1500 steps) ≈ **20-26h** — ajustable bajando el tope de steps.

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
