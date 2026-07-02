# Changelog

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
