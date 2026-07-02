# Estándares de Implementación ETL

1. **Retorno de Estados para el Orquestador**: 
   Cada función principal de un script ETL (ej. `process_generation`, `process_silver`) DEBE finalizar retornando `True` explícitamente. Esto es crítico para que `src/orchestrator.py` pueda validar que la función terminó con éxito, incluso si se saltó debido a la idempotencia.

2. **Nomenclatura y TDD**: 
   Asegurar consistencia exacta con los tests unitarios en la nomenclatura de columnas. Si los tests en `tests/` exigen caracteres especiales del idioma español (ej. `estacion_año`), el código de producción debe respetarlo estrictamente y no intentar normalizarlo a `estacion_ano` u omitir los caracteres especiales.
