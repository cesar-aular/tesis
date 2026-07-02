# Model Training Workflow Mandates

Al ejecutar scripts relacionados con el entrenamiento o evaluación de modelos (Módulos `00` al `06`), adhiérete estrictamente a los siguientes mandatos de flujo:

1. **Strategic Scaling**: Antes de ejecutar una suite de entrenamiento completa, SIEMPRE verifica la lógica del código en un entorno a pequeña escala usando `STRATEGY=toy`.
2. **Idempotency**: Todos los scripts deben comprobar la existencia de salidas previas (por ejemplo, `results/lopo_*.csv`) para admitir la reanudación de pipelines interrumpidos. No sobrescribas resultados sin solicitar confirmación.
3. **Error Handling**: Cada ejecución de entrenamiento debe estar envuelta en bloques `try-except`. Los fallos deben registrarse en `resultados/execution_errors.log` sin detener la ejecución global del pipeline.
4. **XAI Integration**: Los modelos basados en Temporal Fusion Transformer (TFT) deben exportar obligatoriamente sus pesos de atención (attention weights) para permitir su validación física posterior (según el paso 2 del plan del anteproyecto).
5. **Automated Validation**: Después de realizar CUALQUIER modificación en los scripts de `00` a `06`, DEBES ejecutar `.venv\Scripts\python.exe -m pytest tests/` para comprobar que las restricciones matemáticas y de data leakage no se han visto comprometidas.
