# Change Management Workflow

Siempre que un usuario solicite un cambio que requiera planificación o que involucre cambios de código importantes, sigue estrictamente este flujo de trabajo:

1. **Elaborar el Plan**: Redacta el plan de implementación propuesto dentro de `docs/plan.md`. Detente y espera la aprobación explícita del usuario.
2. **Definir Tareas**: Una vez aprobado el plan, desglosa los pasos técnicos como una lista de verificación en `docs/tasks.md`.
3. **Ejecutar y Rastrear**: Implementa los cambios en el código. Actualiza `docs/tasks.md` marcando los elementos como `[/]` (en progreso) y `[x]` (completado) a medida que avanzas.
4. **Registrar Cambios (Changelog)**: Al finalizar, añade un resumen de todos los cambios aplicados en `docs/changelog.md` para conservar un historial detallado de las modificaciones realizadas.
