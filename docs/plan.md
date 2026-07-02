# Plan de Implementación: Orquestador Central

## Objetivo
Desarrollar el núcleo de ejecución del proyecto (`src/orchestrator.py`). Este orquestador será el único punto de entrada para ejecutar las fases del proyecto. Garantizará la **integridad** (ejecutando y validando los tests antes del código de producción) y la **idempotencia** (verificando la existencia de datos previos para no reprocesar innecesariamente a menos que se le instruya).

## Estrategia e "Inyección de Memoria"
1. **Memoria Agentica:** Se creará un nuevo archivo de reglas `.gemini/rules/orchestration-mandates.md` para que los agentes futuros sepan que el orquestador es la columna vertebral de ejecución y que NUNCA deben ejecutarse scripts aislados en producción.
2. **Pipeline de Integridad (Tests First):** El orquestador correrá `pytest` internamente. Si los tests fallan (rojo), la ejecución de la tubería de datos se aborta de inmediato.
3. **Control de Idempotencia:** El orquestador validará si los archivos de salida (ej. `data/bronze/maestro_generacion.parquet`) ya existen. Si existen, saltará la fase a menos que se invoque con un flag `--force`.

## Cambios Propuestos

### 1. Inyección de Regla
- **[NUEVO]** `.gemini/rules/orchestration-mandates.md`: Documentará las reglas de integridad, idempotencia y el rol del orquestador.

### 2. Desarrollo del Orquestador
- **[NUEVO]** `src/orchestrator.py`
  - Utilizará `pytest.main()` para correr los tests de forma programática.
  - Implementará funciones de validación de estado (idempotencia) verificando qué outputs ya están generados en `data/bronze/`.
  - Agrupará las ejecuciones (ETL Generación y ETL Exógenas) en un bloque unificado.
  - Soportará parámetros de CLI (ej. `python src/orchestrator.py --force` para forzar re-ejecución, o `python src/orchestrator.py` para ejecución continua/segura).

### 3. Actualización de Pipelines
- Modificaremos ligeramente `landing_to_bronze_generation.py` y `landing_to_bronze_exogenous.py` para que sus funciones devuelvan el estado de ejecución y puedan ser importadas y manejadas elegantemente por el orquestador.

## Preguntas y Revisión Requerida
> [!IMPORTANT]
> 1. Al decir "ejecutar de forma continua", ¿te refieres a que el script se quede corriendo en un bucle infinito (como un *daemon* o servicio de fondo vigilando la carpeta `landing`), o a que el orquestador sea un "pipeline continuo" (que ejecuta TDD -> ETL1 -> ETL2 de un solo golpe cada vez que tú lo invoques)?
> 2. He diseñado el orquestador para soportar un flag `--force`. ¿Estás de acuerdo con este mecanismo de idempotencia para poder sobrescribir resultados anteriores de forma deliberada?

Por favor, confirma estas dos preguntas para sellar el plan, actualizar tus reglas (`.gemini/`) y construir el orquestador.

---

# Fase 2: Capa Silver (Macrozona & Feature Engineering)

El objetivo de esta fase es diseñar la arquitectura de la Capa Silver, adaptando la metodología solicitada: **abandonar la asociación por K-Nearest Neighbors (KNN)** en favor de un enfoque por **Macrozonas Geográficas**, y construir la matriz unificada de features.

> [!IMPORTANT]
> **Cambio Metodológico Significativo**
> Este enfoque asume un comportamiento climático más homogéneo por macrozona y reduce la carga computacional del cruce KNN. Disminuye la sensibilidad a microclimas muy específicos, pero mejora la generalización del modelo.

## 1. Mapeo de Macrozonas

Para poder agrupar tanto las plantas solares (que tienen `region`) como las estaciones meteorológicas (que solo tienen `latitud`/`longitud`), estableceremos un puente:

### 1.1 Plantas Solares (basado en Región)
- **Norte Grande:** Arica y Parinacota, Tarapacá, Antofagasta
- **Norte Chico:** Atacama, Coquimbo
- **Zona Central:** Valparaíso, Metropolitana, O'Higgins, Maule
- **Zona Sur:** Ñuble, Biobío, La Araucanía, Los Ríos, Los Lagos
- **Zona Austral:** Aysén, Magallanes

### 1.2 Estaciones Meteorológicas / Exógenas (basado en Latitud)
- **Norte Grande:** Latitud > -26.0°
- **Norte Chico:** Latitud entre -26.0° y -32.1°
- **Zona Central:** Latitud entre -32.1° y -36.2°
- **Zona Sur:** Latitud entre -36.2° y -44.0°
- **Zona Austral:** Latitud < -44.0°

## 2. Feature Engineering (bronze_to_silver.py)

### 2.1 Variables Temporales Cíclicas
Para capturar la estacionalidad y los ciclos temporales, calcularemos transformaciones `sin` y `cos` de:
- **Hora del día** (ciclo de 24 horas).
- **Día del mes** (ciclo de 31 días).
- **Día del año** (ciclo de 365 días).
- **Mes** (ciclo de 12 meses).
- **Estación del año** (se preservará el valor categórico/limpio y adicionalmente se generará su codificación cíclica, asumiendo 4 estaciones).

### 2.2 Variables de Rezago (Lags) - *Postergadas*
> [!NOTE]
> La generación de Lags (rezagos) y el manejo del "Cold-Start" han sido removidos de la Capa Silver. Esta transformación computacionalmente intensiva se realizará de manera dinámica usando heurísticas directamente durante el pipeline de **Entrenamiento (Fase de Machine Learning)**, para no inflar la dimensionalidad de los archivos Silver.

### 2.3 Variables Categóricas y Estáticas
- `macrozona` como variables Dummy.
- `potencia_neta_mw` (Capacidad instalada) obtenida del maestro.

## 3. Cálculo del Performance Ratio (PR)

El PR es una métrica de eficiencia calculada por **Macrozona** y **Estación del Año**.

Fórmula propuesta para el Feature `PR_Macrozona_Estacional`:
$$ PR = \frac{\text{Generación Real (MWh)}}{\text{Potencia Neta (MW)} \times \text{Horas del período}} $$

## 4. Entregables en la Capa Silver (Unificados)

A diferencia de capas anteriores, la salida será estrictamente unificada en un solo megadataset que facilita el input para los modelos:
1. `silver_unified.parquet`
2. `silver_unified.csv`

## 5. Diseño Orientado a Pruebas (TDD)
De acuerdo a las reglas estipuladas en `GEMINI.md` y `.gemini/rules/add-tdd.md`, crearemos primero `tests/test_bronze_to_silver.py` con fixtures que validen la correcta asignación de macrozonas por región/latitud, la validez de los senos/cosenos y la generación del dataset único.

---
## Preguntas Abiertas para Revisión
> [!IMPORTANT]
> 1. **Cálculo del PR (Performance Ratio):** ¿Estás de acuerdo con calcular el PR usando el Factor de Planta teórico (Generación / Capacidad Instalada)? ¿O tienes en mente una fórmula distinta que utilice la radiación?
> 2. **Límites de Latitud:** ¿Apruebas los límites de latitudes que propuse arriba para agrupar las estaciones meteorológicas en las Macrozonas?
