# Naturaleza de los Datos (Lessons Learned - Fase 1)

Durante la ingesta de la Fase 1 (Landing -> Bronze), hemos detectado e interiorizado las siguientes características críticas de los datos:

## 1. Volumen y Rendimiento (Exógenas - CR2)
- **Granularidad y Peso:** Los archivos crudos de CR2 (variables meteorológicas como radiación, humedad, temperatura) contienen millones de filas (pesando >50MB cada uno).
- **Procesamiento Paralelo:** No es factible parsearlos de forma secuencial en un solo hilo. Se **DEBE** usar `concurrent.futures.ProcessPoolExecutor` para leer, filtrar y exportar los CSV en paralelo, calculando los hilos de manera agnóstica (`max(1, os.cpu_count() - 1)`).
- **Escritura CSV en Hilos:** Pandas `.to_csv(mode='a')` no es thread-safe/process-safe si múltiples hilos escriben sobre el mismo archivo simultáneamente. La estrategia adoptada es que cada proceso retorne un diccionario de DataFrames (`{safe_name: group}`), y el hilo principal se encargue de ensamblar y escribir en disco secuencialmente.

## 2. Fechas y Truncado (Cold-Start)
- **Formato Atípico:** La variable `Momento_medicion` en los datos CR2 viene en un formato puramente textual atípico: `YYYY-MM-DD-HH-MM` (ej. `2013-05-15-12-00`).
- **Conversión Costosa:** Parsear millones de filas de este string con `pd.to_datetime(format='mixed')` es prohibitivamente lento (Memory & CPU bottleneck).
- **Solución Óptima de Truncado:** Dado que nos interesa aislar un período específico (generalmente 2014-2024, dictaminado por los datos de generación), el truncado temporal debe hacerse **antes** de parsear como fecha real, utilizando un filtrado puramente de string:
  ```python
  valid_years = tuple(str(y) for y in range(2014, 2025))
  df[date_col].astype(str).str.startswith(valid_years)
  ```
  Esto reduce dramáticamente el tamaño del dataframe en milisegundos, permitiendo que el pipeline sea veloz.

## 3. Limitaciones del File System (Windows)
- Los nombres de las plantas solares pueden contener caracteres ilegales para Windows (`< > : " / \ | ? * '`).
- Cualquier nombre derivado de los datos crudos debe pasar por un `re.sub()` estricto antes de intentar crear un archivo o directorio.

## 4. Normalizacion Fechas y Data Contracts
- **Problemas de Timezone (CR2):** Usar pd.to_datetime(format='mixed') en fechas del clima chileno genera fallas. Se DEBE forzar utc=True y luego eliminarla con .dt.tz_localize(None).
- **El Periodo Valido:** Rango historico util estricto es **2014 a 2024**.
- **Data Contract Bronze:** Estandarizacion a `ds` y `unique_id`.

## 5. Manejo de Outliers, Nulos y Eficiencia
- **Outliers CR2 (-9999):** Los sensores fallidos o datos inexistentes están fuertemente marcados con `-9999.0` o `-9999`. Éstos deben ser descartados tempranamente en la extracción (Capa Bronze).
- **Imputación (Silver):** Debido al rezago de las instituciones (ej. Dirección Meteorológica) en publicar datos de 2024, el cruce con Generación deja vacíos masivos de clima a final de año. Estos **NO** deben rellenarse con `0.0` (destruye los promedios físicos); se utiliza `ffill` y `bfill` agrupado por macrozona. Obligatoriamente se crea una variable dummy (`<variable>_is_imputed = 1`) para evitar sesgos en los modelos ML. Estas imputaciones deben ser registradas en `data/silver/imputation_log.csv`.
- **Eficiencia de Plantas y Macrozonas:** Reemplazamos el Performance Ratio (PR) horario por estar lleno de ruido. Calculamos una **Eficiencia Estática**: `(Media Generación) / Potencia Neta MW`. Ahora se calculan dos niveles para el modelo: `eficiencia_planta` y `eficiencia_macrozona` (ambos agrupados por la estación del año).

## 6. Arquitectura de Testing (Capa Gold)
- **Train vs Test:** La Capa **Silver** (silver_unified.parquet) es la única fuente de datos para el **Entrenamiento**.
- **Validación LOPO:** La Capa **Gold** se utiliza *exclusivamente* para guardar las particiones de validación Leave-One-Plant-Out. 
- **Estructura Física:** Los datos en Gold se particionan en `data/gold/{macrozona}/{estacion}/{unique_id}_test.parquet`, lo que garantiza que cada planta se evalúe por separado sin contaminación de datos (Leakage) en la fase de inferencia.
