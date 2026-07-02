# KI: Prevención de TypeError en pandas.groupby().mean()

**Contexto:**
Al promediar DataFrames que contienen columnas de timestamp o strings, las versiones modernas de pandas arrojan un `TypeError` fatal si no se ignora explícitamente el texto. En este proyecto, este error detuvo el orquestador entero durante el procesamiento de la Capa Silver.

**Solución Técnica Estandarizada:**
Siempre que se realice una agregación por región o macrozona usando `.mean()` en pandas, se debe forzar el argumento `numeric_only=True`.

**Implementación Aprobada (Obligatoria):**
```python
# CORRECTO
df_agrupado = df_concat.groupby(df_concat.index).mean(numeric_only=True)

# INCORRECTO (Causará caída del orquestador si hay columnas no numéricas)
df_agrupado = df_concat.groupby(df_concat.index).mean()
```
