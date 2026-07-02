# Project Context: Expert Solar Forecasting & MLOps

## Scientific Context (Anteproyecto & Plan)

El objetivo central de este proyecto de título es resolver el problema de **Cold-Start** (Arranque en Frío) en plantas fotovoltaicas recién inauguradas, empleando un enfoque de **Cross-Site Transfer Learning** (Aprendizaje Multi-Sitio).

1. **Cold-Start & Cross-Site Learning**: Las nuevas plantas fotovoltaicas carecen del historial de datos necesario para los modelos de Machine Learning tradicionales (como XGBoost o LSTM), que requieren ser entrenados localmente. Este proyecto utiliza arquitecturas Deep Learning basadas en Transformers (como TFT e Informer) pre-entrenadas a partir de múltiples plantas consolidadas. Este modelo global transfiere los "patrones atmosféricos" aprendidos para proyectar sobre la nueva planta.
2. **Zero-Shot Integrity (LOPO)**: En la evaluación Leave-One-Plant-Out (LOPO), la planta objetivo debe mantenerse estrictamente invisible durante el escalado global del modelo. Cualquier fuga de datos (Data Leakage), como usar medias o estadísticas de la planta objetivo durante la normalización global, se considera un fallo crítico en el rigor de la tesis.
3. **Probabilistic Forecasting**: Las estimaciones no deben ser previsiones de un solo punto (point-forecasts). El sistema utiliza **Quantile Loss (P10, P50, P90)** para generar bandas de incertidumbre, esenciales para el balanceo y la estabilidad de la red eléctrica.
4. **Physical Consistency**: Las predicciones del modelo deben respetar la física de la generación solar (por ejemplo, proyectar estrictamente cero generación durante la noche y respetar la fuerte correlación entre la Irradiancia GHI y la potencia de salida).
5. **Epoch-Based Training**: Los modelos deben iterar un número estadísticamente significativo sobre el dataset completo. Los pasos de entrenamiento deben ser dinámicos: `Steps = (TotalRows / (BatchSize * StepSize)) * Epochs`.
6. **Multimodal Data**: La ingesta de datos integra información multimodal: variables meteorológicas dinámicas exógenas (GHI, temperatura, viento) junto con características espaciales estáticas (coordenadas geográficas).
