# Guión de defensa — Examen de Título (20 min)

**César David Aular Muñoz** · Escuela Ingeniería Civil Informática · Universidad Católica del Maule
Presentación: `latex-tesis/defensa.pdf` (defensa) · `latex-tesis/charla_conferencia.pdf` (conferencia)

> **Doble uso — defensa y conferencia.** El mismo deck compila en dos modos (interruptor `\confmode` en `defensa.tex`). El **contenido y los tiempos son idénticos**; solo cambia el encuadre. Para la **conferencia** ajusta cuatro cosas al hablar:
> 1. **Apertura**: en vez de *"hoy defiendo mi proyecto de título"*, di *"presentamos nuestro trabajo sobre pronóstico fotovoltaico en condiciones de arranque en frío"*.
> 2. **Coautoría**: menciona a **Sergio Hernández** como coautor (aparece en portada y cierre del modo conferencia).
> 3. **Encuadre**: habla de *"contribuciones del trabajo"* en lugar de *"objetivos de la tesis"* en la diapositiva de Hipótesis y objetivos.
> 4. **Cierre**: cambia *"quedo atento a sus preguntas"* (comisión) por *"gracias, quedamos atentos a preguntas"* (audiencia). Las diapositivas de respaldo siguen siendo útiles para el turno de preguntas técnico.

---

## Cómo usar este guión

- Texto en *cursiva* = lo que dices en voz alta (puedes parafrasear, no memorizar literal).
- **[acción]** = qué haces o qué señalas en la diapositiva.
- Cada bloque indica su **tiempo objetivo** y el **acumulado**. Meta total: **20:00**.
- Habla ~130 palabras/min. Si te retrasas, recorta la Metodología (no los Resultados ni las Conclusiones).
- Las **diapositivas de respaldo** (RMSE, calibración de XGBoost, cómputo, tabla completa) NO se presentan: se usan solo si la comisión pregunta. Ver la sección final "Ante preguntas".

### Reparto de tiempo por bloque

| Bloque | Diapositivas | Tiempo | Acumulado |
|--------|--------------|--------|-----------|
| Apertura | Portada + Agenda | 1:00 | 1:00 |
| 1. El problema | 3 | 3:15 | 4:15 |
| 2. Metodología | 6 (incl. mapa) | 6:15 | 10:30 |
| 3. Resultados | 5 | 7:00 | 17:30 |
| 4. Conclusiones | 2 | 2:00 | 19:30 |
| Cierre | Gracias | 0:30 | 20:00 |

> **Nota de tiempos.** Se añadió la diapositiva del **mapa** (0:45) en Metodología; se
> compensa recortando 15 s en El problema y 30 s en la Semilla sintética. El total sigue
> siendo 20:00. Si vas apretado, el mapa es la primera candidata a acortar (basta con
> señalar el contraste norte-grande / centro-pequeño).

---

## APERTURA (1:00)

### Portada — *(0:40)*
**[De pie, mirando a la comisión, sin leer aún la diapositiva.]**

*Buenos días. Muchas gracias, profesores, por su tiempo. Mi nombre es César Aular y hoy defiendo mi proyecto de título: un análisis comparativo entre un enfoque multi-sitio y modelos locales para pronosticar, a corto plazo, la generación de plantas fotovoltaicas cuando casi no hay datos disponibles.*

*La pregunta que motiva todo el trabajo es muy concreta y muy operativa: cuando una planta solar recién se conecta a la red y no tiene historia propia, ¿podemos pronosticar su generación aprovechando lo aprendido de otras plantas? Y sobre todo: ¿eso realmente funciona mejor que un modelo local, o no?*

### Agenda — *(0:20)*
**[Avanzas a la Agenda.]**

*La estructura es simple: primero planteo el problema del arranque en frío; luego la metodología, que es donde están las decisiones clave del trabajo; después los resultados sobre 94 plantas reales; y cierro con las conclusiones y lo que rescato de todo esto.*

---

## BLOQUE 1 — EL PROBLEMA (3:30)

### Diapositiva "El problema del Cold-Start fotovoltaico" — *(1:15)*

*El pronóstico solar clásico entrena un modelo por cada planta, usando meses o años de su propia historia. El problema aparece con una planta nueva: en el instante de la conexión no tiene historial de generación, así que un modelo local o no se puede entrenar, o entrega pronósticos muy malos.*

*Esto no es un detalle técnico menor. La incertidumbre aparece justo cuando más se necesita certeza —al arranque— y se traduce en mayores costos de balance para el operador del sistema y en riesgo financiero, en especial para los pequeños generadores distribuidos, los PMGD, que en Chile han crecido muchísimo.*

**[Señalas la figura.]** *Este es el clásico problema de "arranque en frío", el mismo concepto que en los sistemas de recomendación: hay que decidir sin evidencia sobre el caso nuevo.*

### Diapositiva "La idea: Cross-Site" — *(1:15)*

*La idea para salir de esto es abandonar el supuesto de "un modelo por planta". En lugar de eso se entrena un único modelo global sobre muchas plantas a la vez. Ese modelo absorbe los patrones compartidos entre clima y potencia, y luego los transfiere a una planta que nunca vio, de modo que puede pronosticar desde la primera hora de conexión.*

*Ahora bien —y esto es central en la tesis— yo no quise asumir que eso funciona. La pregunta la planteé de forma adversarial: ¿el modelo global realmente supera a una línea base local madura, o el premio de la transferencia se desvanece cuando la comparación se hace de forma justa? Esa honestidad en el planteamiento es la que define todo el diseño experimental.*

### Diapositiva "Hipótesis y objetivos" — *(1:00)*

*La hipótesis formal fue optimista: el modelo global multi-sitio sería más preciso que un modelo entrenado localmente.*

*Los objetivos específicos fueron tres: adaptar una arquitectura Transformer al entrenamiento simultáneo sobre muchas series; evaluarla contra los modelos locales bajo un protocolo de arranque en frío estricto; e integrar datos multimodales sin fuga de información.*

*Y la brecha que detecté en la literatura es que rara vez se combinan tres cosas a la vez: una evaluación verdaderamente ciega a la planta objetivo, sin ajuste fino; un contraste justo contra líneas base locales maduras; y una caracterización probabilística de la incertidumbre. Mi trabajo cierra esas tres a la vez.*

---

## BLOQUE 2 — METODOLOGÍA (6:00)

### Diapositiva "Arquitectura del sistema" — *(1:00)*

*Antes de los modelos, una palabra sobre la ingeniería, porque fue una parte grande del trabajo. Todo el sistema está gobernado por un orquestador único, con un pipeline de datos tipo Medallion —landing, bronze, silver, gold—, una capa de utilidades compartidas que centraliza la lógica sensible a la integridad científica, y un gate de pruebas automatizadas que detiene la ejecución si algo se rompe. Es un pipeline reproducible, no un conjunto de scripts sueltos.*

### Diapositiva "Protocolo Leave-One-Plant-Out" — *(1:15)*

*El protocolo de evaluación es Leave-One-Plant-Out. Para cada planta objetivo, los modelos globales se entrenan con las demás plantas —las N menos 1— usando toda la historia de 2014 a 2024, y la planta objetivo queda completamente invisible durante el entrenamiento. Reporto la mediana sobre las 94 plantas y solo mido sobre observaciones reales.*

**[Enfatizas el bloque verde.]** *La integridad zero-shot fue la restricción no negociable de la tesis: ninguna variable puede ser función de la generación de la planta objetivo. Y esto no lo dejé a la buena fe: está verificado por pruebas automatizadas.*

### Diapositiva "La lección de la fuga de datos" — *(1:30)*

*Y aquí quiero ser transparente con la comisión, porque es la lección metodológica más importante del trabajo. En una versión temprana usé un "prior" de eficiencia calculado por fila desde la propia planta objetivo —básicamente generación dividida por capacidad—. Esa variable tenía correlación 1.0 con lo que quería predecir.*

**[Señalas los dos bloques.]** *El resultado era un error espectacular, del orden de 0,8 %. Espectacular... y falso. Era fuga de información pura. Al detectarlo y corregirlo, el error honesto saltó a alrededor de 56 %.*

*La corrección fue calcular ese prior regional únicamente sobre las plantas de entrenamiento, agregado por macrozona y estación, nunca por fila del objetivo. Menciono esto abiertamente porque distingue un resultado creíble de uno que no lo es, y toda la validez de la tesis descansa sobre este punto.*

### Diapositiva "Semilla Cold-Start sintética" — *(1:15)*

*Un problema práctico: los modelos profundos necesitan una ventana de contexto para arrancar la inferencia, y una planta nueva justamente no la tiene. La solución fue sembrarla de forma sintética: capacidad por el performance ratio regional por un perfil solar determinista.*

*La clave es qué usa datos reales y qué no: solo el clima exógeno —que es conocido o pronosticable— y la geografía estática usan valores reales del objetivo. La generación del contexto es 100 % sintética. Así se garantiza que hay cero fuga, incluso en la propia inicialización del modelo.*

### Diapositiva "Ocho configuraciones y ventanas" — *(1:00)*

*Comparo ocho configuraciones bajo el mismo protocolo: cinco globales —TFT, Informer, N-HiTS, LSTM y un XGBoost global— contra tres líneas base estrictamente locales, entrenadas solo con la historia propia de cada planta.*

*Evalúo en dos horizontes: day1, a 24 horas, y rollout de 7 días autorregresivo. Y, como refinamiento que surgió al ver los datos, distingo ventanas: la Raw, que incluye la rampa de puesta en marcha; la Operacional, desde la primera producción sostenida; y cuatro ventanas estacionales.*

---

## BLOQUE 3 — RESULTADOS (7:00)

### Diapositiva "¿Dónde están las plantas?" (mapa) — *(0:45)*

**[Señalas el mapa.]** *Antes de los resultados, dónde estamos parados. Estas son las 94 plantas sobre el mapa de Chile, a lo largo de todo el gradiente solar, desde Arica hasta la zona sur.*

*Fíjense en el contraste: en el norte hay pocas plantas pero muy grandes, de escala utility, hasta 200 megawatts, con altísima irradiancia. En la zona central hay muchísimas plantas pequeñas —los PMGD, con una mediana de unos 3 megawatts—, que son justamente el segmento que más sufre el arranque en frío, porque son los actores con menos capacidad de análisis.*

*Y un detalle metodológico: la macrozona es la única señal espacial que ven los modelos, y es la clave sobre la que se agrega el prior regional que siembra el contexto sintético.*

### Diapositiva "Precisión puntual" — *(2:00)*

*Vamos a los resultados, que son el corazón de la presentación.*

**[Señalas la tabla y la figura.]** *Primero, una aclaración de métrica. Reportamos el error normalizado por la capacidad instalada, el NRMSE, porque es directamente interpretable: es el error como porcentaje de la potencia nominal de la planta. El rRMSE que aparece al lado divide por la generación media, que incluye las horas de noche, y por eso se dispara sobre 100 %.*

*Con eso: el modelo más preciso es el XGBoost local, con 13,2 % de la potencia nominal. Le sigue el XGBoost global, con 17,1 %, que es el mejor de los modelos globales. Las redes profundas forman un grupo claramente peor, entre 22 y 31 %.*

*Y aquí hay una lección de medición: si uno ordena por rRMSE, el LSTM global parece claramente mejor que el local. Si ordena por NRMSE, el orden se invierte. Comparar medianas marginales de un cociente cuyo denominador cambia por planta no es una base segura para rankear —y eso es exactamente lo que nos llevó a los tests pareados.*

### Diapositiva "Sensibilidad estacional" — *(1:30)*

*Estratifiqué por estación del año, y aquí hay un hallazgo muy claro: el invierno es sistemáticamente la estación más difícil, para todas las arquitecturas sin excepción.*

**[Señalas los saltos.]** *El XGBoost global pasa de 52 % en verano a 134 % en invierno; el local, de 40 a 115 %. La razón física es la mayor variabilidad nubosa y la menor irradiación invernal, que debilitan la señal determinista del perfil solar y amplifican el peso relativo de los errores.*

*La implicación operativa es directa y útil: la incertidumbre de un parque recién conectado depende fuertemente de la estación en que se pone en marcha. Conectar en invierno es el escenario más adverso.*

### Diapositiva "Calidad probabilística" — *(1:30)*

*No basta con el error puntual; en un mercado eléctrico importa la incertidumbre. Y aquí hay otro hallazgo crítico: la calibración probabilística no se transfiere bien.*

*El nominal es 90 %. Ningún paradigma calibra idealmente, pero fallan en direcciones opuestas. Las redes globales quedan subcalibradas —coberturas de 0,36 a 0,65—: sus bandas son demasiado estrechas. El XGBoost global, al otro extremo, sobre-cubre con 0,99: bandas tan anchas que casi nunca fallan, pero que pierden utilidad. Quiero ser claro: eso es un defecto simétrico al de las redes, no una virtud.*

*El más cercano al nominal es el XGBoost local, con 0,86 y la menor pérdida pinball. La razón de fondo en las redes es que su escalador se ajusta sobre el contexto sintético, cuya varianza es menor que la real, y por eso las bandas salen estrechas.*

### Diapositiva "Contraste de la hipótesis — con test estadístico" — *(2:00)*

*Con todo esto, contrasto la hipótesis. Y ahora no con medianas, sino con un test.*

*El diseño es pareado y balanceado: las 94 plantas evaluadas con las 8 configuraciones sobre la misma grilla horaria. Eso permite un Friedman como test global y luego Wilcoxon pareado con corrección de Holm.*

**[Bloque rojo.]** *La hipótesis se rechaza, y ahora con significancia: el XGBoost local gana al XGBoost global en 77 de 94 plantas, con p menor a diez a la menos siete. No es un artefacto de cómo agregamos.*

**[Bloque azul.]** *A igualdad de arquitectura el resultado es, ahora sí, inequívoco: la N-HiTS local supera a la global en 88 de 94 plantas. Y el par LSTM —que en la versión anterior parecía favorecer al global— es un empate estadístico: 49 contra 45 plantas, p igual a 1,00. Quiero ser transparente: el matiz que reportábamos antes no sobrevive al test pareado. La diferencia de medianas la producían unas pocas plantas donde el LSTM local falla mucho, no una ventaja sistemática del global.*

*Dicho de otro modo: no hay ninguna arquitectura en la que la transferencia global supere significativamente a su contraparte local.*

**[Bloque verde.]** *Y aquí está el punto que quiero que quede: el valor real del modelo global no es ganar en precisión, es la disponibilidad inmediata. En el arranque en frío puro, con historia local nula, el modelo global simplemente no tiene competidor local posible, porque el modelo local no puede existir. La contribución no es coronar un ganador absoluto, sino cuantificar con rigor ese compromiso entre disponibilidad inmediata y precisión asintótica.*

### Diapositiva "Ejemplo de roll-out" — *(0:60)*

**[Señalas la figura.]** *Para que se vea concreto: este es un roll-out de 7 días de arranque en frío. El modelo global, sin haber visto jamás esta planta, reproduce el ciclo diario y la magnitud de los picos usando solo el clima entrante y la geografía estática, con el forzamiento físico a cero durante la noche. Es una predicción utilizable desde el día uno.*

---

## BLOQUE 4 — CONCLUSIONES (2:00)

### Diapositiva "Conclusiones" — *(1:15)*

*En síntesis. Primero, este es un resultado negativo, pero un resultado negativo riguroso: bajo un protocolo estrictamente ciego, un XGBoost local maduro sigue siendo el estándar a batir, y la hipótesis se rechaza. Lejos de invalidar el trabajo, esto es un aporte: somete a prueba estricta la creencia extendida en la superioridad de los modelos atencionales.*

*Segundo, el paradigma global conserva un régimen propio y valioso: el arranque en frío puro, donde entrega pronósticos coherentes y accionables desde la hora cero.*

*Y tercero, dejo cuatro hallazgos metodológicos transferibles: las rampas de puesta en marcha distorsionan las métricas; el invierno es la estación más difícil; la calibración probabilística no se transfiere a las redes profundas; y la integridad del target —nunca imputar la generación— es esencial para la honestidad de las métricas.*

### Diapositiva "Trabajo futuro" — *(0:45)*

*El rechazo de la hipótesis no cierra la línea; señala dónde mejorar. La vía más prometedora es el ajuste fino en dos etapas: pre-entrenar global y luego ajustar con los primeros días reales de la planta. También propongo máscaras de intermitencia en la atención para que la red no gaste capacidad aprendiendo la noche; recalibración conformal de los intervalos; y, con más datos y cómputo, verificar si el orden entre arquitecturas se mantiene.*

---

## CIERRE (0:30)

### Diapositiva "Gracias"

*En resumen: pregunté de forma honesta si la transferencia entre plantas vence al modelo local, la respuesta es "no en precisión, pero sí donde el local no puede existir", y lo cuantifiqué sobre 94 plantas reales. Muchas gracias. Quedo atento a sus preguntas.*

---

## ANTE PREGUNTAS — diapositivas de respaldo

Salta a la diapositiva de respaldo correspondiente según lo que pregunte la comisión.

- **"¿Cómo definió el RMSE / por qué penaliza así?"** → *Backup — Definición del RMSE.*
  *El RMSE eleva al cuadrado cada error antes de promediar, así que penaliza de forma cuadrática las desviaciones grandes —no lineal ni logarítmica—. En fotovoltaica eso es deseable: castiga con fuerza los errores en los picos de mediodía.*

- **"¿Por qué el rRMSE pasa del 100 %?"** → *Backup — rRMSE > 100 %.*
  *Porque normaliza por la generación media horaria de la ventana, que incluye las horas nocturnas nulas; eso deprime la media e infla el indicador. Valores sobre 100 % son esperables; el valor está en la comparación relativa sobre ventanas idénticas.*

- **"¿Por qué el XGBoost global sobre-cubre y el local no? ¿No es mejor cubrir más?"** → *Backup — Calibración de XGBoost.*
  *Sobre-cubrir con 0,99 significa bandas demasiado anchas: casi nunca fallan, pero pierden utilidad operativa. Es un defecto simétrico al de las redes. El local, ajustado a la variabilidad real de su planta, queda en 0,86, cerca del nominal, y con menor pinball.*

- **"¿No estará sub-entrenando los Transformers en una GPU de 6 GB?"** → *Backup — Detalles de cómputo.*
  *Es una limitación reconocida y una de las explicaciones candidatas. La VRAM la gobierna batch × windows_batch, no el tamaño del corpus, así que no trunqué la historia; pero TFT e Informer pudieron quedar sub-entrenados, y reentrenar con más presupuesto es trabajo futuro explícito.*

- **"¿Las diferencias son estadísticamente significativas?"** → *Backup — Tests estadísticos.*
  *Friedman da chi-cuadrado 260,7 con p menor a diez a la menos cincuenta, así que las configuraciones no son equivalentes. El ranking medio pone al XGBoost local primero con 1,80. Y por planta, el XGBoost local es el mejor en 60 de 94. Los contrastes por pares usan Wilcoxon con corrección de Holm.*

- **"¿Los baselines locales no están viendo el futuro?"** → *Backup — Causalidad temporal.*
  *Buena observación, y la respuesta honesta es que sí en parte: se entrenan con la historia completa menos las ventanas de evaluación, así que incluyen horas posteriores a la ventana. Los lags hacia las ventanas están enmascarados, o sea no hay fuga del target, pero no es estrictamente causal. Es una elección de diseño: queríamos el rival local más fuerte posible. Y lo importante es la dirección del sesgo: favorece al local, que gana, mientras que el espejo —LOPO oculta la planta pero no el calendario— favorece al global, que pierde igual. Ambos sesgos empujan contra nuestra conclusión, así que la refuerzan.*

- **"¿Por qué NRMSE y no solo rRMSE?"** → *Backup — rRMSE > 100 %.*
  *Mantuvimos rRMSE por continuidad, pero agregamos NRMSE normalizado por capacidad instalada porque es acotado e interpretable como porcentaje de la potencia nominal, y está definido para todas las plantas —el rRMSE queda indefinido cuando la media de la ventana es cero.*

- **"¿Me muestra todos los números?"** → *Backup — Tabla completa (day1 y roll-out).*

### Reglas para el turno de preguntas
- Si no sabes algo, dilo con honestidad y reconduce a lo que sí mediste.
- Vuelve siempre al mensaje central: **rechazo en precisión absoluta, valor en disponibilidad inmediata, todo bajo un protocolo estrictamente ciego.**
- No defiendas la hipótesis: el aporte es haberla refutado con rigor.
