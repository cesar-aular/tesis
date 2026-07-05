# Pronóstico Fotovoltaico Cross-Site bajo Escasez de Datos (Cold-Start)

Proyecto de título — Ingeniería Civil Informática, Universidad Católica del Maule.
**Autor:** César David Aular Muñoz.

> *Análisis comparativo de un enfoque multi-sitio frente a modelos locales para la
> proyección a corto plazo de generación fotovoltaica bajo escasez de datos.*

Este repositorio contiene el pipeline completo (datos, entrenamiento, evaluación y
generación de reportes) que sustenta la tesis, junto con el documento LaTeX.

---

## 🎯 Objetivo científico

Resolver el problema de **arranque en frío** (*Cold-Start*) de plantas fotovoltaicas
recién conectadas —que no tienen historial de generación propio— mediante
**aprendizaje multi-sitio** (*Cross-Site Learning*): un modelo global de Deep Learning
se pre-entrena sobre muchas plantas consolidadas y transfiere los patrones atmosféricos
a una planta nueva con cero historia (inferencia *zero-shot*).

El enfoque se contrasta contra **líneas base estrictamente locales** (XGBoost y las
mismas redes profundas entrenadas de forma aislada) mediante validación
**Leave-One-Plant-Out (LOPO)** sobre 94 plantas del Sistema Eléctrico Nacional chileno.

### 🔴 Principio no negociable: cero fuga de datos (*Zero-Shot Integrity*)

La validez de la tesis depende de que la planta objetivo sea **verdaderamente invisible**
durante el entrenamiento global:

- En LOPO, el modelo global se entrena con **todas las plantas excepto la objetivo**.
- Ninguna característica puede derivarse del `y` (generación) de la planta objetivo.
- El contexto de inferencia Cold-Start es **100 % sintético**
  (`y = capacidad × PR_regional × perfil_solar`), nunca la generación real.
- La generación **jamás se imputa**: si un dato no está, no está; las métricas solo
  puntúan observaciones reales.

Una suite de pruebas anti-fuga (`tests/`) actúa como *gate*: si una regla se rompe, el
pipeline se detiene antes de ejecutarse.

---

## 🏗️ Arquitectura del pipeline

ETL en capas *Medallion*, todo en Parquet, con un único punto de entrada:

```
landing → bronze → silver → gold → ML (LOPO)
```

- **Bronze** (`src/etl/landing_to_bronze_*`): parseo paralelo de los CSV crudos del
  Coordinador Eléctrico Nacional (generación) y de la red meteorológica CR2
  (radiación, temperatura, humedad); recorte 2014–2024; agregación multi-unidad.
- **Silver** (`bronze_to_silver.py`): agrupación geográfica por macrozona, features
  cíclicas seno/coseno, imputación trazable de exógenas (nunca de `y`).
- **Gold / DL** (`silver_to_gold_split.py`, `ml/prepare_dl_dataset.py`): particiones
  LOPO por planta y dataset numérico `silver_dl` para las redes profundas.
- **ML** (`ml/orchestrator_ml.py`): por cada planta objetivo entrena los modelos globales
  (`xgb_global`, `lstm`, `nhits`, `tft`, `informer`) y las líneas base locales
  (`xgb_local`, `lstm_local`, `nhits_local`), y evalúa horizontes `day1` y `rollout7d`
  sobre ventanas raw / operacional / estacionales.

### Modelos

| Paradigma | Modelos |
|-----------|---------|
| **Global (Cross-Site)** | XGBoost global, LSTM, N-HiTS, TFT, Informer |
| **Local (línea base)**  | XGBoost local, LSTM local, N-HiTS local |

Salida **probabilística** (P5/P50/P90) tanto en las redes (`MQLoss`) como en XGBoost
(regresión cuantílica), con consistencia física (0 nocturno, no-negatividad).

---

## 📁 Estructura del repositorio

```
├── src/
│   ├── orchestrator.py          # Entrypoint de producción (ETL → ML)
│   ├── etl/                     # Landing → Bronze → Silver → Gold
│   └── ml/
│       ├── orchestrator_ml.py   # Bucle LOPO por planta
│       ├── prepare_dl_dataset.py
│       ├── xgb_global/ xgb_local/ lstm/ nhits/ tft/ informer/
│       ├── utils/               # grid, eval_window, metrics, regional_prior, ...
│       ├── visualize.py         # Reporte consolidado de métricas
│       └── report_tesis.py      # Figuras y tablas .tex para la tesis
├── tests/                       # Suite TDD (incluye anti-fuga)
├── latex-tesis/                 # Documento de tesis (LaTeX) + figuras/tablas
├── docs/                        # plan.md, tasks.md, changelog.md
├── context/                     # Anteproyecto y contexto del problema
├── requirements.txt
└── CLAUDE.md                    # Mandatos del proyecto (reglas de rigor)
```

> **Nota:** `data/`, `models/`, `results/`, `lightning_logs/` y `visuales/` están
> excluidos del control de versiones (`.gitignore`): son artefactos pesados y
> regenerables por el pipeline.

---

## ⚙️ Instalación

Requiere **Python 3.12** y una GPU NVIDIA con **CUDA 12.1** (probado en RTX 2060, 6 GB;
funciona en CPU con menor velocidad).

```bash
# 1. Clonar
git clone https://github.com/cesar-aular/tesis.git
cd tesis

# 2. Entorno virtual
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# 3. Dependencias (incluye torch cu121 desde el índice de PyTorch)
pip install -r requirements.txt
```

---

## ▶️ Uso

El orquestador es el **único** punto de entrada de producción. Estrategias de escala:

| Estrategia | Plantas evaluadas | Uso |
|------------|-------------------|-----|
| `toy`      | 2                 | Prueba de humo rápida |
| `half`     | 47                | Corrida intermedia |
| `total`    | 94                | Corrida completa |

En todas las estrategias los modelos globales se entrenan con la **base histórica
completa** (2014–2024) de las N−1 plantas; lo que varía es el número de plantas
objetivo evaluadas y el presupuesto de iteraciones.

```bash
# Prueba de humo (rápida)
python -m src.ml.orchestrator_ml --strategy toy

# Corrida completa (requiere Silver ya construido)
python -m src.ml.orchestrator_ml --strategy total

# Pipeline completo desde el ETL
python src/orchestrator.py --strategy toy
```

Flags útiles: `--force` (ignora idempotencia y re-ejecuta), `--clean` (limpia salidas),
`--skip-tests` (omite el *gate* de pruebas).

Las corridas son **idempotentes**: al relanzarse, saltan los modelos/plantas ya
completados mediante marcadores de completitud.

### Pruebas

```bash
python -m pytest tests/ -q
```

### Reportes y artefactos de tesis

```bash
# Consolida métricas de una corrida en results/<estrategia>/visuals/
# (se ejecuta automáticamente al final del orquestador)

# Genera figuras (.png) y tablas (.tex) para el documento
python -m src.ml.report_tesis --strategy total
```

---

## 📊 Resultados (estrategia `total`, 94 plantas)

rRMSE mediano (roll-out 7 días, ventana operacional). Menor es mejor:

| Modelo | rRMSE (%) | Paradigma |
|--------|-----------|-----------|
| XGBoost local | 62 | Local |
| XGBoost global | 95 | Cross-Site |
| N-HiTS local | 104 | Local |
| LSTM global | 118 | Cross-Site |
| Informer global | 129 | Cross-Site |
| TFT global | 133 | Cross-Site |

**Veredicto matizado:** un modelo local maduro (XGBoost con historia propia) conserva la
mayor precisión absoluta; sin embargo, el enfoque Cross-Site es la única opción viable
en el régimen genuino de arranque en frío —pronósticos utilizables y calibrados desde la
primera hora de conexión, donde el paradigma local no puede existir—. El invierno es
sistemáticamente la estación de mayor dificultad para todas las arquitecturas.

---

## 📄 Documento de tesis

El informe LaTeX está en `latex-tesis/`. Compilación (MiKTeX / TeX Live):

```bash
cd latex-tesis
pdflatex formato_proyecto.tex
bibtex formato_proyecto
pdflatex formato_proyecto.tex
pdflatex formato_proyecto.tex
```

---

## 🛠️ Stack tecnológico

Python · PyTorch (cu121) · NeuralForecast · MLForecast / XGBoost · Optuna · pandas ·
PyArrow · scikit-learn · matplotlib / seaborn · pytest.
