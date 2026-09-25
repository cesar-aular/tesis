# Presentaciones — tres variantes desde un mismo fuente

Todo sale de `defensa.tex`, controlado por dos interruptores (`\confmode` y `\minmode`).
El **contenido base es el mismo**, así que cualquier corrección se propaga a las tres.

| PDF | Modo | Slides | Para qué |
|---|---|---|---|
| `defensa.pdf` | `confmode=0` | 30 | Examen de título (histórico) |
| `charla_conferencia.pdf` | `confmode=1` | 30 | **CWPR 2026 — versión completa** |
| `charla_minima.pdf` | `confmode=1`, `minmode=1` | 28 | **CWPR 2026 — versión conservadora** |

## Cómo compilar

```bash
# Defensa (por defecto)
pdflatex defensa.tex

# Conferencia — versión completa
pdflatex -jobname=charla_conferencia "\def\confmode{1}\input{defensa.tex}"

# Conferencia — versión mínima
pdflatex -jobname=charla_minima "\def\confmode{1}\def\minmode{1}\input{defensa.tex}"
```
Compilar **dos veces** cada una (el índice de secciones necesita la segunda pasada).

## Qué cambia cada interruptor

### `\confmode` — defensa vs. conferencia
Solo afecta portada y cierre; el cuerpo es idéntico.

| | `confmode=0` (defensa) | `confmode=1` (conferencia) |
|---|---|---|
| Título | Título de la tesis | Título del paper |
| Autores | César solo | **César + Sergio Hernández** |
| Afiliación | Escuela Ing. Civil Informática | Departamento de Computación e Industrias |
| Evento | Examen de Título — Junio 2026 | **Chilean Workshop on Pattern Recognition (CWPR) — 2026** |

### `\minmode` — completa vs. mínima
La versión mínima es el deck de la defensa **más el mapa**, sin material nuevo.

| | `minmode=0` (completa) | `minmode=1` (mínima) |
|---|---|---|
| Mapa de Chile | sí | sí |
| Tabla NRMSE + rRMSE (94 plantas) | sí | sí |
| Diapositiva *Contraste* | con tests pareados ($p$, Wilcoxon-Holm) | **sin mencionar tests** |
| Respaldo *Tests estadísticos* | sí | **omitido** |
| Respaldo *Causalidad temporal* | sí | **omitido** |

## Datos

Las **tres** usan los números corregidos: ventana *cold-start* sobre las **94 plantas**
(Operational donde hay rampa >24 h, Raw en el resto), NRMSE normalizado por capacidad, y
las cifras probabilísticas recalculadas sobre esa misma base.

En ninguna se afirma ya que *"el LSTM global supera al local"*: ese matiz de la defensa
**no sobrevivió al test pareado** ($p=1{,}00$) y el paper lo desmiente explícitamente.

## Figuras propias de la charla

`make_figs_charla.py` genera dos figuras que **no existen en la tesis** (no toca ninguna
figura de la tesis, que está congelada):

```bash
./.venv/Scripts/python.exe latex-tesis/make_figs_charla.py
```
- `fig_mapa_plantas.png` — ubicación real de las plantas sobre el mapa de Chile con
  fronteras regionales (87 de 94 georreferenciadas).
- `fig_resultados_nrmse_es.png` — ranking por NRMSE sobre 94 plantas, en español.

Los contornos (`chile_boundary.geojson`, `chile_regions.geojson`) vienen de Natural Earth
(dominio público) y están versionados, así que la figura se regenera **sin red y sin
geopandas/cartopy** — es matplotlib puro leyendo los polígonos.

## Trampa conocida

Dentro de un **título de bloque** no uses `\rj{}`, `\lo{}` ni `\gl{}`: el fondo ya es de
color y el texto queda invisible (rojo sobre rojo). Usa `\textbf{}`, que hereda el blanco.
