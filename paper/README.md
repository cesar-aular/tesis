# Paper IEEE-CS — Cross-Site Transfer Learning for Cold-Start PV Forecasting

Condensación del proyecto de título en formato **IEEE Computer Society conference**
(`IEEEtran`, doble columna, inglés). Longitud: **4 páginas** (límite: 8).

## Archivos
- `paper.tex` — fuente principal (IEEEtran, modo `conference`).
- `paper.bib` — 19 referencias (subconjunto de la bibliografía de la tesis).
- `make_paper_figs.py` — genera las figuras en inglés bajo `figuras/`.
- `figuras/fig_method_en.png` — protocolo Cold-Start LOPO (contexto sintético + horizontes + ventanas).
- `figuras/fig_results_en.png` — rRMSE mediano (roll-out 7 días) por configuración.

## Compilar
```bash
# 1) (re)generar figuras — requiere el venv del proyecto
./.venv/Scripts/python.exe paper/make_paper_figs.py

# 2) compilar (desde paper/)
pdflatex paper.tex
bibtex   paper
pdflatex paper.tex
pdflatex paper.tex
```
Salida: `paper.pdf`. Compilación limpia: 0 referencias sin resolver, 0 *overfull boxes*.

## Contenido
Estructura IMRaD: Introduction → Related Work → Methodology (LOPO, semilla Cold-Start
sintética, 8 configuraciones, ventanas/métricas) → Results (Tabla I operacional, Tabla II
estacional, calibración probabilística, contraste de hipótesis) → Discussion/Limitations →
Conclusion. Los números provienen de la corrida `total` (94 plantas) de la tesis.
