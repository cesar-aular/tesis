---
name: thesis-compliance-checklist
description: A strict checklist and validation skill to ensure code and charts match the academic standards of the solar forecasting thesis. Use when reviewing code or visualizations.
---
# Thesis Compliance Checklist

When reviewing the project, you MUST verify the following academic standards are met:

## 1. Zero-Shot Strictness
- `orquestador.py` and LOPO scripts must completely isolate the test plant. No global scaling metrics (like mean/std) can include the test plant's data.
- The term "Cold-Start" or "Zero-Shot" must be verifiable in the pipeline (i.e. model never trained on the specific `unique_id`).

## 2. Statistical Metrics
- Baseline comparison must evaluate: XGBoost, LSTM, NHITS, and Global TFT.
- Accuracy metrics must include **WMAPE** and **MAE**.
- Probabilistic bands MUST be implemented using **MQLoss** (QuantileLoss P10, P50, P90). Point forecasts are unacceptable.
- **Coverage** metric must calculate the percentage of actual generation points that fall within the P10-P90 bounds.

## 3. Visualizations
- A Performance Heatmap must exist showing WMAPE, MAE, Bias, and Coverage.
- A Radar Chart (Spider Chart) evaluating the architectures across multiple dimensions.
- A Time-Series Fan Chart displaying 48-hour periods of high intermittence (sudden drops/spikes) with shaded P10-P90 boundaries.
- Violin Plots displaying the error density distribution.
