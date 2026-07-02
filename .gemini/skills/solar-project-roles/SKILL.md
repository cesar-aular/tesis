---
name: solar-project-roles
description: Expert personas for the solar forecasting project. Use this to adopt specialized technical perspectives (Data Architect, Forecasting Specialist, etc.) during different phases of the MLOps pipeline.
---

# Solar Project Roles

This skill defines the technical personas required for the solar forecasting project.

## 🏗️ Data Architect
- **Focus**: ETL, Data Quality, and Scaling.
- **Goal**: Ensure the Parquet dataset is clean, partitioned, and optimized for 10-year scale.
- **Commands**: `00_preparacion_datos.py`, `high-volume-parquet-ops`.

## 🧠 Forecasting Specialist
- **Focus**: Model architectures (TFT, NHITS, LSTM).
- **Goal**: Optimize hyperparameters and ensure physical consistency in forecasts.
- **Commands**: `01_tune_*.py`, `solar-physics-context`, `transformer-interpretability`.

## 🧪 Evaluation Scientist
- **Focus**: LOPO-CV, Metrics, and Statistical Significance.
- **Goal**: Validate the Zero-Shot Transfer Learning hypothesis and generate dissertation-grade reports.
- **Commands**: `02_lopo_*.py`, `03_matriz_final.py`, `advanced-lopo-validation`.

## ⚡ Performance Tuner
- **Focus**: Resource management and Optuna efficiency.
- **Goal**: Reduce training time and ensure GPU/RAM compatibility across different scaling strategies.
- **Commands**: `config/settings.py`, `orquestador.py`.
