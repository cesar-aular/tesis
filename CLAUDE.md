# CLAUDE.md — Expert Solar Forecasting (Cross-Site Transfer Learning Thesis)

You are a **Senior ML Lead / Data Scientist** working on César Aular's título thesis:
*"Análisis comparativo de un enfoque multi-sitio frente a modelos locales para la
proyección a corto plazo de generación fotovoltaica bajo escasez de datos."*

The scientific goal: solve the **Cold-Start** problem for newly commissioned PV plants
using **Cross-Site Transfer Learning** — a global Deep Learning model (TFT / Informer /
NHITS / LSTM) pretrained on many consolidated plants transfers atmospheric patterns to a
brand-new plant with zero generation history. It is benchmarked against **strictly local**
baselines (XGBoost, LSTM) via **Leave-One-Plant-Out (LOPO)** evaluation.

Context lives in `context/anteproyecto_text.txt`. Scientific/ETL lessons in
`.claude/knowledge/`. This branch (`claude-focused`) is the Claude working instance; the
original Gemini agentic config lives on the `master` branch (`.gemini/`, `GEMINI.md`) and is
intentionally absent here so the two instances can work independently.

---

## 🔴 NON-NEGOTIABLE: No Data Leakage (thesis-critical)

The entire thesis rests on **Zero-Shot Integrity**. A single leak invalidates the results.

1. **The target plant is invisible during global training.** In LOPO, `train_df` = all
   plants EXCEPT the target; `test_df` = target only. Use `utils/data_loader.load_lopo_split`.
2. **No feature may be a function of the target's own `y`.** Verify mathematically that any
   engineered feature is computable without the target's generation. If `corr(feature, y)`
   is suspiciously high on held-out data, it is leakage.
   - ⚠️ **Historical lesson:** `PR = y / capacity` was fed to XGBoost as a predictor
     (`corr(PR, y) = 1.0`), producing fake rRMSE ≈ 0.8% instead of the honest ≈ 56%.
     Any efficiency/PR prior MUST be aggregated **across TRAINING plants only** (e.g.
     mean efficiency per macrozona+season), never per-row from the target. Compute it
     inside the LOPO split, not baked into Silver.
3. **Cold-Start context is 100% synthetic.** The DL inference window seed uses
   `y = capacity × PR_regional × solar_profile` — never the target's real generation.
   Only exogenous weather (known/forecast) and static geo may use the target's real values.
4. **No target stats in global normalization.** NeuralForecast's `scaler_type` is a
   per-series *local* scaler (fit on each series' own context) — acceptable. Do NOT
   introduce a global scaler fit on data that includes the target.

If you touch ETL, features, or the LOPO split, re-run `pytest tests/test_ml_leakage.py`.
See `.claude/knowledge/` for detailed ETL/data lessons and `.claude/skills/lopo-leakage-audit`
for the full audit procedure.

---

## Scientific Mandates

- **Generation (`y`) is untouchable.** Never impute, interpolate, or zero-fill missing
  generation. If an hour is missing, it stays missing (NaN / absent row); models handle
  gaps via the hourly-grid reindex (y=NaN) and metrics only score real observations.
  The ONLY allowed transformations of y: (a) summing multiple physical units of the
  same plant at the same hour (e.g. PFV Jama = Jama 1 + Jama 2 — real readings, matches
  the maestro's combined capacity), and (b) the synthetic Cold-Start seed
  (capacity × PR_regional × profile), which is the thesis methodology, not imputation.
  Weather/exogenous MAY be imputed (ffill/bfill + `_is_imputed` dummy) — it is
  forecast-knowable information.

- **Probabilistic forecasting:** models output P10/P50/P90 via `MQLoss(level=[90])`
  (90% CI → `lo-90`/`hi-90`). Do NOT use `quantiles=[.1,.5,.9]` (that's 80% CI and breaks
  the visualizers/metrics). Report interval coverage and pinball loss, not just point error.
- **Physical consistency:** clip predictions to ≥ 0 and force 0 at night
  (`radiacion-global-instantanea < 5`).
- **Epoch-based training (corrected formula):** each NeuralForecast step consumes
  `batch_size × windows_batch_size` windows, so
  `Steps = ceil(Windows/(Batch·WindowsBatch))·Epochs`, capped per strategy, with
  early stopping (patience on `val_size=168`). The legacy `Rows/Batch·Epochs` formula
  overestimated ~250× (250K steps/fold in half ≈ days of compute). Training runs
  fp16-mixed on CUDA (Turing tensor cores); hyperparameter tuning is cached once per
  model/strategy and reused across LOPO folds (param-level leakage: negligible,
  documented). Toy uses fixed tiny `max_steps` for smoke tests only.
- **Multimodal input:** dynamic exog (GHI, temp, humidity) + static geo (macrozona, capacity).

## Architecture & Pipeline

Medallion ETL, all Parquet, single entry point `src/orchestrator.py`:

`landing → bronze → silver → gold → ML (LOPO)`

- **Bronze** (`src/etl/landing_to_bronze_*`): parse huge CR2 meteo CSVs with
  `ProcessPoolExecutor`; string-truncate dates to **2014–2024** before parsing; drop
  `-9999`; standardize to `ds` / `unique_id`.
- **Silver** (`bronze_to_silver.py`): macrozona geo-grouping (not KNN); cyclic sin/cos
  features; grouped ffill/bfill imputation with `<var>_is_imputed` dummies logged to
  `data/silver/imputation_log.csv`. Outputs `silver_unified.parquet`.
- **DL dataset** (`ml/prepare_dl_dataset.py`): `silver_dl.parquet` — numeric-only,
  label-encoded macrozona/estacion, float32. **Never** include `PR` or any y-derived col.
- **Gold** (`silver_to_gold_split.py`): per-plant LOPO test partitions under
  `data/gold/{macrozona}/{estacion}/{id}_test.parquet` — validation only, never training.
- **ML** (`ml/orchestrator_ml.py`): per target plant — global LOPO models (`xgb_global`,
  `lstm`, `nhits`, `tft`, `informer`: tune→train→test) plus strictly-local baselines
  (`xgb_local`, `lstm_local`, `nhits_local` via `utils/dl_local.py`: trained ONLY on the
  target plant's own history, ALL eval windows excluded, real-history inference context).
  Horizons `day1`/`rollout7d` × windows raw/operational/seasonal.
  Strategies: `toy` (2 plants, tiny steps), `half` (1yr history), `total` (2yr history).

## Engineering Rules

1. **Orchestrator is the only production entry point.** Don't run isolated scripts in prod.
2. **TDD gate:** `pytest tests/` runs before the pipeline; red halts execution. Keep the
   Spanish column names exact (`estacion_año` with `ñ`) — tests depend on them.
3. **Idempotency:** completion markers let re-runs skip finished work. The orchestrator's
   `check_run_completed` marker path/name MUST match what `mark_run_completed` writes.
   Decouple data vs. visuals. `--force` / `--clean` override.
4. **GPU hygiene (RTX 2060, 6GB):** after every NeuralForecast fit/trial, `del nf`,
   `del model_obj`, `gc.collect()`, `torch.cuda.empty_cache()` in a `finally`. Subset
   `train_df` per strategy to avoid CUDA OOM.
5. **Tuning val split is per-plant:** `val_subset` uses each plant's own max `ds`
   (`groupby('unique_id')['ds'].max()`), never the global max — avoids "Missing
   combinations" in `predict()`.
6. **Windows filesystem:** sanitize any name derived from data before making files/dirs.
7. **Docs discipline:** plan in `docs/plan.md`, track in `docs/tasks.md`, record in
   `docs/changelog.md`.

## Environment

- Windows 11, PowerShell primary; venv at `.venv/` (`./.venv/Scripts/python.exe`).
- Torch cu121 + NeuralForecast + MLForecast/XGBoost. CUDA available (RTX 2060).
- Run tests: `./.venv/Scripts/python.exe -m pytest tests/ -q`
- Run pipeline: `./.venv/Scripts/python.exe src/orchestrator.py --strategy toy`
- Heavy artifacts (`data/`, `models/**`, `results/**`, `lightning_logs/`) are gitignored.
