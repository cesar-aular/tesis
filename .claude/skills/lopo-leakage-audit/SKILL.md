---
name: lopo-leakage-audit
description: Checklist and procedure to audit the solar-forecasting pipeline for data leakage and LOPO integrity. Use before trusting any benchmark result, after touching ETL/features/splits, or when a model's held-out error looks implausibly low.
---

# LOPO / Cold-Start Leakage Audit

The thesis is only valid if the target plant is truly unseen. Run this audit whenever
results look "too good," or after changing `src/etl/*`, `src/ml/utils/features.py`,
`data_loader.load_lopo_split`, or any model's `test.py`.

## 1. Feature leakage (the #1 risk)
For each feature column fed to a model, ask: *can it be computed WITHOUT the target
plant's `y`?* If not, it leaks.
```python
import pandas as pd, numpy as np
df = pd.read_parquet('data/silver/silver_unified.parquet')
for c in df.select_dtypes('number').columns:
    if c == 'y': continue
    r = np.corrcoef(df[c].fillna(0), df['y'].fillna(0))[0,1]
    if abs(r) > 0.95: print('SUSPECT LEAK:', c, round(r,3))
```
- Known trap: `PR = y/capacity` (corr = 1.0). Any PR/efficiency prior must be a
  macrozona+season aggregate over **training plants only**, joined inside the LOPO split.

## 2. Split integrity
- `load_lopo_split(df, target)` → target NOT in `train_df['unique_id'].unique()`; `test_df`
  is target-only. (`tests/test_ml_leakage.py` asserts this — keep it green.)
- Global normalization/scaling must not be fit on data including the target. NeuralForecast's
  per-series local scaler is fine; a global scaler over all rows is not.

## 3. Cold-Start context
- The DL history seed `y` must be synthetic (`capacity × PR_regional × solar_profile`), not
  real generation. Only exogenous weather + static geo may be the target's real values.
  (`tests/test_ml_coldstart.py` asserts syntheticity.)

## 4. Roll-out honesty
- The 7-day autoregressive roll-out must feed back the model's own prediction as the next
  history `y` — never the real future `y`.

## 5. Sanity thresholds
- Solar day-ahead rRMSE is realistically ~30–60%. If a "cold-start global" model reports
  < 5% rRMSE, assume leakage until proven otherwise and re-run this audit.
