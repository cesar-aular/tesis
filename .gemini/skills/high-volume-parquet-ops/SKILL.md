---
name: high-volume-parquet-ops
description: Expert strategies for handling large-scale (TB+) time-series data using Parquet and Row Group filtering. Use this when the project scales to the 'total' strategy (10 years of data).
---

# High-Volume Parquet Operations

Efficiency mandates for processing the 10-year solar dataset.

## Storage Optimization
- **Partitioning**: Partition data by `año` and `macrozona` to allow surgical reads.
- **Data Types**: Use `float32` instead of `float64` to halve memory footprint and I/O time without losing significant precision.

## Reading Strategies
- **Row-Group Filtering**: Use `filters` in `pd.read_parquet` to skip irrelevant plants or time periods before loading into RAM.
- **Memory Mapping**: Set `memory_map=True` for faster access on local SSDs.

## Dask Integration
When RAM limits are reached (Strategy `total`), transition from `pandas` to `dask.dataframe`:
```python
import dask.dataframe as dd
ddf = dd.read_parquet('data/*.parquet')
# Vectorized FE remains identical
ddf['hora_sin'] = np.sin(2 * np.pi * ddf['hora'] / 24.0)
```
