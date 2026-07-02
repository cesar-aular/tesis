import pandas as pd
from pathlib import Path
from src.ml.lstm.tune import run_tune

print("Loading data...")
silver_dl = pd.read_parquet("data/silver/silver_dl.parquet")

planta = "alto_solar"
train_df = silver_dl[silver_dl['unique_id'] == planta].copy()

print("Running tune...")
run_tune(train_df, strategy="toy")
print("Done!")
