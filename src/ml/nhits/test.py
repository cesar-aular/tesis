import pandas as pd
from pathlib import Path

from src.ml.utils.dl_test import run_dl_test


def run_test(test_df: pd.DataFrame, silver_dl: pd.DataFrame, results_dir: Path,
             macrozona: str, estacion: str, planta: str, strategy: str = "toy",
             regional_pr: float = 0.75):
    run_dl_test("NHITS", test_df, results_dir, macrozona, estacion, planta,
                strategy=strategy, regional_pr=regional_pr)
