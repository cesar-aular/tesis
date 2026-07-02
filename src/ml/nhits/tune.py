import pandas as pd

from src.ml.utils.dl_models import run_dl_tune


def run_tune(train_df: pd.DataFrame, strategy: str = "toy"):
    return run_dl_tune("nhits", train_df, strategy)
