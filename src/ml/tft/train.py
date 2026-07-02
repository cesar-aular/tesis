import pandas as pd

from src.ml.utils.dl_models import run_dl_train


def run_train(train_df: pd.DataFrame, strategy: str = "toy", force_fp32: bool = False):
    run_dl_train("tft", train_df, strategy, force_fp32=force_fp32)
