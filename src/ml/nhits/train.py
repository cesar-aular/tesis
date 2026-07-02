import pandas as pd

from src.ml.utils.dl_models import run_dl_train


def run_train(train_df: pd.DataFrame, strategy: str = "toy"):
    run_dl_train("nhits", train_df, strategy)
