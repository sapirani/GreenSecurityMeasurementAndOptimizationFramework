import pandas as pd

from measurements_model.config import TIME_COLUMN_NAME


def merge_dfs(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(left, right, on=TIME_COLUMN_NAME, how='inner')
