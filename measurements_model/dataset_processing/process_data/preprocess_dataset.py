import pandas as pd

from measurements_model.config import ProcessColumns


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[ProcessColumns.ENERGY_USAGE_PROCESS_COL] >= 0]
    return df