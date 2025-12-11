import pandas as pd


def extract_x_y(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
