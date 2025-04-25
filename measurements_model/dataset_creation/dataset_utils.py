import pandas as pd


def save_df_to_excel(df: pd.DataFrame, path: str):
    df.to_csv(path)


def extract_df_from_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)
