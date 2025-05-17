import pandas as pd


def save_df_to_excel(df: pd.DataFrame, path: str):
    df.to_csv(path)


def extract_df_from_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def create_sample_from_mapping(df: pd.DataFrame, mapping: dict[str, str], total_df_column: str) -> dict[str, any]:
    return {required_name: df.loc[current_name, total_df_column] for required_name, current_name in mapping.items() if
            current_name in df.index}
