import pandas as pd


def save_df_to_excel(df: pd.DataFrame, path: str):
    df.to_csv(path)


def extract_df_from_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def return_dict_as_sample(features: list[str], info: list[any]) -> dict[str, any]:
    sample = {}
    for index, feature in enumerate(features):
        sample[feature] = float(info[index])

    return sample