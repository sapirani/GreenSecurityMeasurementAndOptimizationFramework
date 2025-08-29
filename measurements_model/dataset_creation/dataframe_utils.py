import pandas as pd

from measurements_model.config import TIME_COLUMN_NAME, PROCESS_COLUMN_SUFFIX, SYSTEM_COLUMN_SUFFIX
from utils.general_consts import BatteryColumns

PROCESS_INDEX_CSV = r"dataset_creation/processes_index.csv"
SYSTEM_INDEX_CSV = r"dataset_creation/systems_index.csv"

MERGE_ON_COLUMNS = ["session_id", "hostname", "timestamp"]


def get_process_features_dataframe() -> pd.DataFrame:
    df = pd.read_csv(PROCESS_INDEX_CSV)
    df = df[df["process_of_interest"]]
    return df


def get_system_features_dataframe() -> pd.DataFrame:
    df = pd.read_csv(SYSTEM_INDEX_CSV)
    return df


def get_full_features_dataframe() -> pd.DataFrame:
    df_process = get_process_features_dataframe()
    df_system = get_system_features_dataframe()

    full_df = pd.merge(df_process, df_system, how='left', on=MERGE_ON_COLUMNS, suffixes=(PROCESS_COLUMN_SUFFIX, SYSTEM_COLUMN_SUFFIX))
    full_df[TIME_COLUMN_NAME] = pd.to_datetime(full_df[TIME_COLUMN_NAME], format="%b %d, %Y @ %H:%M:%S.%f")

    return full_df
