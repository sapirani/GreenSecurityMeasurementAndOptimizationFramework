import pandas as pd

from measurements_model.config import SystemColumns, IDLEColumns
from measurements_model.dataset_processing.feature_selection.feature_selector import FeatureSelector


class AllFeaturesNoEnergy(FeatureSelector):
    """
    Includes all process, hardware, idle and system features.
    Doesn't include energy consumption for idle and system.
    The system features are NOT a subtraction of idle and process.
    """

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop([SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL, IDLEColumns.ENERGY_TOTAL_USAGE_IDLE_COL], axis=1)
