import pandas as pd

from measurements_model.config import SystemColumns, IDLEColumns
from measurements_model.dataset_processing.process_data.feature_selection.feature_selector import FeatureSelector


class AllFeaturesNoEnergy(FeatureSelector):
    # no subtraction in system column + idle features
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop([SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL, IDLEColumns.ENERGY_TOTAL_USAGE_IDLE_COL], axis=1)
