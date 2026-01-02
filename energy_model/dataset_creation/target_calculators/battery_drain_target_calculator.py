import pandas as pd

from energy_model.configs.columns import SystemColumns
from energy_model.dataset_creation.dataset_creation_config import AggregationName
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator

FIRST_SAMPLE_TARGET_COLUMN = f"{SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL}_{AggregationName.FIRST_SAMPLE}"
LAST_SAMPLE_TARGET_COLUMN = f"{SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL}_{AggregationName.LAST_SAMPLE}"

class BatteryDrainTargetCalculator(TargetCalculator):
    def __init__(self):
        super().__init__(target_column=SystemColumns.ENERGY_USAGE_SYSTEM_COL,
                         must_appear_columns=[FIRST_SAMPLE_TARGET_COLUMN, LAST_SAMPLE_TARGET_COLUMN])

    def _add_target_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self._target_column] = df[FIRST_SAMPLE_TARGET_COLUMN] - df[LAST_SAMPLE_TARGET_COLUMN]
        return df

    def get_name(self) -> str:
        return "battery_drain_based_target_calculator"
