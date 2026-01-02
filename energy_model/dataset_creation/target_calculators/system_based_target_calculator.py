import pandas as pd

from energy_model.configs.columns import SystemColumns
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator


class SystemBasedTargetCalculator(TargetCalculator):
    def __init__(self):
        super().__init__(target_column=SystemColumns.ENERGY_USAGE_SYSTEM_COL,
                         must_appear_columns=[SystemColumns.DURATION_COL,
                                              SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL])
    def _add_target_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self._target_column] = (df[SystemColumns.DURATION_COL] *
                                   df[SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL])

        return df

    def get_name(self) -> str:
        return "system_based_target_calculator"