import pandas as pd

from energy_model.configs.columns import SystemColumns
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator


class SystemBasedTargetCalculator(TargetCalculator):
    def __init__(self):
        super().__init__(target_column=SystemColumns.ENERGY_USAGE_SYSTEM_COL,
                         must_appear_columns=[SystemColumns.DURATION_COL,
                                              SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL])
    def _add_target_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the energy usage of each sample.
        Before calling this method, the dataframe needs to contain the following columns:
            * SystemColumns.DURATION_COL - The duration of the sample (in seconds).
            * SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL - The battery drain of the batch divided by the batch's duration.
        This calculator multiplies between the two columns, to receive:
        E_system = ((df[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL][0] - df[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL[-1]) \ batch_duration) * sample_duration
        Input:
            df - the full telemetry dataset.
        Output:
            df with another column that represents the calculated target, E_system.
        """
        df[self._target_column] = (df[SystemColumns.DURATION_COL] *
                                   df[SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL])

        return df

    def get_name(self) -> str:
        return "system_based_target_calculator"