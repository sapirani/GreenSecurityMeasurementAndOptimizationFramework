import pandas as pd

from energy_model.configs.columns import SystemColumns
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator


class BatteryDrainTargetCalculator(TargetCalculator):
    def __init__(self):
        super().__init__(target_column=SystemColumns.ENERGY_USAGE_SYSTEM_COL,
                         must_appear_columns=[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL])

    def _add_target_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the energy usage of each sample.
        Before calling this method, the dataframe needs to contain the following column:
            * SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL - The aggregated value of the battery capacity of the system during the execution of a process.
        This calculator returns the aggregated value, to receive:
        E_system = df[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL][0] - df[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL[-1]
        Input:
            df - the full telemetry dataset.
        Output:
            df with another column that represents the calculated target, E_system.
        """
        df[self._target_column] = df[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL]
        return df

    def get_name(self) -> str:
        return "battery_drain_based_target_calculator"
