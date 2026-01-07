from dataclasses import fields

import pandas as pd

from DTOs.aggregators_features.energy_model_features.idle_energy_model_features import IdleEnergyModelFeatures
from DTOs.aggregators_features.energy_model_features.process_energy_model_features import ProcessEnergyModelFeatures
from energy_model.configs.columns import ProcessColumns, SystemColumns
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator
from energy_model.energy_model_utils.resource_energy_calculator import ResourceEnergyCalculator

DEFAULT_ENERGY_PER_SECOND_IDLE_MEASUREMENT = 1.57
DEFAULT_ENERGY_RATIO = 1.0


class IdleBasedTargetCalculator(TargetCalculator):
    def __init__(self):
        super().__init__(target_column=ProcessColumns.ENERGY_USAGE_PROCESS_COL,
                         must_appear_columns=[SystemColumns.DURATION_COL,
                                              SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL,
                                              SystemColumns.ENERGY_RATIO_SHARE])

        self.__idle_details = IdleEnergyModelFeatures(
            energy_per_second=DEFAULT_ENERGY_PER_SECOND_IDLE_MEASUREMENT
        )

    def _add_target_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the energy usage of each sample.
        Before calling this method, the dataframe needs to contain the following columns:
            * SystemColumns.DURATION_COL - The duration of the sample (in seconds).
            * SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL - The battery drain of the batch divided by the batch's duration.
        This calculator multiplies between the two columns and subtracts the Idle energy usage for the same duration, to receive:
        E_process = (((df[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL][0] - df[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL[-1]) \ batch_duration) * sample_duration) - (sample_duration * idle_energy_per_secon)
        Input:
            df - the full telemetry dataset.
        Output:
            df with another column that represents the calculated target, E_process.
        """
        df[self._target_column] = (df[SystemColumns.DURATION_COL] *
                                   df[SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL]) - \
                                  (df[SystemColumns.DURATION_COL] * self.__idle_details.energy_per_second)

        df[self._target_column] = df[self._target_column] * df[SystemColumns.ENERGY_RATIO_SHARE]
        return df

    def get_name(self) -> str:
        return "idle_based_target_calculator"
