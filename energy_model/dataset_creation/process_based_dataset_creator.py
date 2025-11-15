from dataclasses import fields
from overrides import overrides
import pandas as pd

from DTOs.aggregators_features.energy_model_features.idle_energy_model_features import IdleEnergyModelFeatures
from DTOs.aggregators_features.energy_model_features.process_energy_model_features import ProcessEnergyModelFeatures
from energy_model.dataset_creation.dataset_creator import DatasetCreator
from energy_model.configs.columns import ProcessColumns, SystemColumns

# todo: extend this logic when we want to use a baseline background activity instead of idle.
# todo: extend to reading idle sessions from elastic and calculate the average energy per second
DEFAULT_ENERGY_PER_SECOND_IDLE_MEASUREMENT = 1.57
DEFAULT_ENERGY_RATIO = 1.0


# todo: change to consumer interface
class ProcessBasedDatasetCreator(DatasetCreator):
    def __init__(self):
        super().__init__()
        self.__idle_details = IdleEnergyModelFeatures(
            energy_per_second=DEFAULT_ENERGY_PER_SECOND_IDLE_MEASUREMENT
        )

    def _add_target_to_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        batch_df = self.__add_energy_ratio_column(batch_df)

        batch_df[ProcessColumns.ENERGY_USAGE_PROCESS_COL] = (batch_df[SystemColumns.DURATION_COL] *
                                                             batch_df[
                                                                 SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL]) - \
                                                            (batch_df[SystemColumns.DURATION_COL] *
                                                             self.__idle_details.energy_per_second)

        batch_df[ProcessColumns.ENERGY_USAGE_PROCESS_COL] = batch_df[ProcessColumns.ENERGY_USAGE_PROCESS_COL] * \
                                                            batch_df[SystemColumns.ENERGY_RATIO_SHARE]
        return batch_df

    def __add_energy_ratio_column(self, df: pd.DataFrame) -> pd.DataFrame:
        unique_procs = df[ProcessColumns.PROCESS_ID_COL].nunique()
        if unique_procs > 1:
            df = self.__calculate_energy_ratio_by_resources(df)
        else:
            df[SystemColumns.ENERGY_RATIO_SHARE] = DEFAULT_ENERGY_RATIO

        return df

    def __calculate_energy_ratio_by_resources(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        # todo: a key can be (pid, process_name) in the future
        group_by_energy_columns = [energy_field.name for energy_field in fields(ProcessEnergyModelFeatures)]
        resources_per_process_df = (
            batch_df.groupby(ProcessColumns.PROCESS_ID_COL)[group_by_energy_columns].agg(lambda s: sum(s))
        )

        processes_energy_by_resources = {
            row.name: self.__resource_energy_calculator.calculate_total_energy_by_resources(
                ProcessEnergyModelFeatures.from_pandas_series(row))
            for _, row in resources_per_process_df.iterrows()
        }

        sum_energy_processes_by_resources = sum(processes_energy_by_resources.values())
        if sum_energy_processes_by_resources > 0:
            energy_ratio_per_process = {pid: process_energy / sum_energy_processes_by_resources
                                        for pid, process_energy in processes_energy_by_resources.items()}
        else:
            # all energy values for processes are 0 and have the same ratio compared the total energy
            energy_ratio_per_process = {pid: 1 / len(processes_energy_by_resources)
                                        for pid, _ in processes_energy_by_resources.items()}

        batch_df[SystemColumns.ENERGY_RATIO_SHARE] = batch_df[ProcessColumns.PROCESS_ID_COL].map(
            energy_ratio_per_process)

        return batch_df

    @overrides
    def _remove_temporary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_without_super_columns = super()._remove_temporary_columns(df)
        return df_without_super_columns.drop([SystemColumns.ENERGY_RATIO_SHARE],
                                             axis=1)
