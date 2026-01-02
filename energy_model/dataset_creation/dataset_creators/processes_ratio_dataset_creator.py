import pandas as pd
from dataclasses import fields
from overrides import override

from DTOs.aggregators_features.energy_model_features.process_energy_model_features import ProcessEnergyModelFeatures
from energy_model.configs.columns import ProcessColumns, SystemColumns
from energy_model.dataset_creation.dataset_creators.basic_dataset_creator import BasicDatasetCreator
from energy_model.dataset_creation.dataset_readers.dataset_reader import DatasetReader
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator
from energy_model.energy_model_utils.resource_energy_calculator import ResourceEnergyCalculator

DEFAULT_ENERGY_RATIO = 1.0


class ProcessesRatioDatasetCreator(BasicDatasetCreator):
    """
    This class represents the basic reading from elastic.
    Reading only process of interest logs.
    No aggregations on samples.
    Calculating the impact of each sample of each process in case where multiple processes appear in a single batch.
    """

    def __init__(self, target_calculator: TargetCalculator, dataset_reader: DatasetReader,
                 batch_time_intervals: list[int] = None):
        super().__init__(target_calculator, dataset_reader, batch_time_intervals)
        self.__resource_energy_calculator = ResourceEnergyCalculator()

    @override
    def _add_energy_necessary_columns(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        df_without_process_ratio = super()._add_energy_necessary_columns(df, batch_duration_seconds)
        unique_procs = df_without_process_ratio[ProcessColumns.PROCESS_ID_COL].nunique()
        if unique_procs > 1:
            df_without_process_ratio = self.__calculate_energy_ratio_by_resources(df)
        else:
            df_without_process_ratio[SystemColumns.ENERGY_RATIO_SHARE] = DEFAULT_ENERGY_RATIO

        return df_without_process_ratio

    def __calculate_energy_ratio_by_resources(self, df: pd.DataFrame) -> pd.DataFrame:
        # todo: a key can be (pid, process_name) in the future
        group_by_energy_columns = [energy_field.name for energy_field in fields(ProcessEnergyModelFeatures)]
        resources_per_process_df = (
            df.groupby(ProcessColumns.PROCESS_ID_COL)[group_by_energy_columns].agg(lambda s: sum(s))
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

        df[SystemColumns.ENERGY_RATIO_SHARE] = df[ProcessColumns.PROCESS_ID_COL].map(
            energy_ratio_per_process)

        return df

    @override
    def _remove_temporary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_without_basic_columns = super()._remove_temporary_columns(df)
        return df_without_basic_columns.drop([SystemColumns.ENERGY_RATIO_SHARE], axis=1)
