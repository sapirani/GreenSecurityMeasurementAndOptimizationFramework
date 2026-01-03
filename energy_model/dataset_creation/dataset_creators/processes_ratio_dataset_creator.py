import pandas as pd
from dataclasses import fields
from overrides import override

from DTOs.aggregators_features.energy_model_features.process_energy_model_features import ProcessEnergyModelFeatures
from energy_model.configs.columns import ProcessColumns, SystemColumns
from energy_model.dataset_creation.dataset_creation_config import DEFAULT_FILTERING_SINGLE_PROCESS
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
                 batch_time_intervals: list[int] = None, single_process_only: bool = DEFAULT_FILTERING_SINGLE_PROCESS):
        super().__init__(target_calculator=target_calculator, dataset_reader=dataset_reader,
                         batch_time_intervals=batch_time_intervals, single_process_only=single_process_only)
        self.__resource_energy_calculator = ResourceEnergyCalculator()

    def get_name(self) -> str:
        return "process_ratio_dataset_creator"

    @override
    def _add_energy_necessary_columns(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        """
            For each batch:
                - Calculate the total energy consumption per second of that batch, by calculating the battery drain during that batch.
                - Adding the calculated result as new column.
                - Calculate the ratio of a specific sample on the overall batch's energy usage.
                    * If the batch has single process - the ratio is 1
                    * If the batch has multiple process - the ratio is calculated based on resource consumption.
        """
        df_with_basic_columns = super()._add_energy_necessary_columns(df, batch_duration_seconds)
        uniques_per_batch = (
            df.groupby(SystemColumns.BATCH_ID_COL)[ProcessColumns.PROCESS_ID_COL]
            .agg(lambda s: (s.nunique()))
            .rename(SystemColumns.NUMBER_OF_UNIQUE_PROCESSES)
        )

        df_with_basic_columns = df_with_basic_columns.merge(uniques_per_batch, on=SystemColumns.BATCH_ID_COL, how="left")

        df_with_basic_columns = df_with_basic_columns.groupby(SystemColumns.BATCH_ID_COL, group_keys=False).apply(
            self.__calculate_energy_ratio)

        return df_with_basic_columns

    def __calculate_energy_ratio(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        if batch_df[SystemColumns.NUMBER_OF_UNIQUE_PROCESSES].iloc[0] > 1:
            batch_df = self.__calculate_energy_ratio_by_resources(batch_df)
        else:
            batch_df[SystemColumns.ENERGY_RATIO_SHARE] = DEFAULT_ENERGY_RATIO

        return batch_df

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
        return df_without_basic_columns.drop(
            [SystemColumns.ENERGY_RATIO_SHARE, SystemColumns.NUMBER_OF_UNIQUE_PROCESSES],
            axis=1)
