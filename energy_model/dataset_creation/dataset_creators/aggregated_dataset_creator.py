import pandas as pd
from overrides import override

from energy_model.configs.columns import ProcessColumns, SystemColumns
from energy_model.dataset_creation.dataset_creation_config import DEFAULT_FILTERING_SINGLE_PROCESS, AggregationName
from energy_model.dataset_creation.dataset_creators.basic_dataset_creator import BasicDatasetCreator
from energy_model.dataset_creation.dataset_readers.dataset_reader import DatasetReader
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator


class AggregatedDatasetCreator(BasicDatasetCreator):
    """
    This class represents the basic reading from elastic for the sake of dataset creation.
    Reading only process of interest logs.
    Aggregations on every process telemetry per batch.
    """
    def __init__(self, target_calculator: TargetCalculator, dataset_reader: DatasetReader,
                 batch_time_intervals: list[int] = None, single_process_only: bool = DEFAULT_FILTERING_SINGLE_PROCESS):
        super().__init__(target_calculator=target_calculator, dataset_reader=dataset_reader,
                         batch_time_intervals=batch_time_intervals, single_process_only=single_process_only)

    def get_name(self) -> str:
        return "aggregated_dataset_creator"

    @override
    def _add_energy_necessary_columns(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        """
        For each batch:
            - Calculate the total energy consumption per second of that batch, by calculating the battery drain during that batch.
            - Adding the calculated result as new column.
            - aggregate over each process's telemetry.
        """

        df_without_aggregations = super()._add_energy_necessary_columns(df, batch_duration_seconds)
        necessary_aggregations = self._get_necessary_aggregations(df_without_aggregations.columns.to_list())
        df_grouped = (
            df_without_aggregations.groupby([SystemColumns.BATCH_ID_COL, ProcessColumns.PROCESS_ID_COL], as_index=False)
            .agg(necessary_aggregations)
        )
        return df_grouped
