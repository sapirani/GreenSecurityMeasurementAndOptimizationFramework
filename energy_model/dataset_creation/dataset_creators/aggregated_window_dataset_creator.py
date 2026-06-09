import pandas as pd
from overrides import override
from pandas._libs.window import aggregations
from pandas.core.groupby import DataFrameGroupBy

from energy_model.configs.columns import ProcessColumns, SystemColumns
from energy_model.dataset_creation.dataset_creation_config import DEFAULT_FILTERING_SINGLE_PROCESS, AggregationName
from energy_model.dataset_creation.dataset_creators.energy_per_second_dataset_creator import \
    EnergyPerSecondDatasetCreator
from energy_model.dataset_creation.raw_telemetry_readers.raw_telemetry_reader import RawTelemetryReader
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator

WINDOW_SIZES = (2, 3, 5)
COLUMNS_TO_GROUP_BY = [SystemColumns.BATCH_ID_COL, ProcessColumns.PROCESS_ID_COL]
TEMP_NEW_INDEX_COLUMN_NAME = "_row_id"


class AggregatedWindowDatasetCreator(EnergyPerSecondDatasetCreator):
    """
    This class represents the basic reading from elastic for the sake of dataset creation.
    Aggregations on every process telemetry per batch - aggregating a sliding window over all samples.
    The chosen windows are 2, 3 and 5.
    Meaning that the basic dataframe is extended with 3 dataframes with:
        * n - 1 new samples - every two samples in the original dataframe are combined into a new sample.
        * n - 2 new samples - every three samples in the original dataframe are combined into a new sample.
        * n - 4 new samples - every five samples in the original dataframe are combined into a new sample.
    """

    def __init__(self, target_calculator: TargetCalculator, dataset_reader: RawTelemetryReader,
                 batch_time_intervals: list[int] = None, single_process_only: bool = DEFAULT_FILTERING_SINGLE_PROCESS):
        super().__init__(target_calculator=target_calculator, dataset_reader=dataset_reader,
                         batch_time_intervals=batch_time_intervals, single_process_only=single_process_only)

    def get_name(self) -> str:
        return "window_aggregated_dataset_creator"

    @override
    def _add_energy_necessary_columns(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        """
        For each batch:
            - Calculate the total energy consumption per second of that batch, by calculating the battery drain during that batch.
            - Adding the calculated result as new column.
            - aggregate over each process's telemetry.
        """

        df_without_aggregations = super()._add_energy_necessary_columns(df, batch_duration_seconds)
        df = df_without_aggregations.copy()

        df = df.reset_index(drop=False).rename(columns={"index": TEMP_NEW_INDEX_COLUMN_NAME}).set_index(
            TEMP_NEW_INDEX_COLUMN_NAME)

        necessary_aggregations = self._get_necessary_aggregations(df.columns.to_list())
        necessary_aggregations[SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL] = AggregationName.SUM.value
        rolling_aggs = {
            col: agg for col, agg in necessary_aggregations.items()
            if agg != AggregationName.FIRST_SAMPLE.value and col not in COLUMNS_TO_GROUP_BY
        }
        metadata_cols = [col for col, agg in necessary_aggregations.items() if col not in rolling_aggs]
        grouped_df = df.groupby(COLUMNS_TO_GROUP_BY, group_keys=False)
        dfs = self.__handle_aggregations_on_rolling_windows(df, grouped_df, metadata_cols, rolling_aggs)

        df_orig = df.reset_index(drop=True).copy()
        final_df = pd.concat([df_orig, *dfs], ignore_index=True)
        return final_df

    def __handle_aggregations_on_rolling_windows(self, original_df: pd.DataFrame, grouped_df: DataFrameGroupBy, metadata_columns: list[str],
                         rolling_aggregations: dict[str, str]) -> list[pd.DataFrame]:
        dfs = []
        for w in WINDOW_SIZES:
            rolled_df = grouped_df.rolling(window=w).agg(rolling_aggregations).dropna().reset_index()

            rolled_df = rolled_df.dropna()

            rolled_df[metadata_columns] = original_df.loc[
                rolled_df[TEMP_NEW_INDEX_COLUMN_NAME], metadata_columns
            ].values

            rolled_df = rolled_df.drop(columns=TEMP_NEW_INDEX_COLUMN_NAME)
            dfs.append(rolled_df)
        return dfs
