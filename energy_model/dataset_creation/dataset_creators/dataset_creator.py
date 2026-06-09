import sys
from abc import ABC, abstractmethod
from typing import Callable, Union

import pandas as pd

from energy_model.configs.columns import ProcessColumns, SystemColumns
from energy_model.dataset_creation.dataset_creation_config import DEFAULT_BATCH_INTERVAL_SECONDS, TIMESTAMP_COLUMN_NAME, \
    MINIMAL_BATCH_DURATION, AggregationName, COLUMNS_TO_CALCULATE_DIFF, COLUMNS_TO_SUM, \
    DEFAULT_FILTERING_SINGLE_PROCESS, AggregationValue
from energy_model.dataset_creation.raw_telemetry_readers.raw_telemetry_reader import RawTelemetryReader
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator

DEFAULT_DURATIONS_BETWEEN_SAMPLES = (sys.maxsize,)


class DatasetCreator(ABC):
    """
    Class for processing the telemetry data and calculating the energy usage of each sample.
    The data is retrieved from elastic.
    """

    def __init__(self, target_calculator: TargetCalculator, dataset_reader: RawTelemetryReader,
                 batch_time_intervals: list[int] = None, single_process_only: bool = DEFAULT_FILTERING_SINGLE_PROCESS):
        if batch_time_intervals is None:
            batch_time_intervals = DEFAULT_BATCH_INTERVAL_SECONDS

        self.__durations_thresholds = DEFAULT_DURATIONS_BETWEEN_SAMPLES
        self.__batch_time_intervals = batch_time_intervals
        self.__target_calculator = target_calculator
        self.__dataset_reader = dataset_reader
        self.__single_process_only = single_process_only

    def create_dataset(self) -> pd.DataFrame:
        df = self.__dataset_reader.read_dataset()

        full_df = pd.DataFrame()
        for max_aggregated_duration_in_batch in self.__batch_time_intervals:
            full_df_for_interval = self.__get_batch_based_df(df, max_aggregated_duration_in_batch)
            full_df = pd.concat([full_df, full_df_for_interval], ignore_index=True)

        return full_df

    def __get_batch_based_df(self, full_df: pd.DataFrame, max_aggregated_duration_in_batch: int) -> pd.DataFrame:
        """
        For a given batch size, the method splits the given dataframe into sub-dataframes by duration thresholds. For each sub-dataframe it performs processing.
        The concatenation helps with extending the dataset.

        ! In the future, for finding a duration threshold that defines when to use real-time model and long-term model:
        We need to train regression models on different duration ranges.
        In that case, we don't need to concat the dataframes, and we should return each one of them separately.
        We can concat different dataframes based on these durations.
        Input:
            - full_df: the full original dataframe
            - max_aggregated_duration_in_batch: the duration of each batch of samples
        Output:
            - concatenated dataframe with all duration-based thresholds for the given batch size
        """
        full_df_for_interval = pd.DataFrame()
        previous_duration_threshold = 0

        for current_duration_threshold in self.__durations_thresholds:
            full_df_for_duration = self.__get_duration_based_df(full_df, max_aggregated_duration_in_batch,
                                                                previous_duration_threshold,
                                                                current_duration_threshold)
            previous_duration_threshold = current_duration_threshold
            full_df_for_interval = pd.concat([full_df_for_interval, full_df_for_duration], ignore_index=True)

        return full_df_for_interval

    def __get_duration_based_df(self, full_df: pd.DataFrame, max_aggregated_duration_in_batch: int,
                                minimal_duration: int, maximal_duration: int) -> pd.DataFrame:
        """
        The method processes samples that their duration matches the given range, while splitting these samples into batches of max_aggregated_duration_in_batch total duration.
        Input:
            - full_df: the full original dataframe
            - max_aggregated_duration_in_batch: the duration of each batch of samples
            - minimal_duration: the processed sample's duration should be grater than this value
            - maximal_duration: the processed sample's duration should be lower or equal to this value
        Output:
            - Processed dataframe representing samples with duration in a given range.

        For example, given minimal duration of 10 seconds and maximal duration of 30 seconds, we filter from the original dataset the samples with 10 < duration <= 30.
        Then, we split them into batches, where the total duration of the batch (the sum over the duration of all samples in that batch) is batch_size.
        We process each batch separately.
        """
        duration_based_df = full_df[(full_df[SystemColumns.DURATION_COL] > minimal_duration) &
                                    (full_df[SystemColumns.DURATION_COL] <= maximal_duration)]
        full_df_for_duration = self.__process_single_time_interval(duration_based_df, max_aggregated_duration_in_batch)
        return full_df_for_duration

    def __process_single_time_interval(self, df: pd.DataFrame, max_aggregated_duration_in_batch: int) -> pd.DataFrame:
        """
        The method processes a single dataframe by splitting it to batches and handling each batch with:
        * adding batch id
        * adding target column
        * filter last record in the batch if the last batch duration is much smaller than max_aggregated_duration_in_batch
        * remove unnecessary columns
        Input:
            - df: The dataframe to be processed
            - max_aggregated_duration_in_batch: the duration of each batch of samples. Total duration of all samples in this batch should be <= max_aggregated_duration_in_batch
        Output:
            - Processed dataframe
        """
        full_df_with_batch_id = self.__add_batch_id(df, max_aggregated_duration_in_batch)
        self.__check_dataset_validity(full_df_with_batch_id)

        # todo: handle energy calculations with several sessions in the same batch
        full_df_for_interval = self.__extend_df_with_target(full_df_with_batch_id, max_aggregated_duration_in_batch)
        full_df_for_interval = self.__filter_last_batch_records(full_df_for_interval)
        full_df_for_interval = self._remove_temporary_columns(full_df_for_interval)
        return full_df_for_interval

    def __add_batch_id(self, df: pd.DataFrame, max_aggregated_duration_in_batch: int) -> pd.DataFrame:
        df = df.copy()

        # Group by session id.
        # For each group, calculate time delta of each sample (time passed since the beginning of the session).
        # Then, split the time delta by max_aggregated_duration_in_batch to define the index of the batch.
        df[SystemColumns.BATCH_ID_COL] = df[SystemColumns.SESSION_ID_COL] + '_' + (
            df.groupby(SystemColumns.SESSION_ID_COL)[TIMESTAMP_COLUMN_NAME]
            .transform(
                lambda x: ((x - x.min()).dt.total_seconds() // max_aggregated_duration_in_batch).astype(int).astype(
                    str))
        )

        return df

    def __check_dataset_validity(self, df: pd.DataFrame):
        # count unique session_id per batch
        session_counts = df.groupby(SystemColumns.BATCH_ID_COL)[SystemColumns.SESSION_ID_COL].nunique()

        # batches with more than 1 session_id
        bad_batches = session_counts[session_counts > 1]

        if not bad_batches.empty:
            print("⚠️ Warning: Some batches contain multiple session_ids!")
            for batch_id in bad_batches.index:
                batch_df = df[df[SystemColumns.BATCH_ID_COL] == batch_id]

                session_ids = batch_df[SystemColumns.SESSION_ID_COL].unique()
                start_time = batch_df[TIMESTAMP_COLUMN_NAME].min()
                end_time = batch_df[TIMESTAMP_COLUMN_NAME].max()

                print(
                    f" - Batch {batch_id} has {len(session_ids)} session_ids: {list(session_ids)} "
                    f"(from {start_time} to {end_time})"
                )

    def __extend_df_with_target(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        df_with_necessary_columns = self._add_energy_necessary_columns(df, batch_duration_seconds)
        removed_samples = 0
        results = []
        # Handle batches separately depending on process_id count
        for batch_id, batch_df in df_with_necessary_columns.groupby(SystemColumns.BATCH_ID_COL, group_keys=False):
            if self.__single_process_only:
                # Filter out batches with more than 1 processes
                unique_procs = batch_df[ProcessColumns.PROCESS_ID_COL].nunique()
                if unique_procs > 1:
                    removed_samples += len(df[df[SystemColumns.BATCH_ID_COL] == batch_id])
                    continue

            batch_df_with_target = self.__target_calculator.add_target_to_dataframe(batch_df)
            results.append(batch_df_with_target)

        df_with_target = pd.concat(results, ignore_index=True)
        print(f"Used {df.shape[0] - removed_samples}/{df.shape[0]} samples while extending the dataset with target.")
        return df_with_target

    def __filter_last_batch_records(self, df: pd.DataFrame) -> pd.DataFrame:
        # get last batch
        last_batch_id = df[SystemColumns.BATCH_ID_COL].max()
        last_batch = df[df[SystemColumns.BATCH_ID_COL] == last_batch_id]

        # compute its duration (max - min timestamp)
        duration = (last_batch[TIMESTAMP_COLUMN_NAME].max() - last_batch[TIMESTAMP_COLUMN_NAME].min()).total_seconds()

        # check if it's shorter than MINIMAL_BATCH_DURATION minutes
        if duration < MINIMAL_BATCH_DURATION:
            df = df[df[SystemColumns.BATCH_ID_COL] != last_batch_id]

        return df

    def _remove_temporary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop([SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL,
                        SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL,
                        SystemColumns.BATCH_ID_COL, TIMESTAMP_COLUMN_NAME,
                        SystemColumns.SESSION_ID_COL, ProcessColumns.PROCESS_ID_COL],
                       axis=1)

    def _get_necessary_aggregations(self, available_columns: list[str]) -> dict[str, Union[list[str], str, Callable]]:
        consts_columns = list(set(available_columns) - set(COLUMNS_TO_SUM))
        consts_columns = list(set(consts_columns) - set(COLUMNS_TO_CALCULATE_DIFF))
        columns_aggregations: dict[str, AggregationValue] = {
            col: AggregationName.SUM.value for col in available_columns if col in COLUMNS_TO_SUM
        }
        columns_aggregations.update({
            col: AggregationName.FIRST_SAMPLE.value for col in available_columns if col in consts_columns
        })
        columns_aggregations.update({
            col: lambda x: x.iloc[0] - x.iloc[-1] for col in available_columns if col in COLUMNS_TO_CALCULATE_DIFF
        })
        return columns_aggregations

    def get_dataset_file_name(self, dir_path: str) -> str:
        return f"{dir_path}\\{self.get_name()}_{self.__dataset_reader.get_name()}_{self.__target_calculator.get_name()}.csv"

    @abstractmethod
    def _add_energy_necessary_columns(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        """
        This method requires calculating the relevant features for calculating the target column.
        Input:
            df - pandas dataframe with all raw information.
            batch_duration_seconds - duration of each batch in seconds.
        Output:
            pandas dataframe with columns that are relevant for calculating the target.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
