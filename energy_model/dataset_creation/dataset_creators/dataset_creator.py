from abc import ABC, abstractmethod
from typing import Callable, Union

import pandas as pd

from energy_model.configs.columns import ProcessColumns, SystemColumns
from energy_model.dataset_creation.dataset_creation_config import DEFAULT_BATCH_INTERVAL_SECONDS, TIMESTAMP_COLUMN_NAME, \
    MINIMAL_BATCH_DURATION, AggregationName, COLUMNS_TO_CALCULATE_DIFF, COLUMNS_TO_SUM, \
    DEFAULT_FILTERING_SINGLE_PROCESS, AggregationValue
from energy_model.dataset_creation.dataset_readers.dataset_reader import DatasetReader
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator


class DatasetCreator(ABC):
    """
    Class for processing the telemetry data and calculating the energy usage of each sample.
    """
    def __init__(self, target_calculator: TargetCalculator, dataset_reader: DatasetReader,
                 batch_time_intervals: list[int] = None, single_process_only: bool = DEFAULT_FILTERING_SINGLE_PROCESS):
        if batch_time_intervals is None:
            batch_time_intervals = DEFAULT_BATCH_INTERVAL_SECONDS

        self.__batch_time_intervals = batch_time_intervals
        self.__target_calculator = target_calculator
        self.__dataset_reader = dataset_reader
        self.__single_process_only = single_process_only

    def create_dataset(self) -> pd.DataFrame:
        df = self.__dataset_reader.read_dataset()

        full_df = pd.DataFrame()
        for batch_interval in self.__batch_time_intervals:
            full_df_for_interval = self.__handle_single_time_interval(df, batch_interval)
            full_df = pd.concat([full_df, full_df_for_interval], ignore_index=True)

        return full_df

    def __handle_single_time_interval(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        full_df_with_batch_id = self.__add_batch_id(df, batch_duration_seconds)
        self.__check_dataset_validity(full_df_with_batch_id)

        # todo: handle energy calculations with several sessions in the same batch
        full_df_for_interval = self.__extend_df_with_target(full_df_with_batch_id, batch_duration_seconds)
        full_df_for_interval = self.__filter_last_batch_records(full_df_for_interval)
        full_df_for_interval = self._remove_temporary_columns(full_df_for_interval)
        return full_df_for_interval

    def __add_batch_id(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        df = df.copy()

        # Group by session id.
        # For each group, calculate time delta of each sample (time passed since the beginning of the session).
        # Then, split the time delta by batch_duration_seconds to define the index of the batch.
        df[SystemColumns.BATCH_ID_COL] = df[SystemColumns.SESSION_ID_COL] + '_' + (
            df.groupby(SystemColumns.SESSION_ID_COL)[TIMESTAMP_COLUMN_NAME]
            .transform(lambda x: ((x - x.min()).dt.total_seconds() // batch_duration_seconds).astype(int).astype(str))
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
            col: AggregationName.SUM for col in available_columns if col in COLUMNS_TO_SUM
        }
        columns_aggregations.update({
            col: AggregationName.FIRST_SAMPLE for col in available_columns if col in consts_columns
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
