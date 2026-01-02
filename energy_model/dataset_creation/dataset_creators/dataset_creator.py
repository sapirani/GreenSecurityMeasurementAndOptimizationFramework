from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime

import pandas as pd

from DTOs.aggregators_features.energy_model_features.full_energy_model_features import ExtendedEnergyModelFeatures
from DTOs.process_info import ProcessIdentity
from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from DTOs.session_host_info import SessionHostIdentity
from elastic_reader.consts import ElasticIndex
from elastic_reader.elastic_reader import ElasticReader
from elastic_reader.elastic_reader_parameters import time_picker_input_strategy, preconfigured_time_picker_input
from energy_model.configs.columns import ProcessColumns, SystemColumns
from energy_model.dataset_creation.dataset_creation_config import DEFAULT_BATCH_INTERVAL_SECONDS, TIMESTAMP_COLUMN_NAME, \
    MINIMAL_BATCH_DURATION, IDLE_SESSION_ID_NAME
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator
from energy_model.energy_model_parameters import FULL_DATASET_BEFORE_PROCESSING_PATH
from energy_model.energy_model_utils.energy_model_convertor import EnergyModelConvertor
from energy_model.energy_model_utils.energy_model_feature_extractor import EnergyModelFeatureExtractor
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input


class DatasetCreator(ABC):
    def __init__(self, target_calculator: TargetCalculator, batch_time_intervals: list[int] = None):
        if batch_time_intervals is None:
            batch_time_intervals = DEFAULT_BATCH_INTERVAL_SECONDS

        self.__elastic_reader_iterator = ElasticReader(
            get_time_picker_input(time_picker_input_strategy, preconfigured_time_picker_input),
            [ElasticIndex.PROCESS, ElasticIndex.SYSTEM]).read()
        self.__processes_features_extractor_mapping: dict[ProcessIdentity, EnergyModelFeatureExtractor] = defaultdict(
            lambda: EnergyModelFeatureExtractor())
        self.__batch_time_intervals = batch_time_intervals
        self.__target_calculator = target_calculator

    def create_dataset(self) -> pd.DataFrame:
        all_samples_features = self.__create_all_dataset_objects()
        df = self.__convert_objects_to_dataframe(all_samples_features)

        full_df = pd.DataFrame()
        for batch_interval in self.__batch_time_intervals:
            full_df_for_interval = self.__handle_single_time_interval(df, batch_interval)
            full_df = pd.concat([full_df, full_df_for_interval], ignore_index=True)

        full_df.to_csv(FULL_DATASET_BEFORE_PROCESSING_PATH)
        return full_df

    def __create_all_dataset_objects(self) -> list[ExtendedEnergyModelFeatures]:
        all_samples = []
        for sample in self.__elastic_reader_iterator:
            metadata = sample.metadata
            if sample.system_raw_results is None or IDLE_SESSION_ID_NAME in metadata.session_host_identity.session_id:
                continue

            # todo: fix duration handling in case of multiple sessions and hostnames running at the same time (single iteration raw results may contain samples from different measurements)
            iteration_samples = self._extract_iteration_samples(sample.system_raw_results,
                                                                list(sample.processes_raw_results.values()),
                                                                metadata.timestamp, metadata.session_host_identity)

            all_samples.extend(iteration_samples)

        return all_samples

    @staticmethod
    def __convert_objects_to_dataframe(all_samples_features: list[ExtendedEnergyModelFeatures]):
        samples_as_df = [EnergyModelConvertor.convert_complete_features_to_pandas(sample, timestamp=sample.timestamp,
                                                                                  session_id=sample.session_id,
                                                                                  hostname=sample.hostname,
                                                                                  pid=sample.pid,
                                                                                  battery_capacity_mwh_system=sample.battery_remaining_capacity_mWh,
                                                                                  **asdict(sample.hardware_features))
                         for sample in all_samples_features]
        full_df = pd.concat(samples_as_df, ignore_index=True)
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

        results = []
        # Step 2: Handle batches separately depending on process_id count
        for batch_id, batch_df in df_with_necessary_columns.groupby(SystemColumns.BATCH_ID_COL, group_keys=False):
            batch_df_with_target = self.__target_calculator.add_target_to_dataframe(batch_df)
            results.append(batch_df_with_target)

        df_with_target = pd.concat(results, ignore_index=True)
        return df_with_target

    @staticmethod
    def __filter_last_batch_records(df: pd.DataFrame) -> pd.DataFrame:
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

    @abstractmethod
    def _extract_iteration_samples(self, system_raw_results: SystemRawResults,
                                   processes_raw_results: list[ProcessRawResults],
                                   timestamp: datetime, session_host_identity: SessionHostIdentity) -> \
                                   list[ExtendedEnergyModelFeatures]:
        pass

    @abstractmethod
    def _add_energy_necessary_columns(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        pass
