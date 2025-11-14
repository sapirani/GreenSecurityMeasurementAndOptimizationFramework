from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Dict

import pandas as pd

from DTOs.aggregators_features.empty_features import EmptyFeatures
from DTOs.aggregators_features.energy_model_features.full_energy_model_features import ExtendedEnergyModelFeatures
from DTOs.process_info import ProcessIdentity
from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from DTOs.session_host_info import SessionHostIdentity
from elastic_reader.consts import ElasticIndex
from elastic_reader.elastic_reader import ElasticReader
from elastic_reader.elastic_reader_parameters import time_picker_input_strategy, preconfigured_time_picker_input
from measurements_model_pipeline.dataset_creation.dataset_constants import TIMESTAMP_COLUMN_NAME, IDLE_SESSION_ID_NAME
from measurements_model_pipeline.dataset_parameters import FULL_DATASET_PATH
from measurements_model_pipeline.column_names import ProcessColumns, SystemColumns
from measurements_model_pipeline.energy_model_convertor import EnergyModelConvertor
from measurements_model_pipeline.energy_model_feature_extractor import EnergyModelFeatureExtractor
from measurements_model_pipeline.resource_energy_calculator import ResourceEnergyCalculator
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input
from utils.general_consts import MINUTE

DEFAULT_BATCH_INTERVAL_SECONDS = 5 * MINUTE
MINIMAL_BATCH_DURATION = DEFAULT_BATCH_INTERVAL_SECONDS * 0.5

# todo: change to consumer interface
class SystemDatasetCreator:
    def __init__(self):
        self.__elastic_reader_iterator = ElasticReader(
            get_time_picker_input(time_picker_input_strategy, preconfigured_time_picker_input),
            [ElasticIndex.PROCESS, ElasticIndex.SYSTEM]).read()

        self.__processes_features_extractor_mapping: Dict[ProcessIdentity, EnergyModelFeatureExtractor] = defaultdict(
            lambda: EnergyModelFeatureExtractor())
        self.__resource_energy_calculator = ResourceEnergyCalculator()

    def __create_system_process_dataset(self) -> list[ExtendedEnergyModelFeatures]:
        all_samples = []
        for sample in self.__elastic_reader_iterator:
            metadata = sample.metadata
            if sample.system_raw_results is None or IDLE_SESSION_ID_NAME in metadata.session_host_identity.session_id:
                continue

            # todo: fix duration handling in case of multiple sessions and hostnames running at the same time (single iteration raw results may contain samples from different measurements)
            iteration_samples = self.__extract_iteration_samples(sample.system_raw_results,
                                                                 sample.processes_raw_results,
                                                                 metadata.timestamp, metadata.session_host_identity)

            all_samples.extend(iteration_samples)

        return all_samples

    def __extract_iteration_samples(self, system_raw_results: SystemRawResults,
                                    processes_raw_results: list[ProcessRawResults],
                                    timestamp: datetime, session_host_identity: SessionHostIdentity) -> list[
        ExtendedEnergyModelFeatures]:
        iteration_samples = []
        for process_result in processes_raw_results:
            if not process_result.process_of_interest:
                continue

            sample_raw_results = ProcessSystemRawResults(system_raw_results=system_raw_results,
                                                         process_raw_results=process_result)
            process_id = ProcessIdentity.from_raw_results(process_result)
            process_feature_extractor = self.__processes_features_extractor_mapping[process_id]
            sample_features = process_feature_extractor.extract_extended_energy_model_features(
                raw_results=sample_raw_results, timestamp=timestamp, session_host_identity=session_host_identity)

            if isinstance(sample_features, EmptyFeatures):
                continue

            iteration_samples.append(sample_features)

        return iteration_samples

    def __convert_objects_to_dataframe(self, all_samples_features: list[ExtendedEnergyModelFeatures]):
        samples_as_df = [EnergyModelConvertor.convert_features_to_pandas(sample,
                                                                         timestamp=sample.timestamp,
                                                                         session_id=sample.session_id,
                                                                         hostname=sample.hostname,
                                                                         pid=sample.pid,
                                                                         battery_capacity_mwh_system=sample.battery_remaining_capacity_mWh,
                                                                         **asdict(sample.hardware_features))
                         for sample in all_samples_features]
        full_df = pd.concat(samples_as_df, ignore_index=True)
        return full_df

    def create_dataset(self) -> pd.DataFrame:
        all_samples_features = self.__create_system_process_dataset()
        df = self.__convert_objects_to_dataframe(all_samples_features)
        full_df_with_batch_id = self.__add_batch_id(df, DEFAULT_BATCH_INTERVAL_SECONDS)
        self.__check_dataset_validity(full_df_with_batch_id)
        # todo: handle energy calculations with several sessions in the same batch
        full_df = self.__extend_df_with_target(full_df_with_batch_id, DEFAULT_BATCH_INTERVAL_SECONDS)
        full_df = self.__filter_last_batch_records(full_df)
        full_df = self.__remove_temporary_columns(full_df)
        full_df.to_csv(FULL_DATASET_PATH)
        return full_df

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
        """
        Extend the given DataFrame with energy usage targets.

        Steps:
        1. Calculate the average system energy consumption per second for each batch.
        2. Compute the process-level energy usage for each row, adjusted for idle consumption.
        """

        # Step 1: Calculate system energy consumption rate (mWh/sec) for each batch
        energy_per_batch = (
            df.groupby(SystemColumns.BATCH_ID_COL)[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL]
            .agg(lambda s: (s.iloc[0] - s.iloc[-1]) / batch_duration_seconds)
            .rename(SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL)
        )

        # Merge batch-level system energy rates back into the main DataFrame
        df = df.merge(energy_per_batch, on=SystemColumns.BATCH_ID_COL, how="left")

        results = []
        # Step 2: Handle batches separately depending on process_id count
        for batch_id, batch_df in df.groupby(SystemColumns.BATCH_ID_COL, group_keys=False):
            batch_df[SystemColumns.ENERGY_USAGE_SYSTEM_COL] = (batch_df[SystemColumns.DURATION_COL] *
                                                               batch_df[
                                                                   SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL])

            results.append(batch_df)

        df = pd.concat(results, ignore_index=True)
        return df

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

    @staticmethod
    def __remove_temporary_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop([SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL,
                        SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL,
                        SystemColumns.BATCH_ID_COL, TIMESTAMP_COLUMN_NAME,
                        SystemColumns.SESSION_ID_COL, ProcessColumns.PROCESS_ID_COL,
                        SystemColumns.ENERGY_RATIO_SHARE],
                       axis=1)
