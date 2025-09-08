from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from DTOs.aggregators_features.empty_features import EmptyFeatures
from DTOs.aggregators_features.energy_model_features.full_energy_model_features import ExtendedEnergyModelFeatures
from DTOs.aggregators_features.energy_model_features.idle_energy_model_features import IdleEnergyModelFeatures
from DTOs.process_info import ProcessIdentity
from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from elastic_reader.consts import ElasticIndex
from elastic_reader.elastic_reader import ElasticReader
from elastic_reader.elastic_reader_parameters import time_picker_input_strategy, preconfigured_time_picker_input
from measurements_model.config import TIME_COLUMN_NAME, ProcessColumns, IDLE_SESSION_ID_NAME, FULL_DATASET_PATH, \
    SystemColumns
from measurements_model.energy_model_convertor import EnergyModelConvertor
from measurements_model.energy_model_feature_extractor import EnergyModelFeatureExtractor
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input
from utils.general_consts import MINUTE

DEFAULT_BATCH_INTERVAL_SECONDS = 4 * MINUTE
DEFAULT_ENERGY_PER_SECOND_IDLE_MEASUREMENT = 2.921666667 # todo: extend this logic when we want to use a baseline background activity instead of idle.
ENERGY_MINIMAL_VALUE = 0


# todo: change to consumer interface
class DatasetCreator:
    def __init__(self, idle_session_path: str):
        self.__elastic_reader_iterator = ElasticReader(
            get_time_picker_input(time_picker_input_strategy, preconfigured_time_picker_input),
            [ElasticIndex.PROCESS, ElasticIndex.SYSTEM]).read()
        self.__idle_details = IdleEnergyModelFeatures(
            energy_per_second=DEFAULT_ENERGY_PER_SECOND_IDLE_MEASUREMENT
        )
        self.__processes_features_extractor_mapping: Dict[ProcessIdentity, EnergyModelFeatureExtractor] = defaultdict(
            lambda: EnergyModelFeatureExtractor())

    def __create_system_process_dataset(self) -> list[ExtendedEnergyModelFeatures]:
        all_samples = []
        for sample in self.__elastic_reader_iterator:
            metadata = sample.metadata
            print("TIMESTAMP: ", metadata.timestamp)
            if sample.system_raw_results is None or IDLE_SESSION_ID_NAME in metadata.session_host_identity.session_id:
                continue

            # todo: fix duration handling in case of several sessions running at the same time (single iteration raw results may contain samples from different measurements)
            iteration_samples = self.__extract_iteration_samples(sample.system_raw_results,
                                                                 sample.processes_raw_results,
                                                                 metadata.timestamp)

            all_samples.extend(iteration_samples)

        return all_samples

    def __extract_iteration_samples(self, system_raw_results: SystemRawResults,
                                    processes_raw_results: list[ProcessRawResults],
                                    timestamp: datetime) -> list[ExtendedEnergyModelFeatures]:
        iteration_samples = []
        for process_result in processes_raw_results:
            if not process_result.process_of_interest:
                continue

            sample_raw_results = ProcessSystemRawResults(system_raw_results=system_raw_results,
                                                         process_raw_results=process_result)
            process_id = ProcessIdentity.from_raw_results(process_result)
            process_feature_extractor = self.__processes_features_extractor_mapping[process_id]
            sample_features = process_feature_extractor.extract_extended_energy_model_features(
                raw_results=sample_raw_results, timestamp=timestamp)

            if isinstance(sample_features, EmptyFeatures):
                continue

            iteration_samples.append(sample_features)

        return iteration_samples

    def __convert_objects_to_dataframe(self, all_samples_features: list[ExtendedEnergyModelFeatures]):
        samples_as_df = [EnergyModelConvertor.convert_features_to_pandas(sample,
                                                                         timestamp=sample.timestamp,
                                                                         battery_capacity_mwh_system=sample.battery_remaining_capacity_mWh,
                                                                         **asdict(sample.hardware_features))
                         for sample in all_samples_features]
        full_df = pd.concat(samples_as_df, ignore_index=True)
        return full_df

    def create_dataset(self) -> pd.DataFrame:
        all_samples_features = self.__create_system_process_dataset()
        df = self.__convert_objects_to_dataframe(all_samples_features)
        full_df = self.__extend_df_with_target(df, DEFAULT_BATCH_INTERVAL_SECONDS)
        full_df.to_csv(FULL_DATASET_PATH)
        return full_df

    def __extend_df_with_target(self, df: pd.DataFrame, time_per_batch: int) -> pd.DataFrame:
        # TODO: beautify this code
        """
        Extend the given DataFrame with energy usage targets.

        Steps:
        1. Assign each sample to a batch based on `time_per_batch`.
        2. Calculate the average system energy consumption per second for each batch.
        3. Compute the process-level energy usage for each row, adjusted for idle consumption.
        """

        df = df.copy()

        # Step 1: Assign batch IDs (integer division of timestamp by batch duration)
        df[SystemColumns.BATCH_ID_COLUMN] = (
                df[TIME_COLUMN_NAME].astype("int64") // 10 ** 9 // time_per_batch
        ).astype(int)

        # Step 2: Calculate system energy consumption rate (mWh/sec) for each batch
        energy_per_batch = (
            df.groupby(SystemColumns.BATCH_ID_COLUMN)[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL]
            .agg(lambda s: (s.iloc[0] - s.iloc[-1]) / time_per_batch)
            .rename(SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL)
        )

        # Merge batch-level system energy rates back into the main DataFrame
        df = df.merge(energy_per_batch, on=SystemColumns.BATCH_ID_COLUMN, how="left")

        # Step 3: Calculate process energy usage
        df[ProcessColumns.ENERGY_USAGE_PROCESS_COL] = (
                df[SystemColumns.DURATION_COL] * df[SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL]
                - df[SystemColumns.DURATION_COL] * self.__idle_details.energy_per_second
        ).clip(lower=ENERGY_MINIMAL_VALUE)

        return df
