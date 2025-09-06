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
from measurements_model.config import TIME_COLUMN_NAME, ProcessColumns
from measurements_model.energy_model_convertor import EnergyModelConvertor
from measurements_model.energy_model_feature_extractor import EnergyModelFeatureExtractor
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input

DEFAULT_TIME_PER_BATCH = 150
DEFAULT_ENERGY_PER_SECOND_IDLE_MEASUREMENT = 1753


class DatasetCreator:
    def __init__(self, idle_session_path: str):
        self.__elastic_reader_iterator = ElasticReader(
            get_time_picker_input(time_picker_input_strategy, preconfigured_time_picker_input),
            [ElasticIndex.PROCESS, ElasticIndex.SYSTEM]).read()
        self.__idle_details = IdleEnergyModelFeatures(
            energy_per_second=DEFAULT_ENERGY_PER_SECOND_IDLE_MEASUREMENT
        )
        self.__processes_extractor_mapping: Dict[ProcessIdentity, EnergyModelFeatureExtractor] = defaultdict(
            lambda: EnergyModelFeatureExtractor())

    def __create_system_process_dataset(self) -> list[ExtendedEnergyModelFeatures]:
        all_samples = []
        duration = 0
        last_timestamp: Optional[datetime] = None
        for sample in self.__elastic_reader_iterator:
            metadata = sample.metadata
            current_timestamp = metadata.timestamp
            if sample.system_raw_results is None or "idle" in metadata.session_host_identity.session_id:
                continue

            if last_timestamp is not None:
                duration += (current_timestamp - last_timestamp).total_seconds()
            last_timestamp = current_timestamp

            iteration_samples = self.__extract_iteration_samples(sample.system_raw_results,
                                                                 sample.processes_raw_results,
                                                                 duration, current_timestamp)

            all_samples.extend(iteration_samples)
            if len(all_samples) > 5000:  # todo: remove after finding way to exit loop
                break
            print("TIMESTAMP: ", current_timestamp)

        return all_samples

    def __extract_iteration_samples(self, system_raw_results: SystemRawResults,
                                    processes_raw_results: list[ProcessRawResults],
                                    duration: float, timestamp: datetime) -> list[ExtendedEnergyModelFeatures]:
        iteration_samples = []
        for process_result in processes_raw_results:
            if not process_result.process_of_interest:
                continue
                
            sample_raw_results = ProcessSystemRawResults(system_raw_results=system_raw_results,
                                                         process_raw_results=process_result)
            process_id = ProcessIdentity.from_raw_results(process_result)
            process_feature_extractor = self.__processes_extractor_mapping[process_id]
            sample_features = process_feature_extractor.extract_extended_energy_model_features(
                raw_results=sample_raw_results, timestamp=timestamp, duration=duration)

            if isinstance(sample_features, EmptyFeatures):
                continue

            iteration_samples.append(sample_features)

        return iteration_samples

    def __convert_objects_to_dataframe(self, all_samples_features: list[ExtendedEnergyModelFeatures]):
        samples_as_df = [EnergyModelConvertor.convert_features_to_pandas(sample,
                                                                         battery_capacity_mwh_system=sample.battery_remaining_capacity_mWh,
                                                                         **asdict(sample.hardware_features))
                         for sample in all_samples_features]
        full_df = pd.concat(samples_as_df, ignore_index=True)
        return full_df

    def create_dataset(self) -> pd.DataFrame:
        all_samples_features = self.__create_system_process_dataset()
        df = self.__convert_objects_to_dataframe(all_samples_features)
        full_df = self.__extend_df_with_target(df, DEFAULT_TIME_PER_BATCH)
        return full_df

    def __extend_df_with_target(self, df: pd.DataFrame, time_per_batch: int) -> pd.DataFrame:
        df = df.copy()
        # todo: make it look better
        df["batch_id"] = (df[TIME_COLUMN_NAME] // time_per_batch).astype(int)

        # Step 4: compute energy usage per second per batch
        energy_per_batch = (
            df.groupby("batch_id")["battery_capacity_mwh_system"]
            .agg(lambda s: (s.iloc[0] - s.iloc[-1]) / time_per_batch)
            .rename("energy_per_sec_system")
        )
        df = df.merge(energy_per_batch, on="batch_id", how="left")

        # Step 5: energy usage per row
        df[ProcessColumns.ENERGY_USAGE_PROCESS_COL] = df["duration"] * df["energy_per_sec_system"] - \
                                                      df["duration"] * self.__idle_details.energy_per_second
        return df
