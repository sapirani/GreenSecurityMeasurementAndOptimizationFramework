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
from energy_model.dataset_creation.dataset_creation_config import IDLE_SESSION_ID_NAME
from energy_model.energy_model_utils.energy_model_convertor import EnergyModelConvertor
from energy_model.energy_model_utils.energy_model_feature_extractor import EnergyModelFeatureExtractor
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input


class DatasetReader(ABC):
    def __init__(self):
        self.__elastic_reader_iterator = ElasticReader(
            get_time_picker_input(time_picker_input_strategy, preconfigured_time_picker_input),
            [ElasticIndex.PROCESS, ElasticIndex.SYSTEM]).read()

        self._processes_features_extractor_mapping: dict[ProcessIdentity, EnergyModelFeatureExtractor] = defaultdict(
            lambda: EnergyModelFeatureExtractor())

    def read_dataset(self) -> pd.DataFrame:
        all_samples_features = self.__create_all_dataset_objects()
        df = self.__convert_objects_to_dataframe(all_samples_features)
        return df

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

    @abstractmethod
    def _extract_iteration_samples(self, system_raw_results: SystemRawResults,
                                   processes_raw_results: list[ProcessRawResults],
                                   timestamp: datetime, session_host_identity: SessionHostIdentity) -> \
            list[ExtendedEnergyModelFeatures]:
        pass
