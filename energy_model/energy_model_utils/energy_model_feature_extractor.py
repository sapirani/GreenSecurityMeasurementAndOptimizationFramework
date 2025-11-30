from datetime import datetime
from typing import Optional, Union

from DTOs.aggregators_features.empty_features import EmptyFeatures
from DTOs.aggregators_features.energy_model_features.full_energy_model_features import CompleteEnergyModelFeatures, \
    ExtendedEnergyModelFeatures
from DTOs.aggregators_features.energy_model_features.process_energy_model_features import ProcessEnergyModelFeatures
from DTOs.aggregators_features.energy_model_features.system_energy_model_features import SystemEnergyModelFeatures
from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from DTOs.session_host_info import SessionHostIdentity
from energy_model.dataset_creation.data_extractors.hardware_extractor import HardwareExtractor
from utils.general_consts import KB


class EnergyModelFeatureExtractor:
    def __init__(self):
        self.__hardware_details = HardwareExtractor.extract()
        self.__previous_process_sample: Optional[ProcessRawResults] = None
        self.__previous_system_sample: Optional[SystemRawResults] = None
        self.__previous_timestamp: Optional[datetime] = None

    def extract_system_energy_model_features(self, system_raw_results: SystemRawResults, timestamp: datetime) -> Union[
        SystemEnergyModelFeatures, EmptyFeatures]:
        system_features = EmptyFeatures()
        try:
            if self.__previous_timestamp is None:
                return system_features

            duration = (timestamp - self.__previous_timestamp).total_seconds()

            system_features = self.__extract_system_features(
                system_raw_results,
                duration
            )
        finally:
            self.__set_previous_sample(None, system_raw_results, timestamp)
            return system_features

    def extract_process_energy_model_features(self, raw_results: ProcessSystemRawResults,
                                              timestamp: datetime) -> Union[CompleteEnergyModelFeatures, EmptyFeatures]:
        process_features, system_features, duration = self.__extract_process_system_features(raw_results, timestamp)
        if not process_features or not system_features or not duration:
            return EmptyFeatures()

        return CompleteEnergyModelFeatures(
            process_features=process_features,
            system_features=system_features
        )

    def extract_extended_energy_model_features(self, raw_results: ProcessSystemRawResults,
                                               timestamp: datetime, session_host_identity: SessionHostIdentity) -> \
            Union[ExtendedEnergyModelFeatures, EmptyFeatures]:
        process_features, system_features, duration = self.__extract_process_system_features(raw_results, timestamp)
        if not process_features or not system_features or not duration:
            return EmptyFeatures()

        current_battery_capacity = raw_results.system_raw_results.battery_remaining_capacity_mWh
        if not current_battery_capacity:
            raise ValueError("Expected energy stats for system from the device.")

        return ExtendedEnergyModelFeatures(
            duration=duration,
            process_features=process_features,
            system_features=system_features,
            session_id=session_host_identity.session_id,
            hostname=session_host_identity.hostname,
            pid=raw_results.process_raw_results.pid,
            hardware_features=self.__hardware_details,
            timestamp=timestamp,
            battery_remaining_capacity_mWh=current_battery_capacity
        )

    def __extract_process_system_features(self, raw_results: ProcessSystemRawResults, current_timestamp: datetime) -> \
            tuple[Optional[ProcessEnergyModelFeatures], Optional[SystemEnergyModelFeatures], Optional[float]]:
        process_features = None
        system_features = None
        duration = None
        try:
            if self.__previous_timestamp is None:
                return process_features, system_features, duration

            duration = (current_timestamp - self.__previous_timestamp).total_seconds()
            process_features = self.__extract_process_features(
                raw_results.process_raw_results,
                duration
            )

            system_features = self.__extract_system_features(
                raw_results.system_raw_results,
                duration
            )
        finally:
            self.__set_previous_sample(raw_results.process_raw_results, raw_results.system_raw_results,
                                       current_timestamp)
            return process_features, system_features, duration

    def __set_previous_sample(self, process_raw_results: Optional[ProcessRawResults],
                              system_raw_results: SystemRawResults, timestamp: datetime):
        self.__previous_timestamp = timestamp
        self.__previous_process_sample = process_raw_results
        self.__previous_system_sample = system_raw_results

    def __extract_process_features(self, process_data: ProcessRawResults,
                                   duration: float) -> ProcessEnergyModelFeatures:
        process_cpu_time_seconds = EnergyModelFeatureExtractor.__calculate_integral_value(
            process_data.cpu_percent_sum_across_cores,
            self.__previous_process_sample.cpu_percent_sum_across_cores,
            duration) / 100
        process_memory_relative_usage = process_data.used_memory_mb - self.__previous_process_sample.used_memory_mb
        return ProcessEnergyModelFeatures(
            duration=duration,
            cpu_usage_seconds_process=process_cpu_time_seconds,
            memory_mb_relative_process=process_memory_relative_usage,
            disk_read_kb_process=process_data.disk_read_kb,
            disk_write_kb_process=process_data.disk_write_kb,
            disk_read_count_process=process_data.disk_read_count,
            disk_write_count_process=process_data.disk_write_count,
            number_of_page_faults_process=process_data.page_faults,
            network_kb_received_process=process_data.network_kb_received,
            network_packets_received_process=process_data.packets_received,
            network_kb_sent_process=process_data.network_kb_sent,
            network_packets_sent_process=process_data.packets_sent)

    def __extract_system_features(self, system_data: SystemRawResults, duration: float) -> SystemEnergyModelFeatures:
        system_cpu_time = EnergyModelFeatureExtractor.__calculate_integral_value(
            system_data.cpu_percent_sum_across_cores,
            self.__previous_system_sample.cpu_percent_sum_across_cores,
            duration) / 100
        system_memory_relative_usage_mb = (
                                                      system_data.total_memory_gb - self.__previous_system_sample.total_memory_gb) * KB
        return SystemEnergyModelFeatures(
            duration=duration,
            cpu_seconds_system=system_cpu_time,
            memory_mb_relative_system=system_memory_relative_usage_mb,
            disk_read_kb_system=system_data.disk_read_kb,
            disk_write_kb_system=system_data.disk_write_kb,
            disk_read_count_system=system_data.disk_read_count,
            disk_write_count_system=system_data.disk_write_count,
            network_kb_sent_system=system_data.network_kb_sent,
            network_packets_sent_system=system_data.packets_sent,
            network_kb_received_system=system_data.packets_received,
            network_packets_received_system=system_data.packets_received,
            disk_read_time_ms_system=system_data.disk_read_time,
            disk_write_time_ms_system=system_data.disk_write_time
        )

    @staticmethod
    def __calculate_integral_value(current_val: float, previous_val: float, duration: float) -> float:
        return (current_val + previous_val) * duration / 2
