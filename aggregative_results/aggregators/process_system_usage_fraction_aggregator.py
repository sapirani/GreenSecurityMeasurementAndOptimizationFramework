from typing import List

from aggregative_results.DTOs.aggregated_results_dtos.process_share_usage_from_total import ProcessShareUsageFromTotal
from aggregative_results.DTOs.aggregators_features.process_system_usage_fraction_features import \
    ProcessSystemUsageFractionFeatures
from aggregative_results.aggregators.abstract_aggregator import AbstractAggregator
from aggregative_results.DTOs.raw_results_dtos.iteration_info import IterationMetadata
from aggregative_results.DTOs.raw_results_dtos.system_processes_raw_results import FullScopeRawResults


class ProcessSystemUsageFractionAggregator(AbstractAggregator):
    def extract_features(
            self,
            raw_results: FullScopeRawResults,
            iteration_metadata: IterationMetadata
    ) -> ProcessSystemUsageFractionFeatures:
        processes_cpu = [p.cpu_percent_sum_across_cores for p in raw_results.processes_raw_results]
        processes_memory_mb = [p.used_memory_mb for p in raw_results.processes_raw_results]
        processes_disk_read_count = [p.disk_read_count for p in raw_results.processes_raw_results]
        processes_disk_write_count = [p.disk_write_count for p in raw_results.processes_raw_results]
        processes_disk_read_kb = [p.disk_read_kb for p in raw_results.processes_raw_results]
        processes_disk_write_kb = [p.disk_write_kb for p in raw_results.processes_raw_results]
        processes_network_kb_sent = [p.network_kb_sent for p in raw_results.processes_raw_results]
        processes_packets_sent = [p.packets_sent for p in raw_results.processes_raw_results]
        processes_network_kb_received = [p.network_kb_received for p in raw_results.processes_raw_results]
        processes_packets_received = [p.packets_received for p in raw_results.processes_raw_results]

        return ProcessSystemUsageFractionFeatures(
            desired_process_cpu=raw_results.desired_process_raw_results.cpu_percent_sum_across_cores,
            processes_cpu=processes_cpu,

            desired_process_memory_mb=raw_results.desired_process_raw_results.used_memory_mb,
            processes_memory_mb=processes_memory_mb,

            desired_process_disk_read_count=raw_results.desired_process_raw_results.disk_read_count,
            processes_disk_read_count=processes_disk_read_count,

            desired_process_disk_write_count=raw_results.desired_process_raw_results.disk_write_count,
            processes_disk_write_count=processes_disk_write_count,

            desired_process_disk_read_kb=raw_results.desired_process_raw_results.disk_read_kb,
            processes_disk_read_kb=processes_disk_read_kb,

            desired_process_disk_write_kb=raw_results.desired_process_raw_results.disk_write_kb,
            processes_disk_write_kb=processes_disk_write_kb,

            desired_process_network_kb_sent=raw_results.desired_process_raw_results.network_kb_sent,
            processes_network_kb_sent=processes_network_kb_sent,

            desired_process_packets_sent=raw_results.desired_process_raw_results.packets_sent,
            processes_packets_sent=processes_packets_sent,

            desired_process_network_kb_received=raw_results.desired_process_raw_results.network_kb_received,
            processes_network_kb_received=processes_network_kb_received,

            desired_process_packets_received=raw_results.desired_process_raw_results.packets_received,
            processes_packets_received=processes_packets_received,
        )

    @staticmethod
    def safe_division(desired_process_result: float, all_processes_results: List[float]) -> float:
        all_processes_results_sum = sum(all_processes_results)
        if all_processes_results_sum == 0:
            return 0.0
        return desired_process_result / all_processes_results_sum

    def process_sample(self, sample: ProcessSystemUsageFractionFeatures) -> ProcessShareUsageFromTotal:

        return ProcessShareUsageFromTotal(
            cpu_usage_share=self.safe_division(sample.desired_process_cpu, sample.processes_cpu),
            memory_usage_share=self.safe_division(sample.desired_process_memory_mb, sample.processes_memory_mb),
            disk_read_count_share=self.safe_division(sample.desired_process_disk_read_count, sample.processes_disk_read_count),
            disk_write_count_share=self.safe_division(sample.desired_process_disk_write_count, sample.processes_disk_write_count),
            disk_read_volume_share=self.safe_division(sample.desired_process_disk_read_kb, sample.processes_disk_read_kb),
            disk_write_volume_share=self.safe_division(sample.desired_process_disk_write_kb, sample.processes_disk_write_kb),
            network_volume_sent_share=self.safe_division(sample.desired_process_network_kb_sent, sample.processes_network_kb_sent),
            packets_sent_share=self.safe_division(sample.desired_process_packets_sent, sample.processes_packets_sent),
            network_volume_received_share=self.safe_division(sample.desired_process_network_kb_received, sample.processes_network_kb_received),
            packets_received_share=self.safe_division(sample.desired_process_packets_received, sample.processes_packets_received),
        )
