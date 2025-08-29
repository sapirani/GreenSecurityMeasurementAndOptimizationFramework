from dataclasses import dataclass

from DTOs.aggregated_results_dtos.abstract_aggregation_results import AbstractAggregationResult


@dataclass
class ProcessShareUsageFromTotal(AbstractAggregationResult):
    """
    All results are in the range of [0, 1].
    """
    cpu_usage_share: float
    memory_usage_share: float
    disk_read_count_share: float
    disk_write_count_share: float
    disk_read_volume_share: float
    disk_write_volume_share: float
    network_volume_sent_share: float
    packets_sent_share: float
    network_volume_received_share: float
    packets_received_share: float
