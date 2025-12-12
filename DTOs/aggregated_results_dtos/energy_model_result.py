from dataclasses import dataclass

from DTOs.aggregated_results_dtos.abstract_aggregation_results import AbstractAggregationResult


@dataclass
class EnergyModelResult(AbstractAggregationResult):
    energy_mwh: float
    cpu_energy_consumption: float
    ram_energy_consumption: float
    disk_io_read_energy_consumption: float
    disk_io_write_energy_consumption: float
    network_io_received_energy_consumption: float
    network_io_sent_energy_consumption: float
