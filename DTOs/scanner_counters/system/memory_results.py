from dataclasses import dataclass

from DTOs.scanner_counters.metric_results import MetricResult


@dataclass
class SystemMemoryResults(MetricResult):
    total_memory_gb: float
    total_memory_percent: float
