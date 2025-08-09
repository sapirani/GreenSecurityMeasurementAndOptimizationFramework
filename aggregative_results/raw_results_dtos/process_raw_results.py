from dataclasses import dataclass, fields
from typing import List, Dict, Any

from aggregative_results.raw_results_dtos.abstract_raw_results import AbstractRawResults


@dataclass
class ProcessRawResults(AbstractRawResults):
    pid: int
    process_name: str
    arguments: List[str]
    cpu_percent_sum_across_cores: float
    cpu_percent_mean_across_cores: float
    threads_num: int
    used_memory_mb: float
    used_memory_percent: float
    disk_read_count: int
    disk_write_count: int
    disk_read_kb: float
    disk_write_kb: float
    page_faults: int
    network_kb_sent: float
    packets_sent: int
    network_kb_received: float
    packets_received: int
    process_of_interest: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessRawResults':
        init_kwargs = {}

        for f in fields(cls):
            if f.name not in data:
                raise ValueError(f"Missing required field: {f.name}")
            init_kwargs[f.name] = data[f.name]

        return cls(**init_kwargs)
