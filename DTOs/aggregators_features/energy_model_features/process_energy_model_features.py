from dataclasses import dataclass, fields

import pandas as pd


@dataclass
class ProcessEnergyModelFeatures:
    cpu_usage_seconds_process: float
    memory_mb_relative_process: float
    disk_read_kb_process: float
    disk_read_count_process: int
    disk_write_kb_process: float
    disk_write_count_process: int
    number_of_page_faults_process: int
    network_kb_sent_process: float
    network_packets_sent_process: int
    network_kb_received_process: float
    network_packets_received_process: int

    @classmethod
    def from_pandas_series(cls, row: pd.Series) -> "ProcessEnergyModelFeatures":
        init_kwargs = {}

        for f in fields(cls):
            if f.name not in row:
                raise ValueError(f"Missing required field: {f.name}")
            init_kwargs[f.name] = row[f.name]

        return cls(**init_kwargs)
