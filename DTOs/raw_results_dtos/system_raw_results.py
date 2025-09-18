from dataclasses import dataclass, field, fields, MISSING
from typing import Dict, Any, List, Optional
from DTOs.raw_results_dtos.abstract_raw_results import AbstractRawResults


def get_cores_metrics(data: Dict[str, Any]):
    core_fields = {}
    for key, value in data.items():
        if key.startswith("core_") and key.endswith("_percent"):
            try:
                core_index = int(key.split("_")[1])
                core_fields[core_index] = value
            except (IndexError, ValueError) as e:
                raise RuntimeError(
                    f"Invalid core key format: '{key}'. Expected 'core_N_percent'"
                ) from e

    if not core_fields:
        raise ValueError("No core percent fields found.")

    # Validate contiguous indices starting at 0
    expected_indices = list(range(len(core_fields)))
    actual_indices = sorted(core_fields.keys())

    if actual_indices != expected_indices:
        raise ValueError(
            f"Core indices must start at 0 and be contiguous. Found: {actual_indices}"
        )

    return [core_fields[i] for i in expected_indices]


@dataclass
class SystemRawResults(AbstractRawResults):
    cpu_percent_mean_across_cores: float
    cpu_percent_sum_across_cores: float
    number_of_cores: int
    total_memory_gb: float
    total_memory_percent: float
    disk_read_count: int
    disk_write_count: int
    disk_read_kb: float
    disk_write_kb: float
    disk_read_time: int
    disk_write_time: int
    packets_sent: int
    packets_received: int
    network_kb_sent: float
    network_kb_received: float

    battery_percent: Optional[float] = None
    battery_remaining_capacity_mWh: Optional[float] = None
    battery_voltage_mV: Optional[float] = None
    core_percents: List[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemRawResults':
        init_kwargs = {}

        for f in fields(cls):
            is_optional = (
                f.default is not MISSING or
                f.default_factory is not MISSING
            )

            if f.name in data:
                init_kwargs[f.name] = data[f.name]
            elif not is_optional:
                raise ValueError(f"Missing required field: {f.name}")

        # Create object and compute duration
        system_raw_results = SystemRawResults(**init_kwargs)
        system_raw_results.core_percents = get_cores_metrics(data)

        return system_raw_results
