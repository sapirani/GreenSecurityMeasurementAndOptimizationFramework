import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict

from DTOs.scanner_counters.metric_results import MetricResult


@dataclass
class SystemCPUResults(MetricResult):
    cpu_percent_mean_across_cores: float
    cpu_percent_sum_across_cores: float
    number_of_cores: int
    per_core_percent: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        """Flatten the core percentages"""
        flat = dataclasses.asdict(self)
        # Flatten per_core_percent: core0_percent, core1_percent, ...
        per_core = {
            # CHANGE get_core_name in general consts if you change this variable name
            f"core_{idx}_percent": val for idx, val in enumerate(flat.pop("per_core_percent"))
        }
        flat.update(per_core)
        return flat
