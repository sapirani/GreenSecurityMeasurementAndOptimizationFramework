from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class PreviousSampleFeatures:
    cpu_usage_process: float
    cpu_usage_system: float

    memory_mb_usage_process: float
    memory_gb_usage_system: float

    timestamp: datetime
    battery_remaining_capacity_mWh: Optional[float] = None