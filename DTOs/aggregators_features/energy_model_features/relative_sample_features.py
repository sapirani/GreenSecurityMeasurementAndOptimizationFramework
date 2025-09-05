from dataclasses import dataclass
from datetime import datetime


@dataclass
class RelativeSampleFeatures:
    cpu_usage_process: float
    cpu_usage_system: float

    memory_usage_process: float
    memory_usage_system: float

    timestamp: datetime