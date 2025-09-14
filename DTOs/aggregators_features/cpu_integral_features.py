from dataclasses import dataclass
from datetime import datetime


@dataclass
class CPUIntegralFeatures:
    date: datetime
    cpu_percent_sum_across_cores: float
