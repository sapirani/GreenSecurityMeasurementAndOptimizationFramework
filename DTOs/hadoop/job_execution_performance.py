from dataclasses import dataclass


@dataclass(frozen=True)
class JobExecutionPerformance:
    running_time_sec: float
    energy_use_mwh: float

    def __str__(self):
        return f"running_time_sec={self.running_time_sec}, energy_use_mwh={self.energy_use_mwh}"
