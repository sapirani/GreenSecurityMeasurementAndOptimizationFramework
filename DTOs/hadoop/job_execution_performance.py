from pydantic import BaseModel


class JobExecutionPerformance(BaseModel):
    running_time_sec: float
    energy_use_mwh: float

    model_config = {
        "frozen": True  # ensures that this class is immutable
    }

    def __str__(self):
        return f"running_time_sec={self.running_time_sec}, energy_use_mwh={self.energy_use_mwh}"
