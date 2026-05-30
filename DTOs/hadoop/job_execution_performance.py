from typing import Optional

from pydantic import BaseModel, model_validator


class JobExecutionPerformance(BaseModel):
    running_time_sec: float
    energy_use_mwh: float
    simulated: bool = False
    std_running_time_sec: Optional[float] = None
    std_energy_mwh: Optional[float] = None
    number_of_simulation_participants: Optional[int] = None

    model_config = {
        "frozen": True  # ensures that this class is immutable
    }

    def __str__(self):
        return f"running_time_sec={self.running_time_sec}, energy_use_mwh={self.energy_use_mwh}"

    @model_validator(mode="after")
    def validate_simulation_fields(self):
        simulation_fields = [
            self.std_running_time_sec,
            self.std_energy_mwh,
            self.number_of_simulation_participants,
        ]

        if self.simulated:
            # must ALL be present (not None)
            if any(v is None for v in simulation_fields):
                raise ValueError(
                    "When simulated=True, all simulation fields must be provided"
                )

        else:
            # must ALL be None
            if any(v is not None for v in simulation_fields):
                raise ValueError(
                    "When simulated=False, simulation fields must be None"
                )

        return self
