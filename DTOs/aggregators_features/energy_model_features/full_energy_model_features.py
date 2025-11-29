from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from DTOs.aggregators_features.energy_model_features.hardware_energy_model_features import HardwareEnergyModelFeatures
from DTOs.aggregators_features.energy_model_features.process_energy_model_features import ProcessEnergyModelFeatures
from DTOs.aggregators_features.energy_model_features.system_energy_model_features import SystemEnergyModelFeatures


@dataclass
class CompleteEnergyModelFeatures:
    system_features: SystemEnergyModelFeatures
    process_features: ProcessEnergyModelFeatures
    duration: timedelta = field(init=False)

    def __post_init__(self):
        system_duration = self.system_features.duration
        process_duration = self.process_features.duration
        if not system_duration == process_duration:
            raise Exception("Invalid initialization of duration in CompleteEnergyModelFeatures!")
        object.__setattr__(self, 'duration', system_duration)


@dataclass
class ExtendedEnergyModelFeatures(CompleteEnergyModelFeatures):
    session_id: str
    hostname: str
    pid: int
    timestamp: datetime
    hardware_features: HardwareEnergyModelFeatures
    battery_remaining_capacity_mWh: Optional[float] = None
