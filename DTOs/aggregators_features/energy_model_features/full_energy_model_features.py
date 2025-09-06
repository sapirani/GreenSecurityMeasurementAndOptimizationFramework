from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from DTOs.aggregators_features.energy_model_features.hardware_energy_model_features import HardwareEnergyModelFeatures
from DTOs.aggregators_features.energy_model_features.process_energy_model_features import ProcessEnergyModelFeatures
from DTOs.aggregators_features.energy_model_features.system_energy_model_features import SystemEnergyModelFeatures


@dataclass
class EnergyModelFeatures:
    duration: float

    system_features: SystemEnergyModelFeatures
    process_features: ProcessEnergyModelFeatures


@dataclass
class ExtendedEnergyModelFeatures(EnergyModelFeatures):
    hardware_features: HardwareEnergyModelFeatures

    timestamp: datetime
    battery_remaining_capacity_mWh: Optional[float] = None