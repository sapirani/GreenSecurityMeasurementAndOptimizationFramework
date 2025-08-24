from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from aggregative_results.DTOs.aggregators_features.energy_model_features.hardware_energy_model_features import \
    HardwareEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.idle_energy_model_features import \
    IdleEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.process_energy_model_features import \
    ProcessEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures


@dataclass
class EnergyModelFeatures:
    timestamp: datetime

    system_features: SystemEnergyModelFeatures
    process_features: ProcessEnergyModelFeatures

    hardware_features: Optional[HardwareEnergyModelFeatures] = None
    idle_features: Optional[IdleEnergyModelFeatures] = None