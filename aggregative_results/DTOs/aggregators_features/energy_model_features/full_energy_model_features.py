from dataclasses import dataclass
from datetime import datetime

from aggregative_results.DTOs.aggregators_features.energy_model_features.process_energy_model_features import \
    ProcessEnergyModelFeatures
from aggregative_results.DTOs.aggregators_features.energy_model_features.system_energy_model_features import \
    SystemEnergyModelFeatures


@dataclass
class EnergyModelFeatures:
    timestamp: datetime

    system_features: SystemEnergyModelFeatures
    process_features: ProcessEnergyModelFeatures
