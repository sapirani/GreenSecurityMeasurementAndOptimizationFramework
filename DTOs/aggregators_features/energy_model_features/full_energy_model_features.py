from dataclasses import dataclass

from DTOs.aggregators_features.energy_model_features.process_energy_model_features import ProcessEnergyModelFeatures
from DTOs.aggregators_features.energy_model_features.system_energy_model_features import SystemEnergyModelFeatures


@dataclass
class EnergyModelFeatures:
    duration: float

    system_features: SystemEnergyModelFeatures
    process_features: ProcessEnergyModelFeatures
