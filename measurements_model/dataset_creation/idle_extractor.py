from aggregative_results.DTOs.aggregators_features.energy_model_features.idle_energy_model_features import \
    IdleEnergyModelFeatures


class IdleExtractor:
    def extract(self, idle_directory_path: str) -> IdleEnergyModelFeatures:
        return IdleEnergyModelFeatures()