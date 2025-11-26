from dataclasses import asdict

import pandas as pd

from DTOs.aggregators_features.energy_model_features.full_energy_model_features import CompleteEnergyModelFeatures
from DTOs.aggregators_features.energy_model_features.system_energy_model_features import SystemEnergyModelFeatures

class EnergyModelConvertor:
    @staticmethod
    def convert_complete_features_to_pandas(sample: CompleteEnergyModelFeatures, **kwargs) -> pd.DataFrame:
        sample_dict = {"duration": sample.duration,
                       **asdict(sample.process_features),
                       **asdict(sample.system_features),
                       **kwargs}

        return EnergyModelConvertor.convert_dict_to_df(sample_dict)

    @staticmethod
    def convert_system_features_to_pandas(sample: SystemEnergyModelFeatures, **kwargs) -> pd.DataFrame:
        sample_dict = {
            **asdict(sample.system_features),
            **kwargs
        }
        return EnergyModelConvertor.convert_dict_to_df(sample_dict)

    @staticmethod
    def convert_dict_to_df(data: dict) -> pd.DataFrame:
        if any(value is None for key, value in data.items()):
            raise ValueError("Invalid sample, there is at least one empty field.")

        return pd.DataFrame([data])
