from dataclasses import asdict

import pandas as pd

from DTOs.aggregators_features.energy_model_features.full_energy_model_features import EnergyModelFeatures


class EnergyModelConvertor:
    @staticmethod
    def convert_features_to_pandas(sample: EnergyModelFeatures, **kwargs) -> pd.DataFrame:
        sample_dict = {"duration": sample.duration,
                       **asdict(sample.process_features),
                       **asdict(sample.system_features),
                       **kwargs}
        if any(value is None for key, value in sample_dict.items()):
            raise ValueError("Invalid sample, there is at least one empty field.")

        return pd.DataFrame([sample_dict])
