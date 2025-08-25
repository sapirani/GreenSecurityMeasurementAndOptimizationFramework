import pandas as pd

from aggregative_results.DTOs.aggregators_features.energy_model_features.idle_energy_model_features import \
    IdleEnergyModelFeatures
from utils.general_consts import BatteryColumns


class IdleExtractor:
    def extract(self, idle_session_path: str) -> IdleEnergyModelFeatures:
        idle_df = pd.read_csv(idle_session_path)

        initial_capacity = idle_df.iloc[0].at[BatteryColumns.CAPACITY]
        final_capacity = idle_df.iloc[len(idle_df) - 1].at[BatteryColumns.CAPACITY]
        energy_used = initial_capacity - final_capacity

        duration = idle_df.iloc[len(idle_df) - 1].at[BatteryColumns.TIME]

        return IdleEnergyModelFeatures(
            energy_per_second=energy_used/duration
        )
