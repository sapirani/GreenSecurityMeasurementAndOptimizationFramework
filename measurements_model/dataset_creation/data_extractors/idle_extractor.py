import pandas as pd

from aggregative_results.DTOs.aggregators_features.energy_model_features.idle_energy_model_features import \
    IdleEnergyModelFeatures
from measurements_model.config import TIME_COLUMN_NAME
from utils.general_consts import BatteryColumns


class IdleExtractor:
    @staticmethod
    def extract(idle_session_path: str) -> IdleEnergyModelFeatures:
        idle_df = pd.read_csv(idle_session_path)
        idle_df[TIME_COLUMN_NAME] = pd.to_datetime(idle_df[TIME_COLUMN_NAME])
        idle_df = idle_df.sort_values(by=TIME_COLUMN_NAME)

        beginning_time = idle_df.iloc[0].at[TIME_COLUMN_NAME]
        end_time = idle_df.iloc[len(idle_df) - 1].at[TIME_COLUMN_NAME]
        duration = (end_time - beginning_time).total_seconds()

        initial_capacity = idle_df.iloc[0].at[BatteryColumns.CAPACITY]
        final_capacity = idle_df.iloc[len(idle_df) - 1].at[BatteryColumns.CAPACITY]
        energy_used = initial_capacity - final_capacity

        return IdleEnergyModelFeatures(
            energy_per_second=energy_used/duration
        )
