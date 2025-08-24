import os

import pandas as pd

from aggregative_results.DTOs.aggregators_features.energy_model_features.idle_energy_model_features import \
    IdleEnergyModelFeatures
from measurements_model.config import BATTERY_STATUS_CSV
from measurements_model.dataset_creation.data_extractors.summary_extractors.abstract_summary_extractor import \
    AbstractSummaryExtractor


class IdleExtractor:
    def extract(self, idle_summary_dir_path: str) -> IdleEnergyModelFeatures:
        idle_battery_file_path = os.path.join(idle_summary_dir_path, BATTERY_STATUS_CSV)
        idle_battery_df = pd.read_csv(idle_battery_file_path)

        initial_capacity = idle_battery_df.iloc[0].at["battery_remaining_capacity_mWh"]
        final_capacity = idle_battery_df.iloc[len(idle_battery_df) - 1].at["battery_remaining_capacity_mWh"]
        energy_used = initial_capacity - final_capacity

        duration = idle_battery_df.iloc[len(idle_battery_df) - 1].at["seconds_from_start"]

        return IdleEnergyModelFeatures(
            energy_per_second=energy_used/duration
        )
