import pandas as pd

from energy_model.configs.columns import SystemColumns
from energy_model.dataset_creation.dataset_creators.basic_dataset_creator import BasicDatasetCreator


class SystemBasicDatasetCreator(BasicDatasetCreator):
    def _add_target_to_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        batch_df[SystemColumns.ENERGY_USAGE_SYSTEM_COL] = (batch_df[SystemColumns.DURATION_COL] *
                                                           batch_df[
                                                               SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL])

        return batch_df