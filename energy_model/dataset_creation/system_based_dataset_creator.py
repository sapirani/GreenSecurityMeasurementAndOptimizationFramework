import pandas as pd

from energy_model.dataset_creation.dataset_creator import DatasetCreator
from energy_model.configs.columns import SystemColumns


# todo: change to consumer interface
class SystemBasedDatasetCreator(DatasetCreator):
    def _add_target_to_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        batch_df[SystemColumns.ENERGY_USAGE_SYSTEM_COL] = (batch_df[SystemColumns.DURATION_COL] *
                                                           batch_df[
                                                               SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL])

        return batch_df
