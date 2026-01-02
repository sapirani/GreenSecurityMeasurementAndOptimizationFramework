from datetime import datetime
import pandas as pd

from energy_model.configs.columns import SystemColumns
from DTOs.aggregators_features.energy_model_features.full_energy_model_features import ExtendedEnergyModelFeatures
from DTOs.process_info import ProcessIdentity
from DTOs.aggregators_features.empty_features import EmptyFeatures
from DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from DTOs.session_host_info import SessionHostIdentity
from energy_model.dataset_creation.dataset_creators.dataset_creator import DatasetCreator


class BasicDatasetCreator(DatasetCreator):
    """
    This class represents the basic reading from elastic.
    Reading only process of interest logs.
    No special aggregations on the data.
    """
    def _add_energy_necessary_columns(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        """
            This method calculates the target column and adds it to the dataframe.
            First, split every session in the dataframe into batches.
            For each batch:
                - Calculate the total energy consumption per second of that batch, by calculating the battery drain during that batch.
                - Calculate energy consumption of each record in that batch, depending on the dataset creator class.

            Input:
            df: pandas dataframe containing all samples.
            batch_duration_seconds: the duration (in seconds) of each batch.
        """
        # Step 1: Calculate system energy consumption rate (mWh/sec) for each batch
        energy_per_batch = (
            df.groupby(SystemColumns.BATCH_ID_COL)[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL]
            .agg(lambda s: (s.iloc[0] - s.iloc[-1]) / batch_duration_seconds)
            .rename(SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL)
        )

        # Merge batch-level system energy rates back into the main DataFrame
        df = df.merge(energy_per_batch, on=SystemColumns.BATCH_ID_COL, how="left")
        return df
