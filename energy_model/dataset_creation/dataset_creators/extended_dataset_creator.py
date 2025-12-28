from abc import abstractmethod

import pandas as pd

from energy_model.configs.columns import ProcessColumns, SystemColumns
from energy_model.dataset_creation.dataset_creators.abstract_dataset_creator import AbstractDatasetCreator
from energy_model.dataset_creation.dataset_creation_config import TIMESTAMP_COLUMN_NAME, COLUMNS_TO_AVOID_SUM, \
    AggregationName


class ExtendedDatasetCreator(AbstractDatasetCreator):
    def _extend_df_with_target(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        """
        This method calculates the target column and adds it to the dataframe.
        First, split every session in the dataframe into batches.
        For each batch:
            - Calculate the total energy consumption per second of that batch, by calculating the battery drain during that batch.
            - For each process in the batch:
                * Turn all process samples into a single sample by summing all telemetry columns and duration column.
            - Calculate energy consumption of each record in that batch (record per process), depending on the dataset creator class.

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

        # Step 2: Calculate sum across all column to receive single sample per batch and process id
        columns_aggregations = {
            col: AggregationName.SUM for col in df.columns if col not in COLUMNS_TO_AVOID_SUM
        }
        df = df.groupby([SystemColumns.BATCH_ID_COL, ProcessColumns.PROCESS_ID_COL], as_index=False).agg(
            columns_aggregations)

        results = []
        # Step 3: Handle batches separately depending on process_id count
        for batch_id, batch_df in df.groupby(SystemColumns.BATCH_ID_COL, group_keys=False):
            batch_df_with_target = self._add_target_to_batch(batch_df)
            results.append(batch_df_with_target)

        df = pd.concat(results, ignore_index=True)
        return df

    @abstractmethod
    def _add_target_to_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extend the given DataFrame with energy usage targets.
        """
        pass
