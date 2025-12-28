import pandas as pd

from energy_model.configs.columns import ProcessColumns, SystemColumns
from energy_model.dataset_creation.dataset_creation_config import TIMESTAMP_COLUMN_NAME, COLUMNS_TO_AVOID_SUM, \
    AggregationName
from energy_model.dataset_creation.dataset_creators.abstract_dataset_creator import AbstractDatasetCreator


class EnergyExtendedDatasetCreator(AbstractDatasetCreator):
    def _extend_df_with_target(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        """
        This method calculates the target column and adds it to the dataframe.
        First, split every session in the dataframe into batches.
        For each batch and for each process in that batch:
            * Turn all process samples into a single sample by summing all telemetry columns and duration column.
            * Calculate the energy consumption of the new sample by calculating the battery drain
              between the first and last samples of the process in that batch.

        Input:
        df: pandas dataframe containing all samples.
        """
        df = df.sort_values([SystemColumns.BATCH_ID_COL, ProcessColumns.PROCESS_ID_COL, TIMESTAMP_COLUMN_NAME])

        # Step 1: Calculate sum across all column to receive single sample per batch and process id
        columns_aggregations = {
            col: AggregationName.SUM for col in df.columns if col not in COLUMNS_TO_AVOID_SUM
        }
        columns_aggregations[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL] = [AggregationName.FIRST_SAMPLE,
                                                                               AggregationName.LAST_SAMPLE]
        df = df.groupby([SystemColumns.BATCH_ID_COL, ProcessColumns.PROCESS_ID_COL], as_index=False).agg(
            columns_aggregations)

        # Step 2: Calculate energy usage of each process in each batch based on difference between first and last samples
        df[SystemColumns.ENERGY_USAGE_SYSTEM_COL] = df[(
            SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL, AggregationName.FIRST_SAMPLE)] \
                                                    - df[(
            SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL, AggregationName.LAST_SAMPLE)]

        return df

    def _add_target_to_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("No need to add target column per batch separately")
