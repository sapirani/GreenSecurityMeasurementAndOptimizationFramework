import pandas as pd

from energy_model.configs.columns import SystemColumns
from energy_model.dataset_creation.dataset_creators.dataset_creator import DatasetCreator


class BasicDatasetCreator(DatasetCreator):
    """
    This class represents the basic reading from elastic.
    Reading only process of interest logs.
    No special aggregations on the data.
    """

    def _add_energy_necessary_columns(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        """
        For each batch:
            - Calculate the total energy consumption per second of that batch, by calculating the battery drain during that batch.
            - Adding the calculated result as new column.
        """

        # Calculate system energy consumption rate (mWh/sec) for each batch
        energy_per_batch = (
            df.groupby(SystemColumns.BATCH_ID_COL)[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL]
            .agg(lambda s: (s.iloc[0] - s.iloc[-1]) / batch_duration_seconds)
            .rename(SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL)
        )

        # Merge batch-level system energy rates back into the main DataFrame
        df = df.merge(energy_per_batch, on=SystemColumns.BATCH_ID_COL, how="left")
        return df
