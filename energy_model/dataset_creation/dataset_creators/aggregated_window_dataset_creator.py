import pandas as pd
from overrides import override

from energy_model.configs.columns import ProcessColumns, SystemColumns
from energy_model.dataset_creation.dataset_creation_config import DEFAULT_FILTERING_SINGLE_PROCESS
from energy_model.dataset_creation.dataset_creators.energy_per_second_dataset_creator import \
    EnergyPerSecondDatasetCreator
from energy_model.dataset_creation.raw_telemetry_readers.raw_telemetry_reader import RawTelemetryReader
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator

LIST_OF_WINDOWS = [2, 3, 5]


class AggregatedWindowDatasetCreator(EnergyPerSecondDatasetCreator):
    """
    This class represents the basic reading from elastic for the sake of dataset creation.
    Aggregations on every process telemetry per batch.
    """

    def __init__(self, target_calculator: TargetCalculator, dataset_reader: RawTelemetryReader,
                 batch_time_intervals: list[int] = None, single_process_only: bool = DEFAULT_FILTERING_SINGLE_PROCESS):
        super().__init__(target_calculator=target_calculator, dataset_reader=dataset_reader,
                         batch_time_intervals=batch_time_intervals, single_process_only=single_process_only)

    def get_name(self) -> str:
        return "window_aggregated_dataset_creator"

    @override
    def _add_energy_necessary_columns(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        """
        For each batch:
            - Calculate the total energy consumption per second of that batch, by calculating the battery drain during that batch.
            - Adding the calculated result as new column.
            - aggregate over each process's telemetry.
        """

        df_without_aggregations = super()._add_energy_necessary_columns(df, batch_duration_seconds)
        df = df_without_aggregations.copy()

        df = df.reset_index(drop=False).rename(columns={"index": "_row_id"})

        necessary_aggregations = self._get_necessary_aggregations(df.columns.to_list())
        necessary_aggregations.pop("batch_id", None)
        necessary_aggregations.pop("pid", None)

        rolling_aggs = {
            k: v
            for k, v in necessary_aggregations.items()
            if v != "first"
        }

        metadata_cols = [col for col, agg in necessary_aggregations.items() if col not in rolling_aggs]

        dfs = []  # start with original df

        for w in LIST_OF_WINDOWS:
            rolled = (
                df
                .groupby([SystemColumns.BATCH_ID_COL, ProcessColumns.PROCESS_ID_COL], group_keys=False)
                .rolling(window=w)
                .agg(rolling_aggs)
                .dropna()
                .reset_index()
            )
            rolled = rolled.dropna()


            # -----------------------------------------------------
            # 4. Attach metadata using preserved row index
            # -----------------------------------------------------
            rolled[metadata_cols] = df.loc[
                rolled["_row_id"], metadata_cols
            ].values

            # -----------------------------------------------------
            # 6. Optional: drop helper column
            # -----------------------------------------------------
            rolled = rolled.drop(columns="_row_id")
            dfs.append(rolled)

        # ---------------------------------------------------------
        # 7. Prepare original dataframe for concat
        # ---------------------------------------------------------
        df_orig = df.drop(columns="_row_id").copy()

        # ---------------------------------------------------------
        # 8. Combine
        # ---------------------------------------------------------
        final_df = pd.concat([df_orig, *dfs], ignore_index=True)

        return final_df
