import pandas as pd
from overrides import override

from energy_model.dataset_creation.dataset_creators.basic_dataset_creator import BasicDatasetCreator
from energy_model.configs.columns import ProcessColumns, SystemColumns

class AggregatedDatasetCreator(BasicDatasetCreator):
    @override
    def _add_energy_necessary_columns(self, df: pd.DataFrame, batch_duration_seconds: int) -> pd.DataFrame:
        df_without_aggregations = super()._add_energy_necessary_columns(df, batch_duration_seconds)
        necessary_aggregations = self._get_necessary_aggregations(df_without_aggregations.columns.to_list())
        df_grouped = (
            df_without_aggregations.groupby([SystemColumns.BATCH_ID_COL, ProcessColumns.PROCESS_ID_COL], as_index=False)
            .agg(necessary_aggregations)
        )
        return df_grouped
