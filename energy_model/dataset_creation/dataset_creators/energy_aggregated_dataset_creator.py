from typing import Callable, Union

from overrides import override

from energy_model.configs.columns import SystemColumns
from energy_model.dataset_creation.dataset_creation_config import AggregationName
from energy_model.dataset_creation.dataset_creators.aggregated_dataset_creator import AggregatedDatasetCreator
from energy_model.dataset_creation.dataset_readers.dataset_reader import DatasetReader
from energy_model.dataset_creation.target_calculators.battery_drain_target_calculator import \
    BatteryDrainTargetCalculator


class EnergyAggregatedDatasetCreator(AggregatedDatasetCreator):
    """
    This class represents the basic reading from elastic.
    Reading only process of interest logs.
    Aggregations on every process telemetry per batch.
    The system energy per batch per process is calculated by the battery drain between the first and last samples of each process in the batch.
    """

    def __init__(self, dataset_reader: DatasetReader, batch_time_intervals: list[int] = None):
        super().__init__(target_calculator=BatteryDrainTargetCalculator(), dataset_reader=dataset_reader, batch_time_intervals=batch_time_intervals)

    @override
    def _get_necessary_aggregations(self, available_columns: list[str]) -> dict[str, Union[list[str], str, Callable]]:
        aggregations_dict = super()._get_necessary_aggregations(available_columns)
        aggregations_dict[SystemColumns.BATTERY_CAPACITY_MWH_SYSTEM_COL] = [AggregationName.FIRST_SAMPLE,
                                                                            AggregationName.LAST_SAMPLE]
        return aggregations_dict
