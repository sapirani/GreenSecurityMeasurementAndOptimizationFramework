from collections import defaultdict
from threading import Lock
from typing import Optional, cast, Dict

from DTOs.aggregated_results_dtos.empty_aggregation_results import EmptyAggregationResults
from DTOs.aggregated_results_dtos.energy_model_result import EnergyModelResult
from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.raw_results_dtos.iteration_info import IterationRawResults
from DTOs.aggregation_types import AggregationType
from elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer
from hadoop_optimizer.erros import NoEnergyMeasurements


class EnergyTracker(AbstractElasticConsumer):
    def __init__(self):
        self.__hostname_to_energy = defaultdict(float)
        self.lock = Lock()

    def consume(self, iteration_raw_results: IterationRawResults,
                iteration_aggregation_results: Optional[IterationAggregatedResults]):
        # TODO: CREATE AN ABSTRACT CLASS FOR THE VALIDATION?
        print("Consuming telemetry (energy tracker)")

        if not iteration_aggregation_results:
            raise ValueError("Aggregations cannot be None in state computations")

        if iteration_raw_results.metadata != iteration_aggregation_results.iteration_metadata:
            raise ValueError("Received inconsistent metadata between raw results and aggregations")

        energy_aggregator = iteration_aggregation_results.system_results[AggregationType.SystemEnergyModelAggregator]
        # used to ignore first iteration as the aggregations need more than one sample
        if isinstance(energy_aggregator, EmptyAggregationResults):
            return

        energy_model_result = cast(
            EnergyModelResult,
            energy_aggregator
        )

        hostname = iteration_raw_results.metadata.session_host_identity.hostname

        with self.lock:
            self.__hostname_to_energy[hostname] += energy_model_result.energy_mwh

    def get_energy_consumption(self) -> Dict[str, float]:
        with self.lock:
            if not self.__hostname_to_energy:
                raise NoEnergyMeasurements()

            return dict(self.__hostname_to_energy)

    def reset_tracker(self) -> None:
        with self.lock:
            self.__hostname_to_energy = defaultdict(float)

