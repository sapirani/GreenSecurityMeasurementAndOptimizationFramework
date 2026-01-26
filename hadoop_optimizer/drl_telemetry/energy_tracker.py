import threading
from collections import defaultdict
from typing import Optional, cast, Dict
from DTOs.aggregated_results_dtos.empty_aggregation_results import EmptyAggregationResults
from DTOs.aggregated_results_dtos.energy_model_result import EnergyModelResult
from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.hadoop.training_metadata import TrainingMetadata
from DTOs.raw_results_dtos.iteration_info import IterationRawResults
from DTOs.aggregation_types import AggregationType
from elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer
from hadoop_optimizer.erros import NoEnergyMeasurements


class EnergyTracker(AbstractElasticConsumer):
    def __init__(self):
        self.__hostname_to_energy = defaultdict(float)
        self.measurement_session_id = None

        self._cond = threading.Condition()
        self._finished_hosts = set()
        self.session_done = False
        self.current_training_metadata: Optional[TrainingMetadata] = None

    def consume(
            self,
            iteration_raw_results: IterationRawResults,
            iteration_aggregation_results: Optional[IterationAggregatedResults]
    ):
        """
        This function is being called by a different thread
        """
        # TODO: CREATE AN ABSTRACT CLASS FOR THE VALIDATION?
        if not iteration_aggregation_results:
            raise ValueError("Aggregations cannot be None in state computations")

        if iteration_raw_results.metadata != iteration_aggregation_results.iteration_metadata:
            raise ValueError("Received inconsistent metadata between raw results and aggregations")

        if iteration_raw_results.metadata.session_host_identity.session_id != self.measurement_session_id:
            raise ValueError(
                f"Received measurement from unexpected session id, "
                f"expected: {self.measurement_session_id}, "
                f"received: {iteration_raw_results.metadata.session_host_identity.session_id}"
            )

        received_training_metadata = TrainingMetadata.from_dict(iteration_aggregation_results.system_extras)
        if self.current_training_metadata != received_training_metadata:
            raise ValueError(
                f"Received unexpected training metadata, "
                f"expected: {self.current_training_metadata}, "
                f"received: {received_training_metadata}"
            )

        energy_aggregator = iteration_aggregation_results.system_results[AggregationType.SystemEnergyModelAggregator]
        # used to ignore first iteration as the aggregations need more than one sample
        if isinstance(energy_aggregator, EmptyAggregationResults):
            return

        energy_model_result = cast(
            EnergyModelResult,
            energy_aggregator
        )

        hostname = iteration_raw_results.metadata.session_host_identity.hostname

        with self._cond:
            self._cond.wait_for(lambda: not self.session_done)

            self.__hostname_to_energy[hostname] += energy_model_result.energy_mwh

            if iteration_raw_results.is_last_iteration:
                self._finished_hosts.add(iteration_raw_results.metadata.session_host_identity.hostname)

                # session ends when all seen hosts finished
                if self._finished_hosts >= self.__hostname_to_energy.keys():
                    self.session_done = True
                    self._cond.notify_all()

    def get_energy_consumption(self) -> Dict[str, float]:
        """
        This function is supposed to be called right after the job has terminated (or even while the job is running).
        It is important to do so to avoid blocking the consume function
        """
        with self._cond:
            self._cond.wait_for(lambda: self.session_done)

            if not self.__hostname_to_energy:
                raise NoEnergyMeasurements()

            self.measurement_session_id = None
            self.session_done = False
            self._cond.notify_all()

            return dict(self.__hostname_to_energy)

    def reset_tracker(self, measurement_session_id: str, current_training_metadata: TrainingMetadata) -> None:
        """
        Scanner is assumed to start running after calling this function
        """
        with self._cond:
            self.__hostname_to_energy.clear()
            self._finished_hosts.clear()
            self.measurement_session_id = measurement_session_id
            self.current_training_metadata = current_training_metadata
            self.session_done = False
            self._cond.notify_all()
