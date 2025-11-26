from collections import defaultdict
from typing import List, Dict, Callable, Type, DefaultDict, Optional, TypeAlias

from DTOs.aggregated_results_dtos.abstract_aggregation_results import AbstractAggregationResult
from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from DTOs.session_host_info import SessionHostIdentity
from aggregators.energy_model_aggregators.system_energy_model_aggregator import SystemEnergyModelAggregator
from elastic_reader.aggregators.abstract_aggregator import AbstractAggregator
from DTOs.process_info import ProcessIdentity, ProcessMetadata
from DTOs.aggregated_results_dtos.aggregated_process_results import AggregatedProcessResults
from elastic_reader.aggregators.cpu_integral_aggregator import CPUIntegralAggregator
from aggregators.energy_model_aggregators.process_energy_model_aggregator import ProcessEnergyModelAggregator
from DTOs.raw_results_dtos.iteration_info import IterationMetadata, IterationRawResults
from DTOs.raw_results_dtos.abstract_raw_results import AbstractRawResults
from DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from DTOs.raw_results_dtos.system_processes_raw_results import FullScopeRawResults


PerProcesAggregators: TypeAlias = DefaultDict[ProcessIdentity, List[AbstractAggregator]]
SessionHostProcessAggregators: TypeAlias = DefaultDict[SessionHostIdentity, PerProcesAggregators]
SessionHostSystemAggregators: TypeAlias = DefaultDict[SessionHostIdentity, List[AbstractAggregator]]
PerProcessAggregationResults: TypeAlias = Dict[ProcessIdentity, AggregatedProcessResults]


class AggregationManager:
    """
    This class is accountable of mapping the raw results into the relevant aggregator instances.
    For example, each process will have a separate instance for computing cpu integrals, and this class is responsible
    for choosing the right instance for the incoming sample.
    """
    SYSTEM_AGGREGATOR_TYPES = [CPUIntegralAggregator]
    PROCESS_ONLY_AGGREGATOR_TYPES = [CPUIntegralAggregator]
    PROCESS_SYSTEM_AGGREGATORS_TYPES = [ProcessEnergyModelAggregator, SystemEnergyModelAggregator]
    FULL_SCOPE_AGGREGATORS_TYPES = []

    def __init__(self):
        self.system_aggregators: SessionHostSystemAggregators = defaultdict(self.__get_system_aggregators)

        self.process_only_aggregators: SessionHostProcessAggregators = \
            defaultdict(lambda: defaultdict(self.__get_process_only_aggregators))

        self.process_system_aggregators: SessionHostProcessAggregators = \
            defaultdict(lambda: defaultdict(self.__get_process_system_aggregators))

        self.full_scope_aggregators: SessionHostProcessAggregators = \
            defaultdict(lambda: defaultdict(self.__get_full_scope_aggregators))

    def __get_system_aggregators(self) -> List[AbstractAggregator]:
        return self.__get_initialized_aggregators(self.SYSTEM_AGGREGATOR_TYPES)

    def __get_process_only_aggregators(self) -> List[AbstractAggregator]:
        return self.__get_initialized_aggregators(self.PROCESS_ONLY_AGGREGATOR_TYPES)

    def __get_process_system_aggregators(self) -> List[AbstractAggregator]:
        return self.__get_initialized_aggregators(self.PROCESS_SYSTEM_AGGREGATORS_TYPES)

    def __get_full_scope_aggregators(self) -> List[AbstractAggregator]:
        return self.__get_initialized_aggregators(self.FULL_SCOPE_AGGREGATORS_TYPES)

    @staticmethod
    def __get_initialized_aggregators(aggregator_types: List[Type[AbstractAggregator]]) -> List[AbstractAggregator]:
        return [cls() for cls in aggregator_types]

    def aggregate_iteration_raw_results(self, iteration_raw_results: IterationRawResults) -> IterationAggregatedResults:
        """
        This function receives the full raw metrics measured as part of a full iteration of the scanner (on a specific host).
        It is accountable for calculating all kinds of aggregations, and log these aggregations into a separate index
        in Elastic.
        """

        system_aggregated_results = self.__aggregate_system_metrics(
            iteration_raw_results.system_raw_results,
            iteration_raw_results.metadata
        )

        combined_process_results = self.__combine_process_results(
            self.__aggregate_from_process_metrics_only(iteration_raw_results),
            self.__aggregate_from_process_and_system_metrics(iteration_raw_results),
            self.__aggregate_from_full_scope_metrics(iteration_raw_results)
        )

        return IterationAggregatedResults(
            processes_results=combined_process_results,
            system_aggregated_results=system_aggregated_results,
            iteration_metadata=iteration_raw_results.metadata
        )

    def __aggregate_system_metrics(
            self,
            system_iteration_results: Optional[SystemRawResults],
            iteration_metadata: IterationMetadata
    ) -> List[AbstractAggregationResult]:
        """
        This function receives iteration's system raw metrics and metadata, apply all system metrics aggregations,
        and returns all aggregations results.
        """

        if not system_iteration_results:
            # TODO: REMOVE THIS PRINT WHEN OPTIONAL RESULTS WILL BE FULLY SUPPORTED
            print("Warning! system iteration results are missing")
            return []

        system_aggregation_results = []
        for aggregator in self.system_aggregators[iteration_metadata.session_host_identity]:
            system_aggregation_results.append(self.__process(aggregator, system_iteration_results, iteration_metadata))

        return system_aggregation_results

    def __aggregate_process_metrics_generic(
            self,
            processes_iteration_results: List[ProcessRawResults],
            iteration_metadata: IterationMetadata,
            aggregators_dict: Dict[SessionHostIdentity, Dict[ProcessIdentity, List[AbstractAggregator]]],
            raw_results_combiner: Callable[[ProcessRawResults], ProcessRawResults | ProcessSystemRawResults | FullScopeRawResults]
    ) -> PerProcessAggregationResults:
        """
        This is a generic function that iterates over all processes metrics, 
        apply the relevant aggregators for each process, and returns all aggregations results.
        :param processes_iteration_results: all processes raw metrics within the current iteration
        :param iteration_metadata: general metadata related to the iteration
        :param aggregators_dict: maps between session host to aggregators of each process
        :param raw_results_combiner: a function that receives the raw metrics measured for a process and combine it
        with additional raw metrics to be fed collectively later on to the aggregators. For example, some aggregators
        may require both process raw metrics and system raw metrics.
        :return: a dictionary that maps between a process and its aggregated results
        """

        processes_aggregation_results = {}
        for raw_process_results in processes_iteration_results:
            process_identity = ProcessIdentity.from_raw_results(raw_process_results)
            process_aggregation_results = []
            for aggregator in aggregators_dict[iteration_metadata.session_host_identity][process_identity]:
                process_aggregation_results.append(
                    self.__process(
                        aggregator,
                        raw_results_combiner(raw_process_results),
                        iteration_metadata
                    )
                )

            processes_aggregation_results[process_identity] = AggregatedProcessResults(
                process_metadata=ProcessMetadata(
                    process_of_interest=raw_process_results.process_of_interest,
                    arguments=raw_process_results.arguments
                ),
                aggregation_results=process_aggregation_results
            )

        return processes_aggregation_results

    def __aggregate_from_process_metrics_only(
            self,
            iteration_raw_results: IterationRawResults,
    ) -> PerProcessAggregationResults:
        """
        The most basic process aggregation method.
        This function applies aggregations on each process individually, relying on the process's raw metrics solely
        without relying on any additional context.
        """

        if not iteration_raw_results.processes_raw_results:
            return {}

        return self.__aggregate_process_metrics_generic(
            iteration_raw_results.processes_raw_results,
            iteration_raw_results.metadata,
            self.process_only_aggregators,
            lambda raw_process_results: raw_process_results
        )

    def __aggregate_from_process_and_system_metrics(
            self, iteration_raw_results: IterationRawResults,
    ) -> PerProcessAggregationResults:
        """
        This function applies aggregations on each process individually, 
        while taking the raw system metrics as an additional context
        """

        if not iteration_raw_results.system_raw_results or not iteration_raw_results.processes_raw_results:
            return {}

        return self.__aggregate_process_metrics_generic(
            iteration_raw_results.processes_raw_results,
            iteration_raw_results.metadata,
            self.process_system_aggregators,
            lambda raw_process_results: ProcessSystemRawResults(
                            process_raw_results=raw_process_results,
                            system_raw_results=iteration_raw_results.system_raw_results
                        ),
        )

    def __aggregate_from_full_scope_metrics(
            self, iteration_raw_results: IterationRawResults,
    ) -> PerProcessAggregationResults:
        """
        This function applies aggregations on each process individually, 
        while taking the raw system metrics and other processes' raw metrics as an additional context
        """
        if not iteration_raw_results.system_raw_results or not iteration_raw_results.processes_raw_results:
            return {}

        return self.__aggregate_process_metrics_generic(
            iteration_raw_results.processes_raw_results,
            iteration_raw_results.metadata,
            self.full_scope_aggregators,
            lambda raw_process_results: FullScopeRawResults(
                            desired_process_raw_results=raw_process_results,
                            processes_raw_results=iteration_raw_results.processes_raw_results,
                            system_raw_results=iteration_raw_results.system_raw_results
                        ),
        )

    @staticmethod
    def __process(
            aggregator: AbstractAggregator,
            raw_results: AbstractRawResults,
            iteration_metadata: IterationMetadata
    ) -> AbstractAggregationResult:
        relevant_sample_features = aggregator.extract_features(raw_results, iteration_metadata)
        return aggregator.process_sample(relevant_sample_features)

    @staticmethod
    def __combine_process_results(
            processes_basic_aggregated_results: PerProcessAggregationResults,
            system_process_aggregated_results: PerProcessAggregationResults,
            system_processes_aggregated_results: PerProcessAggregationResults
    ) -> PerProcessAggregationResults:
        all_dicts = (
            processes_basic_aggregated_results,
            system_process_aggregated_results,
            system_processes_aggregated_results
        )

        combined_results = {}
        for process_aggregations_dict in all_dicts:
            for process_identity, aggregated_results in process_aggregations_dict.items():
                if process_identity in combined_results:
                    combined_results[process_identity].merge(aggregated_results)
                else:
                    combined_results[process_identity] = aggregated_results

        return combined_results
