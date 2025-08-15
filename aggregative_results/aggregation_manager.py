from collections import defaultdict
from dataclasses import asdict
from logging import getLogger
from typing import List, Dict, Callable, Type

from aggregative_results.aggregators.abstract_aggregator import AbstractAggregator
from aggregative_results.dtos import ProcessIdentity, ProcessMetadata
from aggregative_results.dtos.aggregated_results_dtos.aggregated_process_results import AggregatedProcessResults
from aggregative_results.dtos.aggregated_results_dtos import AggregationResult
from aggregative_results.aggregators.cpu_integral_aggregator import CPUIntegralAggregator
from aggregative_results.aggregators.process_system_usage_fraction_aggregator import \
    ProcessSystemUsageFractionAggregator
from aggregative_results.dtos.raw_results_dtos import ProcessRawResults, IterationMetadata, IterationRawResults, \
    SystemRawResults
from aggregative_results.dtos.raw_results_dtos.abstract_raw_results import AbstractRawResults
from aggregative_results.dtos.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from aggregative_results.dtos.raw_results_dtos.system_processes_raw_results import FullScopeRawResults
from utils.general_consts import LoggerName

logger = getLogger(LoggerName.METRICS_AGGREGATIONS)


class AggregationManager:
    """
    This class is accountable of mapping the raw results into the relevant aggregator instances.
    For example, each process will have a separate instance for computing cpu integrals, and this class is responsible
    for choosing the right instance for the incoming sample.
    """
    SYSTEM_AGGREGATOR_TYPES = [CPUIntegralAggregator]
    PROCESS_ONLY_AGGREGATOR_TYPES = [CPUIntegralAggregator]
    PROCESS_SYSTEM_AGGREGATORS_TYPES = []   # TODO: ADD ENERGY ESTIMATIONS
    FULL_SCOPE_AGGREGATORS_TYPES = [ProcessSystemUsageFractionAggregator]

    def __init__(self):
        self.system_aggregators = self._get_system_aggregators()

        self.process_only_aggregators = defaultdict(self._get_process_only_aggregators)
        self.process_system_aggregators = defaultdict(self._get_process_system_aggregators)
        self.full_scope_aggregators = defaultdict(self._get_full_scope_aggregators)

    def _get_system_aggregators(self) -> List[AbstractAggregator]:
        return self._get_initialized_aggregators(self.SYSTEM_AGGREGATOR_TYPES)

    def _get_process_only_aggregators(self) -> List[AbstractAggregator]:
        return self._get_initialized_aggregators(self.PROCESS_ONLY_AGGREGATOR_TYPES)

    def _get_process_system_aggregators(self) -> List[AbstractAggregator]:
        return self._get_initialized_aggregators(self.PROCESS_SYSTEM_AGGREGATORS_TYPES)

    def _get_full_scope_aggregators(self) -> List[AbstractAggregator]:
        return self._get_initialized_aggregators(self.FULL_SCOPE_AGGREGATORS_TYPES)

    @staticmethod
    def _get_initialized_aggregators(aggregator_types: List[Type[AbstractAggregator]]) -> List[AbstractAggregator]:
        return [cls() for cls in aggregator_types]

    def aggregate_iteration_raw_results(self, iteration_raw_results: IterationRawResults):
        """
        This function receives the full raw metrics measured as part of a full iteration of the scanner (on a specific host).
        It is accountable for calculating all kinds of aggregations, and log these aggregations into a separate index
        in Elastic.
        """

        system_aggregated_results = self._aggregate_system_metrics(
            iteration_raw_results.system_raw_results,
            iteration_raw_results.metadata
        )

        combined_process_results = self._combine_process_results(
            self._aggregate_from_process_metrics_only(iteration_raw_results),
            self._aggregate_from_process_and_system_metrics(iteration_raw_results),
            self._aggregate_from_full_scope_metrics(iteration_raw_results)
        )

        self._log_aggregated_iteration_results(
            combined_process_results,
            system_aggregated_results,
            iteration_raw_results.metadata
        )

    @staticmethod
    def _log_aggregated_iteration_results(
            combined_process_results: Dict[ProcessIdentity, AggregatedProcessResults],
            system_aggregated_results: List[AggregationResult],
            iteration_raw_results: IterationMetadata
    ):
        for process_identity, process_results in combined_process_results.items():
            logger.info(
                "Process Aggregation Results",
                extra=
                {
                    **asdict(iteration_raw_results),
                    **asdict(process_identity),
                    **asdict(process_results.process_metadata),
                    **{
                        result_name: result_val
                        for aggregation_result in process_results.aggregation_results for result_name, result_val in
                        asdict(aggregation_result).items()
                    }
                }
            )
        logger.info(
            "System Aggregation Results",
            extra=
            {
                **asdict(iteration_raw_results),
                **{key: value for aggregation_result in system_aggregated_results for key, value in
                   asdict(aggregation_result).items()}
            }
        )

    def _aggregate_system_metrics(
            self,
            system_iteration_results: SystemRawResults,
            iteration_metadata: IterationMetadata
    ) -> List[AggregationResult]:
        """
        This function receives iteration's system raw metrics and metadata, apply all system metrics aggregations,
        and returns all aggregations results.
        """

        system_aggregation_results = []
        for aggregator in self.system_aggregators:
            system_aggregation_results.append(self._process(aggregator, system_iteration_results, iteration_metadata))

        return system_aggregation_results

    def _aggregate_process_metrics_generic(
            self,
            processes_iteration_results: List[ProcessRawResults],
            iteration_metadata: IterationMetadata,
            aggregators_dict: Dict[ProcessIdentity, List[AbstractAggregator]],
            raw_results_combiner: Callable[[ProcessRawResults], ProcessRawResults | ProcessSystemRawResults | FullScopeRawResults]
    ) -> Dict[ProcessIdentity, AggregatedProcessResults]:
        """
        This is a generic function that iterates over all processes metrics, 
        apply the relevant aggregators for each process, and returns all aggregations results.
        :param processes_iteration_results: all processes raw metrics within the current iteration
        :param iteration_metadata: general metadata related to the iteration
        :param aggregators_dict: maps between process to all its aggregators
        :param raw_results_combiner: a function that receives the raw metrics measured for a process and combine it
        with additional raw metrics to be fed collectively later on to the aggregators. For example, some aggregators
        may require both process raw metrics and system raw metrics.
        :return: a dictionary that maps between a process and its aggregated results
        """

        processes_aggregation_results = {}
        for raw_process_results in processes_iteration_results:
            process_identity = ProcessIdentity.from_raw_results(raw_process_results)
            process_aggregation_results = []
            for aggregator in aggregators_dict[process_identity]:
                process_aggregation_results.append(
                    self._process(
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

    def _aggregate_from_process_metrics_only(
            self,
            iteration_raw_results: IterationRawResults,
    ) -> Dict[ProcessIdentity, AggregatedProcessResults]:
        """
        The most basic process aggregation method.
        This function applies aggregations on each process individually, relying on the process's raw metrics solely
        without relying on any additional context.
        """

        return self._aggregate_process_metrics_generic(
            iteration_raw_results.processes_raw_results,
            iteration_raw_results.metadata,
            self.process_only_aggregators,
            lambda raw_process_results: raw_process_results
        )

    def _aggregate_from_process_and_system_metrics(
            self, iteration_raw_results: IterationRawResults,
    ) -> Dict[ProcessIdentity, AggregatedProcessResults]:
        """
        This function applies aggregations on each process individually, 
        while taking the raw system metrics as an additional context
        """

        return self._aggregate_process_metrics_generic(
            iteration_raw_results.processes_raw_results,
            iteration_raw_results.metadata,
            self.process_system_aggregators,
            lambda raw_process_results: ProcessSystemRawResults(
                            processes_raw_results=raw_process_results,
                            system_raw_results=iteration_raw_results.system_raw_results
                        ),
        )

    def _aggregate_from_full_scope_metrics(
            self, iteration_raw_results: IterationRawResults,
    ) -> Dict[ProcessIdentity, AggregatedProcessResults]:
        """
        This function applies aggregations on each process individually, 
        while taking the raw system metrics and other processes' raw metrics as an additional context
        """

        return self._aggregate_process_metrics_generic(
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
    def _process(
            aggregator: AbstractAggregator,
            raw_results: AbstractRawResults,
            iteration_metadata: IterationMetadata
    ) -> AggregationResult:
        relevant_sample_features = aggregator.extract_features(raw_results, iteration_metadata)
        return aggregator.process_sample(relevant_sample_features)

    @staticmethod
    def _combine_process_results(
            processes_basic_aggregated_results: Dict[ProcessIdentity, AggregatedProcessResults],
            system_process_aggregated_results: Dict[ProcessIdentity, AggregatedProcessResults],
            system_processes_aggregated_results: Dict[ProcessIdentity, AggregatedProcessResults]
    ) -> Dict[ProcessIdentity, AggregatedProcessResults]:
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
