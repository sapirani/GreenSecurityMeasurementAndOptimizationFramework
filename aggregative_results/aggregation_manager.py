from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from logging import StreamHandler
from typing import List, Dict, Tuple

from aggregative_results.aggregators.abstract_aggregator import AggregationResult, AbstractAggregator
from aggregative_results.aggregators.cpu_integral_aggregator import CPUIntegralAggregator
from aggregative_results.aggregators.process_system_usage_fraction_aggregator import \
    ProcessSystemUsageFractionAggregator
from aggregative_results.raw_results_dtos import IterationRawResults, SystemRawResults, Metadata, ProcessRawResults
from aggregative_results.raw_results_dtos.abstract_raw_results import AbstractRawResults
from aggregative_results.raw_results_dtos.system_process_raw_results import SystemProcessRawResults
from aggregative_results.raw_results_dtos.system_processes_raw_results import SystemProcessesRawResults
from application_logging import get_measurement_logger, get_elastic_logging_handler
from application_logging.formatters.pretty_extra_formatter import PrettyExtrasFormatter
from utils.general_consts import LoggerName, IndexName

# TODO: REMOVE FROM HERE
ES_USER = "elastic"
ES_PASS = "SVR4mUZl"
ES_URL = "http://127.0.0.1:9200"

logger = get_measurement_logger(
    logger_name=LoggerName.METRICS_AGGREGATIONS,
    # TODO: REMOVE TIMESTAMP
    logger_handler=get_elastic_logging_handler(ES_USER, ES_PASS, ES_URL, IndexName.METRICS_AGGREGATIONS, datetime.now().timestamp()),
)

handler = StreamHandler()
handler.setFormatter(PrettyExtrasFormatter())
logger.addHandler(handler)


class AggregationManager:
    """
    This class is accountable of mapping the raw results into the relevant aggregator instances.
    For example, each process will have a separate instance for computing cpu integrals, and this class is responsible
    for choosing the right instance for the incoming sample.
    """

    def __init__(self):
        self.system_aggregators = [CPUIntegralAggregator()]    # TODO: ADD SYSTEM AGGREGATORS (E.G., MEMORY INTEGRAL)

        # TODO: TRY TO FIND MORE DYNAMIC WAY TO INITIALIZE THESE LISTS
        basic_process_aggregators_types = [CPUIntegralAggregator]   # TODO: ADD MEMORY INTEGRAL (MAYBE COMBINE THE CPU AND MEMTORT INSTANCES BY CREATING AN ABSTRACT INTEGRAL CACULATOR)
        system_process_aggregators_types = []
        system_processes_aggregators_types = [ProcessSystemUsageFractionAggregator]

        self.basic_process_aggregators = defaultdict(lambda: [cls() for cls in basic_process_aggregators_types])
        self.system_process_aggregators = defaultdict(lambda: [cls() for cls in system_process_aggregators_types])   # TODO: ADD ENERGY ESTIMATIONS
        self.system_processes_aggregators = defaultdict(lambda: [cls() for cls in system_processes_aggregators_types])

    def feed_full_iteration_raw_data(self, iteration_raw_results: IterationRawResults):
        system_aggregated_results = self._feed_system_aggregators(iteration_raw_results.system_raw_results, iteration_raw_results.metadata)

        # TODO: WE MIGHT IMPLEMENT A COMBINED FUNCTION FOR ALL 'feed' FUNCTIONS AND JUST SEND THE RELEVANT PARAMETERS
        processes_basic_aggregated_results = self._feed_processes_basic_aggregators(iteration_raw_results.processes_raw_results, iteration_raw_results.metadata)
        system_process_aggregated_results = self._feed_system_process_aggregators(iteration_raw_results)
        system_processes_aggregated_results = self._feed_system_processes_aggregators(iteration_raw_results)

        combined_process_results = self._combine_process_results(
            processes_basic_aggregated_results,
            system_process_aggregated_results,
            system_processes_aggregated_results
        )

        for (pid, process_name), process_results in combined_process_results.items():
            logger.info(
                "Process Aggregation Results",
                extra=
                {
                    **asdict(iteration_raw_results.metadata),
                    "pid": pid,
                    "process_name": process_name,
                    # TODO: ADD PROCESS_OF_INTEREST AND MAYBE PROGRAM ARGUMENTS
                    **{key: value for aggregation_result in process_results for key, value in
                       asdict(aggregation_result).items()}
                }
            )

        logger.info(
            "System Aggregation Results",
            extra=
            {
                **asdict(iteration_raw_results.metadata),
                **{key: value for aggregation_result in system_aggregated_results for key, value in
                   asdict(aggregation_result).items()}
            }
        )

    def _feed_system_aggregators(
            self,
            system_iteration_results: SystemRawResults,
            iteration_metadata: Metadata
    ) -> List[AggregationResult]:

        system_aggregation_results = []
        for aggregator in self.system_aggregators:
            system_aggregation_results.append(self._process(aggregator, system_iteration_results, iteration_metadata))

        return system_aggregation_results

    def _feed_processes_basic_aggregators(
            self, processes_iteration_results: List[ProcessRawResults],
            iteration_metadata: Metadata
    ) -> Dict[Tuple[int, str], List[AggregationResult]]:     # TODO: REPLACE TYPING WITH DATACLASS
        """
        Assuming pid and process_name are unique
        """
        processes_aggregation_results = {}
        for raw_process_results in processes_iteration_results:
            process_aggregation_results = []
            for aggregator in self.basic_process_aggregators[raw_process_results.pid, raw_process_results.process_name]:
                process_aggregation_results.append(self._process(aggregator, raw_process_results, iteration_metadata))

            processes_aggregation_results[raw_process_results.pid, raw_process_results.process_name] = process_aggregation_results

        return processes_aggregation_results

    def _feed_system_process_aggregators(
            self, iteration_raw_results: IterationRawResults,
    ) -> Dict[Tuple[int, str], List[AggregationResult]]:     # TODO: REPLACE TYPING WITH DATACLASS
        """
        Assuming pid and process_name are unique
        """
        processes_aggregation_results = {}
        for raw_process_results in iteration_raw_results.processes_raw_results:
            process_aggregation_results = []
            for aggregator in self.system_process_aggregators[raw_process_results.pid, raw_process_results.process_name]:
                process_aggregation_results.append(
                    self._process(
                        aggregator,
                        SystemProcessRawResults(
                            processes_raw_results=raw_process_results,
                            system_raw_results=iteration_raw_results.system_raw_results
                        ),
                        iteration_raw_results.metadata
                    )
                )

            processes_aggregation_results[raw_process_results.pid, raw_process_results.process_name] = process_aggregation_results

        return processes_aggregation_results

    def _feed_system_processes_aggregators(
            self, iteration_raw_results: IterationRawResults,
    ) -> Dict[Tuple[int, str], List[AggregationResult]]:     # TODO: REPLACE TYPING WITH DATACLASS
        """
        Assuming pid and process_name are unique
        """
        processes_aggregation_results = {}
        for raw_process_results in iteration_raw_results.processes_raw_results:
            process_aggregation_results = []
            for aggregator in self.system_processes_aggregators[raw_process_results.pid, raw_process_results.process_name]:
                process_aggregation_results.append(
                    self._process(
                        aggregator,
                        SystemProcessesRawResults(
                            desired_process_raw_results=raw_process_results,
                            processes_raw_results=iteration_raw_results.processes_raw_results,
                            system_raw_results=iteration_raw_results.system_raw_results
                        ),
                        iteration_raw_results.metadata
                    )
                )

            processes_aggregation_results[raw_process_results.pid, raw_process_results.process_name] = process_aggregation_results

        return processes_aggregation_results

    @staticmethod
    def _process(
            aggregator: AbstractAggregator,
            raw_results: AbstractRawResults,
            iteration_metadata: Metadata
    ) -> AggregationResult:
        relevant_sample_features = aggregator.extract_features(raw_results, iteration_metadata)
        return aggregator.process_sample(relevant_sample_features)

    @staticmethod
    def _combine_process_results(
            processes_basic_aggregated_results: Dict[Tuple[int, str], List[AggregationResult]],
            system_process_aggregated_results: Dict[Tuple[int, str], List[AggregationResult]],
            system_processes_aggregated_results: Dict[Tuple[int, str], List[AggregationResult]]
    ) -> Dict[Tuple[int, str], List[AggregationResult]]:
        all_dicts = (
            processes_basic_aggregated_results,
            system_process_aggregated_results,
            system_processes_aggregated_results
        )

        combined_results = defaultdict(list)
        for process_aggregations_dict in all_dicts:
            for (pid, process_name), aggregated_results in process_aggregations_dict.items():
                combined_results[(pid, process_name)].extend(aggregated_results)

        return combined_results

