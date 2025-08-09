from dataclasses import asdict
from datetime import datetime
from typing import List

from aggregative_results.aggregators.abstract_aggregator import AggregationResult, AbstractAggregator
from aggregative_results.aggregators.cpu_integral_aggregator import CPUIntegralAggregator
from aggregative_results.raw_results_dtos import IterationRawResults, SystemRawResults, IterationMetadata
from aggregative_results.raw_results_dtos.abstract_raw_results import AbstractRawResults
from application_logging import get_measurement_logger, get_elastic_logging_handler
from utils.general_consts import LoggerName, IndexName

# TODO: REMOVE
ES_USER = "elastic"
ES_PASS = "SVR4mUZl"
ES_URL = "http://127.0.0.1:9200"

# TODO: ADD STREAM HANDLER
logger = get_measurement_logger(
        logger_name=LoggerName.METRICS_AGGREGATIONS,
        # TODO: REMOVE TIMESTAMP
        logger_handler=get_elastic_logging_handler(ES_USER, ES_PASS, ES_URL, IndexName.METRICS_AGGREGATIONS, datetime.now().timestamp())
    )


class AggregationManager:
    """
    This class is accountable of mapping the raw results into the relevant aggregator instances.
    For example, each process will have a separate instance for computing cpu integrals, and this class is responsible
    for choosing the right instance for the incoming sample.
    """

    def __init__(self):
        self.system_aggregators = [CPUIntegralAggregator()]    # TODO: ADD SYSTEM AGGREGATORS
        self.basic_process_aggregators = {}     # TODO: ADD MAPPINGS FROM PID TO AGGREGATORS
        self.system_process_aggregators = {}    # TODO: ADD MAPPINGS FROM PID TO AGGREGATORS
        self.cross_processes_aggregators = {}   # TODO: ADD MAPPINGS FROM PID TO AGGREGATORS

    def feed_full_iteration_raw_data(self, iteration_raw_results: IterationRawResults):
        # TODO: think about how to print all process-related aggregations in one document per process
        # TODO: think about how to print system-related in a separate document
        # TODO: all aggregations should be sent to the same index.



        # both can use the same aggregators, just different instances per process and a separate one for the system
        system_aggregated_results = self._feed_system_aggregators(iteration_raw_results.system_raw_results, iteration_raw_results.metadata)
        # processes_basic_aggregated_results = self.feed_processes_basic_aggregators(iteration_raw_results.raw_process_samples)
        #
        # feed_system_process_aggregators(iteration_raw_results)   # will be logged as process document
        # feed_cross_processes_aggregators(iteration_raw_results)   # will be logged as process document

        # TODO: chain all process results together here. Probably through taking all values from the pid key of all results dictionaries
        # for process_results in processes_basic_aggregated_results:
        #     logger.info(
        #         "Process Aggregation Results",
        #         extra=
        #         {
        #             "date": iteration_raw_results.date,
        #             "pid": pid,  # TODO: extract process identifier somehow
        #             **{key: value for aggregation_result in process_results for key, value in
        #                asdict(processes_basic_aggregated_results).items()}
        #         }
        #     )

        # TODO: REMOVE
        print({
                **asdict(iteration_raw_results.metadata),
                **{key: value for aggregation_result in system_aggregated_results for key, value in
                   asdict(aggregation_result).items()}
            })
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
            system_iteration_input: SystemRawResults,
            iteration_metadata: IterationMetadata
    ) -> List[AggregationResult]:

        system_aggregation_results = []
        for aggregator in self.system_aggregators:
            system_aggregation_results.append(self._process(aggregator, system_iteration_input, iteration_metadata))

        return system_aggregation_results

    def feed_processes_basic_aggregators(self, processes_iteration_input):
        processes_aggregation_results = []
        for raw_sample in processes_iteration_input:
            process_aggregation_results = []
            for aggregator in self.get_process_aggregators(raw_sample.pid):
                process_aggregation_results.append(self._process(aggregator, raw_sample))

            # TODO: append the pid of the process here somehow, maybe a dictionary that maps pid to results
            processes_aggregation_results.append(process_aggregation_results)

        return processes_aggregation_results

    @staticmethod
    def _process(
            aggregator: AbstractAggregator,
            raw_results: AbstractRawResults,
            iteration_metadata: IterationMetadata
    ) -> AggregationResult:
        relevant_sample_features = aggregator.extract_features(raw_results, iteration_metadata)
        return aggregator.process_sample(relevant_sample_features)

