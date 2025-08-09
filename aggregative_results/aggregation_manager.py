import logging
from dataclasses import asdict

from aggregative_results.aggregators.cpu_integral_aggregator import CPUIntegralAggregator
from utils.general_consts import LoggerName

logger = logging.getLogger(LoggerName.METRICS_AGGREGATIONS)


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

    def feed_full_iteration_raw_data(self, iteration_input: InputDataclass):
        # TODO: think about how to print all process-related aggregations in one document per process
        # TODO: think about how to print system-related in a separate document
        # TODO: all aggregations should be sent to the same index.



        # both can use the same aggregators, just different instances per process and a separate one for the system
        system_aggregated_results = self.feed_system_aggregators(iteration_input)
        # processes_basic_aggregated_results = self.feed_processes_basic_aggregators(iteration_input.raw_process_samples)
        #
        # feed_system_process_aggregators(iteration_input)   # will be logged as process document
        # feed_cross_processes_aggregators(iteration_input)   # will be logged as process document

        # TODO: chain all process results together here. Probably through taking all values from the pid key of all results dictionaries
        # for process_results in processes_basic_aggregated_results:
        #     logger.info(
        #         "Process Aggregation Results",
        #         extra=
        #         {
        #             "date": iteration_input.date,
        #             "pid": pid,  # TODO: extract process identifier somehow
        #             **{key: value for aggregation_result in process_results for key, value in
        #                asdict(processes_basic_aggregated_results).items()}
        #         }
        #     )

        logger.info(
            "System Aggregation Results",
            extra=
            {
                "date": iteration_input.date,
                **{key: value for aggregation_result in system_aggregated_results for key, value in
                   asdict(aggregation_result).items()}
            }
        )

    def feed_system_aggregators(self, system_iteration_input):
        system_aggregation_results = []
        for aggregator in self.system_aggregators:
            system_aggregation_results.append(self._process(aggregator, system_iteration_input))

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
    def _process(aggregator, raw_input):
        relevant_sample_features = aggregator.extract_features(raw_input)
        return aggregator.process_sample(relevant_sample_features)

