from typing import Optional

from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.raw_results_dtos.iteration_info import IterationRawResults
from elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer


# TODO: IMPLEMENT LOCKS INSIDE CONSUME (A BACKGROUND TASK) AND GET BEST CONFIGURATION (API), AS THE HAPPEN CONCURRENTLY
class DRLModel(AbstractElasticConsumer):
    def __init__(self):
        self.raw_results = []
        self.aggregation_results = []
        print("inside constructor")

    def consume(
            self,
            iteration_raw_results: IterationRawResults,
            iteration_aggregation_results: Optional[IterationAggregatedResults]
    ):
        print("Inside DRL model")
        self.raw_results.append(iteration_raw_results)
        self.aggregation_results.append(iteration_aggregation_results)

    def get_best_configuration(self, param1, param2):
        print("Returning the best configuration")
        print(self.raw_results)
        print(self.aggregation_results)
