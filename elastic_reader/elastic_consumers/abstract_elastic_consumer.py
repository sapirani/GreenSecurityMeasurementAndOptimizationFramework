from abc import ABC, abstractmethod
from typing import Optional

from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.raw_results_dtos.iteration_info import IterationRawResults


# TODO: CONSIDER USING extract_features METHOD AS IT WAS DONE IN THE AGGREGATIONS WHERE EACH AGGREGATOR EXTRACTS THE
# TODO: INFORMATION IT IS INTERESTED IN
class AbstractElasticConsumer(ABC):
    @abstractmethod
    def consume(
            self,
            iteration_raw_results: IterationRawResults,
            iteration_aggregation_results: Optional[IterationAggregatedResults]
    ):
        """
        This function is the main function that each consumer should implement.
        The consumer receives the iteration results and decides what to do with them.
        """
        pass

    def post_processing(self):
        """
        This function is called after all results are fetched and transferred to the consumer via the consume function.
        This function allows performing operations when knowing no more data is going to be fetched.
        """
        pass
