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
            iteration_aggregations: Optional[IterationAggregatedResults]
    ):
        pass
