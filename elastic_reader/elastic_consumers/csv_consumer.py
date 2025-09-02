from typing import Optional

from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.raw_results_dtos.iteration_info import IterationRawResults
from elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer


# TODO: IMPLEMENT
class CSVConsumer(AbstractElasticConsumer):
    def consume(
            self,
            iteration_raw_results: IterationRawResults,
            iteration_aggregation_results: Optional[IterationAggregatedResults]
    ):
        pass
