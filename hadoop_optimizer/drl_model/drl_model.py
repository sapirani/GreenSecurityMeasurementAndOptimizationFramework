from typing import Optional
import pandas as pd
from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.raw_results_dtos.iteration_info import IterationRawResults
from elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer
from hadoop_optimizer.drl_model.drl_state import DRLState


class DRLModel(AbstractElasticConsumer):
    def __init__(self, drl_state: DRLState):
        self.drl_state = drl_state

    def consume(
            self,
            iteration_raw_results: IterationRawResults,
            iteration_aggregation_results: Optional[IterationAggregatedResults]
    ):
        print("Inside DRL model")
        self.drl_state.update_state(iteration_raw_results, iteration_aggregation_results)

    def get_best_configuration(self, param1, param2):
        with pd.option_context('display.max_columns', None):
            print(self.drl_state.state)
