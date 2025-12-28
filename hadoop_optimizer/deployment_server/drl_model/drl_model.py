from typing import Optional
from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.raw_results_dtos.iteration_info import IterationRawResults
from elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer
from hadoop_optimizer.DTOs.job_properties import JobProperties
from hadoop_optimizer.deployment_server.drl_model.drl_state import DRLState


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

    def determine_best_job_configuration(self, job_properties: JobProperties):
        drl_state = self.drl_state.retrieve_state_entries(job_properties)

        print("state shape:", drl_state.shape, ", is index unique:", drl_state.index.is_unique)
        print(drl_state.to_string())

