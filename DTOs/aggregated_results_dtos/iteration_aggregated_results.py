from dataclasses import dataclass
from typing import Dict

from DTOs.aggregated_results_dtos.abstract_aggregation_results import AbstractAggregationResult
from DTOs.aggregated_results_dtos.aggregated_process_results import AggregatedProcessResults
from DTOs.process_info import ProcessIdentity
from DTOs.raw_results_dtos.iteration_info import IterationMetadata
from elastic_reader.aggregators.aggregation_types import AggregationType


@dataclass
class IterationAggregatedResults:
    iteration_metadata: IterationMetadata
    system_aggregated_results: Dict[AggregationType, AbstractAggregationResult]
    processes_results: Dict[ProcessIdentity, AggregatedProcessResults]
