from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from aggregative_results.DTOs.raw_results_dtos import IterationMetadata
from aggregative_results.DTOs.raw_results_dtos.abstract_raw_results import AbstractRawResults
from aggregative_results.DTOs.aggregated_results_dtos import AggregationResult

T = TypeVar('T')


class AbstractAggregator(ABC, Generic[T]):
    """
    Note, this class might save a state, regarding previous samples.
    It means that each instance should consistently receive inputs regrading to the same measured entity.
    This class is not accountable for that consistency (which should be managed by the callee).

    For example, suppose we create a CPU integral aggregator per each process and also at the system-level.
    This aggregator must save the previous iteration information to be able to compute integral over CPU measurements.
    You must create a separate instance for each process + another one for the system,
    thereby each instance manages only one entity.
    In this example, it is the callee responsibility to provide samples of the right entity its dedicated instance.
    Otherwise, results will be incorrect (for example, computing an integral where samples are taken from 2 different processes)
    """

    @abstractmethod
    def extract_features(self, raw_results: AbstractRawResults, iteration_metadata: IterationMetadata) -> T:
        """
        This function aims to extract the relevant features only per aggregator
        :return: the extracted features relevant to the aggregator
        """
        pass

    @abstractmethod
    def process_sample(self, sample: T) -> AggregationResult:
        """
        This function receives the relevant features to the aggregation strategy and outputs the aggregation result.
        :param sample: the relevant features to the specific aggregator.
        :return: aggregation results
        """
        pass
