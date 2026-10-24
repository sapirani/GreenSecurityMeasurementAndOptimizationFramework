from abc import ABC, abstractmethod
from typing import List, Union

from DTOs.scanner_counters.metric_results import MetricResult


class MetricRecorder(ABC):
    @abstractmethod
    def get_current_metrics(self) -> Union[MetricResult, List[MetricResult]]:
        pass
