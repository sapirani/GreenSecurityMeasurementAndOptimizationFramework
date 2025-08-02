import dataclasses
from abc import ABC, abstractmethod
from typing import List, Union
from dataclasses import dataclass


@dataclass
class MetricResult:
    def to_dict(self):
        return dataclasses.asdict(self)


class MetricRecorder(ABC):
    @abstractmethod
    def get_current_metrics(self) -> Union[MetricResult, List[MetricResult]]:
        pass
