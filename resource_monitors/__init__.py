import abc
import dataclasses
from abc import ABC
from typing import List, Union


@dataclasses.dataclass
class MetricResult:
    def to_dict(self):
        return dataclasses.asdict(self)


class MetricRecorder(ABC):
    @abc.abstractmethod
    def get_current_metrics(self) -> Union[MetricResult, List[MetricResult]]:
        pass
