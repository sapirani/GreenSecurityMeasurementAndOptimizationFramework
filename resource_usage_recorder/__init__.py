import dataclasses
from abc import ABC, abstractmethod
from typing import List, Union, Set
from dataclasses import dataclass


@dataclass
class MetricResult:
    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def get_columns(cls) -> Set[str]:
        return {f.name for f in dataclasses.fields(cls)}


class MetricRecorder(ABC):
    @abstractmethod
    def get_current_metrics(self) -> Union[MetricResult, List[MetricResult]]:
        pass
