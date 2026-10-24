import dataclasses
from dataclasses import dataclass
from typing import Set


@dataclass
class MetricResult:
    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def get_columns(cls) -> Set[str]:
        return {f.name for f in dataclasses.fields(cls)}
