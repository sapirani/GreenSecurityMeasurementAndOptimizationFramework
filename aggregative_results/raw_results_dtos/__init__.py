from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd

from aggregative_results.raw_results_dtos.process_raw_results import ProcessRawResults
from aggregative_results.raw_results_dtos.system_raw_results import SystemRawResults


def parse_time(ts_str: str) -> datetime:
    return pd.to_datetime(ts_str)


@dataclass
class IterationMetadata:
    timestamp: datetime
    start_date: datetime
    hostname: str
    session_id: str

    def __post_init__(self):
        self.seconds_from_starting_measurement: timedelta = self.timestamp - self.start_date

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IterationMetadata':
        init_kwargs = {}

        for f in fields(cls):
            try:
                value = data[f.name]

                # Parse to datetime if necessary
                if f.type is datetime and isinstance(value, str):
                    value = parse_time(value)

                init_kwargs[f.name] = value
            except KeyError:
                raise ValueError(f"Missing required field: {f.name}")

        return IterationMetadata(**init_kwargs)


@dataclass
class IterationRawResults:
    metadata: IterationMetadata
    system_raw_results: SystemRawResults
    processes_raw_results: List[ProcessRawResults]
