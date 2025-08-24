from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from typing import Dict, Any, List

import pandas as pd

from aggregative_results.DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from aggregative_results.DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from aggregative_results.DTOs.session_host_info import SessionHostIdentity


def parse_time(date: str) -> datetime:
    return pd.to_datetime(date)


@dataclass(frozen=True)
class IterationMetadata:
    timestamp: datetime
    start_date: datetime
    session_host_identity: SessionHostIdentity

    seconds_from_starting_measurement: timedelta = field(init=False)

    def __post_init__(self):
        delta = self.timestamp - self.start_date
        object.__setattr__(self, 'seconds_from_starting_measurement', delta.total_seconds())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IterationMetadata':
        init_kwargs = {}

        for f in fields(cls):
            if not f.init:
                continue  # Skip fields not set through constructor

            try:
                # Handle session_host_identity creation from dictionary
                if f.name == 'session_host_identity':
                    session_host_identity = SessionHostIdentity(
                        hostname=data.get('hostname'),
                        session_id=data.get('session_id')
                    )
                    init_kwargs[f.name] = session_host_identity
                else:
                    value = data[f.name]
                    # Parse to datetime if necessary
                    if f.type is datetime and isinstance(value, str):
                        value = parse_time(value)

                    init_kwargs[f.name] = value
            except KeyError:
                print(data)
                raise ValueError(f"Missing required field: {f.name}")

        return IterationMetadata(**init_kwargs)


@dataclass
class IterationRawResults:
    metadata: IterationMetadata
    system_raw_results: SystemRawResults
    processes_raw_results: List[ProcessRawResults]
