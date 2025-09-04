from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class ReadingMode(str, Enum):
    REALTIME = "realtime"
    OFFLINE = "offline"
    SINCE = "since"


@dataclass
class TimePickerChosenInput:
    start: datetime
    end: Optional[datetime]
    mode: ReadingMode
    LOCAL_TIMEZONE = datetime.now().astimezone().tzinfo

    def __str__(self) -> str:
        return (
            "-------------- Chosen Configuration --------------\n"
            f"Selected mode: {self.mode}\n"
            f"Selected start time: {self.start.astimezone(self.LOCAL_TIMEZONE)}\n"
            f"Selected end time: {self.end.astimezone(self.LOCAL_TIMEZONE) if self.end else None}\n"
            "--------------------------------------------------\n"
        )


class AbstractTimePicker(ABC):
    def __init__(self):
        self.user_input: Optional[TimePickerChosenInput] = None

    @abstractmethod
    def _inner_get_input(self) -> TimePickerChosenInput:
        pass

    def get_input(self) -> TimePickerChosenInput:
        self.user_input = self._inner_get_input()
        return self.user_input
