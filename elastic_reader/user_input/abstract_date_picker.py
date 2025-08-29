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


class AbstractTimePicker(ABC):
    LOCAL_TIMEZONE = datetime.now().astimezone().tzinfo

    def __init__(self):
        self.user_input: Optional[TimePickerChosenInput] = None

    @abstractmethod
    def _inner_get_input(self):
        pass

    def get_input(self) -> TimePickerChosenInput:
        self.user_input = self._inner_get_input()
        return self.user_input

    def __str__(self) -> str:
        if not self.user_input:
            raise ValueError("You must call get_input before calling this function")

        return (
            "-------------- Chosen Configuration --------------\n"
            f"Selected mode: {self.user_input.mode}\n"
            f"Selected start time: {self.user_input.start.astimezone(self.LOCAL_TIMEZONE)}\n"
            f"Selected end time: {self.user_input.end.astimezone(self.LOCAL_TIMEZONE) if self.user_input.end else None}\n"
            "--------------------------------------------------\n"
        )
