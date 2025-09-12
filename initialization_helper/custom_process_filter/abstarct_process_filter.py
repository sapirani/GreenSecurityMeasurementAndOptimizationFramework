from abc import ABC, abstractmethod

import psutil


class AbstractProcessFilter(ABC):
    """
    This interface is dedicated for filtering process.
    Users who want to filter processes must implement this interface in a self-created module inside this directory
    (which should be referenced in the program_parameters.py file).
    """
    @abstractmethod
    def should_ignore_process(self, process: psutil.Process) -> bool:
        pass
