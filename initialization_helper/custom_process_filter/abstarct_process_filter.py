from abc import ABC, abstractmethod

import psutil


class AbstractProcessFilter(ABC):
    """
    This interface is dedicated for filtering process.
    Users who want to filter processes must implement this interface in a self-created module inside this directory.
    Then, the user should add another type in the Enum CustomFilter (in general_consts.py), handle this type in
    initialization factories, and add the newly created type in program_parameters.py
    """
    @abstractmethod
    def should_ignore_process(self, process: psutil.Process) -> bool:
        pass
