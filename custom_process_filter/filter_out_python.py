import psutil
from custom_process_filter.abstarct_process_filter import AbstractProcessFilter


class FilterOutPythonProcesses(AbstractProcessFilter):
    def should_ignore_process(self, process: psutil.Process) -> bool:
        return "python" in process.name()
