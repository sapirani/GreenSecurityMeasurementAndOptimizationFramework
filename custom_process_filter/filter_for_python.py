import psutil

from custom_process_filter.abstarct_process_filter import AbstractProcessFilter


class FilterForPythonProcesses(AbstractProcessFilter):
    def should_ignore_process(self, process: psutil.Process) -> bool:
        return "python" not in process.name()
