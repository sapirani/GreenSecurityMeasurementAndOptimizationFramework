import psutil
from custom_process_filter.abstarct_process_filter import AbstractProcessFilter


class FilterForSplunkProcesses(AbstractProcessFilter):
    def should_ignore_process(self, process: psutil.Process) -> bool:
        return "splunk" not in process.name()
