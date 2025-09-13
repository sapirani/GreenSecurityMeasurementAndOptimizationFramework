import psutil

from custom_process_filter.abstarct_process_filter import AbstractProcessFilter


class FilterOutCMDProcesses(AbstractProcessFilter):
    def should_ignore_process(self, process: psutil.Process) -> bool:
        return process.name() == "cmd.exe"
