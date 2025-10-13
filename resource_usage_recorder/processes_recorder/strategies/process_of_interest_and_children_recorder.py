from typing import List

import psutil
from overrides import override

from resource_usage_recorder.processes_recorder.strategies.abstract_processes_recorder import AbstractProcessResourceUsageRecorder, \
    ProcessMetrics


class ProcessesOfInterestAndChildrenRecorder(AbstractProcessResourceUsageRecorder):
    def get_current_metrics(self) -> List[ProcessMetrics]:
        """
        This function gets telemetry from processes we interest of and the sub-processes that they open
        (which where set using the set_processes_to_mark function)
        """
        return self._get_current_metrics(self.mark_processes)

    @override
    def set_processes_to_mark(self, processes: List[psutil.Process]):
        processes_children = [parent_process.children(recursive=True) for parent_process in processes]
        self.mark_processes = processes + processes_children
