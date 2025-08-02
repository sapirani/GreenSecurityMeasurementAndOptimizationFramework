from typing import List

from resource_usage_recorder.processes_recorder.strategies.abstract_processes_recorder import AbstractProcessResourceUsageRecorder, \
    ProcessMetrics


class ProcessesOfInterestOnlyRecorder(AbstractProcessResourceUsageRecorder):
    def get_current_metrics(self) -> List[ProcessMetrics]:
        """
        This function gets telemetry from processes we interest of only
        (which where set using the set_processes_to_mark function)
        """
        return self._get_current_metrics(self.mark_processes)
