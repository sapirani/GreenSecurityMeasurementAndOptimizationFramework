from typing import List

import psutil

from resource_usage_recorder.processes_recorder.strategies.abstract_processes_recorder import AbstractProcessResourceUsageRecorder, \
    ProcessMetrics


class AllProcessesResourceUsageRecorder(AbstractProcessResourceUsageRecorder):
    def get_current_metrics(self) -> List[ProcessMetrics]:
        """
        This function gets telemetry from all processes in the system
        (that are not filtered by the should_ignore_process predicate)
        """
        return self._get_current_metrics(psutil.process_iter())
