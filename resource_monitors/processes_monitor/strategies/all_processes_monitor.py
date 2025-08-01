import psutil
from resource_monitors.processes_monitor.strategies.abstract_processes_monitor import AbstractProcessMonitor


class AllProcessesMonitor(AbstractProcessMonitor):
    def get_current_metrics(self):
        """
        This function gets all processes running in the system and order them by their cpu usage
        """
        return self._get_current_metrics(psutil.process_iter())
