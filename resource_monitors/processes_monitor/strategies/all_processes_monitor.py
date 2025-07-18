import psutil
from resource_monitors.processes_monitor.strategies.abstract_processes_monitor import AbstractProcessMonitor


class AllProcessesMonitor(AbstractProcessMonitor):
    def save_current_processes_statistics(self) -> None:
        """
        This function gets all processes running in the system and order them by their cpu usage
        """
        self.monitor_relevant_processes(psutil.process_iter())
