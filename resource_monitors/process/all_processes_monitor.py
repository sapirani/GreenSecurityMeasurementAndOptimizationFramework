import psutil
from resource_monitors.process.abstract_process_monitor import AbstractProcessMonitor


class AllProcessesMonitor(AbstractProcessMonitor):
    def save_current_processes_statistics(self) -> None:
        """
        This function gets all processes running in the system and order them by their cpu usage
        """
        self.add_to_processes_dataframe(psutil.process_iter())
