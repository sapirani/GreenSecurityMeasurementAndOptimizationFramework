from resource_monitors.process.abstract_process_monitor import AbstractProcessMonitor


class ProcessesOfInterestOnlyMonitor(AbstractProcessMonitor):
    def save_current_processes_statistics(self) -> None:
        """
        This function gets all processes running in the system and order them by thier cpu usage
        """
        self.monitor_relevant_processes(self.mark_processes)
