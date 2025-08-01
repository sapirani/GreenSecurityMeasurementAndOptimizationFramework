from resource_monitors.processes_monitor.strategies.abstract_processes_monitor import AbstractProcessMonitor


class ProcessesOfInterestOnlyMonitor(AbstractProcessMonitor):
    def save_current_processes_statistics(self):
        """
        This function gets all processes running in the system and order them by thier cpu usage
        """
        self.monitor_relevant_processes(self.mark_processes)
