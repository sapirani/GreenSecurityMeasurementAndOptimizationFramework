from resource_monitors.processes_monitor.strategies.abstract_processes_monitor import AbstractProcessMonitor


class ProcessesOfInterestOnlyMonitor(AbstractProcessMonitor):
    def get_current_metrics(self):
        """
        This function gets all processes running in the system and order them by thier cpu usage
        """
        return self._get_current_metrics(self.mark_processes)
