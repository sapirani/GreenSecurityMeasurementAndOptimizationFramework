from typing import List

import psutil

from resource_usage_recorder.processes_recorder.strategies.abstract_processes_recorder import AbstractProcessResourceUsageRecorder, \
    ProcessMetrics


class SplunkProcessesResourceUsageRecorder(AbstractProcessResourceUsageRecorder):
    def get_current_metrics(self) -> List[ProcessMetrics]:
        """
        This function gets telemetry from all processes in the system
        (that are not filtered by the should_ignore_process predicate)
        """
        processes = []
        for p in psutil.process_iter():
            try:
                # print(f"{p.name()} - {p.pid} - {p.status()} {'splunk' in p.name().lower()}")
                if 'splunk' in p.name().lower():
                    # print(p.name().lower())
                    
                    processes.append(p)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                # print(f"Error accessing process {p.name()}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error accessing process {p.name()}: {e}")
                continue
        return self._get_current_metrics(processes)
