from typing import Callable

import psutil
from scapy.interfaces import get_working_ifaces

from general_consts import ProcessMonitorType
from operating_systems.abstract_operating_system import AbstractOSFuncs
from process_connections import ProcessNetworkMonitor
from resource_monitors.process.abstract_process_monitor import AbstractProcessMonitor
from resource_monitors.process.all_processes_monitor import AllProcessesMonitor
from resource_monitors.process.process_of_interest_only_monitor import ProcessesOfInterestOnlyMonitor


def process_monitor_factory(process_monitor_type: ProcessMonitorType, running_os: AbstractOSFuncs,
                            should_ignore_process: Callable[[psutil.Process], bool]) -> AbstractProcessMonitor:
    interfaces_for_packets_capturing = get_working_ifaces()
    process_network_monitor = ProcessNetworkMonitor(interfaces_for_packets_capturing)

    if process_monitor_type == ProcessMonitorType.FULL:
        return AllProcessesMonitor(process_network_monitor, running_os, should_ignore_process)
    elif process_monitor_type == ProcessMonitorType.PROCESSES_OF_INTEREST_ONLY:
        return ProcessesOfInterestOnlyMonitor(process_network_monitor, running_os, should_ignore_process)

    raise Exception("Selected process monitor type is not supported")
