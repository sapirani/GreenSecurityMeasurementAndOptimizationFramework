import platform
from typing import Callable

import psutil
from scapy.interfaces import get_working_ifaces

from operating_systems.abstract_operating_system import AbstractOSFuncs
from operating_systems.os_linux import LinuxOS
from operating_systems.os_windows import WindowsOS
from process_connections import ProcessNetworkMonitor
from program_parameters import antivirus_type, scan_type, custom_scan_path, recursive, should_optimize, \
    should_mitigate_timestomping, ids_type, interface_name, pcap_list_dirs, log_path, configuration_file_path, \
    model_name, model_action, script_relative_path, installation_dir, cpu_percent_to_consume, RUNNING_TIME, \
    memory_chunk_size, consumption_speed, time_interval
from resource_monitors.process.abstract_process_monitor import AbstractProcessMonitor
from resource_monitors.process.all_processes_monitor import AllProcessesMonitor
from resource_monitors.process.process_of_interest_only_monitor import ProcessesOfInterestOnlyMonitor
from resource_monitors.system_monitor.battery.battery_monitor import BatteryMonitor
from resource_monitors.system_monitor.battery.null_battery_monitor import NullBatteryMonitor
from summary_builder import DuduSummary, OtherSummary
from general_consts import SummaryType, ProgramToScan, AntivirusType, IDSType, ProcessMonitorType, BatteryMonitorType
from tasks.program_classes.abstract_program import ProgramInterface
from tasks.program_classes.antiviruses.clam_av_program import ClamAVProgram
from tasks.program_classes.antiviruses.defender_program import DefenderProgram
from tasks.program_classes.antiviruses.dummy_antivirus_program import DummyAntivirusProgram
from tasks.program_classes.antiviruses.sophos_av_program import SophosAVProgram
from tasks.program_classes.dummy_cpu_consumer_program import CPUConsumer
from tasks.program_classes.dummy_io_writer_consumer_program import IOWriteConsumer
from tasks.program_classes.dummy_memory_consumer_program import MemoryConsumer
from tasks.program_classes.ids.snort_program import SnortProgram
from tasks.program_classes.ids.suricata_program import SuricataProgram
from tasks.program_classes.log_anomaly_detection_program import LogAnomalyDetection
from tasks.program_classes.network_receiver_program import NetworkReceiver
from tasks.program_classes.network_sender_program import NetworkSender
from tasks.program_classes.no_scan_program import NoScanProgram
from tasks.program_classes.perfmon_monitoring_program import PerfmonProgram
from tasks.program_classes.server_program import PythonServer
from tasks.program_classes.splunk_program import SplunkProgram
from tasks.program_classes.user_activity_program import UserActivityProgram


def running_os_factory(is_inside_container: bool) -> AbstractOSFuncs:
    if platform.system() == "Linux":
        return LinuxOS(is_inside_container=is_inside_container)
    elif platform.system() == "Windows":
        return WindowsOS()

    raise Exception("Operating system is not supported")


def summary_builder_factory(summary_type: SummaryType):
    if summary_type == SummaryType.DUDU:
        return DuduSummary()
    elif summary_type == SummaryType.OTHER:
        return OtherSummary()

    raise Exception("Selected summary builder is not supported")


def process_monitor_factory(
        process_monitor_type: ProcessMonitorType,
        running_os: AbstractOSFuncs,
        should_ignore_process: Callable[[psutil.Process], bool]
) -> AbstractProcessMonitor:
    interfaces_for_packets_capturing = get_working_ifaces()
    process_network_monitor = ProcessNetworkMonitor(interfaces_for_packets_capturing)

    if process_monitor_type == ProcessMonitorType.FULL:
        return AllProcessesMonitor(process_network_monitor, running_os, should_ignore_process)
    elif process_monitor_type == ProcessMonitorType.PROCESSES_OF_INTEREST_ONLY:
        return ProcessesOfInterestOnlyMonitor(process_network_monitor, running_os, should_ignore_process)

    raise Exception("Selected process monitor type is not supported")


def battery_monitor_factory(battery_monitor_type: BatteryMonitorType, running_os: AbstractOSFuncs):
    if battery_monitor_type == BatteryMonitorType.FULL:
        return BatteryMonitor(running_os)
    elif battery_monitor_type == BatteryMonitorType.WITHOUT_BATTERY:
        return NullBatteryMonitor()

    raise Exception("Selected process monitor type is not supported")


def program_to_scan_factory(program_type: ProgramToScan) -> ProgramInterface:
    """
    Return the class that represents the program that the user wishes to run and send its dedicated parameters
    :param program_type: The program specified by the user
    :return: The dedicated class
    """

    if program_type == ProgramToScan.ANTIVIRUS and antivirus_type == AntivirusType.DEFENDER:
        return DefenderProgram(scan_type, custom_scan_path)
    if program_type == ProgramToScan.ANTIVIRUS and antivirus_type == AntivirusType.ClamAV:
        return ClamAVProgram(scan_type, custom_scan_path, recursive, should_optimize, should_mitigate_timestomping)
    if program_type == ProgramToScan.ANTIVIRUS and antivirus_type == AntivirusType.SOPHOS:
        return SophosAVProgram(scan_type, custom_scan_path)
    if program_type == ProgramToScan.IDS and ids_type == IDSType.SURICATA:
        return SuricataProgram(interface_name, pcap_list_dirs, log_path)
    if program_type == ProgramToScan.IDS and ids_type == IDSType.SNORT:
        return SnortProgram(interface_name, pcap_list_dirs, log_path, configuration_file_path=configuration_file_path)
    if program_type == ProgramToScan.DummyANTIVIRUS:
        return DummyAntivirusProgram(custom_scan_path)
    if program_type == ProgramToScan.NO_SCAN:
        return NoScanProgram()
    if program_type == ProgramToScan.Perfmon:
        return PerfmonProgram()
    if program_type == ProgramToScan.UserActivity:
        return UserActivityProgram()
    if program_type == ProgramToScan.LogAnomalyDetection:
        return LogAnomalyDetection(model_name, model_action, script_relative_path, installation_dir)
    if program_type == ProgramToScan.Splunk:
        return SplunkProgram()
    if program_type == ProgramToScan.CPUConsumer:
        return CPUConsumer(cpu_percent_to_consume, RUNNING_TIME)
    if program_type == ProgramToScan.MemoryConsumer:
        return MemoryConsumer(memory_chunk_size, consumption_speed, RUNNING_TIME)
    if program_type == ProgramToScan.IOWriteConsumer:
        return IOWriteConsumer(custom_scan_path)
    if program_type == ProgramToScan.PythonServer:
        return PythonServer()
    if program_type == ProgramToScan.NetworkReceiver:
        return NetworkReceiver()
    if program_type == ProgramToScan.NetworkSender:
        return NetworkSender(time_interval=time_interval, running_time=RUNNING_TIME)

    raise Exception("choose program to scan from ProgramToScan enum")
