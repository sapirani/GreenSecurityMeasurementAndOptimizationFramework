import platform
from typing import Callable

import psutil
from scapy.interfaces import get_working_ifaces

from operating_systems.abstract_operating_system import AbstractOSFuncs
from operating_systems.os_linux import LinuxOS
from operating_systems.os_windows import WindowsOS
from program_parameters import antivirus_type, scan_type, custom_scan_path, recursive, should_optimize, \
    should_mitigate_timestomping, ids_type, interface_name, pcap_list_dirs, log_path, configuration_file_path, \
    model_name, model_action, script_relative_path, installation_dir, cpu_percent_to_consume, RUNNING_TIME, \
    dummy_task_rate, dummy_task_unit_size
from resource_usage_recorder.processes_recorder.process_network_usage_recorder import ProcessNetworkUsageRecorder
from resource_usage_recorder.processes_recorder.strategies.abstract_processes_recorder import AbstractProcessResourceUsageRecorder
from resource_usage_recorder.processes_recorder.strategies.all_processes_recorder import AllProcessesResourceUsageRecorder
from resource_usage_recorder.processes_recorder.strategies.process_of_interest_only_recorder import \
    ProcessesOfInterestOnlyRecorder
from resource_usage_recorder.processes_recorder.strategies.splunk_processes_recorder import SplunkProcessesResourceUsageRecorder
from resource_usage_recorder.system_recorder.battery.battery_usage_recorder import SystemBatteryUsageRecorder
from resource_usage_recorder.system_recorder.battery.null_battery_recorder import NullBatteryUsageRecorder
from summary_builder import SystemResourceIsolationSummaryBuilder, NativeSummaryBuilder
from tasks.program_classes.resources_consumers_programs.disk_io_read_program import DiskIOReadConsumer
from tasks.program_classes.resources_consumers_programs.disk_io_write_program import DiskIOWriteConsumer
from tasks.program_classes.resources_consumers_programs.memory_releaser_program import MemoryReleaser
from utils.general_consts import SummaryType, ProgramToScan, AntivirusType, IDSType, ProcessMonitorType, BatteryMonitorType
from tasks.program_classes.abstract_program import ProgramInterface
from tasks.program_classes.antiviruses.clam_av_program import ClamAVProgram
from tasks.program_classes.antiviruses.defender_program import DefenderProgram
from tasks.program_classes.antiviruses.dummy_antivirus_program import DummyAntivirusProgram
from tasks.program_classes.antiviruses.sophos_av_program import SophosAVProgram
from tasks.program_classes.resources_consumers_programs.cpu_consumer_program import CPUConsumer
from tasks.program_classes.dummy_io_writer_consumer_program import IOWriteConsumer
from tasks.program_classes.resources_consumers_programs.memory_consumer_program import MemoryConsumer
from tasks.program_classes.ids.snort_program import SnortProgram
from tasks.program_classes.ids.suricata_program import SuricataProgram
from tasks.program_classes.log_anomaly_detection_program import LogAnomalyDetection
from tasks.program_classes.resources_consumers_programs.network_receiver_program import NetworkReceiver
from tasks.program_classes.resources_consumers_programs.network_sender_program import NetworkSender
from tasks.program_classes.baseline_measurement_program import BaselineMeasurementProgram
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
    if summary_type == SummaryType.ISOLATE_SYSTEM_RESOURCES:
        return SystemResourceIsolationSummaryBuilder()
    elif summary_type == SummaryType.NATIVE:
        return NativeSummaryBuilder()

    raise Exception("Selected summary builder is not supported")


def process_resource_usage_recorder_factory(
        process_monitor_type: ProcessMonitorType,
        running_os: AbstractOSFuncs,
        should_ignore_process: Callable[[psutil.Process], bool]
) -> AbstractProcessResourceUsageRecorder:
    interfaces_for_packets_capturing = get_working_ifaces()
    process_network_monitor = ProcessNetworkUsageRecorder(interfaces_for_packets_capturing)

    if process_monitor_type == ProcessMonitorType.FULL:
        return AllProcessesResourceUsageRecorder(process_network_monitor, running_os, should_ignore_process)
    elif process_monitor_type == ProcessMonitorType.PROCESSES_OF_INTEREST_ONLY:
        return ProcessesOfInterestOnlyRecorder(process_network_monitor, running_os, should_ignore_process)
 
    raise Exception("Selected process monitor type is not supported")


def battery_usage_recorder_factory(battery_monitor_type: BatteryMonitorType, running_os: AbstractOSFuncs):
    if battery_monitor_type == BatteryMonitorType.FULL:
        return SystemBatteryUsageRecorder(running_os)
    elif battery_monitor_type == BatteryMonitorType.WITHOUT_BATTERY:
        return NullBatteryUsageRecorder()

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
    if program_type == ProgramToScan.BASELINE_MEASUREMENT:
        return BaselineMeasurementProgram()
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
        return MemoryConsumer(consumption_speed=dummy_task_rate, memory_chunk_size=dummy_task_unit_size)
    if program_type == ProgramToScan.MemoryReleaser:
        return MemoryReleaser(releasing_speed=dummy_task_rate, memory_chunk_size=dummy_task_unit_size)
    if program_type == ProgramToScan.DiskIOWriteConsumer:
        return DiskIOWriteConsumer(rate=dummy_task_rate, file_size=dummy_task_unit_size)
    if program_type == ProgramToScan.DiskIOReadConsumer:
        return DiskIOReadConsumer(rate=dummy_task_rate, file_size=dummy_task_unit_size)
    if program_type == ProgramToScan.PythonServer:
        return PythonServer()
    if program_type == ProgramToScan.NetworkReceiver:
        return NetworkReceiver(rate=dummy_task_rate, buffer_size=dummy_task_unit_size)
    if program_type == ProgramToScan.NetworkSender:
        return NetworkSender(rate=dummy_task_rate, packet_size=dummy_task_unit_size)

    raise Exception("choose program to scan from ProgramToScan enum")
