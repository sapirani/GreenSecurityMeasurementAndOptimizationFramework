import argparse
import json
import os.path
import platform
import shutil
import signal
import threading
import time
import warnings
from typing import TextIO, Tuple, List, Dict, Optional

from human_id import generate_id
from prettytable import PrettyTable
from threading import Thread, Timer
import pandas as pd

from application_logging.logging_utils import get_measurement_logger
from application_logging.handlers.elastic_handler import get_elastic_logging_handler
from application_logging.filters.scanner_filter import ScannerLoggerFilter
from initialization_helper import *
from datetime import date, datetime, timezone
from pathlib import Path

from resource_usage_recorder import MetricResult
from resource_usage_recorder.system_recorder.cpu.cpu_usage_recorder import SystemCpuUsageRecorder
from resource_usage_recorder.system_recorder.disk.disk_usage_recorder import SystemDiskUsageRecorder
from resource_usage_recorder.system_recorder.memory.memory_usage_recorder import SystemMemoryUsageRecorder
from resource_usage_recorder.system_recorder.network.network_usage_recorder import SystemNetworkUsageRecorder
from tasks.program_classes.abstract_program import ProgramInterface
from utils.general_functions import EnvironmentImpact, BatteryDeltaDrain
from operating_systems.abstract_operating_system import AbstractOSFuncs

base_dir, GRAPHS_DIR, STDOUT_FILES_DIR, STDERR_FILES_DIR, PROCESSES_CSV, TOTAL_MEMORY_EACH_MOMENT_CSV, \
    DISK_IO_EACH_MOMENT, NETWORK_IO_EACH_MOMENT, BATTERY_STATUS_CSV, GENERAL_INFORMATION_FILE, TOTAL_CPU_CSV, \
    SUMMARY_CSV = result_paths()

BACKUP_DIR_PATH_BEFORE_BATTERY_DEPLETION = "backup_before_battery_depletion"
BACKED_UP_SCANNING_TIMESTAMPS_PATH = os.path.join(BACKUP_DIR_PATH_BEFORE_BATTERY_DEPLETION, "scanning timestamps.json")

program.set_results_dir(base_dir)

# ======= Program Global Parameters =======
done_scanning_event = threading.Event()
starting_time = 0
main_process_id = None
max_timeout_reached = False
main_process = None
system_metrics_logger = None
process_metrics_logger = None
application_flow_logger = None
session_id: str = ""
start_date: datetime = datetime.now(timezone.utc)

# include main programs and background
processes_ids = []
processes_names = []

processes_df = pd.DataFrame(columns=processes_columns_list)
processes_df = processes_df.astype({ProcessesColumns.PROCESS_OF_INTEREST: "bool"})

memory_df = pd.DataFrame(columns=memory_columns_list)

disk_io_each_moment_df = pd.DataFrame(columns=disk_io_columns_list)

network_io_each_moment_df = pd.DataFrame(columns=network_io_columns_list)

battery_df = pd.DataFrame(columns=battery_columns_list)

cpu_df = pd.DataFrame(columns=cpu_columns_list)

finished_scanning_time = []


def handle_sigint(signum, frame):
    """
    This function captures a signal (typically SIGINT), and terminates the program gracefully.
    It sends signal to the main program, instructing it to exit gracefully and set the global event to instruct this
    program's threads to exit.
    """
    print("Got signal, writing results and terminating")
    if main_process:
        running_os.kill_process_gracefully(main_process.pid)  # killing the main process
    done_scanning_event.set()


def dataframe_append(df: pd.DataFrame, element: Dict) -> pd.DataFrame:
    """

    :param df: dataframe to append to
    :param element: element to append
    """
    return pd.concat([df, pd.DataFrame([{"seconds_from_start": time_since_start(), **element}])], ignore_index=True)


def time_since_start() -> float:
    """
    :return: the total seconds elapsed since the measurement session started
    NOTE: In the case where this measurement is continuing a previous measurement that was early stopped
    (for example, due to low battery) and both measurements represent the same measurement session, the returned
    time will be the time elapsed since the beginning of the entire measurement session
    """
    return time.time() - starting_time


def scan_time_passed():
    """
    Checks if the minimum scan time has passed
    :return: True if the minimum scan time has passed, False otherwise
    """
    return time_since_start() >= RUNNING_TIME


def save_data_when_too_low_battery():
    """
    Stores metadata before battery depletes, to be able to continue the same measurement session from the same point
    when the device will be recharged.
    """
    with open(GENERAL_INFORMATION_FILE, 'a') as f:
        f.write("EARLY TERMINATION DUE TO LOW BATTERY!!!!!!!!!!\n\n")
    finished_scanning_time.append(time_since_start())

    print("NOTE! backing up program metadata")
    Path(BACKUP_DIR_PATH_BEFORE_BATTERY_DEPLETION).mkdir(parents=True, exist_ok=True)  # create dir for saving metadata before termination
    with open(BACKED_UP_SCANNING_TIMESTAMPS_PATH, "w") as f:
        backup_data = {
            "session_id": session_id,
            "start_timestamp": starting_time,
            "end_timestamp": time.time()
        }
        json.dump(backup_data, f, indent=4)

    if main_process:
        running_os.kill_process_gracefully(main_process.pid)  # killing the main process
    done_scanning_event.set()


# TODO: maybe use done_scanning_event.is_set() directly instead of returning True / False
# So this function will only set done_scanning_event according to termination conditions
# (e.g., predefined scanning time has passed)
def should_scan(battery_capacity: Optional[float]) -> bool:
    """
    Checks whether the measurements thread should continue to measure
    :return: True if measurement thread should perform another iteration or False if it should terminate
    """
    if battery_usage_recorder.is_battery_too_low(battery_capacity):
        save_data_when_too_low_battery()
        return False

    if main_program_to_scan == ProgramToScan.BASELINE_MEASUREMENT:
        return not scan_time_passed() and not done_scanning_event.is_set()
    elif scan_option == ScanMode.ONE_SCAN:
        return not done_scanning_event.is_set()
    elif scan_option == ScanMode.CONTINUOUS_SCAN:
        return not (scan_time_passed() and is_delta_capacity_achieved()) and not done_scanning_event.is_set()
        # return not scan_time_passed() and not is_delta_capacity_achieved()


def save_metrics_results(
        processes_results: List[MetricResult],
        system_cpu_results: MetricResult,
        system_memory_results: MetricResult,
        system_disk_results: MetricResult,
        system_network_results: MetricResult,
        system_battery_results: MetricResult
):
    global processes_df
    global cpu_df
    global memory_df
    global disk_io_each_moment_df
    global network_io_each_moment_df
    global battery_df

    iteration_timestamp = datetime.now(timezone.utc).isoformat()

    for process_results in processes_results:
        process_metrics_logger.info(
            "Process Measurements",
            extra={
                "timestamp": iteration_timestamp,
                **process_results.to_dict()
            }
        )
        processes_df = dataframe_append(processes_df, process_results.to_dict())

    system_metrics_logger.info(
        "System Measurements",
        extra={
            "timestamp": iteration_timestamp,
            **system_cpu_results.to_dict(),
            **system_memory_results.to_dict(),
            **system_disk_results.to_dict(),
            **system_network_results.to_dict(),
            **system_battery_results.to_dict()
        }
    )
    cpu_df = dataframe_append(cpu_df, system_cpu_results.to_dict())
    memory_df = dataframe_append(memory_df, system_memory_results.to_dict())
    disk_io_each_moment_df = dataframe_append(disk_io_each_moment_df, system_disk_results.to_dict())
    network_io_each_moment_df = dataframe_append(network_io_each_moment_df, system_network_results.to_dict())
    battery_df = dataframe_append(battery_df, system_battery_results.to_dict())


def continuously_measure():
    """
    This function runs in a different thread. It accounts for measuring the full resource consumption of the system
    """
    running_os.init_thread()

    # init prev_disk_io by first disk io measurements (before scan)
    # TODO: lock thread until process starts
    battery_capacity = None

    system_cpu_monitor = SystemCpuUsageRecorder(running_os, done_scanning_event, is_inside_container)
    system_memory_monitor = SystemMemoryUsageRecorder(running_os, is_inside_container)
    system_disk_usage_recorder = SystemDiskUsageRecorder()
    system_network_usage_recorder = SystemNetworkUsageRecorder()

    with processes_resource_usage_recorder:
        while should_scan(battery_capacity):
            # Create a delay
            time.sleep(SLEEP_BETWEEN_ITERATIONS_SECONDS)

            print("iteration starts:", time_since_start())
            processes_results = processes_resource_usage_recorder.get_current_metrics()
            print("after processes starts:", time_since_start())
            system_cpu_results = system_cpu_monitor.get_current_metrics()
            print("after cpu starts:", time_since_start())
            system_memory_results = system_memory_monitor.get_current_metrics()
            print("after memory starts:", time_since_start())
            system_disk_results = system_disk_usage_recorder.get_current_metrics()
            print("after disk starts:", time_since_start())
            system_network_results = system_network_usage_recorder.get_current_metrics()
            print("after network starts:", time_since_start())
            system_battery_results = battery_usage_recorder.get_current_metrics()
            print("after battery starts:", time_since_start())

            save_metrics_results(
                processes_results,
                system_cpu_results,
                system_memory_results,
                system_disk_results,
                system_network_results,
                system_battery_results
            )

            battery_capacity = system_battery_results.battery_remaining_capacity_mWh

    done_scanning_event.set()   # releasing waiting threads / processes


def save_general_disk(f: TextIO):
    """
    This function writes disk info to a file.
    :param f: text file to write the battery info
    """
    f.write("----Disk----\n")
    disk_table = PrettyTable(["Total(GB)", "Used(GB)",
                              "Available(GB)", "Percentage"])
    disk_stat = psutil.disk_usage('/')
    disk_table.add_row([
        f'{disk_stat.total / GB:.3f}',
        f'{disk_stat.used / GB:.3f}',
        f'{disk_stat.free / GB:.3f}',
        disk_stat.percent
    ])
    f.write(str(disk_table))
    f.write('\n')


def save_general_system_information(f: TextIO):
    """
    This function saves general info about the platform to a file.
    :param f: text file to write the battery info
    """
    platform_system = platform.uname()

    f.write("======System Information======\n")

    running_os.save_system_information(f)

    f.write(f"Machine Type: {platform_system.machine}\n")
    f.write(f"Device Name: {platform_system.node}\n")

    f.write("\n----Operating System Information----\n")
    f.write(f"Operating System: {platform_system.system}\n")
    f.write(f"Release: {platform_system.release}\n")
    f.write(f"Version: {platform_system.version}\n")

    f.write("\n----CPU Information----\n")
    f.write(f"Processor: {platform_system.processor}\n")
    f.write(f"Physical cores: {psutil.cpu_count(logical=False)}\n")
    f.write(f"Total cores: {NUMBER_OF_CORES}\n")
    cpufreq = psutil.cpu_freq()
    f.write(f"Max Frequency: {cpufreq.max:.2f} MHz\n")
    f.write(f"Min Frequency: {cpufreq.min:.2f} MHz\n")

    f.write("\n----RAM Information----\n")
    f.write(f"Total RAM: {psutil.virtual_memory().total / GB} GB\n")

    running_os.save_physical_memory(f)
    running_os.save_disk_information(f)


def save_general_information_before_scanning():
    """
    This function writes general battery, disk, ram, os, etc. information
    """
    with open(GENERAL_INFORMATION_FILE, 'w') as f:
        # dd/mm/YY
        f.write(f"Session_id: {session_id}\n")
        f.write(f"Hostname: {running_os.get_hostname()}\n")
        f.write(f'Date: {date.today().strftime("%d/%m/%Y")}\n')
        f.write(f'Scanner Version: {get_scanner_version_name(battery_monitor_type, process_monitor_type)}\n\n')

        # TODO: add background_programs general_information_before_measurement(f)
        program.general_information_before_measurement(f)

        save_general_system_information(f)

        f.write('\n======Before Scanning======\n')
        battery_usage_recorder.save_general_battery(f)
        f.write('\n')
        save_general_disk(f)
        f.write('\n\n')


def save_general_information_after_scanning():
    """
    save processes names and ids, disk and battery info, scanning times
    """
    with open(GENERAL_INFORMATION_FILE, 'a') as f:
        f.write('======After Scanning======\n')
        if main_process_id is not None:
            f.write(f'{PROCESS_ID_PHRASE}: {processes_names[0]}({main_process_id})\n')

        f.write(f'{BACKGROUND_ID_PHRASE}: ')
        for background_process_id, background_process_name in zip(processes_ids[1:-1], processes_names[1:-1]):
            f.write(f'{background_process_name}({background_process_id}), ')

        if len(processes_ids) > 1:  # not just main program
            f.write(f"{processes_names[-1]}({processes_ids[-1]})\n\n")

        save_general_disk(f)

        if not battery_df.empty:
            f.write('\n------Battery------\n')
            battery_drain = BatteryDeltaDrain.from_battery_drain(battery_df)
            f.write(f'Amount of Battery Drop: {battery_drain.mwh_drain} mWh, {battery_drain.percent_drain}%\n')
            f.write('Approximately equivalent to -\n')
            environment_impact = EnvironmentImpact.from_mwh(battery_drain.mwh_drain)
            f.write(f'  CO2 emission: {environment_impact.co2} kg\n')
            f.write(f'  Coal burned: {environment_impact.coal_burned} kg\n')
            f.write(f'  Number of smartphone charged: {environment_impact.number_of_smartphones_charged}\n')
            f.write(f'  Kilograms of wood burned: {environment_impact.kg_of_woods_burned}\n')

        if main_program_to_scan == ProgramToScan.BASELINE_MEASUREMENT:
            measurement_time = finished_scanning_time[-1]
            f.write(f'\nMeasurement duration: {measurement_time} seconds, '
                    f'{measurement_time / 60} minutes\n')

        else:
            f.write('\n------Scanning Times------\n')
            if max_timeout_reached:
                f.write("Scanned program reached the maximum time so we terminated it\n")
            f.write(f'Scan number 1, finished at: {finished_scanning_time[0]} seconds, '
                    f'{finished_scanning_time[0] / 60} minutes\n')
            for i, scan_time in enumerate(finished_scanning_time[1:]):
                f.write(f'Scan number {i + 2}, finished at: {scan_time}.'
                        f' Duration of Scanning: {scan_time - finished_scanning_time[i]} seconds, '
                        f'{(scan_time - finished_scanning_time[i]) / 60} minutes\n')


def prepare_summary_csv():
    """Prepare the summary csv file"""
    summary_df = summary_builder.prepare_summary_csv(
        processes_df, cpu_df, memory_df, disk_io_each_moment_df, network_io_each_moment_df, battery_df,
        processes_names, finished_scanning_time, processes_ids
    )

    color_func = summary_builder.colors_func

    styled_summary_df = summary_df.style.apply(color_func, axis=0)

    styled_summary_df.to_excel(SUMMARY_CSV, engine='openpyxl', index=False)


def ignore_last_results():
    """
    Remove the last sample from each dataframe because the main process may be finished before the sample,
    so that sample is not relevant
    """
    global processes_df
    global memory_df
    global disk_io_each_moment_df
    global network_io_each_moment_df
    global cpu_df
    global battery_df

    if processes_df.empty:
        processes_num_last_measurement = 0
    else:
        processes_num_last_measurement = processes_df[ProcessesColumns.TIME].value_counts()[
            processes_df[ProcessesColumns.TIME].max()]
    processes_df = processes_df.iloc[:-processes_num_last_measurement, :]
    memory_df = memory_df.iloc[:-1, :]
    disk_io_each_moment_df = disk_io_each_moment_df.iloc[:-1, :]
    network_io_each_moment_df = network_io_each_moment_df.iloc[:-1, :]

    if not battery_df.empty:
        battery_df = battery_df.iloc[:-1, :]
    cpu_df = cpu_df.iloc[:-1, :]


def save_results_to_files():
    """
    Save all measurements (cpu, memory, disk battery) into dedicated files.
    """
    save_general_information_after_scanning()
    ignore_last_results()

    print(f"Results are saved to {base_dir}")

    processes_df.to_csv(PROCESSES_CSV, index=False)
    memory_df.to_csv(TOTAL_MEMORY_EACH_MOMENT_CSV, index=False)
    disk_io_each_moment_df.to_csv(DISK_IO_EACH_MOMENT, index=False)
    network_io_each_moment_df.to_csv(NETWORK_IO_EACH_MOMENT, index=False)

    if not battery_df.empty:
        battery_df.to_csv(BATTERY_STATUS_CSV, index=False)
    cpu_df.to_csv(TOTAL_CPU_CSV, index=False)

    prepare_summary_csv()


def is_delta_capacity_achieved():
    """
    Relevant for Continuous Scan
    :return: True if the minimum capacity drain specified by the user is achieved and False otherwise.
    The meaning of False is that another scan should be performed
    """
    if psutil.sensors_battery() is None:  # if desktop computer (has no battery)
        return True

    return BatteryDeltaDrain.from_battery_drain(battery_df).mwh_drain >= MINIMUM_DELTA_CAPACITY


def start_process(program_to_scan: ProgramInterface) -> Tuple[psutil.Popen, int]:
    """
    This function creates a process that runs the given program
    :param program_to_scan: the program to run. Can be either the main program or background program
    :return: process object as returned by subprocesses popen and the pid of the process
    """
    global processes_ids

    program_to_scan.set_processes_ids(processes_ids)

    # create file for stdout text
    with open(f"{os.path.join(STDERR_FILES_DIR, f'{program_to_scan.get_program_name()} Stderr.txt')}", "a") as f_stderr:
        with open(f"{os.path.join(STDOUT_FILES_DIR, f'{program_to_scan.get_program_name()} Stdout.txt')}",
                  "a") as f_stdout:
            shell_process, pid = AbstractOSFuncs.popen(program_to_scan.get_command(), program_to_scan.find_child_id,
                                                       program_to_scan.should_use_powershell(), running_os.is_posix(),
                                                       program_to_scan.should_find_child_id(), f_stdout, f_stderr)

            f_stdout.write(f"Process ID: {pid}\n\n")
            f_stderr.write(f"Process ID: {pid}\n\n")

    # save the process names and pids in global arrays
    if pid is not None:
        processes_ids.append(pid)
        original_program_name = program_to_scan.get_program_name()
        iteration_num = len(finished_scanning_time) + 1
        processes_names.append(original_program_name if iteration_num == 1 else
                               f"{original_program_name} - iteration {iteration_num}")

    return shell_process, pid


def start_background_processes() -> List[Tuple[psutil.Popen, int]]:
    """
    Start a process per each background program using start_process function above.
    If there is an error when creating a process, terminate all process and notify user
    :return: list of tuples. each tuple contains a process object as returned by psutil popen
    and the pid of the process
    """
    background_processes = [start_process(background_program) for background_program in background_programs]
    # TODO: think how to check if there are errors without sleeping - waiting for process initialization
    time.sleep(5)

    for (background_process, child_process_id), background_program in zip(background_processes, background_programs):
        if background_process.poll() is not None:  # if process has not terminated
            err = background_process.stderr.read().decode()
            if err:
                terminate_due_to_exception(background_processes, background_program.get_program_name(), err)

    return background_processes


def terminate_due_to_exception(background_processes, program_name, err):
    """
    When an exception is raised from one of the processes, we will terminate all other process and stop measuring
    :param background_processes:
    :param program_name: the name of the program that had an error
    :param err: explanation about the error occurred
    """
    done_scanning_event.set()

    # terminate the main process if it still exists
    try:
        p = psutil.Process(main_process_id)
        p.terminate()  # or p.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    # terminate other processes (background processes)
    kill_background_processes(background_processes)

    # stop measuring - raise exception to user
    after_scanning_operations(should_save_results=False)
    raise Exception("An error occurred in child program %s: %s", program_name, err)


# TODO: unify all kill methods, and ensure that we are not missing setting done_scanning_event.set()
def kill_process(child_process_id, powershell_process):
    try:
        if child_process_id is None:
            powershell_process.kill()

        else:
            p = psutil.Process(child_process_id)
            p.terminate()  # or p.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def wait_and_write_running_time_to_file(child_process_id, powershell_process):
    try:
        if child_process_id is None:
            powershell_process.wait()

        else:
            p = psutil.Process(child_process_id)
            p.wait()  # or p.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def kill_background_processes(background_processes):
    waiting_threads = []
    for (powershell_process, child_process_id), background_program in zip(background_processes, background_programs):
        if kill_background_process_when_main_finished:
            kill_process(child_process_id, powershell_process)

        else:
            wait_to_process_thread = Thread(target=wait_and_write_running_time_to_file,
                                            args=(child_process_id, powershell_process))
            wait_to_process_thread.start()
            waiting_threads.append(wait_to_process_thread)

    for waiting_thread in waiting_threads:
        waiting_thread.join()
        # powershell_process.wait()


def start_timeout(main_shell_process):
    """
    This function terminates the main process if its running time exceeds maximum allowed time
    :param main_shell_process: the process to terminate  
    :return: a timer thread (as returned from Timer function)
    """
    if RUNNING_TIME is None or scan_option != ScanMode.ONE_SCAN:
        return

    def kill_and_terminate(process_id: int):
        running_os.kill_process_gracefully(process_id)
        done_scanning_event.set()

    timeout_thread = Timer(RUNNING_TIME, kill_and_terminate, [main_shell_process.pid])
    timeout_thread.start()
    return timeout_thread


def cancel_timeout_timer(timeout_timer):
    """
    This function cancels the timer thread that kills terminates the main program in case it
     exceeded maximum allowed running time
    :param timeout_timer: the timer thread returned from start_timeout
    """
    global max_timeout_reached

    if timeout_timer is None:
        return

    if not timeout_timer.is_alive():
        max_timeout_reached = True

    timeout_timer.cancel()
    timeout_timer.join()


def scan_and_measure():
    """
    The main function. This function starts a thread that will be responsible for measuring the resource
    consumption of the whole system and per each process. Simultaneously, it starts the main program and
    background programs defined by the user (so the thread will measure them also)
    """
    global main_process
    global main_process_id
    global max_timeout_reached

    measurements_thread = Thread(target=continuously_measure, args=())
    measurements_thread.start()

    while not main_program_to_scan == ProgramToScan.BASELINE_MEASUREMENT and not done_scanning_event.is_set():
        main_process, main_process_id = start_process(program)
        timeout_timer = start_timeout(main_process)
        background_processes = start_background_processes()
        processes_resource_usage_recorder.set_processes_to_mark([main_process] + background_processes)

        print("Waiting for the main process to terminate")
        result = running_os.wait_for_process_termination(main_process, done_scanning_event)

        cancel_timeout_timer(timeout_timer)

        # kill background programs after main program finished
        kill_background_processes(background_processes)

        finished_scanning_time.append(time_since_start())
        # check whether another iteration of scan is needed or not
        if scan_option == ScanMode.ONE_SCAN or (scan_time_passed() and is_delta_capacity_achieved()):
            # if there is no need in another iteration, exit this while and signal the measurement thread to stop
            done_scanning_event.set()
        if result and max_timeout_reached is False:
            print(result)
            print(main_process)
            # errs = main_process.stderr.read().decode()
            # after_scanning_operations(should_save_results=False)
            # raise Exception("An error occurred while scanning: %s", errs)
            # warnings.warn(f"An error occurred while scanning: {errs}", RuntimeWarning)
            warnings.warn(f"errors encountered while scanning, see stderr directory", RuntimeWarning)

    running_os.wait_for_thread_termination(measurements_thread, done_scanning_event)

    if main_program_to_scan == ProgramToScan.BASELINE_MEASUREMENT:
        finished_scanning_time.append(time_since_start())


def can_proceed_towards_measurements():
    """
    Check if user is aware that he is going to delete previous measurements
    :return: True if it is new measurement or if the user agreed to delete the previous measurements
    """
    if os.path.exists(base_dir):

        button_selected = running_os.message_box("Deleting Previous Results",
                                                 "Running the program will override the results of the previous measurement.\n\n"
                                                 "Are you sure you want to continue?", 4)

        if button_selected == YES_BUTTON:
            shutil.rmtree(base_dir)  # remove previous data
            return True
        else:
            return False
    else:
        return True


def initialize_total_cpu():
    """
    CPU measurements are relative to the previous measurement.
    For that reason, the first CPU measurement is performed for initialization and the received value of that first
    measurement is meaningless.
    """
    psutil.cpu_percent(percpu=True)  # first call to psutil with cpu is meaningless
    try:
        if is_inside_container:
            running_os.get_container_total_cpu_usage()  # first call is meaningless
    except NotImplementedError as e:
        print(f"Error occurred: {str(e)}")
        done_scanning_event.set()


def print_warnings_system_adjustments(exception: Exception):
    """
    Should be called upon receiving an exception during pre/post configurations of measured environment.
    For example, when screen brightness could not be modified
    :param exception: the received exception
    """
    print(f"Warning! {exception}")
    print("Warning! Ensure that the parameter is_inside_container is set to True if you run inside container")
    print("Warning! Skipping system adjustments (e.g., brightness, power settings")
    print("Warning! If you run inside WSL, these operations may not be supported")


def before_scanning_operations():
    """
    Pre-configuration of measured environment. For example - modify the screen brightness to a certain value.
    """
    battery_usage_recorder.check_if_battery_plugged()

    if disable_real_time_protection_during_measurement and running_os.is_tamper_protection_enabled():
        raise Exception("You must disable Tamper Protection manually so that the program could control real "
                        "time Protection")

    if not can_proceed_towards_measurements():  # avoid deleting previous measurements
        print("Exiting program")
        return

    try:
        if not is_inside_container:
            running_os.change_power_plan(chosen_power_plan_name, running_os.get_chosen_power_plan_identifier())

        if disable_real_time_protection_during_measurement:
            running_os.change_real_time_protection()

        if not is_inside_container:
            running_os.change_sleep_and_turning_screen_off_settings(NEVER_TURN_SCREEN_OFF, NEVER_GO_TO_SLEEP_MODE)
            import screen_brightness_control as sbc
            sbc.set_brightness(screen_brightness_level)

    # Assuming that if one of the operations is failed, the rest will probably fail too
    except Exception as e:
        print_warnings_system_adjustments(e)

    initialize_total_cpu()

    Path(GRAPHS_DIR).mkdir(parents=True, exist_ok=True)  # create empty results dirs

    Path(STDOUT_FILES_DIR).mkdir(parents=True, exist_ok=True)  # create empty results dirs

    Path(STDERR_FILES_DIR).mkdir(parents=True, exist_ok=True)  # create empty results dirs

    save_general_information_before_scanning()


def after_scanning_operations(should_save_results=True):
    """
    Post-configuration of measured environment. For example - restoring device's power plan to what it was prior
    this measurement.
    """
    if should_save_results:
        save_results_to_files()

    try:
        if not is_inside_container:
            running_os.change_power_plan(running_os.get_default_power_plan_name(),
                                         running_os.get_default_power_plan_identifier())  # return to default power plan

            running_os.change_sleep_and_turning_screen_off_settings()  # return to default - must be after changing power plan

        if disable_real_time_protection_during_measurement:
            running_os.change_real_time_protection(should_disable=False)

    # Assuming that if one of the operations is failed, the rest will probably fail too
    except Exception as e:
        print_warnings_system_adjustments(e)

    if max_timeout_reached:
        print("Scanned program reached the maximum time so we terminated it")


def get_starting_time() -> float:
    """
    This function assumes that if we have a directory located in PATH_FOR_SAVING_BEFORE_BATTERY_DEPLETION,
    it happened due to previous measurement in which the battery was running low.
    The file's content is a json containing:
    {
        session_id: <str>
        start_timestamp: <timestamp>
        end_timestamp: <timestamp>
    }
    :return: the current time minus the total time passed so far from the entire measurement session, even if it
    was previously stopped (for example, due to low battery).
    """
    if os.path.exists(BACKED_UP_SCANNING_TIMESTAMPS_PATH):
        print("NOTE! backup directory exists. It is used for preserving the state of unfinished measurement that"
              " was interrupted due to low battery.\n"
              "If you find it unnecessary - please remove the directory at:", BACKED_UP_SCANNING_TIMESTAMPS_PATH)

        with open(BACKED_UP_SCANNING_TIMESTAMPS_PATH, "r") as f:
            backed_up_data = json.load(f)
            if backed_up_data["session_id"] == session_id:
                if main_program_to_scan == ProgramToScan.BASELINE_MEASUREMENT:
                    print("WARNING! restoring backed-up state from previous unfinished measurement "
                          "is not supported in BASELINE_MEASUREMENT mode.\n ")
                else:
                    print("NOTE! assuming this measurement is a continuation of previously unfinished measurement that"
                          " was interrupted due to low battery")
                    return time.time() - (float(backed_up_data["end_timestamp"]) - float(backed_up_data["start_timestamp"]))
            else:
                print(
                    "WARNING! The current session ID does not match the measurement id "
                    "of the previous unfinished measurement. "
                    "Assuming this is a new, unrelated measurement.\n"
                    "To continue the previous measurement, re-run the program with the same session ID as an argument.\n"
                    "It is recommended to remove the backup directory at:", BACKED_UP_SCANNING_TIMESTAMPS_PATH,
                    "if you find it unuseful."
                )
    return time.time()


def main(user_args):
    global system_metrics_logger
    global process_metrics_logger
    global application_flow_logger
    global starting_time
    print("======== Process Monitor ========")
    print("Session id:", session_id)

    signal.signal(signal.SIGINT, handle_sigint)

    before_scanning_operations()

    logger_filter = ScannerLoggerFilter(
        session_id=session_id,
        hostname=AbstractOSFuncs.get_hostname(),
        user_defined_extras=user_args.logging_constant_extras
    )

    starting_time = get_starting_time()

    system_metrics_logger = get_measurement_logger(
        logger_name=LoggerName.SYSTEM_METRICS,
        custom_filter=logger_filter,
        logger_handler=get_elastic_logging_handler(elastic_username, elastic_password, elastic_url, IndexName.SYSTEM_METRICS, starting_time)
    )

    process_metrics_logger = get_measurement_logger(
        logger_name=LoggerName.PROCESS_METRICS,
        custom_filter=logger_filter,
        logger_handler=get_elastic_logging_handler(elastic_username, elastic_password, elastic_url, IndexName.PROCESS_METRICS, starting_time)
    )

    application_flow_logger = get_measurement_logger(
        logger_name=LoggerName.APPLICATION_FLOW,
        custom_filter=logger_filter,
        logger_handler=get_elastic_logging_handler(elastic_username, elastic_password, elastic_url, IndexName.APPLICATION_FLOW, starting_time)
    )

    application_flow_logger.info("The scanner is starting the measurement")

    scan_and_measure()

    after_scanning_operations()

    application_flow_logger.info("The scanner has finished measuring")

    print("Finished scanning")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This program periodically monitors the resource consumption of the device at the process-level."
                    "It supports CPU, RAN, DISK and NETWORK measurements"
    )

    parser.add_argument("--measurement_session_id",
                        type=str,
                        default=generate_id(word_count=3),
                        help="ip address to listen on")

    parser.add_argument("--logging_constant_extras",
                        type=json.loads,
                        default={},
                        help="User-defined extras (as JSON) to insert into every log")

    args = parser.parse_args()

    session_id = args.measurement_session_id

    main(args)
