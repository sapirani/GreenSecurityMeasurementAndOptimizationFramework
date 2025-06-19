import argparse
import json
import platform
import shutil
import signal
import threading
import time
import warnings
from functools import partial
from logging import LoggerAdapter

from statistics import mean

from human_id import generate_id
from prettytable import PrettyTable
from threading import Thread, Timer
import pandas as pd
from scapy.interfaces import get_working_ifaces

from application_logging import get_measurement_logger, ElasticSearchLogHandler, get_elastic_logging_handler
from application_logging.adapters.scanner_logger_adapter import ScannerLoggerAdapter
from initialization_helper import *
from datetime import date
from pathlib import Path
from general_functions import convert_mwh_to_other_metrics, calc_delta_capacity
from operating_systems.abstract_operating_system import AbstractOSFuncs
from process_connections import ProcessNetworkMonitor

base_dir, GRAPHS_DIR, STDOUT_FILES_DIR, STDERR_FILES_DIR, PROCESSES_CSV, TOTAL_MEMORY_EACH_MOMENT_CSV, \
    DISK_IO_EACH_MOMENT, NETWORK_IO_EACH_MOMENT, BATTERY_STATUS_CSV, GENERAL_INFORMATION_FILE, TOTAL_CPU_CSV, \
    SUMMARY_CSV = result_paths()

program.set_results_dir(base_dir)

# ======= Program Global Parameters =======
done_scanning_event = threading.Event()
starting_time = 0
main_process_id = None
max_timeout_reached = False
main_process = None
logger = None
session_id: str = ""

# include main programs and background
processes_ids = []
processes_names = []

# TODO: maybe its better to calculate MEMORY(%) in the end of scan in order to reduce calculations during scanning
processes_df = pd.DataFrame(columns=processes_columns_list)

memory_df = pd.DataFrame(columns=memory_columns_list)

disk_io_each_moment_df = pd.DataFrame(columns=disk_io_columns_list)

network_io_each_moment_df = pd.DataFrame(columns=network_io_columns_list)

battery_df = pd.DataFrame(columns=battery_columns_list)

cpu_df = pd.DataFrame(columns=cpu_columns_list)

finished_scanning_time = []


def handle_sigint(signum, frame):
    print("Got signal, writing results and terminating")
    if main_process:
        program.kill_process(main_process, running_os.is_posix())  # killing the main process
    done_scanning_event.set()


def save_current_total_memory():
    """_summary_: take memory information and append it to a dataframe
    """
    if is_inside_container:
        memory_used_bytes, memory_used_percent = running_os.get_container_total_memory_usage()
    else:
        vm = psutil.virtual_memory()
        memory_used_bytes, memory_used_percent = vm.used, vm.percent

    logger.info(
        "Total memory measurement",
        extra={"total_memory_gb": memory_used_bytes / GB, "total_memory_percent": memory_used_percent}
    )

    memory_df.loc[len(memory_df.index)] = [
        scanner_imp.calc_time_interval(starting_time),
        f'{memory_used_bytes / GB:.3f}',
        memory_used_percent
    ]


def dataframe_append(df, element):
    """_summary_: append an element to a dataframe

    Args:
        df : dataframe to append to
        element (): element to append
    """
    df.loc[len(df.index)] = element


def save_current_disk_io(previous_disk_io):
    """_summary_: take disk io information and append it to a dataframe

    Args:
        previous_disk_io : previous disk io information

    Returns:
        disk_io_stat: psutil.disk_io_counters
    """
    disk_io_stat = psutil.disk_io_counters()
    disk_io_each_moment_df.loc[len(disk_io_each_moment_df.index)] = [
        scanner_imp.calc_time_interval(starting_time),
        disk_io_stat.read_count - previous_disk_io.read_count,
        disk_io_stat.write_count - previous_disk_io.write_count,
        f'{(disk_io_stat.read_bytes - previous_disk_io.read_bytes) / KB:.3f}',
        f'{(disk_io_stat.write_bytes - previous_disk_io.write_bytes) / KB:.3f}',
        disk_io_stat.read_time - previous_disk_io.read_time,
        disk_io_stat.write_time - previous_disk_io.write_time
    ]

    logger.info(
        "Total disk measurements",
        extra={
            "disk_read_count": disk_io_stat.read_count - previous_disk_io.read_count,
            "disk_write_count": disk_io_stat.write_count - previous_disk_io.write_count,
            "disk_read_bytes": (disk_io_stat.read_bytes - previous_disk_io.read_bytes) / KB,
            "disk_write_bytes": (disk_io_stat.write_bytes - previous_disk_io.write_bytes) / KB,
            "disk_read_time": disk_io_stat.read_time - previous_disk_io.read_time,
            "disk_write_time": disk_io_stat.write_time - previous_disk_io.write_time
        }
    )

    return disk_io_stat


def save_current_network_io(previous_network_io):
    network_io_stat = psutil.net_io_counters()

    dataframe_append(
        network_io_each_moment_df,
        [
            scanner_imp.calc_time_interval(starting_time),
            network_io_stat.packets_sent - previous_network_io.packets_sent,
            network_io_stat.packets_recv - previous_network_io.packets_recv,
            f'{(network_io_stat.bytes_sent - previous_network_io.bytes_sent) / KB:.3f}',
            f'{(network_io_stat.bytes_recv - previous_network_io.bytes_recv) / KB:.3f}',
        ]
    )

    logger.info(
        "Total network measurements",
        extra={
            "packets_sent": network_io_stat.packets_sent - previous_network_io.packets_sent,
            "packets_received": network_io_stat.packets_recv - previous_network_io.packets_recv,
            "bytes_sent": (network_io_stat.bytes_sent - previous_network_io.bytes_sent) / KB,
            "bytes_received": (network_io_stat.bytes_recv - previous_network_io.bytes_recv) / KB
        }
    )

    return network_io_stat


def save_current_processes_statistics(prev_data_per_process, process_network_monitor):
    """
    This function gets all processes running in the system and order them by thier cpu usage
    :param prev_data_per_process: previous read of all processes from io_counters.
    It is dictionary where the key is (pid, name) and the value is the io_counters() read
    :return: a new dictionary that contains the new values from io_counters() for each process
    """
    proc = []

    time_of_sample = scanner_imp.calc_time_interval(starting_time)

    for p in psutil.process_iter():
        try:
            if program.process_ignore_cond(p):  # ignore System Idle Process
                continue

            # trigger cpu_percent() the first time will lead to return of 0.0
            cpu_percent = p.cpu_percent() / NUMBER_OF_CORES
            proc.append((p, cpu_percent))

        except Exception:
            pass

    proc = sorted(proc, key=lambda x: x[1], reverse=True)

    return add_to_processes_dataframe(time_of_sample, proc, prev_data_per_process, process_network_monitor)


def add_to_processes_dataframe(time_of_sample, top_list, prev_data_per_process, process_network_monitor):
    """
    This function saves the relevant data from the process in dataframe (will be saved later as csv files)
    :param time_of_sample: time since starting the program
    :param top_list: list of all processes sorted by cpu usage
    :param prev_data_per_process: previous read of all processes from io_counters.
    It is dictionary where the key is (pid, name) and the value is the io_counters() read
    :return:
    """
    for p, cpu_percent in top_list:

        # While fetching the processes, some subprocesses may exit
        # Hence we need to put this code in try-except block
        try:
            # oneshot to improve info retrieve efficiency
            with p.oneshot():
                process_traffic = process_network_monitor.get_network_stats(p)

                io_stat = p.io_counters()
                page_faults = running_os.get_page_faults(p)

                if (p.pid, p.name()) not in prev_data_per_process:
                    prev_data_per_process[(p.pid, p.name())] = io_stat, page_faults
                    continue  # remove first sample of process (because cpu_percent is meaningless 0)

                prev_io = prev_data_per_process[(p.pid, p.name())][0]
                # TODO - does io_counters return only disk operations or all io operations (include network etc..)
                processes_df.loc[len(processes_df.index)] = [
                    time_of_sample,
                    p.pid,
                    p.name(),
                    f'{cpu_percent:.2f}',
                    p.num_threads(),
                    f'{p.memory_info().rss / MB:.3f}',  # TODO: maybe should use uss/pss instead rss?
                    round(p.memory_percent(), 2),
                    io_stat.read_count - prev_io.read_count,
                    io_stat.write_count - prev_io.write_count,
                    f'{(io_stat.read_bytes - prev_io.read_bytes) / KB:.3f}',
                    f'{(io_stat.write_bytes - prev_io.write_bytes) / KB:.3f}',
                    page_faults - prev_data_per_process[(p.pid, p.name())][1],
                    process_traffic.bytes_sent / KB,
                    process_traffic.packets_sent,
                    process_traffic.bytes_received / KB,
                    process_traffic.packets_received
                ]

                prev_data_per_process[(p.pid, p.name())] = io_stat, page_faults  # after finishing loop

                logger.info(
                    "Process measurements",
                    extra={
                        "pid": p.pid,
                        "process_name": p.name(),
                        "cpu_percent": cpu_percent,
                        "threads_num": p.num_threads(),
                        "used_memory_mb": p.memory_info().rss / MB,
                        "used_memory_percent": round(p.memory_percent(), 2),
                        "disk_read_count": io_stat.read_count - prev_io.read_count,
                        "disk_write_count": io_stat.write_count - prev_io.write_count,
                        "disk_read_bytes": (io_stat.read_bytes - prev_io.read_bytes) / KB,
                        "disk_write_bytes": (io_stat.write_bytes - prev_io.write_bytes) / KB,
                        "page_faults": page_faults - prev_data_per_process[(p.pid, p.name())][1],
                        "bytes_sent": process_traffic.bytes_sent / KB,
                        "packets_sent": process_traffic.packets_sent,
                        "bytes_received": process_traffic.bytes_received / KB,
                        "packets_received": process_traffic.packets_received
                    }
                )

        # Note, we are just ignoring access denied and other exceptions and do not handle them.
        # There will be no results for those processes
        except (psutil.NoSuchProcess, psutil.AccessDenied, ChildProcessError):
            pass

    return prev_data_per_process


def scan_time_passed():
    """_summary_: check if the minimum scan time has passed

    Returns:
        bool: True if the minimum scan time has passed, False otherwise
    """
    return time.time() - starting_time >= RUNNING_TIME


def save_data_when_too_low_battery():
    with open(GENERAL_INFORMATION_FILE, 'a') as f:
        f.write("EARLY TERMINATION DUE TO LOW BATTERY!!!!!!!!!!\n\n")
    finished_scanning_time.append(scanner_imp.calc_time_interval(starting_time))
    save_results_to_files()


def should_scan():
    """_summary_: check what is the scan option

    Returns:
        True if measurement thread should perform another iteration or False if it should terminate
    """
    if scanner_imp.is_battery_too_low(battery_df):
        save_data_when_too_low_battery()
        return False

    if main_program_to_scan == ProgramToScan.NO_SCAN:
        return not scan_time_passed() and not done_scanning_event.is_set()
    elif scan_option == ScanMode.ONE_SCAN:
        return not done_scanning_event.is_set()
    elif scan_option == ScanMode.CONTINUOUS_SCAN:
        return not (scan_time_passed() and is_delta_capacity_achieved()) and not done_scanning_event.is_set()
        # return not scan_time_passed() and not is_delta_capacity_achieved()


def save_current_total_cpu():
    """
    This function saves the total cpu usage of the system
    """
    total_cpu_per_core = psutil.cpu_percent(percpu=True)
    if is_inside_container:
        total_cpu_sum = running_os.get_container_total_cpu_usage()
        number_of_cores = running_os.get_container_number_of_cores()
        total_cpu_mean = total_cpu_sum / number_of_cores
        total_cpu_val = total_cpu_sum
    else:
        total_cpu_mean = mean(total_cpu_per_core)
        total_cpu_sum = sum(total_cpu_per_core)
        number_of_cores = len(total_cpu_per_core)
        total_cpu_val = total_cpu_mean

    cpu_df.loc[len(cpu_df.index)] = [scanner_imp.calc_time_interval(starting_time), total_cpu_val] + total_cpu_per_core

    logger.info(
        "Total CPU measurements",
        extra={
            "mean_cpu_across_cores_percent": total_cpu_mean,
            "sum_cpu_across_cores_percent": total_cpu_sum,
            "number_of_cores": number_of_cores,
            **{f"core{core_index}_percent": core_cpu_usage for core_index, core_cpu_usage in
               enumerate(total_cpu_per_core)}
        }
    )


def continuously_measure():
    """
    This function runs in a different thread. It accounts for measuring the full resource consumption of the system
    """
    running_os.init_thread()

    # init prev_disk_io by first disk io measurements (before scan)
    # TODO: lock thread until process starts
    prev_disk_io = psutil.disk_io_counters()
    prev_network_io = psutil.net_io_counters()
    prev_data_per_process = {}

    # NOTE: CHANGE THIS IF YOU WANT TO MONITOR SPECIFIC INTERFACES
    interfaces_for_packets_capturing = get_working_ifaces()
    process_network_monitor = ProcessNetworkMonitor(interfaces_for_packets_capturing)

    # TODO: think if total tables should be printed only once
    while should_scan():
        # Create a delay
        scanner_imp.scan_sleep(SLEEP_BETWEEN_ITERATIONS_SECONDS)

        scanner_imp.save_battery_stat(battery_df, scanner_imp.calc_time_interval(starting_time))
        prev_data_per_process = save_current_processes_statistics(prev_data_per_process, process_network_monitor)

        try:  # in case of measuring cpu in a windows container
            save_current_total_cpu()
        except NotImplementedError as e:
            print(f"Error occurred: {str(e)}")
            done_scanning_event.set()

        save_current_total_memory()
        prev_disk_io = save_current_disk_io(prev_disk_io)
        prev_network_io = save_current_network_io(prev_network_io)

    process_network_monitor.stop()


def save_general_disk(f):
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


def save_general_system_information(f):
    """
    This function saves general info about the platform to a file.
    :param f:
    :return:
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
        f.write(f'Scanner Version: {get_scanner_version_name(scanner_version)}\n\n')

        # TODO: add background_programs general_information_before_measurement(f)
        program.general_information_before_measurement(f)

        save_general_system_information(f)

        f.write('\n======Before Scanning======\n')
        scanner_imp.save_general_battery(f)
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
            battery_drop = calc_delta_capacity(battery_df)
            f.write(f'Amount of Battery Drop: {battery_drop[0]} mWh, {battery_drop[1]}%\n')
            f.write('Approximately equivalent to -\n')
            conversions = convert_mwh_to_other_metrics(battery_drop[0])
            f.write(f'  CO2 emission: {conversions[0]} kg\n')
            f.write(f'  Coal burned: {conversions[1]} kg\n')
            f.write(f'  Number of smartphone charged: {conversions[2]}\n')
            f.write(f'  Kilograms of wood burned: {conversions[3]}\n')

        if main_program_to_scan == ProgramToScan.NO_SCAN:
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
    summary_df = summary_version_imp.prepare_summary_csv(processes_df, cpu_df, memory_df, disk_io_each_moment_df,
                                                         network_io_each_moment_df,
                                                         battery_df, processes_names, finished_scanning_time,
                                                         processes_ids)

    color_func = summary_version_imp.colors_func

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

    return calc_delta_capacity(battery_df)[0] >= MINIMUM_DELTA_CAPACITY


def start_process(program_to_scan):
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


def start_background_processes():
    """
    Start a process per each background program using start_process function above.
    If there is an error when creating a process, terminate all process and notify user
    :return: list of tuples. each tuple contains a process object as returned by subprocesses popen
    and the pid of the process
    """
    background_processes = [start_process(background_program) for background_program in background_programs]
    # TODO: think how to check if there are errors without sleeping - waiting for process initialization
    scanner_imp.scan_sleep(5)

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

    print(scanner_imp.calc_time_interval(starting_time))


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


def start_timeout(main_shell_process, is_posix):
    """
    This function terminates the main process if its running time exceeds maximum allowed time
    :param main_shell_process: the process to terminate  
    :return: a timer thread (as returned from Timer function)
    """
    if RUNNING_TIME is None or scan_option != ScanMode.ONE_SCAN:
        return

    timeout_thread = Timer(RUNNING_TIME, program.kill_process, [main_shell_process, is_posix])
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
    global starting_time
    global main_process_id
    global max_timeout_reached
    starting_time = time.time()

    measurements_thread = Thread(target=continuously_measure, args=())
    measurements_thread.start()

    while not main_program_to_scan == ProgramToScan.NO_SCAN and not done_scanning_event.is_set():
        main_process, main_process_id = start_process(program)
        timeout_timer = start_timeout(main_process, running_os.is_posix())
        background_processes = start_background_processes()
        print("Waiting for the main process to terminate")
        result = main_process.wait()

        cancel_timeout_timer(timeout_timer)

        # kill background programs after main program finished
        kill_background_processes(background_processes)

        finished_scanning_time.append(scanner_imp.calc_time_interval(starting_time))
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

    running_os.wait_for_measurement_termination(measurements_thread, done_scanning_event)

    if main_program_to_scan == ProgramToScan.NO_SCAN:
        finished_scanning_time.append(scanner_imp.calc_time_interval(starting_time))


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
    psutil.cpu_percent(percpu=True)  # first call to psutil with cpu is meaningless
    try:
        if is_inside_container:
            running_os.get_container_total_cpu_usage()  # first call is meaningless
    except NotImplementedError as e:
        print(f"Error occurred: {str(e)}")
        done_scanning_event.set()


def print_warnings_system_adjustments(exception: Exception):
    print(f"Warning! {exception}")
    print("Warning! Ensure that the parameter is_inside_container is set to True if you run inside container")
    print("Warning! Skipping system adjustments (e.g., brightness, power settings")
    print("Warning! If you run inside WSL, these operations may not be supported")


def before_scanning_operations():
    scanner_imp.check_if_battery_plugged()

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


def main():
    print("======== Process Monitor ========")
    print("Session id:", session_id)

    signal.signal(signal.SIGINT, handle_sigint)

    before_scanning_operations()

    scan_and_measure()

    after_scanning_operations()

    logger.info("The scanner has finished measuring")

    print("Finished scanning")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This program periodically monitors the resource consumption of the device at the process-level."
                    "It supports CPU, RAN, DISK and NETWORK measurements"
    )

    parser.add_argument("-i" ,"--measurement_session_id",
                        type=str,
                        default=generate_id(word_count=3),
                        help="ip address to listen on")

    parser.add_argument("--logging_constant_extras",
                        type=json.loads,
                        default={},
                        help="User-defined extras (as JSON) to insert into every log")

    args = parser.parse_args()

    session_id = args.measurement_session_id

    logger_adapter = partial(
        ScannerLoggerAdapter,
        session_id=session_id,
        hostname=AbstractOSFuncs.get_hostname(),
        user_defined_extras=args.logging_constant_extras
    )

    logger = get_measurement_logger(
        logger_adapter,
        get_elastic_logging_handler(elastic_username, elastic_password, elastic_url)
    )

    logger.info("The scanner is starting the measurement")

    main()
