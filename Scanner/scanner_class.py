import random
import shutil
import sys
import warnings
import screen_brightness_control as sbc
from statistics import mean
from prettytable import PrettyTable
from threading import Thread, Timer
import pandas as pd
import logging
import time
import os
import psutil
import platform
from datetime import date, datetime
from pathlib import Path
from general_functions import convert_mwh_to_other_metrics, calc_delta_capacity

sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/Scanner')
from initialization_helper_class import *
import logging
logger = logging.getLogger(__name__)

class Scanner:
    def __init__(self):
        self.initialize_variables()
        self.initialize_dataframes()
        

    def initialize_variables(self):
        self.running_os, self.scanner_imp, self.summary_version_imp, self.chosen_power_plan_name, \
        self.chosen_power_plan_guid, self.chosen_power_plan_linux_identifier, self.balanced_power_plan_name, \
        self.balanced_power_plan_guid, self.power_save_plan_name, self.power_save_plan_identifier, \
        self.program, self.background_programs, self.base_dir, self.calc_measurement_number, \
        self.result_paths, self.battery_columns_list, self.memory_columns_list, self.cores_names_list, \
        self.cpu_columns_list, self.disk_io_columns_list, self.processes_columns_list = initialize_variables()

        self.base_dir, self.GRAPHS_DIR, self.STDOUT_FILES_DIR, self.PROCESSES_CSV, \
        self.TOTAL_MEMORY_EACH_MOMENT_CSV, self.DISK_IO_EACH_MOMENT, \
        self.BATTERY_STATUS_CSV, self.GENERAL_INFORMATION_FILE, self.TOTAL_CPU_CSV, \
        self.SUMMARY_CSV, self.HARDWARE_CSV = self.result_paths()
        self.program.set_results_dir(self.base_dir)

        self.done_scanning = False
        self.starting_time = 0
        self.main_process_id = None
        self.max_timeout_reached = False
        self.processes_ids = []
        self.processes_names = []

    def initialize_dataframes(self):
        self.processes_df = pd.DataFrame(columns=self.processes_columns_list)
        self.memory_df = pd.DataFrame(columns=self.memory_columns_list)
        self.disk_io_each_moment_df = pd.DataFrame(columns=self.disk_io_columns_list)
        self.battery_df = pd.DataFrame(columns=self.battery_columns_list)
        self.cpu_df = pd.DataFrame(columns=self.cpu_columns_list)
        self.hardware_df = pd.DataFrame()
        self.finished_scanning_time = []

    def save_current_total_memory(self):
        """_summary_: take memory information and append it to a dataframe
        """
        vm = psutil.virtual_memory()
        self.memory_df.loc[len(self.memory_df.index)] = [
            self.scanner_imp.calc_time_interval(self.starting_time),
            f'{vm.used / GB:.3f}',
            vm.percent
        ]

    def dataframe_append(self, df, element):
        """_summary_: append an element to a dataframe

        Args:
            df : dataframe to append to
            element (): element to append
        """
        df.loc[len(df.index)] = element

    def save_current_disk_io(self, previous_disk_io):
        """_summary_: take disk io information and append it to a dataframe

        Args:
            previous_disk_io : previous disk io information

        Returns:
            disk_io_stat: psutil.disk_io_counters
        """
        disk_io_stat = psutil.disk_io_counters()
        self.disk_io_each_moment_df.loc[len(self.disk_io_each_moment_df.index)] = [
            self.scanner_imp.calc_time_interval(self.starting_time),
            disk_io_stat.read_count - previous_disk_io.read_count,
            disk_io_stat.write_count - previous_disk_io.write_count,
            f'{(disk_io_stat.read_bytes - previous_disk_io.read_bytes) / KB:.3f}',
            f'{(disk_io_stat.write_bytes - previous_disk_io.write_bytes) / KB:.3f}',
            disk_io_stat.read_time - previous_disk_io.read_time,
            disk_io_stat.write_time - previous_disk_io.write_time
        ]

        return disk_io_stat

    def save_current_processes_statistics(self, prev_data_per_process):
        """
        This function gets all processes running in the system and order them by thier cpu usage
        :param prev_data_per_process: previous read of all processes from io_counters.
        It is dictionary where the key is (pid, name) and the value is the io_counters() read
        :return: a new dictionary that contains the new values from io_counters() for each process
        """
        proc = []

        time_of_sample = self.scanner_imp.calc_time_interval(self.starting_time)

        for p in psutil.process_iter():
            try:
                if self.program.process_ignore_cond(p):  # ignore System Idle Process
                    continue

                # trigger cpu_percent() the first time will lead to return of 0.0
                cpu_percent = p.cpu_percent() / NUMBER_OF_CORES
                proc.append((p, cpu_percent))

            except Exception:
                pass

        proc = sorted(proc, key=lambda x: x[1], reverse=True)
        return self.add_to_processes_dataframe(time_of_sample, proc, prev_data_per_process)

    def add_to_processes_dataframe(self, time_of_sample, top_list, prev_data_per_process):
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
                    io_stat = p.io_counters()
                    try:
                        page_faults = self.running_os.get_page_faults(p)
                    except Exception:
                        page_faults = 0

                    if (p.pid, p.name()) not in prev_data_per_process:
                        prev_data_per_process[(p.pid, p.name())] = io_stat, page_faults
                        continue  # remove first sample of process (because cpu_percent is meaningless 0)

                    prev_io = prev_data_per_process[(p.pid, p.name())][0]
                    # TODO - does io_counters return only disk operations or all io operations (include network etc..)
                    self.processes_df.loc[len(self.processes_df.index)] = [
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
                        page_faults - prev_data_per_process[(p.pid, p.name())][1]
                    ]

                    prev_data_per_process[(p.pid, p.name())] = io_stat, page_faults  # after finishing loop

            except psutil.NoSuchProcess:
                pass

        return prev_data_per_process

    def scan_time_passed(self):
        """_summary_: check if the minimum scan time has passed

        Returns:
            bool: True if the minimum scan time has passed, False otherwise
        """
        return time.time() - self.starting_time >= RUNNING_TIME

    def save_data_when_too_low_battery(self):
        with open(self.GENERAL_INFORMATION_FILE, 'a') as f:
            f.write("EARLY TERMINATION DUE TO LOW BATTERY!!!!!!!!!!\n\n")
        self.finished_scanning_time.append(self.FullScanner.calc_time_interval(self.scanner_imp, self.starting_time))
        self.save_results_to_files()
        
    def should_scan(self):
        """_summary_: check what is the scan option

        Returns:
            True if measurement thread should perform another iteration or False if it should terminate
        """
        if self.scanner_imp.is_battery_too_low(self.battery_df):
            self.save_data_when_too_low_battery()
            return False
        # if os.path.exists(r"/home/shouei/GreenSecurity-FirstExperiment/should_scan.txt"):
        #     with open(r"/home/shouei/GreenSecurity-FirstExperiment/should_scan.txt", 'r') as f:
        #         line = f.read()
        #         logger.info(f"Read line: {line}")
        #     if line == "save":
        #         self.save_results_to_files()
        #         os.remove(r"/home/shouei/GreenSecurity-FirstExperiment/should_scan.txt")
        #     if line == "finished":
        #         os.remove(r"/home/shouei/GreenSecurity-FirstExperiment/should_scan.txt")
        #         return False
        # if self.done_scanning:
        #     logger.info("Done scanning")
        #     return False
        if main_program_to_scan in no_process_programs:
            self.save_results_to_files()
            return not self.scan_time_passed()
        elif self.scan_option == self.ScanMode.ONE_SCAN:
            return not self.done_scanning
        elif self.scan_option == self.ScanMode.CONTINUOUS_SCAN:
            return not (self.scan_time_passed() and self.is_delta_capacity_achieved())
            # return not scan_time_passed() and not is_delta_capacity_achieved()

    def save_current_total_cpu(self):
        """
        This function saves the total cpu usage of the system
        """
        total_cpu = psutil.cpu_percent(percpu=True)
        self.cpu_df.loc[len(self.cpu_df.index)] = [self.scanner_imp.calc_time_interval(self.starting_time), mean(total_cpu)] + total_cpu

    def continuously_measure(self, conn):
        """
        This function runs in a different thread. It accounts for measuring the full resource consumption of the system
        """
        self.running_os.init_thread()
        try:
            # init prev_disk_io by first disk io measurements (before scan)
            # TODO: lock thread until process starts
            prev_disk_io = psutil.disk_io_counters()
            prev_data_per_process = {}
            msg = "keep going"
            # TODO: think if total tables should be printed only once
            while self.should_scan() and msg != "stop":
                # Create a delay
                if conn.poll():
                    msg = conn.recv()   
                self.scanner_imp.scan_sleep(0.5)

                self.scanner_imp.save_battery_stat(self.battery_df, self.scanner_imp.calc_time_interval(self.starting_time))
                prev_data_per_process = self.save_current_processes_statistics(prev_data_per_process)
                self.save_current_total_cpu()
                self.save_current_total_memory()
                prev_disk_io = self.save_current_disk_io(prev_disk_io)
        except Exception as e:
            conn.send(f"error: {e}")

    def save_general_disk(self, f):
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
            f'{disk_stat.used /GB:.3f}',
            f'{disk_stat.free / GB:.3f}',
            disk_stat.percent
        ])
        f.write(str(disk_table))
        f.write('\n')

    def save_general_system_information(self, f):
        """
        This function saves general info about the platform to a file.
        :param f:
        :return:
        """
        platform_system = platform.uname()

        f.write("======System Information======\n")

        self.hardware_df = self.running_os.save_system_information(f, self.hardware_df)

        f.write(f"Machine Type: {platform_system.machine}\n")
        f.write(f"Device Name: {platform_system.node}\n")

        self.hardware_df[HardwareColumns.MACHINE_TYPE] = [platform_system.machine]
        self.hardware_df[HardwareColumns.DEVICE_NAME] = [platform_system.node]

        f.write("\n----Operating System Information----\n")
        f.write(f"Operating System: {platform_system.system}\n")
        f.write(f"Release: {platform_system.release}\n")
        f.write(f"Version: {platform_system.version}\n")

        self.hardware_df[HardwareColumns.OPERATING_SYSTEM] = [platform_system.system]
        self.hardware_df[HardwareColumns.OPERATING_SYSTEM_RELEASE] = [platform_system.release]
        self.hardware_df[HardwareColumns.OPERATING_SYSTEM_VERSION] = [platform_system.version]

        f.write("\n----CPU Information----\n")
        f.write(f"Processor: {platform_system.processor}\n")

        number_of_physical_cores = psutil.cpu_count(logical=False)
        f.write(f"Physical cores: {number_of_physical_cores}\n")
        f.write(f"Total cores: {NUMBER_OF_CORES}\n")
        cpufreq = psutil.cpu_freq()
        f.write(f"Max Frequency: {cpufreq.max:.2f} MHz\n")
        f.write(f"Min Frequency: {cpufreq.min:.2f} MHz\n")

        self.hardware_df[HardwareColumns.PROCESSOR_NAME] = [platform_system.processor]
        self.hardware_df[HardwareColumns.PROCESSOR_PHYSICAL_CORES] = [number_of_physical_cores]
        self.hardware_df[HardwareColumns.PROCESSOR_TOTAL_CORES] = [NUMBER_OF_CORES]
        self.hardware_df[HardwareColumns.PROCESSOR_MAX_FREQ] = [cpufreq.max]
        self.hardware_df[HardwareColumns.PROCESSOR_MIN_FREQ] = [cpufreq.min]

        f.write("\n----RAM Information----\n")
        total_ram = psutil.virtual_memory().total / GB
        f.write(f"Total RAM: {total_ram} GB\n")

        self.hardware_df[HardwareColumns.TOTAL_RAM] = [total_ram]

        self.running_os.save_physical_memory(f)
        self.running_os.save_disk_information(f, self.hardware_df)

    def save_general_information_before_scanning(self):
        """
        This function writes general battery, disk, ram, os, etc. information
        """
        with open(self.GENERAL_INFORMATION_FILE, 'w') as f:
            # dd/mm/YY
            f.write(f'Date: {date.today().strftime("%d/%m/%Y")}\n')
            f.write(f'Scanner Version: {get_scanner_version_name(scanner_version)}\n\n')

            # TODO: add background_programs general_information_before_measurement(f)
            self.program.general_information_before_measurement(f)

            self.save_general_system_information(f)

            f.write('\n======Before Scanning======\n')
            self.hardware_df = self.scanner_imp.save_general_battery(f, self.hardware_df)
            f.write('\n')
            self.save_general_disk(f)
            f.write('\n\n')

        self.hardware_df.to_csv(self.HARDWARE_CSV)


    def main(self, conn):
        print("======== Process Monitor ========")
        self.before_scanning_operations()
        sys.stdout.flush()
        self.scan_and_measure(conn)
        sys.stdout.flush()
        self.after_scanning_operations()
        print("Finished scanning")
        sys.stdout.flush()
        # return self.processes_df
    
    def main1(self):
        print("======== Process Monitor ========")

        self.before_scanning_operations()
        self.continuously_measure()
        self.after_scanning_operations()
        print("Finished scanning")
        
        

    def before_scanning_operations(self):
        self.scanner_imp.check_if_battery_plugged()

        if disable_real_time_protection_during_measurement and self.running_os.is_tamper_protection_enabled():
            raise Exception("You must disable Tamper Protection manually so that the program could control real "
                            "time Protection")

        if not self.can_proceed_towards_measurements():  # avoid deleting previous measurements
            print("Exiting program")
            return

        # shimon - turned off this line because it is not working on the vm
        # running_os.change_power_plan(chosen_power_plan_name, running_os.get_chosen_power_plan_identifier())

        if disable_real_time_protection_during_measurement:
            self.running_os.change_real_time_protection()

        # shimon - turned off this line because it is not working on the vm
        # running_os.change_sleep_and_turning_screen_off_settings(NEVER_TURN_SCREEN_OFF, NEVER_GO_TO_SLEEP_MODE)
        # sbc.set_brightness(screen_brightness_level)

        psutil.cpu_percent()  # first call is meaningless

        Path(self.GRAPHS_DIR).mkdir(parents=True, exist_ok=True)  # create empty results dirs

        Path(self.STDOUT_FILES_DIR).mkdir(parents=True, exist_ok=True)  # create empty results dirs

        self.save_general_information_before_scanning()

    def after_scanning_operations(self, should_save_results=True):
        if should_save_results:
            self.save_results_to_files()

        # shimon - turned off this line because it is not working on the vm
        # running_os.change_power_plan(running_os.get_default_power_plan_name(),
        #                               running_os.get_default_power_plan_identifier())  # return to default power plan
        # running_os.change_sleep_and_turning_screen_off_settings()  # return to default - must be after changing power plan

        if disable_real_time_protection_during_measurement:
            self.running_os.change_real_time_protection(should_disable=False)

        if self.max_timeout_reached:
            print("Scanned program reached the maximum time so we terminated it")

    def can_proceed_towards_measurements(self):
        """
        Check if user is aware that he is going to delete previous measurements
        :return: True if it is new measurement or if the user agreed to delete the previous measurements
        """
        if os.path.exists(self.base_dir):
            try:
                button_selected = self.running_os.message_box("Deleting Previous Results",
                                                        "Running the program will override the results of the previous measurement.\n\n"
                                                        "Are you sure you want to continue?", 4)
            except Exception:
                button_selected = YES_BUTTON
            if button_selected == YES_BUTTON:
                shutil.rmtree(self.base_dir)  # remove previous data
                return True
            else:
                return False
        else:
            return True

    def save_results_to_files(self):
        """
        Save all measurements (cpu, memory, disk battery) into dedicated files.
        """
        if self.finished_scanning_time:
            self.save_general_information_after_scanning()
            self.ignore_last_results()

        self.processes_df.to_csv(self.PROCESSES_CSV, index=False)
        self.memory_df.to_csv(self.TOTAL_MEMORY_EACH_MOMENT_CSV, index=False)
        self.disk_io_each_moment_df.to_csv(self.DISK_IO_EACH_MOMENT, index=False)
        if not self.battery_df.empty:
            self.battery_df.to_csv(self.BATTERY_STATUS_CSV, index=False)
        self.cpu_df.to_csv(self.TOTAL_CPU_CSV, index=False)
        if self.finished_scanning_time:
            self.prepare_summary_csv()

    def is_delta_capacity_achieved(self):
        """
        Relevant for Continuous Scan
        :return: True if the minimum capacity drain specified by the user is achieved and False otherwise.
        The meaning of False is that another scan should be performed
        """
        if psutil.sensors_battery() is None:  # if desktop computer (has no battery)
            return True

        return calc_delta_capacity(self.battery_df)[0] >= MINIMUM_DELTA_CAPACITY

    def scan_and_measure(self, conn):
        """
        The main function. This function starts a thread that will be responsible for measuring the resource
        consumption of the whole system and per each process. Simultaneously, it starts the main program and
        background programs defined by the user (so the thread will measure them also)
        """
        self.starting_time = time.time()

        measurements_thread = Thread(target=self.continuously_measure, args=(conn,))
        measurements_thread.start()
        print(f"Starting measurement thread {datetime.now()}")
        while not main_program_to_scan in no_process_programs and not self.done_scanning:
            main_process, self.main_process_id = self.start_process(self.program)
            timeout_timer = self.start_timeout(main_process, self.running_os.is_posix())
            background_processes = self.start_background_processes()
            result = main_process.wait()

            self.cancel_timeout_timer(timeout_timer)

            # kill background programs after main program finished
            self.kill_background_processes(background_processes)

            self.finished_scanning_time.append(self.scanner_imp.calc_time_interval(self.starting_time))
            # check whether another iteration of scan is needed or not
            if self.scan_option == self.ScanMode.ONE_SCAN or (self.scan_time_passed() and self.is_delta_capacity_achieved()):
                # if there is no need in another iteration, exit this while and signal the measurement thread to stop
                self.done_scanning = True
            if result and self.max_timeout_reached is False:
                print(result)
                print(main_process)
                errs = main_process.stderr.read().decode()
                self.after_scanning_operations(should_save_results=False)
                # raise Exception("An error occurred while scanning: %s", errs)
                warnings.warn(f"An error occurred while scanning: {errs}", RuntimeWarning)

        # wait for measurement
        measurements_thread.join()
        
        if main_program_to_scan in no_process_programs:
            self.finished_scanning_time.append(self.scanner_imp.calc_time_interval(self.starting_time))

    def start_process(self, program_to_scan):
        """
        This function creates a process that runs the given program
        :param program_to_scan: the program to run. Can be either the main program or background program
        :return: process object as returned by subprocesses popen and the pid of the process
        """
        program_to_scan.set_processes_ids(self.processes_ids)

        # create file for stdout text
        with open(f"{os.path.join(self.STDOUT_FILES_DIR, program_to_scan.get_program_name() + ' Stdout.txt')}", "a") as f:
            shell_process, pid = OSFuncsInterface.popen(program_to_scan.get_command(), program_to_scan.find_child_id,
                                                        program_to_scan.should_use_powershell(), self.running_os.is_posix(),
                                                        program_to_scan.should_find_child_id(), f)

            f.write(f"Process ID: {pid}\n\n")

        # save the process names and pids in global arrays
        if pid is not None:
            self.processes_ids.append(pid)
            original_program_name = program_to_scan.get_program_name()
            iteration_num = len(self.finished_scanning_time) + 1
            self.processes_names.append(original_program_name if iteration_num == 1 else
                                        f"{original_program_name} - iteration {iteration_num}")

        return shell_process, pid

    def start_background_processes(self):
        """
        Start a process per each background program using start_process function above.
        If there is an error when creating a process, terminate all process and notify user
        :return: list of tuples. each tuple contains a process object as returned by subprocesses popen
        and the pid of the process
        """
        background_processes = [self.start_process(background_program) for background_program in self.background_programs]
        # TODO: think how to check if there are errors without sleeping - waiting for process initialization
        self.scanner_imp.scan_sleep(5)

        for (background_process, child_process_id), background_program in zip(background_processes, self.background_programs):
            if background_process.poll() is not None:   # if process has not terminated
                err = background_process.stderr.read().decode()
                if err:
                    self.terminate_due_to_exception(background_processes, background_program.get_program_name(), err)

        return background_processes

    def terminate_due_to_exception(self, background_processes, program_name, err):
        """
        When an exception is raised from one of the processes, we will terminate all other process and stop measuring
        :param background_processes:
        :param program_name: the name of the program that had an error
        :param err: explanation about the error occurred
        """
        self.done_scanning = True

        # terminate the main process if it still exists
        try:
            p = psutil.Process(self.main_process_id)
            p.terminate()  # or p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # terminate other processes (background processes)
        self.kill_background_processes(background_processes)

        # stop measuring - raise exception to user
        self.after_scanning_operations(should_save_results=False)
        raise Exception("An error occurred in child program %s: %s", program_name, err)

    def kill_background_processes(self, background_processes):
        waiting_threads = []
        for (powershell_process, child_process_id), background_program in zip(background_processes, self.background_programs):
            if self.kill_background_process_when_main_finished:
                self.kill_process(child_process_id, powershell_process)

            else:
                wait_to_process_thread = Thread(target=self.wait_and_write_running_time_to_file, args=(child_process_id, powershell_process))
                wait_to_process_thread.start()
                waiting_threads.append(wait_to_process_thread)

        for waiting_thread in waiting_threads:
            waiting_thread.join()
            # powershell_process.wait()

    def kill_process(self, child_process_id, powershell_process):
        try:
            if child_process_id is None:
                powershell_process.kill()

            else:
                p = psutil.Process(child_process_id)
                p.terminate()  # or p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def wait_and_write_running_time_to_file(self, child_process_id, powershell_process):
        try:
            if child_process_id is None:
                powershell_process.wait()

            else:
                p = psutil.Process(child_process_id)
                p.wait()  # or p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        print(self.scanner_imp.calc_time_interval(self.starting_time))

    def start_timeout(self, main_shell_process, is_posix):
        """
        This function terminates the main process if its running time exceeds maximum allowed time
        :param main_shell_process: the process to terminate
        :return: a timer thread (as returned from Timer function)
        """
        if RUNNING_TIME is None or self.scan_option != self.ScanMode.ONE_SCAN:
            return

        timeout_thread = Timer(RUNNING_TIME, self.program.kill_process, [main_shell_process, is_posix, RUNNING_TIME])
        timeout_thread.start()
        return timeout_thread

    def cancel_timeout_timer(self, timeout_timer):
        """
        This function cancels the timer thread that kills terminates the main program in case it
         exceeded maximum allowed running time
        :param timeout_timer: the timer thread returned from start_timeout
        """
        if timeout_timer is None:
            return

        if not timeout_timer.is_alive():
            self.max_timeout_reached = True

        timeout_timer.cancel()
        timeout_timer.join()

    def prepare_summary_csv(self):
        """Prepare the summary csv file"""
        summary_df = self.summary_version_imp.prepare_summary_csv(self.processes_df, self.cpu_df, self.memory_df, self.disk_io_each_moment_df,
                                                             self.battery_df, self.processes_names, self.finished_scanning_time,
                                                             self.processes_ids)

        color_func = self.summary_version_imp.colors_func

        styled_summary_df = summary_df.style.apply(color_func, axis=0)

        styled_summary_df.to_excel(self.SUMMARY_CSV, engine='openpyxl', index=False)

    def ignore_last_results(self):
        """
        Remove the last sample from each dataframe because the main process may be finished before the sample,
        so that sample is not relevant
        """
        if self.processes_df.empty:
            processes_num_last_measurement = 0
        else:
            processes_num_last_measurement = self.processes_df[ProcessesColumns.TIME].value_counts()[
                self.processes_df[ProcessesColumns.TIME].max()]
        self.processes_df = self.processes_df.iloc[:-processes_num_last_measurement, :]
        self.memory_df = self.memory_df.iloc[:-1, :]
        self.disk_io_each_moment_df = self.disk_io_each_moment_df.iloc[:-1, :]
        if not self.battery_df.empty:
            self.battery_df = self.battery_df.iloc[:-1, :]
        self.cpu_df = self.cpu_df.iloc[:-1, :]

    def save_general_information_after_scanning(self):
        """
        save processes names and ids, disk and battery info, scanning times
        """
        with open(self.GENERAL_INFORMATION_FILE, 'a') as f:
            f.write('======After Scanning======\n')
            if self.main_process_id is not None:
                f.write(f'{PROCESS_ID_PHRASE}: {self.processes_names[0]}({self.main_process_id})\n')

            f.write(f'{BACKGROUND_ID_PHRASE}: ')
            for background_process_id, background_process_name in zip(self.processes_ids[1:-1], self.processes_names[1:-1]):
                f.write(f'{background_process_name}({background_process_id}), ')

            if len(self.processes_ids) > 1:  # not just main program
                f.write(f"{self.processes_names[-1]}({self.processes_ids[-1]})\n\n")

            self.save_general_disk(f)

            if not self.battery_df.empty:
                f.write('\n------Battery------\n')
                battery_drop = calc_delta_capacity(self.battery_df)
                f.write(f'Amount of Battery Drop: {battery_drop[0]} mWh, {battery_drop[1]}%\n')
                f.write('Approximately equivalent to -\n')
                conversions = convert_mwh_to_other_metrics(battery_drop[0])
                f.write(f'  CO2 emission: {conversions[0]} kg\n')
                f.write(f'  Coal burned: {conversions[1]} kg\n')
                f.write(f'  Number of smartphone charged: {conversions[2]}\n')
                f.write(f'  Kilograms of wood burned: {conversions[3]}\n')
            if main_program_to_scan in no_process_programs:
                measurement_time = self.finished_scanning_time[-1]
                f.write(f'\nMeasurement duration: {measurement_time} seconds, '
                        f'{measurement_time / 60} minutes\n')

            else:
                f.write('\n------Scanning Times------\n')
                if self.max_timeout_reached:
                    f.write("Scanned program reached the maximum time so we terminated it\n")
                f.write(f'Scan number 1, finished at: {self.finished_scanning_time[0]} seconds, '
                        f'{self.finished_scanning_time[0] / 60} minutes\n')
                for i, scan_time in enumerate(self.finished_scanning_time[1:]):
                    f.write(f'Scan number {i + 2}, finished at: {scan_time}.'
                            f' Duration of Scanning: {scan_time - self.finished_scanning_time[i]} seconds, '
                            f'{(scan_time - self.finished_scanning_time[i]) / 60} minutes\n')