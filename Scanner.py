import psutil
import pythoncom
import wmi
from prettytable import PrettyTable
import subprocess
from threading import Thread
import time
import pandas as pd
from enum import Enum
import os.path
from pathlib import Path
import platform


class ScanOption(Enum):
    NO_SCAN = 1
    ONE_SCAN = 2
    CONTINUOUS_SCAN = 3


class PreviousDiskIO:
    def __init__(self, disk_io):
        self.read_count = disk_io.read_count
        self.write_count = disk_io.write_count
        self.read_bytes = disk_io.read_bytes
        self.write_bytes = disk_io.write_bytes


def calc_dir():
    if scan_option == ScanOption.NO_SCAN:
        return 'no_scan'
    elif scan_option == ScanOption.ONE_SCAN:
        return os.path.join('one_scan', scan_type)
    else:
        return os.path.join('continuous_scan', scan_type)


# ======= Constants =======
MINUTE = 60
ANTIVIRUS_PROCESS_NAME = "MsMpeng"
SYSTEM_IDLE_PROCESS_NAME = "System Idle Process"
SYSTEM_IDLE_PID = 0

GB = 2**30
MB = 2**20
KB = 2**10

done_scanning = False
starting_time = time.time()


# ======= Program Parameters =======
scan_option = ScanOption.CONTINUOUS_SCAN
scan_type = "QuickScan"
MINIMUM_DELTA_CAPACITY = 20
MINIMUM_SCAN_TIME = 1 * MINUTE


# ======= Result Data Paths =======
results_dir = calc_dir()
Path(results_dir).mkdir(parents=True, exist_ok=True)

PROCESSES_CSV = os.path.join(results_dir, 'processes_data.csv')
TOTAL_MEMORY_EACH_MOMENT_CSV = os.path.join(results_dir, 'total_memory_each_moment.csv')
DISK_IO_EACH_MOMENT = os.path.join(results_dir, 'disk_io_each_moment.csv')
BATTERY_STATUS_CSV = os.path.join(results_dir, 'battery_status.csv')
GENERAL_INFORMATION_FILE = os.path.join(results_dir, 'general_information.txt')


# TODO: maybe its better to calculate MEMORY(%) in the end of scan in order to reduce calculations during scanning
processes_df = pd.DataFrame(columns=['Time(sec)', 'PID', 'PNAME', 'CPU(%)', 'NUM THREADS', 'MEMORY(MB)', 'MEMORY(%)',
                                     "READ_IO(#)", "WRITE_IO(#)", "READ_IO(KB)", "WRITE_IO(KB)"])

memory_df = pd.DataFrame(columns=['Time(sec)', 'Used(GB)', 'Percentage'])

disk_io_each_moment_df = pd.DataFrame(columns=['Time(sec)', "READ(#)", "WRITE(#)", "READ(KB)", "WRITE(KB)"])

REMAINING_CAPACITY_MWH = "REMAINING CAPACITY(mWh)"
battery_df = pd.DataFrame(columns=['Time(sec)', "REMAINING BATTERY(%)", REMAINING_CAPACITY_MWH, "Voltage(mV)"])

finished_scanning_time = []


def calc_time_interval():
    return time.time() - starting_time


def save_battery_stat():
    # Fetch the battery information
    battery = psutil.sensors_battery()
    if battery is None:  # if desktop computer (has no battery)
        return

    if battery.power_plugged:
        raise Exception("Unplug charging cable during measurements!")

    t = wmi.WMI(moniker="//./root/wmi")

    new_row_index = len(battery_df.index)

    for i, b in enumerate(t.ExecQuery('Select * from BatteryStatus where Voltage > 0')):
        battery_df.loc[new_row_index + i] = [
            calc_time_interval(),
            battery.percent,
            b.RemainingCapacity,
            b.Voltage
        ]


def save_current_total_memory():
    vm = psutil.virtual_memory()
    memory_df.loc[len(memory_df.index)] = [
        calc_time_interval(),
        f'{vm.used / GB:.3f}',
        vm.percent
    ]


def save_current_disk_io(previous_disk_io):
    disk_io_stat = psutil.disk_io_counters()
    disk_io_each_moment_df.loc[len(disk_io_each_moment_df.index)] = [
        calc_time_interval(),
        disk_io_stat.read_count - previous_disk_io.read_count,
        disk_io_stat.write_count - previous_disk_io.write_count,
        f'{(disk_io_stat.read_bytes - previous_disk_io.read_bytes) / KB:.3f}',
        f'{(disk_io_stat.write_bytes - previous_disk_io.write_bytes) / KB:.3f}'
    ]

    return disk_io_stat


def save_current_processes_statistics(prev_io_per_process):
    proc = []
    system_idle_process = psutil.Process(SYSTEM_IDLE_PID)
    system_idle_process.cpu_percent()

    time_of_sample = calc_time_interval()

    for p in psutil.process_iter():
        try:
            if p.pid == SYSTEM_IDLE_PID:  # ignore System Idle Process
                continue
            # trigger cpu_percent() the first time which leads to return of 0.0
            p.cpu_percent()
            proc.append(p)

        except Exception:
            pass

    # sort by cpu_percent
    top = {}
    time.sleep(0.1)
    for p in proc:
        # trigger cpu_percent() the second time for measurement
        try:
            top[p] = p.cpu_percent() / psutil.cpu_count()
        except psutil.NoSuchProcess:
            pass

    top_list = sorted(top.items(), key=lambda x: x[1])[-20:]
    top_list.append((system_idle_process, system_idle_process.cpu_percent() / psutil.cpu_count()))
    top_list.reverse()

    return add_to_processes_dataframe(time_of_sample, top_list, prev_io_per_process)


def add_to_processes_dataframe(time_of_sample, top_list, prev_io_per_process):
    for p, cpu_percent in top_list:

        # While fetching the processes, some subprocesses may exit
        # Hence we need to put this code in try-except block
        try:
            # oneshot to improve info retrieve efficiency
            with p.oneshot():
                io_stat = p.io_counters()

                if p.pid not in prev_io_per_process:
                    prev_io_per_process[p.pid] = PreviousDiskIO(io_stat)

                prev_io = prev_io_per_process[p.pid]

                # TODO - is io_counters what we are looking for (only disk reads)
                # TODO: calculate all values for total(include memory, read, write, etc...)
                processes_df.loc[len(processes_df.index)] = [
                    time_of_sample,
                    p.pid,
                    p.name() if p.pid != SYSTEM_IDLE_PID else "Total",
                    f'{(cpu_percent if p.pid != SYSTEM_IDLE_PID else 100 - cpu_percent):.2f}',
                    p.num_threads(),
                    f'{p.memory_info().rss / MB:.3f}',  # TODO: maybe should use uss instead rss?
                    round(p.memory_percent(), 2),
                    io_stat.read_count - prev_io.read_count,
                    io_stat.write_count - prev_io.write_count,
                    f'{(io_stat.read_bytes - prev_io.read_bytes) / KB:.3f}',
                    f'{(io_stat.write_bytes - prev_io.write_bytes) / KB:.3f}',
                ]

                prev_io_per_process[p.pid] = PreviousDiskIO(io_stat)

        except Exception:
            pass

    return prev_io_per_process


def min_scan_time_passed():
    return time.time() - starting_time >= MINIMUM_SCAN_TIME


def should_scan():
    return scan_option != ScanOption.NO_SCAN and not done_scanning


def continuously_measure():
    pythoncom.CoInitialize()

    # init PreviousDiskIO by first disk io measurements (before scan)
    prev_disk_io = PreviousDiskIO(psutil.disk_io_counters())
    prev_io_per_process = {}

    # TODO: think if total tables should be printed only once
    while should_scan() or not min_scan_time_passed():
        save_battery_stat()
        prev_io_per_process = save_current_processes_statistics(prev_io_per_process)
        save_current_total_memory()
        prev_disk_io = save_current_disk_io(prev_disk_io)

        # Create a delay
        time.sleep(0.5)


def save_general_battery(f):
    f.write("----Battery----\n")
    c = wmi.WMI()
    t = wmi.WMI(moniker="//./root/wmi")
    batts1 = c.CIM_Battery(Caption='Portable Battery')
    for i, b in enumerate(batts1):
        f.write('Battery %d Design Capacity: %d mWh\n' % (i, b.DesignCapacity or 0))

    batts = t.ExecQuery('Select * from BatteryFullChargedCapacity')
    for i, b in enumerate(batts):
        f.write('Battery %d Fully Charged Capacity: %d mWh\n' % (i, b.FullChargedCapacity))


def save_general_disk(f):
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
    f.write("======System Information======\n")
    my_system = platform.uname()

    f.write(f"System: {my_system.system}\n")
    f.write(f"Node Name: {my_system.node}\n")
    f.write(f"Release: {my_system.release}\n")
    f.write(f"Version: {my_system.version}\n")
    f.write(f"Machine: {my_system.machine}\n")
    f.write(f"Processor: {my_system.processor}\n")


def save_general_information_before_scanning():
    with open(GENERAL_INFORMATION_FILE, 'w') as f:
        save_general_system_information(f)

        f.write('\n======Before Scanning======\n')
        save_general_battery(f)
        f.write('\n')
        save_general_disk(f)
        f.write('\n\n')


def save_general_information_after_scanning():
    with open(GENERAL_INFORMATION_FILE, 'a') as f:
        f.write('======After Scanning======\n')
        save_general_disk(f)

        f.write('\n------Battery------\n')
        f.write('Amount of Battery Drop: %d mWh\n' % calc_delta_capacity())

        f.write('\n------Scanning Times------\n')
        f.write(f'Scan number 1, finished at: {finished_scanning_time[0]}\n')
        for i, scan_time in enumerate(finished_scanning_time[1:]):
            f.write(f'Scan number {i + 2}, finished at: {scan_time}.'
                    f' Duration of Scanning: {scan_time - finished_scanning_time[i]}\n')


def save_to_files():
    save_general_information_after_scanning()
    processes_df.iloc[:-1, :].to_csv(PROCESSES_CSV, index=False)
    memory_df.iloc[:-1, :].to_csv(TOTAL_MEMORY_EACH_MOMENT_CSV, index=False)
    disk_io_each_moment_df.iloc[:-1, :].to_csv(DISK_IO_EACH_MOMENT, index=False)
    battery_df.iloc[:-1, :].to_csv(BATTERY_STATUS_CSV, index=False)


def calc_delta_capacity():
    if battery_df.empty:
        return 0
    before_scanning_capacity = battery_df.iloc[0].at[REMAINING_CAPACITY_MWH]
    current_capacity = battery_df.iloc[len(battery_df) - 1].at[REMAINING_CAPACITY_MWH]
    return before_scanning_capacity - current_capacity


def is_delta_capacity_achieved():
    if psutil.sensors_battery() is None:  # if desktop computer (has no battery)
        return True

    return calc_delta_capacity() >= MINIMUM_DELTA_CAPACITY


def main():
    global done_scanning
    print("======== Process Monitor ========")

    save_general_information_before_scanning()

    measurements_thread = Thread(target=continuously_measure, args=())
    measurements_thread.start()

    while not scan_option == ScanOption.NO_SCAN and not done_scanning:
        # TODO check about capture_output
        result = subprocess.run(["powershell", "-Command", "Start-MpScan -ScanType " + scan_type], capture_output=True)
        finished_scanning_time.append(calc_time_interval())
        if scan_option == ScanOption.ONE_SCAN or (min_scan_time_passed() and is_delta_capacity_achieved()):
            done_scanning = True
        if result.returncode != 0:
            raise Exception("An error occurred while anti virus scan: %s", result.stderr)

    measurements_thread.join()

    save_to_files()

    print("finished scanning")


if __name__ == '__main__':
    main()
