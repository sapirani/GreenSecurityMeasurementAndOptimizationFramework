import psutil
import time
import numpy as np
from subprocess import call

import pythoncom
import wmi
from prettytable import PrettyTable
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess
from multiprocessing import Process
from threading import Thread
import time
import pandas as pd


ANTIVIRUS_PROCESS_NAME = "MsMpeng"
SYSTEM_IDLE_PROCESS_NAME = "System Idle Process"
SYSTEM_IDLE_PID = 0
GB = 2**30
MB = 2**20
KB = 2**10

need_scan = True
isScanDone = not need_scan
SCAN_TIME = 1 * 60  # 10 minutes
starting_time = time.time()

# TODO: maybe its better to calculate MEMORY(%) in the end of scan in order to reduce calculations during scanning
processes_df = pd.DataFrame(columns=['Time(sec)', 'PID', 'PNAME', 'CPU(%)', 'NUM THREADS', 'MEMORY(MB)', 'MEMORY(%)',
                                     'read_count', 'write_count', 'read_bytes', 'write_bytes'])

memory_df = pd.DataFrame(columns=['Time(sec)', 'Total(GB)', 'Used(GB)', 'Available(GB)', 'Percentage'])

disk_io_each_moment_df = pd.DataFrame(columns=['Time(sec)', "READ(#)", "WRITE(#)", "READ(KB)", "WRITE(KB)"])

battery_df = pd.DataFrame(columns=['Time(sec)', "REMAINING BATTERY(%)", "REMAINING CAPACITY(mWh)", "Voltage(mV)"])


class PreviousDiskIO:
    def __init__(self, disk_io):
        self.read_count = disk_io.read_count
        self.write_count = disk_io.write_count
        self.read_bytes = disk_io.read_bytes
        self.write_bytes = disk_io.write_bytes


def calc_time_interval():
    return time.time() - starting_time


# [ [x1, y1] [x2, y2] [x3,y3]] to [x1 x2 x3] and [y1 y2 y3]
def split_to_xy(arr):
    print("This is split")
    print(arr)
    x, y = zip(*arr)
    return x, y


def draw_graph(x, y, y_name):
    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('Time')
    # naming the y axis
    plt.ylabel(y_name)

    # giving a title to my graph
    plt.title('My first graph!')

    # function to show the plot
    plt.show()


def save_battery_stat():
    # if mobile computer
    # Fetch the battery information
    battery = psutil.sensors_battery()
    if battery is None:  # if desktop computer (has no battery)
        return

    t = wmi.WMI(moniker = "//./root/wmi")

    new_row_index = len(battery_df.index)

    """ batts1 = c.CIM_Battery(Caption='Portable Battery')
    for i, b in enumerate(batts1):
        print('Battery %d Design Capacity: %d mWh' % (i, b.DesignCapacity or 0))

    batts = t.ExecQuery('Select * from BatteryFullChargedCapacity')
    for i, b in enumerate(batts):
        print('Battery %d Fully Charged Capacity: %d mWh' % (i, b.FullChargedCapacity))"""

    for i, b in enumerate(t.ExecQuery('Select * from BatteryStatus where Voltage > 0')):
        battery_df.loc[new_row_index + i] = [
            calc_time_interval(),
            battery.percent,
            b.RemainingCapacity,
            b.Voltage
        ]

    if battery.power_plugged:
        raise Exception("Unplug charging cable during measurements!")


def create_total_memory_table():
    vm = psutil.virtual_memory()
    memory_df.loc[len(memory_df.index)] = [
        calc_time_interval(),
        f'{vm.total / GB:.3f}',
        f'{vm.used / GB:.3f}',
        f'{vm.available / GB:.3f}',
        vm.percent
    ]


def create_total_disk_table():
    print("----Disk----")
    disk_table = PrettyTable(["Total(GB)", "Used(GB)",
                              "Available(GB)", "Percentage"])
    disk_stat = psutil.disk_usage('/')
    disk_table.add_row([
        f'{disk_stat.total / GB:.3f}',
        f'{disk_stat.used / GB:.3f}',
        f'{disk_stat.free / GB:.3f}',
        disk_stat.percent
    ])
    # used_total_disk.append([calc_time_interval(), f'{disk_stat.used / GB:.3f}'])
    print(disk_table)


def create_current_disk_io_table(previous_disk_io):
    disk_io_stat = psutil.disk_io_counters()
    disk_io_each_moment_df.loc[len(disk_io_each_moment_df.index)] = [
        calc_time_interval(),
        disk_io_stat.read_count - previous_disk_io.read_count,
        disk_io_stat.write_count - previous_disk_io.write_count,
        f'{(disk_io_stat.read_bytes - previous_disk_io.read_bytes) / KB:.3f}',
        f'{(disk_io_stat.write_bytes - previous_disk_io.write_bytes) / KB:.3f}'
    ]

    return disk_io_stat


def create_process_table():
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

    add_to_processes_dataframe(time_of_sample, top_list)


def add_to_processes_dataframe(time_of_sample, top_list):
    for p, cpu_percent in top_list:

        # While fetching the processes, some subprocesses may exit
        # Hence we need to put this code in try-except block
        # TODO: CHANGE TO DATAFRAME
        """if p.name() == SYSTEM_IDLE_PROCESS_NAME:
            used_total_cpu.append([calc_time_interval(), float(f'{100 - cpu_percent:.2f}')])
        elif need_scan and p.name() == ANTIVIRUS_PROCESS_NAME:
            used_cpu_by_antivirus.append([calc_time_interval(), f'{cpu_percent:.2f}'])"""
        try:
            # oneshot to improve info retrieve efficiency
            with p.oneshot():
                disk_io = p.io_counters()

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
                    disk_io.read_count,
                    disk_io.write_count,
                    disk_io.read_bytes,
                    disk_io.write_bytes
                ]

        except Exception:
            pass


def continuously_measure():
    pythoncom.CoInitialize()

    # init PreviousDiskIO by first disk io measurements (before scan)
    prev_disk_io = PreviousDiskIO(psutil.disk_io_counters())

    # TODO: think if total tables should be printed only once
    while not isScanDone if need_scan else (SCAN_TIME + starting_time >= time.time()):
        save_battery_stat()
        create_process_table()
        create_total_memory_table()
        prev_disk_io = create_current_disk_io_table(prev_disk_io)

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
    # used_total_disk.append([calc_time_interval(), f'{disk_stat.used / GB:.3f}'])
    f.write(str(disk_table))
    f.write('\n')


def save_general_information():
    with open("general_information.txt", 'w') as f:
        save_general_battery(f)
        f.write('\n')
        save_general_disk(f)


def main():
    global isScanDone
    print("==============================Process Monitor\
                ======================================")
    # print_battery_stat()
    save_general_information()

    mainT = Thread(target=continuously_measure, args=())
    mainT.start()

    if need_scan:
        # TODO check about capture_output
        result = subprocess.run(["powershell", "-Command", "Start-MpScan -ScanType QuickScan"], capture_output=True)
        isScanDone = True
        if result.returncode != 0:
            raise Exception("An error occurred while anti virus scan: %s", result.stderr)

    mainT.join()

    # print_battery_stat()
    # create_total_disk_table()

    processes_df.to_csv('processes_data.csv')
    memory_df.to_csv('total_memory.csv')
    disk_io_each_moment_df.to_csv('disk_io_each_moment.csv')
    battery_df.to_csv('battery_status.csv')

    print("finished scanning")

    """print("done waiting")
    print("Time: " + str(time.time() - starting_time))
    print("memory data: ")
    print(used_total_memory)
    x, y = split_to_xy(used_total_memory)
    draw_graph(x, y, "used memory")
    x, y = split_to_xy(used_total_cpu)
    draw_graph(x, y, "used cpu")
"""


if __name__ == '__main__':
    main()
