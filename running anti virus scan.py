import psutil
import time
import numpy as np
from subprocess import call
from prettytable import PrettyTable
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess
from multiprocessing import Process
import time

need_scan = False
isScanDone = not need_scan
SCAN_TIME = 1 * 60  # 10 minutes
starting_time = time.time()
ANTIVIRUS_PROCESS_NAME = "MsMpeng"
SYSTEM_IDLE_PROCESS_NAME = "System Idle Process"
battery_available_precent = []
used_total_memory = []
used_memory_by_antivirus = []
used_total_cpu = []
used_cpu_by_antivirus = []
used_total_disk = []
used_disk_by_antivirus = []
used_total_IO = []
used_IO_by_antivirus = []


def run_antivirus():
    completed = subprocess.run(["powershell", "-Command", "Start-MpScan -ScanType QuickScan"], capture_output=True)
    if completed.returncode == 0:
        isScanDone = True
    else:
        raise Exception("Anti virus scan didn't work")
    return completed


def calc_time_interval():
    interval = time.time() - starting_time
    return interval


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


def print_battery_stat():
    # c = wmi.WMI()
    # t = wmi.WMI(moniker = "//./root/wmi")
    print("==============================Process Monitor\
        ======================================")

    # Fetch the battery information
    # battery = psutil.sensors_battery().percent
    # print("Battery Available: %d " % (battery,) + "%")
    # battery_available_precent.append([calc_time_interval(), battery])

    # batts1 = c.CIM_Battery(Caption='Portable Battery')
    # for i, b in enumerate(batts1):
    #    print('Battery %d Design Capacity: %d mWh' % (i, b.DesignCapacity or 0))

    # batts = t.ExecQuery('Select * from BatteryFullChargedCapacity')
    # for i, b in enumerate(batts):
    #   print('Battery %d Fully Charged Capacity: %d mWh' %
    #        (i, b.FullChargedCapacity))

    # batts = t.ExecQuery('Select * from BatteryStatus where Voltage > 0')
    # for i, b in enumerate(batts):
    #   print('Voltage:           ' + str(b.Voltage))
    #  print('RemainingCapacity: ' + str(b.RemainingCapacity))

    # if b.Charging:
    #    raise Exception("Unplug charging cable during measurements!")


def create_total_memory_table():
    print("----Memory----")
    memory_table = PrettyTable(["Total(GB)", "Used(GB)",
                                "Available(GB)", "Percentage"])
    vm = psutil.virtual_memory()
    memory_table.add_row([
        f'{vm.total / 1e9:.3f}',
        f'{vm.used / 1e9:.3f}',
        f'{vm.available / 1e9:.3f}',
        vm.percent
    ])

    used_total_memory.append([calc_time_interval(), f'{vm.used / 1e9:.3f}'])
    print("added value: ")
    print(used_total_memory[-1])
    print(memory_table)


def create_total_disk_table():
    print("----Disk----")
    disk_table = PrettyTable(["Total(GB)", "Used(GB)",
                              "Available(GB)", "Percentage"])
    disk_stat = psutil.disk_usage('/')
    disk_table.add_row([
        f'{disk_stat.total / 1e9:.3f}',
        f'{disk_stat.used / 1e9:.3f}',
        f'{disk_stat.free / 1e9:.3f}',
        disk_stat.percent
    ])
    used_total_disk.append([calc_time_interval(), f'{disk_stat.used / 1e9:.3f}'])
    print(disk_table)


def create_current_disk_io_table():
    print("----Disk I/O----")
    disk_table = PrettyTable(["read_count", "write_count",
                              "read_bytes(GB)", "write_bytes(GB)"])
    disk_io_stat = psutil.disk_io_counters()
    disk_table.add_row([
        disk_io_stat.read_count,
        disk_io_stat.write_count,
        f'{disk_io_stat.read_bytes / 1e9:.3f}',
        f'{disk_io_stat.write_bytes / 1e9:.3f}'
    ])
    print(disk_table)


def create_process_table():
    print("----Processes----")
    process_table = PrettyTable(['PID', 'PNAME',
                                 'CPU', 'NUM THREADS', 'MEMORY(MB)', 'MEMORY(%)', 'read_count', 'write_count',
                                 'read_bytes', 'write_bytes'])

    proc = []
    # get the pids from last which mostly are user processes
    for p in psutil.process_iter():
        try:
            # trigger cpu_percent() the first time which leads to return of 0.0
            p.cpu_percent()
            proc.append(p)

        except Exception as e:
            pass

    # sort by cpu_percent
    top = {}
    time.sleep(0.1)
    for p in proc:
        # trigger cpu_percent() the second time for measurement
        top[p] = p.cpu_percent() / psutil.cpu_count()

    top_list = sorted(top.items(), key=lambda x: x[1])
    top10 = top_list[-20:]
    top10.reverse()

    for p, cpu_percent in top10:

        # While fetching the processes, some of the subprocesses may exit
        # Hence we need to put this code in try-except block
        if p.name() == SYSTEM_IDLE_PROCESS_NAME:
            used_total_cpu.append([calc_time_interval(), float(f'{100 - cpu_percent:.2f}')])
        elif need_scan and p.name() == ANTIVIRUS_PROCESS_NAME:
            used_cpu_by_antivirus.append([calc_time_interval(), f'{cpu_percent:.2f}'])
        try:
            # oneshot to improve info retrieve efficiency
            with p.oneshot():
                disk_io = p.io_counters()
                process_table.add_row([
                    str(p.pid),
                    p.name(),
                    f'{cpu_percent:.2f}' + "%",
                    p.num_threads(),
                    f'{p.memory_info().rss / 1e6:.3f}',  # TODO: maybe should use uss instead rss?
                    round(p.memory_percent(), 2),
                    disk_io.read_count,
                    disk_io.write_count,
                    disk_io.read_bytes,
                    disk_io.write_bytes
                ])


        except Exception as e:
            pass
    print(process_table)


def main_program():
    # condition =
    # print(condition)
    while not isScanDone if need_scan else (SCAN_TIME + starting_time >= time.time()):
        print("befor")
        create_total_disk_table()
        print("after disk")
        create_current_disk_io_table()
        print("after io")
        create_total_memory_table()
        print("after memory")
        create_process_table()
        print("after process")

        # Create a delay
        time.sleep(0.5)


if __name__ == '__main__':
    mainP = Process(target=main_program, args=())
    mainP.start()

    if need_scan:
        scanP = Process(target=run_antivirus, args=())
        print("before scan")
        scanP.start()
        print("while scan")
        scanP.join()
        print("after join")

    mainP.join()
    print("done waiting")
    print("Time: " + str(time.time() - starting_time))
    print("memory data: ")
    print(used_total_memory)
    x, y = split_to_xy(used_total_memory)
    draw_graph(x, y, "used memory")
    x, y = split_to_xy(used_total_cpu)
    draw_graph(x, y, "used cpu")
