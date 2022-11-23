import shutil
import pythoncom
from statistics import mean
from prettytable import PrettyTable
from threading import Thread
import pandas as pd
from initialization_helper import *
import ctypes
from datetime import date
from pathlib import Path
import screen_brightness_control as sbc
from powershell_helper import get_powershell_result_list_format


# ======= Constants =======
SYSTEM_IDLE_PROCESS_NAME = "System Idle Process"
SYSTEM_IDLE_PID = 0

YES_BUTTON = 6
NO_BUTTON = 7

NEVER_TURN_SCREEN_OFF = 0
NEVER_GO_TO_SLEEP_MODE = 0

base_dir, GRAPHS_DIR, PROCESSES_CSV, TOTAL_MEMORY_EACH_MOMENT_CSV, DISK_IO_EACH_MOMENT, \
BATTERY_STATUS_CSV, GENERAL_INFORMATION_FILE, TOTAL_CPU_CSV, SUMMARY_CSV = result_paths()

# ======= Program Global Parameters =======
done_scanning = False
starting_time = 0
scanning_process_id = None

# TODO: maybe its better to calculate MEMORY(%) in the end of scan in order to reduce calculations during scanning
processes_df = pd.DataFrame(columns=processes_columns_list)

memory_df = pd.DataFrame(columns=memory_columns_list)

disk_io_each_moment_df = pd.DataFrame(columns=disk_io_columns_list)

battery_df = pd.DataFrame(columns=battery_columns_list)

cpu_df = pd.DataFrame(columns=cpu_columns_list)

finished_scanning_time = []


def message_box(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)


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
        f'{(disk_io_stat.write_bytes - previous_disk_io.write_bytes) / KB:.3f}',
        disk_io_stat.read_time - previous_disk_io.read_time,
        disk_io_stat.write_time - previous_disk_io.write_time
    ]

    return disk_io_stat


def save_current_processes_statistics(prev_io_per_process):
    proc = []

    time_of_sample = calc_time_interval()

    for p in psutil.process_iter():
        try:
            if p.pid == SYSTEM_IDLE_PID:  # ignore System Idle Process
                continue

            # trigger cpu_percent() the first time will lead to return of 0.0
            cpu_percent = p.cpu_percent() / NUMBER_OF_CORES
            proc.append((p, cpu_percent))

        except Exception:
            pass

    proc = sorted(proc, key=lambda x: x[1], reverse=True)

    return add_to_processes_dataframe(time_of_sample, proc, prev_io_per_process)


def add_to_processes_dataframe(time_of_sample, top_list, prev_io_per_process):
    for p, cpu_percent in top_list:

        # While fetching the processes, some subprocesses may exit
        # Hence we need to put this code in try-except block
        try:
            # oneshot to improve info retrieve efficiency
            with p.oneshot():
                io_stat = p.io_counters()

                if (p.pid, p.name) not in prev_io_per_process:
                    prev_io_per_process[(p.pid, p.name)] = io_stat
                    continue    # remove first sample of process (because cpu_percent is meaningless 0)

                prev_io = prev_io_per_process[(p.pid, p.name)]

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
                ]

                prev_io_per_process[(p.pid, p.name)] = io_stat

        except psutil.NoSuchProcess:
            pass

    return prev_io_per_process


def min_scan_time_passed():
    return time.time() - starting_time >= MINIMUM_SCAN_TIME


def should_scan():
    if program_to_scan == ProgramToScan.NO_SCAN:
        return not min_scan_time_passed()
    elif scan_option == ScanMode.ONE_SCAN:
        return not done_scanning
    elif scan_option == ScanMode.CONTINUOUS_SCAN:
        return not min_scan_time_passed() and not is_delta_capacity_achieved()


def save_current_total_cpu():
    total_cpu = psutil.cpu_percent(percpu=True)
    cpu_df.loc[len(cpu_df.index)] = [calc_time_interval(), mean(total_cpu)] + total_cpu


def continuously_measure():
    pythoncom.CoInitialize()

    # init prev_disk_io by first disk io measurements (before scan)
    prev_disk_io = psutil.disk_io_counters()
    prev_io_per_process = {}

    # TODO: think if total tables should be printed only once
    while should_scan():
        # Create a delay
        time.sleep(0.5)

        save_battery_stat()
        prev_io_per_process = save_current_processes_statistics(prev_io_per_process)
        save_current_total_cpu()
        save_current_total_memory()
        prev_disk_io = save_current_disk_io(prev_disk_io)


def save_general_battery(f):
    battery = psutil.sensors_battery()
    if battery is None:  # if desktop computer (has no battery)
        return

    if battery.power_plugged:
        raise Exception("Unplug charging cable during measurements!")

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


def save_disk_info(f, c):
    wmi_logical_disks = c.Win32_LogicalDisk()
    result = subprocess.run(["powershell", "-Command",
                             "Get-Disk | Select FriendlyName, Manufacturer, Model,  PartitionStyle, NumberOfPartitions,"
                             " PhysicalSectorSize, LogicalSectorSize, BusType | Format-List"], capture_output=True)
    if result.returncode != 0:
        raise Exception(f'An error occurred while getting disk information', result.stderr)

    logical_disks_info = get_powershell_result_list_format(result.stdout)

    result = subprocess.run(["powershell", "-Command",
                             "Get-PhysicalDisk | Select FriendlyName, Manufacturer, Model, FirmwareVersion, MediaType"
                             " | Format-List"], capture_output=True)
    if result.returncode != 0:
        raise Exception(f'An error occurred while getting disk information', result.stderr)

    physical_disks_info = get_powershell_result_list_format(result.stdout)

    f.write("\n----Physical Disk Information----\n")
    for physical_disk_info in physical_disks_info:
        f.write(f"\nName: {physical_disk_info['FriendlyName']}\n")
        f.write(f"Manufacturer: {physical_disk_info['Manufacturer']}\n")
        f.write(f"Model: {physical_disk_info['Model']}\n")
        f.write(f"Media Type: {physical_disk_info['MediaType']}\n")
        f.write(f"Disk Firmware Version: {physical_disk_info['FirmwareVersion']}\n")

    f.write("\n----Logical Disk Information----\n")
    try:
        for index, logical_disk_info in enumerate(logical_disks_info):
            f.write(f"\nName: {logical_disk_info['FriendlyName']}\n")
            f.write(f"Manufacturer: {logical_disk_info['Manufacturer']}\n")
            f.write(f"Model: {logical_disk_info['Model']}\n")
            f.write(f"Disk Type: {disk_types[wmi_logical_disks[index].DriveType]}\n")
            f.write(f"Partition Style: {logical_disk_info['PartitionStyle']}\n")
            f.write(f"Number Of Partitions: {logical_disk_info['NumberOfPartitions']}\n")
            f.write(f"Physical Sector Size: {logical_disk_info['PhysicalSectorSize']} bytes\n")
            f.write(f"Logical Sector Size: {logical_disk_info['LogicalSectorSize']}  bytes\n")
            f.write(f"Bus Type: {logical_disk_info['BusType']}\n")
            f.write(f"FileSystem: {wmi_logical_disks[index].FileSystem}\n")
    except Exception:
        pass


def save_general_system_information(f):
    platform_system = platform.uname()
    c = wmi.WMI()
    wmi_system = c.Win32_ComputerSystem()[0]
    wmi_physical_memory = c.Win32_PhysicalMemory()

    f.write("======System Information======\n")

    f.write(f"PC Type: {pc_types[wmi_system.PCSystemType]}\n")
    f.write(f"Manufacturer: {wmi_system.Manufacturer}\n")
    f.write(f"System Family: {wmi_system.SystemFamily}\n")
    f.write(f"Model: {wmi_system.Model}\n")
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

    for physical_memory in wmi_physical_memory:
        f.write(f"\nName: {physical_memory.Tag}\n")
        f.write(f"Manufacturer: {physical_memory.Manufacturer}\n")
        f.write(f"Capacity: {int(physical_memory.Capacity) / GB}\n")
        f.write(f"Memory Type: {physical_memory_types[physical_memory.SMBIOSMemoryType]}\n")
        f.write(f"Speed: {physical_memory.Speed} MHz\n")

    save_disk_info(f, c)


def save_general_information_before_scanning():
    with open(GENERAL_INFORMATION_FILE, 'w') as f:
        # dd/mm/YY
        f.write(f'Date: {date.today().strftime("%d/%m/%Y")}\n\n')

        program.general_information_before_measurement(f)

        save_general_system_information(f)

        f.write('\n======Before Scanning======\n')
        save_general_battery(f)
        f.write('\n')
        save_general_disk(f)
        f.write('\n\n')


def convert_mwh_to_other_metrics(amount_of_mwh):
    kwh_to_mwh = 1e6
    # link: https://www.epa.gov/energy/greenhouse-gases-equivalencies-calculator-calculations-and-references
    co2 = (0.709 * amount_of_mwh) / kwh_to_mwh  # 1 kwh = 0.709 kg co2
    coal_burned = (0.453592 * 0.784 * amount_of_mwh) / kwh_to_mwh  # 1 kwh = 0.784 pound coal
    number_of_smartphones_charged = (86.2 * amount_of_mwh) / kwh_to_mwh  # 1 kwh = 86.2 smartphones

    # the following are pretty much the same. Maybe should consider utilization when converting from heat to electricity
    # link: https://www.cs.mcgill.ca/~rwest/wikispeedia/wpcd/wp/w/Wood_fuel.htm
    # link: https://www3.uwsp.edu/cnr-ap/KEEP/Documents/Activities/Energy%20Fact%20Sheets/FactsAboutWood.pdf
    # link: https://stwww1.weizmann.ac.il/energy/%D7%AA%D7%9B%D7%95%D7%9C%D7%AA-%D7%94%D7%90%D7%A0%D7%A8%D7%92%D7%99%D7%94-%D7%A9%D7%9C-%D7%93%D7%9C%D7%A7%D7%99%D7%9D/
    kg_of_woods_burned = amount_of_mwh / (3.5 * kwh_to_mwh)  # 3.5 kwh = 1 kg of wood

    return co2, coal_burned, number_of_smartphones_charged, kg_of_woods_burned


def save_general_information_after_scanning():
    with open(GENERAL_INFORMATION_FILE, 'a') as f:
        f.write('======After Scanning======\n')
        if scanning_process_id is not None:
            f.write(f'{PROCESS_ID_PHRASE}: {scanning_process_id}\n\n')

        save_general_disk(f)

        if not battery_df.empty:
            f.write('\n------Battery------\n')
            battery_drop = calc_delta_capacity()
            f.write(f'Amount of Battery Drop: {battery_drop[0]} mWh, {battery_drop[1]}%\n')
            f.write('Approximately equivalent to -\n')
            conversions = convert_mwh_to_other_metrics(battery_drop[0])
            f.write(f'  CO2 emission: {conversions[0]} kg\n')
            f.write(f'  Coal burned: {conversions[1]} kg\n')
            f.write(f'  Number of smartphone charged: {conversions[2]}\n')
            f.write(f'  Kilograms of wood burned: {conversions[3]}\n')

        if program_to_scan == ProgramToScan.NO_SCAN:
            measurement_time = finished_scanning_time[-1]
            f.write(f'\nMeasurement duration: {measurement_time} seconds, '
                    f'{measurement_time / 60} minutes\n')

        else:
            f.write('\n------Scanning Times------\n')
            f.write(f'Scan number 1, finished at: {finished_scanning_time[0]} seconds, '
                    f'{finished_scanning_time[0] / 60} minutes\n')
            for i, scan_time in enumerate(finished_scanning_time[1:]):
                f.write(f'Scan number {i + 2}, finished at: {scan_time}.'
                        f' Duration of Scanning: {scan_time - finished_scanning_time[i]} seconds, '
                        f'{(scan_time - finished_scanning_time[i]) / 60} minutes\n')


def slice_df(df, percent):
    num = int(len(df.index)*(percent/100))
    return df[num: len(df.index) - num]


def prepare_summary_csv():
    process_finishing_time = processes_df[ProcessesColumns.TIME].iat[-1]
    disk_finishing_time = disk_io_each_moment_df[DiskIOColumns.TIME].iat[-1]
    total_finishing_time = finished_scanning_time[-1]

    sub_cpu_df = slice_df(cpu_df, 5).astype(float)
    sub_memory_df = slice_df(memory_df, 5).astype(float)
    sub_disk_df = slice_df(disk_io_each_moment_df, 0).astype(float)

    scanning_process_df = processes_df[processes_df[ProcessesColumns.PROCESS_ID] == scanning_process_id]
    sub_scanning_process_df = slice_df(scanning_process_df, 5)

    summary_df = pd.DataFrame(columns=["Metric", "Value"])
    summary_df.loc[len(summary_df.index)] = ["Duration", total_finishing_time]

    cpu_process = pd.to_numeric(sub_scanning_process_df[ProcessesColumns.CPU_CONSUMPTION]).mean()
    cpu_total = sub_cpu_df[CPUColumns.USED_PERCENT].mean()
    summary_df.loc[len(summary_df.index)] = ["CPU Process", cpu_process]
    summary_df.loc[len(summary_df.index)] = ["CPU Total", cpu_total]
    summary_df.loc[len(summary_df.index)] = ["CPU Process / CPU Total", cpu_process / cpu_total]

    """for index, core_name in enumerate(cores_names_list):
        summary_df.loc[len(summary_df.index)] = [f"CPU core {index + 1} (%)", sub_cpu_df[core_name].mean()]

    summary_df.loc[len(summary_df.index)] = ["Min CPU Process",
                                             pd.to_numeric(sub_scanning_process_df[ProcessesColumns.CPU_CONSUMPTION]).min()]
    summary_df.loc[len(summary_df.index)] = ["Max CPU Process",
                                             pd.to_numeric(sub_scanning_process_df[ProcessesColumns.CPU_CONSUMPTION]).max()]
    summary_df.loc[len(summary_df.index)] = ["Min CPU Total", sub_cpu_df[CPUColumns.USED_PERCENT].min()]
    summary_df.loc[len(summary_df.index)] = ["Max CPU Total", sub_cpu_df[CPUColumns.USED_PERCENT].max()]"""

    process_memory = pd.to_numeric(sub_scanning_process_df[ProcessesColumns.USED_MEMORY]).mean()
    total_memory = sub_memory_df[MemoryColumns.USED_MEMORY].mean() * KB
    summary_df.loc[len(summary_df.index)] = ["Memory Process (MB)", process_memory]
    summary_df.loc[len(summary_df.index)] = ["Memory Total (MB)", total_memory]
    summary_df.loc[len(summary_df.index)] = ["Process Memory / Memory Total", process_memory / total_memory]

    #summary_df.loc[len(summary_df.index)] = ["IO Read Process (KB per second)", pd.to_numeric(scanning_process_df[ProcessesColumns.READ_BYTES]).sum() / process_finishing_time]
    summary_df.loc[len(summary_df.index)] = ["IO Read Process (KB - sum)", pd.to_numeric(scanning_process_df[ProcessesColumns.READ_BYTES]).sum()]
    #summary_df.loc[len(summary_df.index)] = ["IO Read Count Process (# per second)", pd.to_numeric(scanning_process_df[ProcessesColumns.READ_COUNT]).sum() / process_finishing_time]
    summary_df.loc[len(summary_df.index)] = ["IO Read Count Process (# - sum)", pd.to_numeric(scanning_process_df[ProcessesColumns.READ_COUNT]).sum()]

    #summary_df.loc[len(summary_df.index)] = ["IO Write Process (KB per second)", pd.to_numeric(scanning_process_df[ProcessesColumns.WRITE_BYTES]).sum() / process_finishing_time]
    summary_df.loc[len(summary_df.index)] = ["IO Write Process (KB - sum)", pd.to_numeric(scanning_process_df[ProcessesColumns.WRITE_BYTES]).sum()]
    #summary_df.loc[len(summary_df.index)] = ["IO Write Process Count (# per second)", pd.to_numeric(scanning_process_df[ProcessesColumns.WRITE_COUNT]).sum() / process_finishing_time]
    summary_df.loc[len(summary_df.index)] = ["IO Write Process Count (# - sum)", pd.to_numeric(scanning_process_df[ProcessesColumns.WRITE_COUNT]).sum()]

    #summary_df.loc[len(summary_df.index)] = ["Disk IO Read Total (KB per second)", sub_disk_df[DiskIOColumns.READ_BYTES].sum() / disk_finishing_time]
    summary_df.loc[len(summary_df.index)] = ["Disk IO Read Total (KB - sum)", sub_disk_df[DiskIOColumns.READ_BYTES].sum()]
    #summary_df.loc[len(summary_df.index)] = ["Disk IO Read Count Total (# per second)", sub_disk_df[DiskIOColumns.READ_COUNT].sum() / disk_finishing_time]
    summary_df.loc[len(summary_df.index)] = ["Disk IO Read Count Total (# - sum)", sub_disk_df[DiskIOColumns.READ_COUNT].sum()]

    #summary_df.loc[len(summary_df.index)] = ["Disk IO Write Total (KB per second)", sub_disk_df[DiskIOColumns.WRITE_BYTES].sum() / disk_finishing_time]
    summary_df.loc[len(summary_df.index)] = ["Disk IO Write Total (KB - sum)", sub_disk_df[DiskIOColumns.WRITE_BYTES].sum()]
    #summary_df.loc[len(summary_df.index)] = ["Disk IO Write Count Total (# per second)", sub_disk_df[DiskIOColumns.WRITE_COUNT].sum() / disk_finishing_time]
    summary_df.loc[len(summary_df.index)] = ["Disk IO Write Count Total (# - sum)", sub_disk_df[DiskIOColumns.WRITE_COUNT].sum()]

    summary_df.loc[len(summary_df.index)] = ["IO Read Process / Total (KB - sum)", (pd.to_numeric(scanning_process_df[ProcessesColumns.READ_BYTES]).sum()) / sub_disk_df[DiskIOColumns.READ_BYTES].sum()]
    summary_df.loc[len(summary_df.index)] = ["IO Read Count Process / Total (# - sum)", (pd.to_numeric(scanning_process_df[ProcessesColumns.READ_COUNT]).sum()) / sub_disk_df[DiskIOColumns.READ_COUNT].sum()]
    summary_df.loc[len(summary_df.index)] = ["IO Write Process / Total  (KB - sum)", (pd.to_numeric(scanning_process_df[ProcessesColumns.WRITE_BYTES]).sum()) / sub_disk_df[DiskIOColumns.WRITE_BYTES].sum()]
    summary_df.loc[len(summary_df.index)] = ["IO Write Process Count / Total  (# - sum)", (pd.to_numeric(scanning_process_df[ProcessesColumns.WRITE_COUNT]).sum()) / sub_disk_df[DiskIOColumns.WRITE_COUNT].sum()]

    summary_df.loc[len(summary_df.index)] = ["Disk IO Read Time (ms - sum)", sub_disk_df[DiskIOColumns.READ_TIME].sum()]
    summary_df.loc[len(summary_df.index)] = ["Disk IO Write Time (ms - sum)", sub_disk_df[DiskIOColumns.WRITE_TIME].sum()]

    battery_drop = calc_delta_capacity()
    summary_df.loc[len(summary_df.index)] = ["Energy consumption - total energy(mwh)", battery_drop[0]]
    summary_df.loc[len(summary_df.index)] = ["Battery Drop( %)", battery_drop[1]]
    summary_df.loc[len(summary_df.index)] = ["Trees (KG)", convert_mwh_to_other_metrics(battery_drop[0])[3]]

    def colors_func(df):
        return ['background-color: #FFFFFF'] + ['background-color: #FFFF00' for _ in range(3)] + \
               ['background-color: #9CC2E5' for _ in range(3)] + ['background-color: #66FF66' for _ in range(3)] + \
               ['background-color: #00B050' for _ in range(4)] + ['background-color: #70ad47' for _ in range(4)] + \
               ['background-color: #FFC000' for _ in range(2)] + ['background-color: #FFFFFF']

    styled_summary_df = summary_df.style.apply(colors_func, axis=0)

    styled_summary_df.to_excel(SUMMARY_CSV, engine='openpyxl', index=False)


def ignore_last_results():
    global processes_df
    global memory_df
    global disk_io_each_moment_df
    global cpu_df
    global battery_df

    if processes_df.empty:
        processes_num_last_measurement = 0
    else:
        processes_num_last_measurement = processes_df[ProcessesColumns.TIME].value_counts()[processes_df[ProcessesColumns.TIME].max()]
    processes_df = processes_df.iloc[:-processes_num_last_measurement, :]
    memory_df = memory_df.iloc[:-1, :]
    disk_io_each_moment_df = disk_io_each_moment_df.iloc[:-1, :]
    if not battery_df.empty:
        battery_df = battery_df.iloc[:-1, :]
    cpu_df = cpu_df.iloc[:-1, :]


def save_results_to_files():
    save_general_information_after_scanning()
    ignore_last_results()

    processes_df.to_csv(PROCESSES_CSV, index=False)
    memory_df.to_csv(TOTAL_MEMORY_EACH_MOMENT_CSV, index=False)
    disk_io_each_moment_df.to_csv(DISK_IO_EACH_MOMENT, index=False)
    if not battery_df.empty:
        battery_df.to_csv(BATTERY_STATUS_CSV, index=False)
    cpu_df.to_csv(TOTAL_CPU_CSV, index=False)

    prepare_summary_csv()


def calc_delta_capacity():
    if battery_df.empty:
        return 0, 0
    before_scanning_capacity = battery_df.iloc[0].at[BatteryColumns.CAPACITY]
    current_capacity = battery_df.iloc[len(battery_df) - 1].at[BatteryColumns.CAPACITY]

    before_scanning_percent = battery_df.iloc[0].at[BatteryColumns.PERCENTS]
    current_capacity_percent = battery_df.iloc[len(battery_df) - 1].at[BatteryColumns.PERCENTS]
    return before_scanning_capacity - current_capacity, before_scanning_percent - current_capacity_percent


def is_delta_capacity_achieved():
    if psutil.sensors_battery() is None:  # if desktop computer (has no battery)
        return True

    return calc_delta_capacity()[0] >= MINIMUM_DELTA_CAPACITY


def change_power_plan(name=balanced_power_plan_name, guid=balanced_power_plan_guid):
    result = subprocess.run(["powershell", "-Command", "powercfg /s " + guid], capture_output=True)
    if result.returncode != 0:
        raise Exception(f'An error occurred while switching to the power plan: {name}', result.stderr)


def scan_and_measure():
    global done_scanning
    global starting_time
    global scanning_process_id
    starting_time = time.time()

    measurements_thread = Thread(target=continuously_measure, args=())
    measurements_thread.start()

    while not program_to_scan == ProgramToScan.NO_SCAN and not done_scanning:
        powershell_process = subprocess.Popen(["powershell", "-Command", program.get_command()],
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        scanning_process_id = program.find_child_id(powershell_process.pid)
        result = powershell_process.wait()
        errs = powershell_process.stderr.read().decode()
        #outs, errs = powershell_process.communicate()

        finished_scanning_time.append(calc_time_interval())
        if scan_option == ScanMode.ONE_SCAN or (min_scan_time_passed() and is_delta_capacity_achieved()):
            done_scanning = True
        if result != 0:
            raise Exception("An error occurred while scanning: %s", errs)

    measurements_thread.join()
    if program_to_scan == ProgramToScan.NO_SCAN:
        finished_scanning_time.append(calc_time_interval())


def can_proceed_towards_measurements():
    if os.path.exists(base_dir):

        button_selected = message_box("Deleting Previous Results",
                                      "Running the program will override the results of the previous measurement.\n\n"
                                      "Are you sure you want to continue?", 4)

        if button_selected == YES_BUTTON:
            shutil.rmtree(base_dir)       # remove previous data
            return True
        else:
            return False
    else:
        return True


def change_sleep_and_turning_screen_off_settings(screen_time=DEFAULT_SCREEN_TURNS_OFF_TIME,
                                                 sleep_time=DEFAULT_TIME_BEFORE_SLEEP_MODE):
    result_screen = subprocess.run(["powershell", "-Command", f"powercfg /Change monitor-timeout-dc {screen_time}"],
                                   capture_output=True)
    if result_screen.returncode != 0:
        raise Exception(f'An error occurred while changing turning off the screen to never', result_screen.stderr)

    result_sleep_mode = subprocess.run(["powershell", "-Command", f"powercfg /Change standby-timeout-dc {sleep_time}"],
                                       capture_output=True)
    if result_sleep_mode.returncode != 0:
        raise Exception(f'An error occurred while disabling sleep mode', result_sleep_mode.stderr)


def change_real_time_protection(should_disable=True):
    protection_mode = "1" if should_disable else "0"
    result = subprocess.run(["powershell", "-Command",
                             f'Start-Process powershell -ArgumentList("Set-MpPreference -DisableRealTimeMonitoring {protection_mode}") -Verb runAs -WindowStyle hidden'],
                            capture_output=True)
    if result.returncode != 0:
        raise Exception("Could not change real time protection", result.stderr)


def is_tamper_protection_enabled():
    result = subprocess.run(["powershell", "-Command", "Get-MpComputerStatus | Select IsTamperProtected | Format-List"],
                            capture_output=True)
    if result.returncode != 0:
        raise Exception("Could not check if tamper protection enabled", result.stderr)

    # return bool(re.search("IsTamperProtected\s*:\sTrue", str(result.stdout)))
    return get_powershell_result_list_format(result.stdout)[0]["IsTamperProtected"] == "True"


def main():
    print("======== Process Monitor ========")

    battery = psutil.sensors_battery()
    if battery is not None and battery.power_plugged:  # ensure that charging cable is unplugged in laptop
        raise Exception("Unplug charging cable during measurements!")

    if disable_real_time_protection_during_measurement and is_tamper_protection_enabled():
        raise Exception("You must disable Tamper Protection manually so that the program could control real "
                        "time Protection")

    if not can_proceed_towards_measurements():
        print("Exiting program")
        return

    change_power_plan(chosen_power_plan_name, chosen_power_plan_guid)

    if disable_real_time_protection_during_measurement:
        change_real_time_protection()

    change_sleep_and_turning_screen_off_settings(NEVER_TURN_SCREEN_OFF, NEVER_GO_TO_SLEEP_MODE)

    sbc.set_brightness(screen_brightness_level)

    psutil.cpu_percent()  # first call is meaningless

    Path(GRAPHS_DIR).mkdir(parents=True, exist_ok=True)     # create empty dirs

    save_general_information_before_scanning()

    scan_and_measure()

    save_results_to_files()

    change_power_plan()  # return to balanced

    change_sleep_and_turning_screen_off_settings()  # return to default - must be after changing power plan

    if disable_real_time_protection_during_measurement:
        change_real_time_protection(should_disable=False)

    print("Finished scanning")


if __name__ == '__main__':
    main()
