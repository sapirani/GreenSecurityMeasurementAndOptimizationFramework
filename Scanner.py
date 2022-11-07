import shutil
import psutil
import pythoncom
import wmi
from prettytable import PrettyTable
import subprocess
from threading import Thread
import time
import pandas as pd
import platform
from configurations import *
import ctypes
from datetime import date
from pathlib import Path
import re


class PreviousDiskIO:
    def __init__(self, disk_io):
        self.read_count = disk_io.read_count
        self.write_count = disk_io.write_count
        self.read_bytes = disk_io.read_bytes
        self.write_bytes = disk_io.write_bytes


# ======= Constants =======
SYSTEM_IDLE_PROCESS_NAME = "System Idle Process"
SYSTEM_IDLE_PID = 0

YES_BUTTON = 6
NO_BUTTON = 7

NEVER_TURN_SCREEN_OFF = 0
NEVER_GO_TO_SLEEP_MODE = 0

base_dir, GRAPHS_DIR, PROCESSES_CSV, TOTAL_MEMORY_EACH_MOMENT_CSV, DISK_IO_EACH_MOMENT, \
BATTERY_STATUS_CSV, GENERAL_INFORMATION_FILE, TOTAL_CPU_CSV = result_paths()

# ======= Program Global Parameters =======
done_scanning = False
starting_time = time.time()

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
        f'{(disk_io_stat.write_bytes - previous_disk_io.write_bytes) / KB:.3f}'
    ]

    return disk_io_stat


def save_current_processes_statistics(prev_io_per_process):
    proc = []

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

    sorted_list = sorted(top.items(), key=lambda x: x[1], reverse=True)

    return add_to_processes_dataframe(time_of_sample, sorted_list, prev_io_per_process)


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

                prev_io_per_process[p.pid] = PreviousDiskIO(io_stat)

        except Exception:
            pass

    return prev_io_per_process


def min_scan_time_passed():
    return time.time() - starting_time >= MINIMUM_SCAN_TIME


def should_scan():
    if scan_option == ScanMode.NO_SCAN:
        return not min_scan_time_passed()
    elif scan_option == ScanMode.ONE_SCAN:
        return not done_scanning
    elif scan_option == ScanMode.CONTINUOUS_SCAN:
        return not min_scan_time_passed() and not is_delta_capacity_achieved()


def save_current_total_cpu():
    cpu_df.loc[len(cpu_df.index)] = [
        calc_time_interval(),
        psutil.cpu_percent()
    ]


def continuously_measure():
    pythoncom.CoInitialize()

    # init PreviousDiskIO by first disk io measurements (before scan)
    prev_disk_io = PreviousDiskIO(psutil.disk_io_counters())
    prev_io_per_process = {}

    # TODO: think if total tables should be printed only once
    while should_scan():
        save_battery_stat()
        prev_io_per_process = save_current_processes_statistics(prev_io_per_process)
        save_current_total_cpu()
        save_current_total_memory()
        prev_disk_io = save_current_disk_io(prev_disk_io)

        # Create a delay
        time.sleep(0.5)


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


def save_general_system_information(f):
    f.write("======System Information======\n")
    my_system = platform.uname()

    f.write(f"System: {my_system.system}\n")
    f.write(f"Node Name: {my_system.node}\n")
    f.write(f"Release: {my_system.release}\n")
    f.write(f"Version: {my_system.version}\n")
    f.write(f"Machine: {my_system.machine}\n")
    f.write(f"Processor: {my_system.processor}\n")
    f.write(f"Total RAM: {psutil.virtual_memory().total / GB} GB\n")

    f.write("\n----CPU Information----\n")
    f.write(f"Physical cores: {psutil.cpu_count(logical=False)}\n")
    f.write(f"Total cores: {psutil.cpu_count(logical=True)}\n")
    cpufreq = psutil.cpu_freq()
    f.write(f"Max Frequency: {cpufreq.max:.2f} MHz\n")
    f.write(f"Min Frequency: {cpufreq.min:.2f} MHz\n")


def save_general_information_before_scanning():
    with open(GENERAL_INFORMATION_FILE, 'w') as f:
        # dd/mm/YY
        f.write(f'Date: {date.today().strftime("%d/%m/%Y")}\n\n')
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

        if scan_option == ScanMode.NO_SCAN:
            measurement_time = calc_time_interval()
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


def save_results_to_files():
    save_general_information_after_scanning()
    processes_df.iloc[:-1, :].to_csv(PROCESSES_CSV, index=False)
    memory_df.iloc[:-1, :].to_csv(TOTAL_MEMORY_EACH_MOMENT_CSV, index=False)
    disk_io_each_moment_df.iloc[:-1, :].to_csv(DISK_IO_EACH_MOMENT, index=False)
    if not battery_df.empty:
        battery_df.iloc[:-1, :].to_csv(BATTERY_STATUS_CSV, index=False)
    cpu_df.iloc[:-1, :].to_csv(TOTAL_CPU_CSV, index=False)


def calc_delta_capacity():
    if battery_df.empty:
        return 0
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
    measurements_thread = Thread(target=continuously_measure, args=())
    measurements_thread.start()

    while not scan_option == ScanMode.NO_SCAN and not done_scanning:
        # TODO check about capture_output
        result = subprocess.run(["powershell", "-Command", f"Start-MpScan -ScanType {scan_type}" + custom_scan_query],
                                capture_output=True)
        finished_scanning_time.append(calc_time_interval())
        if scan_option == ScanMode.ONE_SCAN or (min_scan_time_passed() and is_delta_capacity_achieved()):
            done_scanning = True
        if result.returncode != 0:
            raise Exception("An error occurred while anti virus scan: %s", result.stderr)

    measurements_thread.join()


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
    result = subprocess.run(["powershell", "-Command", "Get-MpComputerStatus"], capture_output=True)
    if result.returncode != 0:
        raise Exception("Could not check if tamper protection enabled", result.stderr)

    return bool(re.search("IsTamperProtected\s*:\sTrue", str(result.stdout)))


def main():
    print("======== Process Monitor ========")

    battery = psutil.sensors_battery()
    if battery is not None and battery.power_plugged:  # ensure that charging cable is unplugged in laptop
        raise Exception("Unplug charging cable during measurements!")

    if is_tamper_protection_enabled() and disable_real_time_protection_during_measurement:
        raise Exception("You must disable Tamper Protection manually so that the program could control real "
                        "time Protection")

    if not can_proceed_towards_measurements():
        print("Exiting program")
        return

    change_power_plan(chosen_power_plan_name, chosen_power_plan_guid)

    if disable_real_time_protection_during_measurement:
        change_real_time_protection()

    change_sleep_and_turning_screen_off_settings(NEVER_TURN_SCREEN_OFF, NEVER_GO_TO_SLEEP_MODE)

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
