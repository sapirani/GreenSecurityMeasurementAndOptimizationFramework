from general_consts import *

# ======= Scanner Parameters =======
main_program_to_scan = ProgramToScan.NO_SCAN
background_programs_types = []

power_plan = PowerPlan.POWER_SAVER
scan_option = ScanMode.ONE_SCAN
scan_type = ScanType.CUSTOM_SCAN  # relevant only for one scan or continuous scan
file_type = "pdf"
directory_type = "Duplicated Files"
custom_scan_path = fr'"C:\Users\Administrator\Documents\GitHub\GreenSecurity-FirstExperiment\Data{directory_type}\{file_type}"'  # relevant only for custom scans. On other types, must be empty

MINIMUM_DELTA_CAPACITY = 10
MINIMUM_SCAN_TIME = 10 * MINUTE

measurement_number = NEW_MEASUREMENT    # write number between 1->inf or type NEW_MEASUREMENT

# MUST disable tamper protection manually for this feature to work
disable_real_time_protection_during_measurement = True # must use administrator permissions

screen_brightness_level = 75    # A number between 0 and 100

# return to default settings (can be costumed)
DEFAULT_SCREEN_TURNS_OFF_TIME = 4
DEFAULT_TIME_BEFORE_SLEEP_MODE = 4


ids_type = IdsType.SURICATA
interface_name = "132.73.204.238"
log_dir = "LogFile"
