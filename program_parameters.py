from general_consts import *

# ======= Scanner Parameters =======
scan_command = ScanCommand.antivirus

power_plan = PowerPlan.BALANCED
scan_option = ScanMode.NO_SCAN
scan_type = ScanType.QUICK_SCAN  # relevant only for one scan or continuous scan
custom_scan_path = r""  # relevant only for custom scans. On other types, must be empty

MINIMUM_DELTA_CAPACITY = 20
MINIMUM_SCAN_TIME = 0.5 * MINUTE

measurement_number = NEW_MEASUREMENT    # write number between 1->inf or type NEW_MEASUREMENT

# MUST disable tamper protection manually for this feature to work
disable_real_time_protection_during_measurement = True  # must use administrator permissions

screen_brightness_level = 75    # A number between 0 and 100

# return to default settings (can be costumed)
DEFAULT_SCREEN_TURNS_OFF_TIME = 4
DEFAULT_TIME_BEFORE_SLEEP_MODE = 4
