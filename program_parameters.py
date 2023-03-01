from general_consts import *

# ======= Scanner Parameters =======
main_program_to_scan = ProgramToScan.DummyANTIVIRUS
background_programs_types = []  #[ProgramToScan.DummyANTIVIRUS, ProgramToScan.Perfmon]#[ProgramToScan.Perfmon]

scanner_version = ScannerVersion.LITE

power_plan = PowerPlan.POWER_SAVER
scan_option = ScanMode.ONE_SCAN
scan_type = ScanType.QUICK_SCAN  # relevant only for one scan or continuous scan

file_type = "pdf"
directory_type = "Duplicated Files"
custom_scan_path = fr'"C:\Users\sagib\Downloads"'
#custom_scan_path = fr'"C:\Users\Administrator\Documents\GitHub\GreenSecurity-FirstExperiment\Data{directory_type}\{file_type}"'  # relevant only for custom scans. On other types, must be empty

MAX_SCAN_TIME = None    # insert time (e.g. 0.5 * MINUTE) or type None

MINIMUM_DELTA_CAPACITY = 20
MINIMUM_SCAN_TIME = 0.5 * MINUTE

measurement_number = 9001    # write number between 1->inf or type NEW_MEASUREMENT

# MUST disable tamper protection manually for this feature to work
disable_real_time_protection_during_measurement = False  # must use administrator permissions

screen_brightness_level = 75    # A number between 0 and 100

# return to default settings (can be costumed)
DEFAULT_SCREEN_TURNS_OFF_TIME = 4
DEFAULT_TIME_BEFORE_SLEEP_MODE = 4

# ==== IDS configurations
ids_type = IDSType.SNORT
pcap_list_dirs = ["path-to-pcap"]         # enter list of pcap diretories. If you to wish sniff packets instead, enter empty list or None.
interface_name = None    # "ens33"     # enter interface name to listen on or None if you want to use pcap files
log_path = "/var/log/snort"
configuration_file_path = "/etc/snort/snort.conf"


model_name = "logdeep"
script_relative_path = r"demo.deeplog"
installation_dir = r"C:\Users\Administrator\Repositories"
model_action = 'predict'# train or predict
