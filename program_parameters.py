from general_consts import *

# ======= Scanner Parameters =======
main_program_to_scan = ProgramToScan.NO_SCAN
background_programs_types = []  #[ProgramToScan.DummyANTIVIRUS, ProgramToScan.Perfmon]

kill_background_process_when_main_finished = True
summary_version = SummaryVersion.OTHER

scanner_version = ScannerVersion.WITHOUT_BATTERY

power_plan = PowerPlan.BALANCED
scan_option = ScanMode.ONE_SCAN
scan_type = ScanType.QUICK_SCAN  # relevant only for one scan or continuous scan

antivirus_type = AntivirusType.DEFENDER

file_type = "pdf"
directory_type = "Duplicated Files"
custom_scan_path = r'""'

RUNNING_TIME = 1 * MINUTE # insert time (e.g. 0.5 * MINUTE) or None in case you want to wait until process ends in ONE_SCAN mode

MINIMUM_DELTA_CAPACITY = 200     # in mWh

measurement_number = NEW_MEASUREMENT    # write number between 1->inf or type NEW_MEASUREMENT

# MUST disable tamper protection manually for this feature to work
disable_real_time_protection_during_measurement = False  # must use administrator permissions

screen_brightness_level = 75    # A number between 0 and 100

# return to default settings (can be costumed)
DEFAULT_SCREEN_TURNS_OFF_TIME = 4
DEFAULT_TIME_BEFORE_SLEEP_MODE = 4

is_inside_container = True

# ==== Elastic logging configuration
elastic_url = "http://192.168.140.101:9200"
elastic_username = "elastic"
elastic_password = "71BPiEiQ"

# ==== ClamAV configurations
recursive = True
should_optimize = False
should_mitigate_timestomping = False

# ==== IDS configurations
ids_type = IDSType.SNORT
pcap_list_dirs = [] #["/home/user/Downloads/Friday-WorkingHours.pcap"]         # enter list of pcap diretories. If you to wish sniff packets instead, enter empty list or None.
interface_name = "wlp0s20f3"    # "ens33"  #wlp0s20f3     # enter interface name to listen on or None if you want to use pcap files
log_path = "/var/log/snort"
configuration_file_path = "/etc/snort/snort.conf"


model_name = "logdeep"
script_relative_path = r"demo.deeplog"
installation_dir = r"C:\Users\Administrator\Repositories"
model_action = 'predict'# train or predict

# For CPU CONSUMER program
cpu_percent_to_consume = 20

# For MEMORY CONSUMER program
memory_chunk_size = 1 * MB
consumption_speed = 10 * MB / SECOND

# For Network Sender
time_interval = 0.2
