from dotenv import load_dotenv
import os
from Scanner.general_consts import *

# Load the .env file
load_dotenv()

# Access variables as environment variables. Convert to appropriate types
main_program_to_scan = ProgramToScan[os.getenv("MAIN_PROGRAM_TO_SCAN")]
background_programs_types = [ProgramToScan[x] for x in os.getenv("BACKGROUND_PROGRAMS_TYPES").split(",")] if os.getenv("BACKGROUND_PROGRAMS_TYPES") else []
kill_background_process_when_main_finished = os.getenv("KILL_BACKGROUND_PROCESS_WHEN_MAIN_FINISHED") == "True"
summary_version = SummaryVersion[os.getenv("SUMMARY_VERSION")]
scanner_version = ScannerVersion[os.getenv("SCANNER_VERSION")]
power_plan = getattr(PowerPlan, os.getenv("POWER_PLAN"))
scan_option = ScanMode[os.getenv("SCAN_OPTION")]
scan_type = getattr(ScanType, os.getenv("SCAN_TYPE"))
file_type = os.getenv("FILE_TYPE")
directory_type = os.getenv("DIRECTORY_TYPE")
custom_scan_path = os.getenv("CUSTOM_SCAN_PATH")
RUNNING_TIME = float(os.getenv("RUNNING_TIME")) * MINUTE
MINIMUM_DELTA_CAPACITY = int(os.getenv("MINIMUM_DELTA_CAPACITY"))
measurement_number =  NEW_MEASUREMENT    # write number between 1->inf or type NEW_MEASUREMENT
disable_real_time_protection_during_measurement = os.getenv("DISABLE_REAL_TIME_PROTECTION_DURING_MEASUREMENT") == "True"
screen_brightness_level = int(os.getenv("SCREEN_BRIGHTNESS_LEVEL"))
DEFAULT_SCREEN_TURNS_OFF_TIME = int(os.getenv("DEFAULT_SCREEN_TURNS_OFF_TIME"))
DEFAULT_TIME_BEFORE_SLEEP_MODE = int(os.getenv("DEFAULT_TIME_BEFORE_SLEEP_MODE"))
ids_type = getattr(IDSType, os.getenv("IDS_TYPE"))
pcap_list_dirs = os.getenv("PCAP_LIST_DIRS").split(",") if os.getenv("PCAP_LIST_DIRS") else []
interface_name = os.getenv("INTERFACE_NAME")
log_path = os.getenv("LOG_PATH")
configuration_file_path = os.getenv("CONFIGURATION_FILE_PATH")
model_name = os.getenv("MODEL_NAME")
script_relative_path = os.getenv("SCRIPT_RELATIVE_PATH")
installation_dir = os.getenv("INSTALLATION_DIR")
model_action = os.getenv("MODEL_ACTION")
cpu_percent_to_consume = int(os.getenv("CPU_PERCENT_TO_CONSUME"))
memory_chunk_size = int(os.getenv("MEMORY_CHUNK_SIZE"))
consumption_speed = int(os.getenv("CONSUMPTION_SPEED"))

# Print variables to check
print(f"main_program_to_scan: {main_program_to_scan}")
print(f"background_programs_types: {background_programs_types}")
print(f"kill_background_process_when_main_finished: {kill_background_process_when_main_finished}")
print(f"summary_version: {summary_version}")
print(f"scanner_version: {scanner_version}")
print(f"power_plan: {power_plan}")
print(f"scan_option: {scan_option}")
print(f"scan_type: {scan_type}")
print(f"no_process_programs: {no_process_programs}")
print(f"file_type: {file_type}")
print(f"directory_type: {directory_type}")
print(f"custom_scan_path: {custom_scan_path}")
print(f"running_time: {RUNNING_TIME}")
print(f"minimum_delta_capacity: {MINIMUM_DELTA_CAPACITY}")
print(f"measurement_number: {measurement_number}")
print(f"disable_real_time_protection_during_measurement: {disable_real_time_protection_during_measurement}")
print(f"screen_brightness_level: {screen_brightness_level}")
print(f"default_screen_turns_off_time: {DEFAULT_SCREEN_TURNS_OFF_TIME}")
print(f"default_time_before_sleep_mode: {DEFAULT_TIME_BEFORE_SLEEP_MODE}")
print(f"ids_type: {ids_type}")
print(f"pcap_list_dirs: {pcap_list_dirs}")
print(f"interface_name: {interface_name}")
print(f"log_path: {log_path}")
print(f"configuration_file_path: {configuration_file_path}")
print(f"model_name: {model_name}")
print(f"script_relative_path: {script_relative_path}")
print(f"installation_dir: {installation_dir}")
print(f"model_action: {model_action}")
print(f"cpu_percent_to_consume: {cpu_percent_to_consume}")
print(f"memory_chunk_size: {memory_chunk_size}")
print(f"consumption_speed: {consumption_speed}")
