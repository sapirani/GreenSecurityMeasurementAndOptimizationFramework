import os

from utils.general_consts import ProgramToScan

DEFAULT_NUMBER_OF_EXPERIMENTS = 3
SLEEPING_TIME_BETWEEN_MEASUREMENTS = 15
SLEEPING_TIME_BETWEEN_TASKS = 15
DEFAULT_TASK_UNIT_SIZE = 1024
SCANNER_PROGRAM_FILE = "scanner.py"
SCANNER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), SCANNER_PROGRAM_FILE)
PROGRAM_PARAMETERS_FILE = "program_parameters.py"
PROGRAM_PARAMETERS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), PROGRAM_PARAMETERS_FILE)
DEFAULT_MAIN_PROGRAM = ProgramToScan.BASELINE_MEASUREMENT
DEFAULT_TASKS = [ProgramToScan.MemoryConsumer, ProgramToScan.MemoryReleaser,
                 ProgramToScan.DiskIOReadConsumer, ProgramToScan.DiskIOWriteConsumer,
                 ProgramToScan.CPUConsumer]
