ALL_MEASUREMENTS_DIRS_PATH = r"C:\Users\Administrator\Desktop\green security\tmp" #fr"C:\Users\sapir\Desktop\University\Second Degree\Green Security\Green Security Experiments\Experiments - HeavyLoad combinations - No scan\No scan - heavyLoad - 200525\No scan - heavyLoad - different combinations"

IS_NO_SCAN_MODE = True

NO_SCAN_HEAVYLOAD_PROCESS = "HeavyLoad.exe"
RANDOM_PROCESS_WITH_OTHER_SUMMARY_VERSION = "java"  # todo: change to actual process when there are real results
PROCESS_NAME = NO_SCAN_HEAVYLOAD_PROCESS if IS_NO_SCAN_MODE else RANDOM_PROCESS_WITH_OTHER_SUMMARY_VERSION  # change accordingly to the process that we want to monitor in the all processes file

NETWORK_SENT_BYTES_COLUMN_NAME = None  # ProcessColumns.NETWORK_BYTES_SENT_PROCESS_COL # can be ether this or None (or system same name)
NETWORK_RECEIVED_BYTES_COLUMN_NAME = None  # ProcessColumns.NETWORK_BYTES_RECEIVED_PROCESS_COL # can be ether this or None (or system same name)

MODEL_FILE_NAME = r"C:\Users\Administrator\Desktop\GreenSecurityAll\framework_code\measurements_model\energy_prediction_model.pkl"
