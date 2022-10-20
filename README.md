# GreenSecurity-FirstExperiment

The main goal of this project is to understand the energy consumption of different anti virus scans, on each of the hardware components separately.
This project measures various metrics of computer resource usage (such as: CPU usage, memory, disk and battery) while running anti virus scans. 
We measure the resource usage of the whole system, and of each process separately.

### Modes of execution
There are 3 modes of execution:
1. No scan - the code measures the resource usage without running any anti virus scan. This is the baseline consumption of other processes running in the background.
2. One scan - the code starts a scan using windows defender. Meanwhile, the code keeps measuring the resource usage. It is possible to configure what type of scan to perform, as shown below
3. Continuous scan - the code runs a scan using windows defender, and when the scan is over it immediately starts a new one. The code ends when at least a certain amount of time has elapsed and at least a certain amount of battery capcity has been decreased. It is possible to configure these parameters and the type of scan, as shown below.

The results are saved in the relevant folders. the raw resource usage data is saved in csv and txt files. graphs are saved in png files.

### Scan types
1. Full scan - scans every file and program on the computer. May take few hours or daysto complete, depending on the amount and type of data that needs to be scanned.
2. Quick scan - looks at all the locations where there could be malware registered to start with the system, such as registry keys and known Windows startup folders. Provides strong protection against malware that starts with the system and kernel-level malware. May take a few seconds or minutes.
3. Custom scan - enables to scan exactly the files and folders that you select.

### Code components
The main components of the code:
1. Scanner - responsible to record the raw data in each mode and save it in the relevant csv and txt files
2. Analyzer - responsible to create graphs from raw data saved by the scanner. The graphs are saved as pdf files.


### Program parameters
1. scan_option - enables to choose the mode of execution (No scan, One scan, Continuous scan)
2. scan_type - enables to choose the type of antivirus scan (full scan, quick scan.....)
3. MINIMUM_DELTA_CAPACITY - enables to define the minimum battery drop required before the code ends. Relevant only in continuous scan mode.
4. MINIMUM_SCAN_TIME - enables to define the minimum time required for antivirus scans before the code ends. Relevant only in continuous scan mode.
