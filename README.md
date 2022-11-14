# GreenSecurity-FirstExperiment

The main goal of this project is to understand the energy consumption of different anti virus scans, on each of the hardware components separately.
This project measures various metrics of computer resource usage (such as: CPU usage, memory, disk and battery) while running anti virus scans. 
We measure the resource usage of the whole system, and of each process separately. It is possible to easily configure program parameters inside the configuration file as detailed below.

## To get optimal results, make sure that:
1. any unnecessary program is closed (including backgroud procceses)
2. all result files are closed
3. auto brightness disabled
4. reboot computer after each measurement 

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

### Power plans
It is possible to configure the computer's power plan during measurements. The available plans:
1. High Performance
2. Balanced
3. Power Saver

### Code components
The main components of the code:
1. Scanner - responsible to record the raw data in each mode and save it in the relevant csv and txt files
2. Analyzer - responsible to create graphs from raw data saved by the scanner. The graphs are saved as pdf files.
3. Configurations - responsible to easily configure power plan, scan mode, scan type, results path, minmum time and battary drain (for Continuous scan only), path of dir to scan (for Custom scan only)


### Program parameters (can easily be modified in Configurations file)
1. power_plan - enables to choose the the comuter's power plan (High Performance, Balanced, Power Saver)
2. scan_option - enables to choose the mode of execution (No scan, One scan, Continuous scan)
3. scan_type - enables to choose the type of antivirus scan (full scan, quick scan.....)
4. custom_scan_path - enables to choose the directory / file to scan (for Custom scan only)
5. MINIMUM_DELTA_CAPACITY - enables to define the minimum battery drop required before the code ends. Relevant only in continuous scan mode.
6. MINIMUM_SCAN_TIME - enables to define the minimum time required for antivirus scans before the code ends. Relevant only in continuous scan mode.
7. measurement_number - if equals to NEW_MEASUREMENT, result will be saved in new folder. It is possible to define specific number
8. disable_real_time_protection_during_measurement - if True, code will disable defender's real time protection during measurements and turn it on before ending. IMPORTANT - in this case, Tamper Protection MUST be turned off manually, and Scanner.py must run in admin mode
9. screen_brightness_level - enables to define the brightness of screen - a value between 0 to 100


## Experiment process
1. Disable auto brightness of the device
2. Disable Tamper Protection (can be done once before all experiments)
3. Change configuration.py parameters according to the experiment that you want to perform
4. Before each measurement, restart the device
5. Make sure that the device is connected to the internet
6. Disconnect charging cable (code will verfiy that)
7. Run scanner.py in admin mode (in order to permit code to disable real time protection before each measurement. The code turns it on before ending)
8. Code will change power plan according to configuration file
9. Code will disable real time protection
10. Code will prevent device from sleeping and turnning off screen
11. Code will set screen brightness according to configuration file
12. Code will start defender scan and measure the device's resource consumption 
13. Code will save results into files
14. Code will back to default settings

### For Developers:
* use "git update-index --skip-worktree program_parameters.py" command to ignore changes in this file.
* use "git update-index --no-skip-worktree program_parameters.py" command to resume tracking changes
