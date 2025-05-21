# GreenSecurity-FirstExperiment

The main goal of this project is to understand the energy consumption of different cyber-security solutions (e.g. antivirus scans), on each of the hardware components separately.
This project measures various metrics of computer resource usage (such as: CPU usage, memory, disk and battery) while running cyber-security scans. 
We measure the resource usage of the whole system, and of each process separately. It is possible to easily configure program parameters inside the configuration file as detailed below. You can run multiple programs simultaneously. Both Windows and POSIX operation systems are supported.

## Preliminaries:
1. Please use python 3.8.5. 

2. Create python environment with the packages listed in requirements.txt.
   In windows:
   `pip install -r requirements.txt`
   In Linux:
   `sudo pip install -r requirements.txt`

2. To get optimal results, make sure that:
   * Any unnecessary program is closed (including background processes)
   * All result files are closed
   * Auto brightness disabled
   * Reboot computer after each measurement 


## Modules
The main components of the code:
1. `Scanner` - responsible to record the raw data in each mode and save it in the relevant csv and txt files
2. `Analyzer` - responsible to create graphs from raw data saved by the scanner. The graphs are saved as pdf files.
3. `Program` Parameters - responsible to easily configure power plan, scan mode, scan type, results path, minmum time and battary drain (for Continuous scan only), path of dir to scan (for Custom scan only)

## Running Instructions
1. Disable auto brightness of the device.
2. Disable Tamper Protection (can be done once before all experiments).
3. Change configuration.py parameters according to the experiment that you want to perform.
4. Before each measurement, restart the device.
5. Make sure that the device is connected to the internet.
6. Disconnect charging cable (code will verify that).
7. Run scanner.py in admin mode (in order to permit code to disable real time protection before each measurement. The code turns it on before ending).
8. Code will change power plan according to configuration file.
9. Code will disable real time protection.
10. Code will prevent device from sleeping and turning the screen off.
11. Code will set screen brightness according to configuration file.
12. Code will start the selected programs and measure the device's resource consumption.
13. Code will save results into files.
14. Code will back to default settings.

## Program Parameters
1. `main_program_to_scan` - the main program that you want to measure it's resource consumption
2. `background_programs_types` - a list of programs that will run in parallel to the main program. This list could be empty (meaning that no additional background program will run). Their resource consumption will be also measured
3. `power_plan` - the computer's power plan, [see options](#power-plans).
4. `scan_option` - the mode of execution, [see options](#modes-of-execution).
5. `scan_type` - the type of antivirus scan, [see options](#scan-types).
6. `custom_scan_path` - the directory / file to scan (for Custom scan only)
7. `MINIMUM_DELTA_CAPACITY` - enables to define the minimum battery drop required before the code ends. Relevant only in continuous scan mode.
8. `RUNNING_TIME` - enables to define the exact time of measurements in one scan mode. in continuous scam it will be the minimum time for the measurements [see modes of execution](#modes-of-execution).
9. `measurement_number` - if equals to NEW_MEASUREMENT, result will be saved in new folder. It is possible to define specific number
10. `disable_real_time_protection_during_measurement` - if True, code will disable defender's real time protection during measurements and turn it on before ending. IMPORTANT - in this case, Tamper Protection MUST be turned off manually, and Scanner.py must run in admin mode
11. `screen_brightness_level` - enables to define the brightness of screen - a value between 0 and 100
12. `scanner_version` - define the metrics that will be measured [see options](#measurement-versions)

### Modes of Execution
There are 3 modes of execution:
1. No scan - the code measures the resource usage without running any antivirus scan. This is the baseline consumption of other processes running in the background.
2. One scan - the code starts a scan using windows defender. Meanwhile, the code keeps measuring the resource usage. It is possible to configure what type of scan to perform, as shown below
3. Continuous scan - the code runs a scan using windows defender, and when the scan is over it immediately starts a new one. The code ends when at least a certain amount of time has elapsed and at least a certain amount of battery capcity has been decreased. It is possible to configure these parameters and the type of scan, as shown below.

The results are saved in the relevant folders. the raw resource usage data is saved in csv and txt files. graphs are saved in png files.

### Scan types
1. Full scan - scans every file and program on the computer. May take few hours or days to complete, depending on the amount and type of data that needs to be scanned.
2. Quick scan - looks at all the locations where there could be malware registered to start with the system, such as registry keys and known Windows startup folders. Provides strong protection against malware that starts with the system and kernel-level malware. May take a few seconds or minutes.
3. Custom scan - enables to scan exactly the files and folders that you select.


### Measurement Versions
1. FULL - measures battery, and for each process separately:  CPU usage, RAM usage, number of disk io reads and writes, size of disk io reads and writes
2. LITE - identical to the full version except that this version does not measure the battery capacity
3. WITHOUT_BATTERY - identical to the full version except that this version does not measure the battery capacity


### Power plans
It is possible to configure the computer's power plan during measurements. The available plans:
1. High Performance
2. Balanced
3. Power Saver

### Available Programs
This project currently supports the following programs (it is very easy to add another program - [perform the following steps](#supporting-additional-programs)):
1. Windows Defender Antivirus
2. Snort Intrusion Detection System (IDS)
3. Suricata Intrusion Detection System (IDS)
4. Windows Performance Monitor (another source to validate the measurements results)
5. Typical User Activity (web browsing, edit a word document, etc.)
6. Splunk


### For Developers:
* use "git update-index --skip-worktree program_parameters.py" command to ignore changes in this file.
* use "git update-index --no-skip-worktree program_parameters.py" command to resume tracking changes

#### Supporting Additional Programs:
1. in `general_consts.py` file - add your program in the enum called *ProgramToScan*
2. in `program_parameters` file - add all the parameters that the user can configure in your program
3. in `program class` file - add a class that represents your program and inherits from *ProgramInterface*. You ***MUST*** implement the funtions:  *get_program_name* and *get_command*. The function *get_command* returns a string which is the shell command that runs your program. You can implement any other function of *ProgramInterface*. Note that if you want to run your command in powershell (for Windows programs), implement the function *should_use_powershell* and return True.
4. in `initialization_helper.py` file - add your program in the function called *program_to_scan_factory*

## Execution Example:
1) When you want to run the windows defender antivirus and measure it's energy consumption when scanning a folder (in path dir) once, with no other processes running in the background, while the power mode of the computer should be power saver, you should change the next parameters in the file program_parameters.py:

* `main_program_to_scan` = ProgramToScan.ANTIVIRUS
* `background_programs_types` = []  
* `power_plan` = PowerPlan.POWER_SAVER
* `scan_option` = ScanMode.ONE_SCAN
* `scan_type` = ScanType.CUSTOM_SCAN
* `custom_scan_path` = dir
* `disable_real_time_protection_during_measurement` = true

Next, run in Windows, in the command line the command - `python scanner.py` or in Linux in the terminal the command - `sudo python3 scanner.py`

2) When you want to run the windows defender antivirus and measure its energy consumption when executing a full scan once while there are other processes running in the background (Dummy process, performance monitor process), with power mode equals to power saver, you should change the next parameters in the file `program_parameters.py`:

* `main_program_to_scan` = ProgramToScan.ANTIVIRUS
* `background_programs_types` = [ProgramToScan.DummyANTIVIRUS, ProgramToScan.Perfmon]  
* `power_plan` = PowerPlan.POWER_SAVER
* `scan_option` = ScanMode.ONE_SCAN
* `scan_type` = ScanType.FULL_SCAN
* `custom_scan_path` = ""
* `disable_real_time_protection_during_measurement` = true

Next, run in Windows, in the command line the command - `python scanner.py` or in Linux in the terminal the command - `sudo python3 scanner.py`

# GNS3 
The research continues with measuring and optimizing distributed task. 
The network is simulated by a network that is generated using GNS3 Gui. 
Each node in the network is created using an image, that is built using our code.
Link to the repository of the GNS configuration code:

https://github.com/sapirani/GreenSecurityGNS
