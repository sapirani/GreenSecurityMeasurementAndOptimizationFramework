# GreenSecurity - A Framework for Measurement of Reasource and Energy Consumption of Processes

This project investigates the energy footprint and resource characteristics of secure computations, with a particular focus on the demands of cybersecurity-related schemes (such as antivirus, IDS, Splunk, etc.).
We introduce a general-purpose, process-level measurement framework capable of __profiling CPU, memory, disk, and network usage, alongside energy consumption estimates__.
The framework operates __independently of source__ code and supports fine-grained attribution of resource and energy usage __at both the per-process and system-wide levels__.
These capabilities make our framework broadly applicable and easy to use for profiling arbitrary programs, including, in our case, a range of cybersecurity applications.

To support deeper analysis, we also provide interactive visualizations through ElasticSearch and Kibana that allow users to explore time-series trends and compare schemes across multiple metrics.
Both Windows and POSIX operating systems are supported.

# Cite Us
```
@INPROCEEDINGS{10629013,
  author={Brudni, Sagi and Anidgar, Sapir and Brodt, Oleg and Mimran, Dudu and Shabtai, Asaf and Elovici, Yuval},
  booktitle={2024 IEEE 9th European Symposium on Security and Privacy (EuroS&P)}, 
  title={Green Security: A Framework for Measurement and Optimization of Energy Consumption of Cybersecurity Solutions}, 
  year={2024},
  volume={},
  number={},
  pages={676-696},
  abstract={Information and communication technology (ICT) is playing an expanding and critical role in our modern lives. Due to its proliferation, ICT has a significant impact on global energy consumption, which in turn contributes to air pollution, climate change, water pollution, etc. The proliferation of ICT has been accompanied by the emergence of cybersecurity technologies and solutions, which play an integral role in society's digitalization. Wherever there is ICT, there is a need to secure it, resulting in an increase in global cybersecurity energy consumption as well. This paper discusses the energy-related aspects of cybersecurity solutions and defines a “Green Security” taxonomy. We highlight the inefficiencies stemming from various cybersecu-rity practices, such as processing the same data repeatedly. Within this context, we analyze cybersecurity solutions in common use cases, demonstrating the inherent energy consumption inefficiencies. In addition, we propose a method of measuring the energy consumed by cybersecurity solutions and present several optimization strategies that reduce their energy consumption. We evaluate our proposed optimization strategies and demonstrate their ability to reduce energy consumption while considering the organizational risk profile and maintaining the required security level.},
  keywords={Climate change;Computer security;Information security;Green design;Energy consumption;Information and communication technology;Cybersecurity;Information security;Cyber;Security;Green security;Energy consumption;Energy optimization},
  doi={10.1109/EuroSP60621.2024.00043},
  ISSN={2995-1356},
  month={July},}
```


## Preliminaries
1. Please use python>=3.10.

2. Create python environment with the packages listed in requirements.txt.
   In windows:
   `pip install -r windows_machine_requirements.txt`
   In Linux:
   `sudo pip install -r linux_machine_requirements.txt`

2. To get optimal results, make sure that:
   * Any unnecessary program is closed (including background processes)
   * All result files are closed
   * Auto brightness disabled

## Modules
The main components of the code:
1. `scanner.py` - responsible to periodically record the raw data, log it into ElasticSearch and save it in the relevant csv and txt files
2. `program_parameters.py` - responsible to easily configure power plan, scan mode, specific parameters for encryption schemes, etc.
   
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
3. `kill_background_process_when_main_finished` - gracefully kills background processes upon main process termination
4. `summary_type` - defines the format of outputted csv summary.
5. `battery_monitor_type` - enables to choose whether to allow battery measurements or not.
6. `process_monitor_type` - Monitoring strategy (whether to monitor all processes in the system or just the process of interest (which are the main and the background processes))
7. `custom_process_filter_types` - user-defined filters.
8. `power_plan` - the computer's power plan, [see options](#power-plans).
9. `scan_option` - defines whether to perform a regular scan, or to start the measured program again and again until timeout or draining a certain amount of energy
10. `RUNNING_TIME` - enables to define the exact time of measurements in one scan mode. in continuous scam it will be the minimum time for the measurements [see modes of execution](#modes-of-execution).
11. `MINIMUM_DELTA_CAPACITY` - enables to define the minimum battery drop required before the code ends. Relevant only in continuous scan mode.
12. `measurement_number` - if equals to NEW_MEASUREMENT, result will be saved in new folder. It is possible to define specific number
13. `screen_brightness_level` - enables to define the brightness of screen - a value between 0 and 100
14. `DEFAULT_SCREEN_TURNS_OFF_TIME` - time, in minutes, before screen turns off
15. `DEFAULT_TIME_BEFORE_SLEEP_MODE` - time, in minutes, before sleep mode is activated
16. `scanner_version` - define the metrics that will be measured [see options](#measurement-versions)
17. `elastic_url` - url for your ElasticSearch
18. `elastic_username` - your ElasticSearch username
19. `elastic_password` - your ElasticSearch password
20. other program-specific parameters (that will be sent to the programs lunched by this measurement framework)

### Modes of Execution
There are 3 modes of execution:
1. Baseline measurement - the code measures the resource usage without running any program. This is the baseline consumption of other system processes running in the background.
2. One scan - the code starts scanning. Meanwhile, the code keeps measuring the resource usage.
3. Continuous scan - the code runs a program, and when the scan is over it immediately starts a new one. The code ends when at least a certain amount of time has elapsed and at least a certain amount of battery capacity has drained decreased.

The results are saved in the relevant folders. the raw resource usage data is saved in csv and txt files. graphs are saved in png files.

### Antivirus Scan types
1. Full scan - scans every file and program on the computer. May take few hours or days to complete, depending on the amount and type of data that needs to be scanned.
2. Quick scan - looks at all the locations where there could be malware registered to start with the system, such as registry keys and known Windows startup folders. Provides strong protection against malware that starts with the system and kernel-level malware. May take a few seconds or minutes.
3. Custom scan - enables to scan exactly the files and folders that you select.

### Processes Monitoring Types
1. FULL - measures the resources of all running processes in the system.
2. PROCESSES_OF_INTEREST_ONLY - monitors only main and background processes

### Battery Monitoring Types
1. FULL - measures battery metrics
2. WITHOUT_BATTERY - does not measure battery metrics

### Power plans
It is possible to configure the computer's power plan during measurements. The available plans:
1. High Performance
2. Balanced
3. Power Saver

### Available Programs
This project currently supports the following programs (it is very easy to add another program - [perform the following steps](#supporting-additional-programs)):
1. Windows Defender Antivirus
2. ClamAV (Antivirus)
3. Snort Intrusion Detection System (IDS)
4. Suricata Intrusion Detection System (IDS)
5. Windows Performance Monitor (another source to validate the measurements results)
6. Typical User Activity (web browsing, edit a word document, etc.)
7. Splunk
8. etc.


# Using Elasticsearach and Kibana to Analyze Results:
Use the following link to start Elasticsearch and Kibana locally: [link](https://www.elastic.co/docs/deploy-manage/deploy/self-managed/local-development-installation-quickstart).

Insert your elastic details (username, password, and url - should appear in your terminal when installation is terminated) into the `program_parameters.py` file.
If you lost your credentials, they should appear inside a directory called `elastic-start-local` in your `home` folder. Specifically, look at the `.env` file.

To compare graphs across multiple measurement sessions, you should create a runtime field.
1. Open your broweser and enter Kibana ([link](http://127.0.0.1:5601))
2. Insert your credentials
3. Go to "Stack management" -> Kibana -> Data Views.
4. Choose the scanner data view.
5. Click on create field, insert `seconds_from_scanner_start` as the name of the field.
6. Choose 'long' as the field's type.
7. Toggle the "Set value".
8. Insert the following script:
```
if (doc.containsKey('timestamp') && doc.containsKey('start_date')
    && !doc['timestamp'].empty && !doc['start_date'].empty) {
    long ts = doc['timestamp'].value.toInstant().toEpochMilli();
    long start = doc['start_date'].value.toInstant().toEpochMilli();    
    emit((ts - start) / 1000); // return in seconds
}
```
9. Create a graph such that the new `seconds_from_scanner_start` field is the x-axis.
10. To observe graphs resulting from different measurements onto each other, tap the "Break down by" and choose the session_id.
11. You may add control (inside the dashboard screen) based on the session_id field, to compare specific measurement sessions of your choice.

## Import Dashboards
Inside `kibana_dashboards`, we have exported multiple interactive dashboards to easily estimate the performance of measured processes and system.

To import the dashboard, you should:
1. press the search bar in Kibana ([link](http://127.0.0.1:5601))
2. search for `saved objects`
3. press the import button
4. press `request action on conflict` under `Import options`
5. select a dashboard file from `kibana_dashboards` directory (an ndjson file).
6. you will probably encounter conflicts. Kibana will ask you to align an ID with a name of a dataview. The following represent mappings between IDs you might encounter and the names of the relevant dataviews. If an ID does not appear in your conflicts screen, continue on.
a. choose `metrics_aggregations` near the ID of `1834a0cf-20ff-4e11-b006-62faedde6305`
b. choose `system_metrics` near the ID of `91b6f9d7-ac38-4fe6-824d-1e89c31d129b`
c. choose `process_metrics` near the ID of `b4c611a0-f002-4d56-b2f5-92b4a70a3cb1`
7. press 'confirm all changes' and 'done'
8. search for `dashboards` in the search bar
9. select the imported dashboard
10. you might encounter an error in the controls. If you do, edit the control - choose the relevant dataview and field. For example, if you encounter an error in the `pid` control, choose `process_metrics` as dataview and `pid` as field.

### Export Dashboards
If you wish to export your own dashboard (as we did in the `kibana_dashboards` directory):
1. press the search bar in Kibana ([link](http://127.0.0.1:5601))
2. search for `saved objects`
3. select the the dashboards you wish to export (using the checkboxes)
4. press the export button
5. **DO NOT TOGGLE `include relevant objects`**
6. press 'export'
7. the dashboard will be downloaded (ensure you select keep file in your browser)


#### Control Custome Logging Extras
When logging into elastic, we allow additional, user-defined extras that will be attached to any log produced by the scanner.
Example usage (extras should be given as JSON):

`python scanner.py --logging_constant_extras '{"key_1":"value_1","key_2":"value_2"}'`

#### Supporting Additional Programs:
1. in `general_consts.py` file - add your program in the enum called *ProgramToScan*
2. in `program_parameters.py` file - add all the parameters that the user can configure in your program
3. in `program class` file - add a class that represents your program and inherits from *ProgramInterface*. You ***MUST*** implement the functions:  *get_program_name* and *get_command*. The function *get_command* returns a string which is the shell command that runs your program. You can implement any other function of *ProgramInterface*. Note that if you want to run your command in powershell (for Windows programs), implement the function *should_use_powershell* and return True.
4. in `initialization_helper.py` file - add your program in the function called *program_to_scan_factory*

#### Supporting User-Defined Filters:
1. in `general_consts.py` file - add your program in the enum called *CustomFilterType*
2. in `program_parameters.py` file - add all filter types you wish to apply (inside custom_process_filter_types list).
3. in `custom_process_filter` package - add a module and a class in it that inherits from *AbstractProcessFilter*. You ***MUST*** implement the function:  *should_ignore_process*.
4. in `initialization_factories.py` file - add your filter in the function called *custom_process_filter_factory*.

## Execution Example:
1) Create program_parameters.py file and fill the relevant fields. You may use the program_parameters.py.example file as a reference.
2) When you want to run the windows defender antivirus and measure it's energy consumption when scanning a folder (in path dir) once, with no other processes running in the background, while the power mode of the computer should be power saver, you should change the next parameters in the file program_parameters.py:

* `main_program_to_scan` = ProgramToScan.ANTIVIRUS
* `background_programs_types` = []  
* `power_plan` = PowerPlan.POWER_SAVER
* `scan_option` = ScanMode.ONE_SCAN
* `scan_type` = ScanType.CUSTOM_SCAN
* `custom_scan_path` = dir
* `disable_real_time_protection_during_measurement` = true

Next, run in Windows, in the command line the command - `python scanner.py` or in Linux in the terminal the command - `sudo python3 scanner.py`

3) When you want to run the windows defender antivirus and measure its energy consumption when executing a full scan once while there are other processes running in the background (Dummy process, performance monitor process), with power mode equals to power saver, you should change the next parameters in the file `program_parameters.py`:

* `main_program_to_scan` = ProgramToScan.ANTIVIRUS
* `background_programs_types` = [ProgramToScan.DummyANTIVIRUS, ProgramToScan.Perfmon]  
* `power_plan` = PowerPlan.POWER_SAVER
* `scan_option` = ScanMode.ONE_SCAN
* `scan_type` = ScanType.FULL_SCAN
* `custom_scan_path` = ""
* `disable_real_time_protection_during_measurement` = true

Next, run in Windows, in the command line the command - `python scanner.py` or in Linux in the terminal the command - `sudo python3 scanner.py`

## Idle Session Id
* The user can add the flag `--measurement_session_id` followed by a `session_id` string. This string is determined by the user and defines a name to the executed measurement.
* Make sure to include "idle" in the name of a session that measures the idle behaviour of the running machine.
* This is necessary for the implementation of the energy model, that takes into account the idle stats of the machine.

## Configure Elastic Logs 
The scanner program writes logs into local elastic database.
In order for it to succeed, we should configure containers of elastic that will run localy on the server.
1. go into the local elastic directory using the command: `cd /home/gns3/elastic-start-local`
2. execute `docker compose down` to shut down existing containers (if something is wrong with the previous ones)
3. In the file `docker-compose` change the ip address from 127.0.0.1 to be 0.0.0.0.
4. execute `docker compose up --build --wait` to turn on the containers.
5. Run the command `docker ps` and make sure that 2 new containers with elastic in their name exist.
   * `docker.elastic.co/kibana/kibana:9.0.1` on port 5601
   * `docker.elastic.co/elasticsearch/elasticsearch:9.0.1` on ports 9200 and 9300.
6. Now, you can go to the firefox and run: `localhost:5601`
7. You might need to enter username and password:
   * Username: elastic
   * Password: saved in .env file in the elastic directory (from step 1)
8. Go to the burger sign in the top left of the site and clicke on Discover.
9. In the DataView tab click on `scanner` to view the logs collected by running the scanner.

# Energy Model
This package contains the actual energy prediction model implementation.
The package contains:
   * dataset creators - objects that read the logs from the elastic, transforms them into a dataframe and calculate the target column (the GT).
   * dataset processing - objects that perform filtering, scaling and feature selection on the collected dataframe.
   * evaluation - simple evaluation pipeline with predefined metrics that inherit `AbstractEvaluationMetric`. This package also allows to perform grid-search easily.
   * energy model utils - objects that are used by the dataset creators to process the data read from the elastic. Also used by the `EnergyModelAggregator`.
   * models - the energy model and its wrapper for the energy model aggregator.
   * pipelines - a pipeline that builds and evaluate the various model.

The energy model class consists of a scalar and the actual sklearn model instance.
The model's scaler and the sklearn model after training are saved to `DEFAULT_ENERGY_MODEL_PATH` directory and the energy model is saved to `MODEL_FILE_NAME`.
All paths (including paths for datasets) can be configured in `energy_model_parameters.py`.

## Training the Models

### About the Datasets
It is important to mention that the dataset that was used to train the system energy model contains only system features (no hardware or idle features).

The process' energy model was trained with both process and system features (no hardware or idle).


However, the feature of cpu usage in seconds (both for process and system) is computed as the integral between the current measurement to the previous one divided by 100.
The feature of memory usage (both for process and system) is computed as the difference between the current memory usage and the memory usage of the last measurement (timestamp).

### Training the Process Energy Model
The steps in the implementation of the model are as follows:
1. Train a system-based energy model:
   * Create a dataset using only system features.
   * Split the dataset into batches, each of predefined duration.
   * Compute the ratio of energy per second for each batch, by dividing the battery drain of that batch with the batch's duration.
   * Compute the target by multiplying the energy usage of the system in each batch by the duration of the measurement.
   * Process the dataset (using filters and StandardScaler)
   * Train a regression model using the processed dataset.
2. Build a dataset of the system telemetry minus the process' telemetry.
   * The columns of the dataset are system features only.
   * The value in each column is the value of the original system feature minus the value of the matching process feature.
   * This dataset has no target column.
3. Create process energy usage column
   * Use the dataset created in step 2 and the system energy model trained in step 1, and predict the energy usage of the system without the process.
   * The energy prediction of the system minus process aims to estimate the background's energy consumption, enabling the isolation of the energy consumption of the process itself.
   * Subtract the predictions from the actual energy usage of the system (from the dataset in step 1).
   * The result is the energy usage of the process in each timestamp.
4. Train a process energy model
   * Create a dataset using both process and system features (without energy related columns).
   * The target of this dataset is the "energy usage of the process" that was calculated in step 3.
   * Process the dataset (using filters and StandardScaler)
   * Train a regression model using the processed dataset.

# GNS3 
The research continues with measuring and optimizing distributed task. 
The network is simulated by a network that is generated using GNS3. 
Each node in the network is created using an image, that is built using our code.
Link to the repository of the GNS configuration code:

https://github.com/sapirani/GreenSecurityGNS
