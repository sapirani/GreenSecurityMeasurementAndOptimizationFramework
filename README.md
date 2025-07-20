# Measurement Benchmark - A Framework for Measurement of Reasource and Energy Consumption of Processes

This project investigates the energy footprint and resource characteristics of secure computations, with a particular focus on the demands of homomorphic encryption schemes.
We introduce a general-purpose, process-level measurement framework capable of __profiling CPU, memory, disk, and network usage, alongside energy consumption estimates__.
The framework operates __independently of source code__ and supports fine-grained attribution of resource and energy usage __at both the per-process and system-wide levels__.
These capabilities make our framework broadly applicable and easy to use for profiling  arbitrary programs, including, in our case, a range of cybersecurity applications.

We implemented a wide range of HE, comparing them against symmetric and asymmetric encryption algorithms, in terms of resource consumption and performance.
To support deeper analysis, we also provide interactive visualizations through ElasticSearch and Kibana that allow users to explore time-series trends and compare schemes across multiple metrics.

It is possible to easily configure program parameters inside the configuration file as detailed below. 
You can run multiple programs simultaneously. Both Windows and POSIX operating systems are supported.

## Preliminaries:
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
3. `program_parameters.py` - responsible to easily configure power plan, scan mode, specific parameters for encryption schemes, etc.

## Running Instructions
1. Disable auto brightness of the device.
2. clone `program_parameters.py.example` into a file called `program_parameters.py`
3. Change `program_parameters.py` parameters according to the experiment that you want to perform (ensure that the url and credentials to your ElasticSearch are correct).
4. Disconnect charging cable (code will verify that).
5. Run `scanner.py`. The code might need to be run in admin mode.
6. Code will change power plan according to configuration file.
8. Code will prevent device from sleeping and turning the screen off.
9. Code will set screen brightness according to configuration file.
10. Code will start the selected programs, measure the device's resource consumption and log it into ElasticSearch in parallel.
11. Code will save results into files.
12. Code will restore the default device settings.

## Program Parameters
1. `main_program_to_scan` - the main program that you want to measure it's resource consumption
2. `background_programs_types` - a list of programs that will run in parallel to the main program. This list could be empty (meaning that no additional background program will run). Their resource consumption will be also measured
3. `kill_background_process_when_main_finished` - gracefully kills background processes upon main process termination
4. `summary_type` - defines the format of outputted csv summary.
5. `battery_monitor_type` - enables to choose whether to allow battery measurements or not.
6. `process_monitor_type` - Monitoring strategy (whether to monitor all processes in the system or just the process of interest (which are the main and the background processes))
7. `power_plan` - the computer's power plan, [see options](#power-plans).
8. `scan_option` - defines whether to perform a regular scan, or to start the measured program again and again until timeout or draining a certain amount of energy
9. `RUNNING_TIME` - enables to define the exact time of measurements in one scan mode. in continuous scam it will be the minimum time for the measurements [see modes of execution](#modes-of-execution).
10. `MINIMUM_DELTA_CAPACITY` - enables to define the minimum battery drop required before the code ends. Relevant only in continuous scan mode.
11. `measurement_number` - if equals to NEW_MEASUREMENT, result will be saved in new folder. It is possible to define specific number
12. `screen_brightness_level` - enables to define the brightness of screen - a value between 0 and 100
13. `DEFAULT_SCREEN_TURNS_OFF_TIME` - time, in minutes, before screen turns off
14. `DEFAULT_TIME_BEFORE_SLEEP_MODE` - time, in minutes, before sleep mode is activated
15. `scanner_version` - define the metrics that will be measured [see options](#measurement-versions)
16. `elastic_url` - url for your ElasticSearch
17. `elastic_username` - your ElasticSearch username
18. `elastic_password` - your ElasticSearch password
19. other program-specific parameters (that will be sent to the programs lunched by this measurement framework)

### Modes of Execution
There are 3 modes of execution:
1. Baseline measurement - the code measures the resource usage without running any program. This is the baseline consumption of other system processes running in the background.
2. One scan - the code starts scanning. Meanwhile, the code keeps measuring the resource usage.
3. Continuous scan - the code runs a program, and when the scan is over it immediately starts a new one. The code ends when at least a certain amount of time has elapsed and at least a certain amount of battery capacity has drained decreased.

### Processes Monitoring Types
1. FULL - measures the resources of all running processes in the system.
2. PROCESSES_OF_INTEREST_ONLY - monitors only main and background processes

# Battery Monitoring Types
1. FULL - measures battery metrics
2. WITHOUT_BATTERY - does not measure battery metrics

### Power plans
It is possible to configure the computer's power plan during measurements. The available plans:
1. High Performance
2. Balanced
3. Power Saver

### Available Programs
This project currently supports the following programs (it is very easy to add another program - [perform the following steps](#supporting-additional-programs)):
1. MessageEncryptor - encrypts all messages in the given file and writes the serialized ciphertexts into results file.
2. MessageDecryptor - decrypts all ciphertexts in the given file and writes the plaintexts into results file.
3. EncryptionPipeline - encrypts and decrypts each message in the given file and writes to results file whether the decryption led to the original plaintext.
4. MessageAddition -
  - In Homomorphic algorithms - first encrypts all messages, than sums the ciphertexts and finally decrypts the result.
  - In traditional algorithms - first decrypts all ciphertexts, than sums the plaintexts.
5. MessageMultiplication - Same as MessageAddition, but calculating the multiplication of the messages.

For each program you need to define the algorithm that will be used to execute the operation. The algorithm should be of type `EncryptionType`.
The available algorithms are:
1. Paillier - implemented without using existing libraries.
2. RSA - implemented without using existing libraries.
3. LightPheRSA - the algorithm RSA implemented using the library lightphe.
4. LightPheElGamal - the algorithm Elgamal implemented using the library lightphe.
5. LightPheExponentialElGamal - the algorithm Exponential ElGamal implemented using the library lightphe.
6. LightPhePaillier - the algorithm Paillier implemented using the library lightphe.
7. LightPheDamgardJurik - the algorithm Damgard Jurik implemented using the library lightphe.
8. LightPheOkamotoUchiyama - the algorithm Okamoto Uchiyama implemented using the library lightphe.
9. LightPheBenaloh - the algorithm Benloh implemented using the library lightphe.
10. LightPheNaccacheStern - the algorithm Naccache Stern implemented using the library lightphe.
11. LightPheGoldwasserMicali - the algorithm Goldwasser Micali implemented using the library lightphe.
12. LightPheEllipticCurveElGamal - the algorithm Elliptic Curve Elgamal implemented using the library lightphe.
13. CKKSTenseal - the algorithm CKKS implemented using the TenSeal library.
14. BFVTenseal - the algorithm BFV implemented using the TenSeal library.
15. FernetAES - the algorithm AES implemented using the Fernet library.
16. PycryptoAES - the algorithm AES implemented using the Pycryptodome library.
17. PycryptoDES - the algorithm DES implemented using the Pycryptodome library.
18. PycryptoBlowfish - the algorithm Blowfish implemented using the Pycryptodome library.
19. PycryptoChaCha20 - the algorithm ChaCha implemented using the Pycryptodome library.
20. PycryptoArc4 - the algorithm Arc4 implemented using the Pycryptodome library.
21. PycryptoRSA - the algorithm RSA implemented using the Pycryptodome library.

#### Using Elastic to Analyze Results:
Make sure your elastic details are written inside the program_parameters.py file.

To compare graphs across multiple measurement sessions, you should create a runtime field.
1. Go to "Stack management" -> Kibana -> Data Views.
2. Choose the scanner data view.
3. Click on create field, insert `seconds_from_scanner_start` as the name of the field.
4. Choose 'long' as the field's type.
5. Toggle the "Set value".
6. Insert the following script:
```
if (doc.containsKey('timestamp') && doc.containsKey('start_date')
    && !doc['timestamp'].empty && !doc['start_date'].empty) {
    long ts = doc['timestamp'].value.toInstant().toEpochMilli();
    long start = doc['start_date'].value.toInstant().toEpochMilli();    
    emit((ts - start) / 1000); // return in seconds
}
```
7. Create a graph such that the new `seconds_from_scanner_start` field is the x-axis.
8. To observe graphs resulting from different measurements onto each other, tap the "Break down by" and choose the session_id.
9. You may add control (inside the dashboard screen) based on the session_id field, to compare specific measurement sessions of your choice.

#### Control Custom Logging Extras
When logging into elastic, we allow additional, user-defined extras that will be attached to any log produced by the scanner.
Example usage (extras should be given as JSON):

`python scanner.py --logging_constant_extras '{"key_1":"value_1","key_2":"value_2"}'`

#### Supporting Additional Programs:
1. in `general_consts.py` file - add your program in the enum called *ProgramToScan*
2. in `program_parameters.py` file - add all the parameters that the user can configure in your program
3. in `program classes` add a class that represents your program and inherits from *ProgramInterface*. You ***MUST*** implement the functions:  *get_program_name* and *get_command*. The function *get_command* returns a string which is the shell command that runs your program. You can implement any other function of *ProgramInterface*. Note that if you want to run your command in powershell (for Windows programs), implement the function *should_use_powershell* and return True.
4. in `program_factory.py` - add your program in the function called *program_to_scan_factory*

## Configure ElasticSearch and Kibana 
Our measurement supports logging measured metrics into local elastic database.
In order for it to succeed, we should configure containers of elastic that will run locally.
1. go into the local elastic directory using the command: `cd /<your_home_directory>/elastic-start-local`
2. execute `docker compose down` to shut down existing containers (if something is wrong with the previous ones)
3. In the file `docker-compose` change the ip address from 127.0.0.1 to be 0.0.0.0.
4. execute `docker compose up --build --wait` to turn on the containers (you may also run the `start.sh script`).
5. Run the command `docker ps` and make sure that 2 new containers with elastic in their name exist.
   * `docker.elastic.co/kibana/kibana:9.0.1` on port 5601
   * `docker.elastic.co/elasticsearch/elasticsearch:9.0.1` on ports 9200 and 9300.
6. Now, you can go to your browser and type `localhost:5601` in the search bar.
7. You might need to enter username and password:
   * Username: elastic
   * Password: saved in .env file in the elastic directory (from step 1)
8. Go to the burger sign in the top left of the site and click on Discover.
9. In the DataView tab click on `scanner` to view the logs collected by running the scanner.
