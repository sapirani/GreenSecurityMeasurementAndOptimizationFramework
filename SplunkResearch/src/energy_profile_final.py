import asyncio
from time import sleep
import time

from sympy import true
from SplunkResearch.src.log_generator import LogGenerator
import pandas as pd
import random
import sys
import threading
sys.path.insert(1, '/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch')
# config logging to file
import logging
import subprocess
from application_logging.handlers.elastic_handler import get_elastic_logging_handler
from resources.section_logtypes import section_logtypes
from SplunkResearch.src.splunk_tools import *
from SplunkResearch.src.env_utils import *
from datetime import datetime, timedelta
from config import config

sys.stdout.reconfigure(line_buffering=True)

# Load configuration
base_dir = config.get('paths.base_dir', '/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch')
log_file_path = config.get('paths.energy_profile_log', f'{base_dir}/energy_profile_final.log')

# sys.path.insert(1, '/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/application_logging/handlers')
logging.basicConfig(filename=log_file_path,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# add elastic handler to logger
SPLUNK_BINARY_PATH = config.get('splunk.binary_path', '/opt/splunk/bin/splunk')

# logger = logging.getLogger()

def inject_episodic_logs():
    """Inject episodic logs into environment"""

    # logging.info(f"Waiting for {sum(self.episode_logs.values())} logs to be written")
    # sleep(sum(self.episode_logs.values())/4500)  # wait for logs to be written
    security_log_file_path = log_file_prefix + "wineventlog:security.txt"
    system_log_file_path = log_file_prefix + "wineventlog:system.txt"
    flush_logs(security_log_file_path, log_type="Security")
    flush_logs(system_log_file_path, log_type="System")
    sleep(5)  # wait for logs to be indexed

def flush_logs(log_file_path, log_type="Security"):
    # Configuration
    # Default management URI is localhost:8089. Change IP if Splunk is remote.
    splunk_port = config.get('splunk.port', 8089)
    splunk_mgmt_uri = f"https://{splunk_tools.splunk_host}:{splunk_port}"
    endpoint = f"{splunk_mgmt_uri}/services/receivers/stream"
    print(endpoint)
    # Credentials and Metadata from your existing object
    username = splunk_tools.splunk_username
    password = splunk_tools.splunk_password
    index = splunk_tools.index_name
    sourcetype = "WinEventLog:" + log_type

    # 1. Prepare Parameters (Query String)
    secondary_host = config.get('hosts.secondary', '132.72.81.150')
    params = {
        "index": index,
        "sourcetype": sourcetype,
        "host": secondary_host
    }
    
    # 2. Prepare Headers (Equivalent to -H "x-splunk-input-mode: streaming")
    headers = {
        "x-splunk-input-mode": "streaming"
    }

    # print(f"Streaming {log_file_path} to Splunk via REST API...")

    try:
        # 3. Open file and POST it (Equivalent to -d @/local/path/data.json)
        # We open in 'rb' (read binary) to stream raw data
        with open(log_file_path, 'rb') as f:
            response = requests.post(
                endpoint,
                auth=(username, password), # Equivalent to -u admin:pass
                params=params,             # Adds ?index=...&sourcetype=...
                headers=headers,
                data=f,                    # The file content
                verify=False,              # Equivalent to curl -k
                timeout=30                 # Prevent hanging indefinitely
            )

        # 4. Check for HTTP Errors
        response.raise_for_status()

        # Optional: Log success
        if response.text:
            logger.info(f"Splunk Response: {response.text}")

    except requests.exceptions.HTTPError as e:
        # This runs if Splunk returns 4xx or 5xx
        print(f"Splunk API Error: {e}", file=sys.stderr)
        print(f"Response Body: {e.response.text}", file=sys.stderr)

    except requests.exceptions.ConnectionError:
        print(f"Connection Failed: Could not reach {splunk_mgmt_uri}", file=sys.stderr)
        print("Check if Splunk is running and port 8089 is open.", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}", file=sys.stderr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

def write_logs_to_monitor( logs, log_source):
    with open(f'/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch/monitor_files_{splunk_tools.splunk_host}/{log_source}.txt', 'a') as f:
        for log in logs:
            f.write(f'{log}\n\n')        

def handle_process_output(process, logger):
    """Handle process output in a non-blocking way"""
    def read_output(pipe, log_func):
        try:
            for line in pipe:
                if line.strip():
                    log_func(line.strip())
        except Exception as e:
            logger.error(f'Error reading process output: {str(e)}')

    # Start threads for stdout and stderr
    return threading.Thread(target=read_output, args=(process.stdout, logger.info), daemon=True), threading.Thread(target=read_output, args=(process.stderr, logger.error), daemon=True)


def check_logs_flushed(earliest_time, latest_time, expected_count, timeout=100):
    """Check if all logs are flushed to Splunk within the timeout period"""
    splunk_tools = SplunkTools()
    start_time = time.time()
    while time.time() - start_time < timeout:
        secondary_host = config.get('hosts.secondary', '132.72.81.150')
        default_host = config.get('splunk.default_host', 'dt-splunk')
        search_query = f'index="main" host IN ({secondary_host}, "{default_host}", {splunk_tools.splunk_host}) '
        results = splunk_tools.run_search(search_query, earliest_time, latest_time)
        if results and int(results[0]['log_count']) >= expected_count:
            logging.info(f'All {expected_count} logs have been flushed to Splunk.')
            return True
        time.sleep(2)  # wait before checking again
    logging.warning(f'Timeout reached: Not all logs were flushed to Splunk.')
    return False


# Main execution
def overload_profile(savedsearches,experimented_savedsearches=None, splunk_tools=None): # TODO: Check wht detect new local admin account isnt find events
    top_logtypes = pd.read_csv("/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch/resources/top_logtypes.csv")
    top_logtypes = top_logtypes[top_logtypes['source'].str.lower().isin(['wineventlog:security', 'wineventlog:system'])]
    top_logtypes = top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:50]
    top_logtypes = [(x[0].lower(), str(x[1])) for x in top_logtypes]
    relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]}))
    # concat top_logtypes and relevant_logtypes, while removing duplicates and keeping order
    top_logtypes = sorted(list(dict.fromkeys(relevant_logtypes + top_logtypes)))       
    log_generator = LogGenerator(top_logtypes)
    # quantities = [100, 200]
    quantities = [0, 1000, 10000, 50000, 100000, 250000, 500000, 1000000]
    # diversities = [0, 1]
    # diversities = [0.5, 1, 5]
    diversities = [0, 0.01, 0.1, 0.5, 1 ,2, 5]
    # diversities = [0, 0.5, 1, 5, 10]
    time_range = ("12/04/2021:08:00:00", "12/06/2021:08:00:00")
    # time_range = ("12/04/2024:08:00:00", "12/06/2024:08:00:00")
    logging.info(clean_env(splunk_tools, time_range, host=splunk_tools.splunk_host))
    injection_id = 0
    for rule in experimented_savedsearches:
        for diversity in diversities:
            for i, quantity in enumerate(quantities):
                for j in range(2):  # repeat each setting 3 times
                    logging.info(f"Start time: {datetime.now()}")
                    if quantity == 0 and diversity > 0:
                        # skip this setting
                        logging.info(f"Skipping setting quantity: {quantity}, diversity: {diversity}")
                        continue
                    if j == 0:
                        # quantity_to_add is the quantity of logs since the previous iteration
                        quantity_to_add = quantity if i == 0 else quantity - quantities[i-1]
                        while true:
                            log_type = section_logtypes[rule][0]
                            log_source = log_type[0].lower()
                            eventcode = log_type[1]
                            istrigger = int(diversity > 0)
                            logging.info(f'Generating logs for rule: {rule}, quantity: {quantity_to_add}, diversity: {diversity}')
                            configs = [{
                                'logsource': log_source,
                                'eventcode': eventcode,
                                'istrigger': istrigger,
                                'time_range': time_range,
                                'num_logs': quantity_to_add,
                                'diversity': max(1, int(diversity*100)),
                                'injection_id': injection_id
                            }]
                            
                            for batch in log_generator.generate_massive_stream(configs, batch_size=50000):
                                logs = []
                                for log_entry, source in batch:
                                    logs.append(log_entry)
                                    
                                # 2. Write each bucket to its own file
                                write_logs_to_monitor(logs, log_source) # send to splunk 
                                logging.info(f'Wrote {len(logs)} logs to monitor file for source {log_source}')   
                            inject_episodic_logs()
                            # check if all logs with injection id in splunk
                            results = 0
                            start_date = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S').strftime("%Y-%m-%dT%H:%M:%S")
                            end_date = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S').strftime("%Y-%m-%dT%H:%M:%S")
                            attempt = 0
                            while quantity > results and attempt < 7:
                                sleep(2)
                                secondary_host = config.get('hosts.secondary', '132.72.81.150')
                                default_host = config.get('splunk.default_host', 'dt-splunk')
                                query = f'index=main host IN ({secondary_host}, "{default_host}", {splunk_tools.splunk_host})  | stats count'
                                # query = f'index=main host="dt-splunk" Injection_id={injection_id} | stats count'
                                results = splunk_tools.run_search(query, start_date, end_date)            
                                results = int(results[0]['count'])
                                logging.info(f'Logs injected so far for injection_id {injection_id}: {results}/{quantity}')
                                attempt += 1
                            if quantity == results:
                                logging.info(f'All {quantity} logs for injection_id {injection_id} have been injected successfully.')
                                break
                            else:
                                logging.warning(f'Only {results}/{quantity} logs injected for injection_id {injection_id}. Retrying log generation.')
                                # empty monitor files
                                empty_monitored_files(get_system_monitor_path(splunk_tools.splunk_host))
                                empty_monitored_files(get_security_monitor_path(splunk_tools.splunk_host))
                                quantity_to_add = quantity - results
                                
                    scanner_id = f"{rule}_{log_source}_{eventcode}_{int(diversity*100)}_{quantity}_{j}_{datetime.now()}"
                    # # config logger handler
                    # elastic_handler = ElasticSearchLogHandler(session_id=scanner_id)
                    # # Remove all previous ElasticSearchLogHandlers
                    # for h in logger.handlers[:]:
                    #     if isinstance(h, ElasticSearchLogHandler):
                    #         logger.removeHandler(h)
                    # logger.addHandler(elastic_handler)
                    # check all logs are flushed

                    logging.info(f"scanner_id: {scanner_id}")
                
                    # process = subprocess.Popen(["sudo", "-S", "-E", "env", "PATH=/usr/bin:/bin:/usr/sbin:/sbin:/home/shouei/anaconda3/envs/py38/bin", "/home/shouei/anaconda3/envs/py38/bin/python3", "../scanner.py", "--measurement_session_id", scanner_id],
                    # process = subprocess.Popen([ "/home/shouei/anaconda3/envs/py310_modelenv/bin/python3", "-u", "scanner.py", "--measurement_session_id", scanner_id, "--elastic_pipeline_processes", "enrich_pid_to_rule_name"],
                    #                         stdin=subprocess.PIPE,
                    #                         stdout=subprocess.PIPE,
                    #                         stderr=subprocess.PIPE,
                    #                         text=True,
                    #                         bufsize=1)  # Line buffered

                    # Send the password to sudo without waiting for completion
                    # process.stdin.write(' \n')
                    # process.stdin.flush()

                    # Start non-blocking output handling
                    # thred_1, thred_2 = handle_process_output(process, logging)
                    # thred_1.start()
                    # thred_2.start()
                    # logging.info(f'Process started with PID: {process.pid} {scanner_id}')
                    
                    # sleep(3)
                    earliest_time = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S').timestamp()
                    latest_time = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S').timestamp()
                    logging.info('Running saved searches')
                    results = asyncio.run(splunk_tools.run_saved_search(rule, start_time=earliest_time, end_time=latest_time))
                    # logging.info(f'Terminating scanner {scanner_id}')
                    # subprocess.run(['pkill', 'scanner.py'],
                    #                 # input=' \n',
                    #                 text=True,
                    #                 check=False)
                    # sleep(2)
                    # process.send_signal(9)
                    # logging.warning('Killed all scanner.py processes as last resort')
                    # thred_1.join()
                    # thred_2.join()
                    empty_monitored_files(get_system_monitor_path(splunk_tools.splunk_host))
                    empty_monitored_files(get_security_monitor_path(splunk_tools.splunk_host))
                    injection_id += 1
            # clean env
            logging.info('Cleaning environment')
            logging.info(clean_env(splunk_tools, time_range, host=splunk_tools.splunk_host))
            # empty sid index from elastic

                
                

if __name__ == "__main__":
    # read host from command line argument
    host = sys.argv[1] if len(sys.argv) > 1 else '1'
    # Your existing setup
    savedsearches = config.get('reward.rule_names', [])
    experimented_savedsearches = config.get('reward.rule_names', [])
    splunk_tools = SplunkTools(active_saved_searches=savedsearches, mode=Mode.PROFILE, ip=host)
    global log_file_prefix 
    log_file_prefix = f"/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch/monitor_files_{splunk_tools.splunk_host}/"

    overload_profile(savedsearches, experimented_savedsearches, splunk_tools)
