import asyncio
from time import sleep
import time
from SplunkResearch.src.log_generator import LogGenerator
import pandas as pd
from datetime import datetime, timedelta
import random
import sys
import threading
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')
# config logging to file
import logging
import subprocess
from application_logging.handlers.elastic_handler import get_elastic_logging_handler
from program_parameters import *
from resources.section_logtypes import section_logtypes
from SplunkResearch.src.splunk_tools import SplunkTools
from SplunkResearch.src.env_utils import clean_env
sys.stdout.reconfigure(line_buffering=True)

# sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/application_logging/handlers')
logging.basicConfig(filename='/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/energy_profile_final.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# add elastic handler to logger


# logger = logging.getLogger()


def write_logs_to_monitor( logs, log_source):
    with open(f'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/{log_source}.txt', 'a') as f:
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

# Main execution
async def overload_profile(savedsearches, splunk_tools):
    top_logtypes = pd.read_csv("/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/resources/top_logtypes.csv")
    top_logtypes = top_logtypes[top_logtypes['source'].str.lower().isin(['wineventlog:security', 'wineventlog:system'])]
    top_logtypes = top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:50]
    top_logtypes = [(x[0].lower(), str(x[1])) for x in top_logtypes]
    relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]}))
    # concat top_logtypes and relevant_logtypes, while removing duplicates and keeping order
    top_logtypes = sorted(list(dict.fromkeys(relevant_logtypes + top_logtypes)))       
    log_generator = LogGenerator(top_logtypes)
    # quantities = [100, 200]
    quantities = [1000, 25000, 50000, 75000, 100000, 250000, 500000, 750000, 1000000]
    # diversities = [0, 1]
    diversities = [1/32]
    # diversities = [0, 0.5, 1, 5, 10]
    time_range = ("08/11/2024:09:00:00", "08/13/2024:09:00:00")
    logging.info(clean_env(splunk_tools, time_range))

    for rule in savedsearches:
        for diversity in diversities:
            for i, quantity in enumerate(quantities):
                logging.info(f"Start time: {datetime.now()}")
                # quantity_to_add is the quantity of logs since the previous iteration
                quantity_to_add = quantity if i == 0 else quantity - quantities[i-1]
                log_type = section_logtypes[rule][0]
                log_source = log_type[0].lower()
                eventcode = log_type[1]
                
                logging.info(f'Generating logs for rule: {rule}, quantity: {quantity_to_add}, diversity: {diversity}')
                logs = log_generator.generate_logs(logsource=log_source,
                                                   eventcode=eventcode,
                                                   istrigger=int(diversity > 0),
                                                   num_logs=quantity_to_add,
                                                   diversity=int(diversity*31+1),
                                                   time_range=time_range)
                waiting_time = len(logs)/ 4500  # 4500 logs per second 
                logging.info(f'Waiting for {waiting_time} seconds before executing rules for {rule}')
                write_logs_to_monitor(logs, log_source) # send to splunk

                await asyncio.sleep(waiting_time)
                scanner_id = f"{rule}_{log_source}_{eventcode}_{int(diversity*31+1)}_{quantity}_{datetime.now()}"
                # # config logger handler
                # elastic_handler = ElasticSearchLogHandler(session_id=scanner_id)
                # # Remove all previous ElasticSearchLogHandlers
                # for h in logger.handlers[:]:
                #     if isinstance(h, ElasticSearchLogHandler):
                #         logger.removeHandler(h)
                # logger.addHandler(elastic_handler)
                logging.info(f"scanner_id: {scanner_id}")
               
                # process = subprocess.Popen(["sudo", "-S", "-E", "env", "PATH=/usr/bin:/bin:/usr/sbin:/sbin:/home/shouei/anaconda3/envs/py38/bin", "/home/shouei/anaconda3/envs/py38/bin/python3", "../scanner.py", "--measurement_session_id", scanner_id],
                process = subprocess.Popen(["/home/shouei/anaconda3/envs/py310_modelenv/bin/python3", "-u", "scanner.py", "--measurement_session_id", scanner_id],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        bufsize=1)  # Line buffered
                
                # Send the password to sudo without waiting for completion
                # process.stdin.write(' \n')
                # process.stdin.flush()

                # Start non-blocking output handling
                thred_1, thred_2 = handle_process_output(process, logging)
                thred_1.start()
                thred_2.start()
                logging.info(f'Process started with PID: {process.pid} {scanner_id}')
                
                await asyncio.sleep(3)

                logging.info('Running saved searches')
                results, _ = await splunk_tools.run_saved_searches(time_range, num_measurements=1)
                logging.info(f'Terminating scanner {scanner_id}')
                # subprocess.run(['pkill', 'scanner.py'],
                #                 # input=' \n',
                #                 text=True,
                #                 check=False)
                process.send_signal(2)
                logging.warning('Killed all scanner.py processes as last resort')
                thred_1.join()
                thred_2.join()

            # clean env
            logging.info('Cleaning environment')
            logging.info(clean_env(splunk_tools, time_range))

def routine_profile():
    # This function is measuring the routine. altering the following profiles parameters:
    # 1. what rules are running
    # 2. how many hosts are forwarding logs
    # 3. what frequency are the rules running
    rules_number = [1, 5, 9]
    hosts_number = [1, 10, 50, 100]
    frequency = [1, 5, 10, 15, 60]  # in minutes
    
    for rule in rules_number:
        for host in hosts_number:
            for freq in frequency:
                logging.info(f'Running routine profile with {rule} rules, {host} hosts and {freq} minutes frequency')
                
                
                

if __name__ == "__main__":
    # Your existing setup
    savedsearches = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name",
                 'Windows AD Replication Request Initiated from Unsanctioned Location',
                 'ESCU Windows Rapid Authentication On Multiple Hosts Rule']
    
    splunk_tools = SplunkTools(active_saved_searches=savedsearches, mode="profile")
    asyncio.run(overload_profile(savedsearches, splunk_tools))