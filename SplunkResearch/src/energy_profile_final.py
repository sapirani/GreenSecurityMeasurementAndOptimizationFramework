import asyncio
from time import sleep
from log_generator import LogGenerator
import pandas as pd
from datetime import datetime, timedelta
import random
import sys
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')
# config logging to file
import logging

logging.basicConfig(filename='/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/energy_profile_final.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


from resources.section_logtypes import section_logtypes
from splunk_tools import SplunkTools
from env_utils import clean_env

def write_logs_to_monitor( logs, log_source):
    with open(f'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/{log_source}.txt', 'a') as f:
        for log in logs:
            f.write(f'{log}\n\n')        




# Main execution
if __name__ == "__main__":
    # Your existing setup
    top_logtypes = pd.read_csv("/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/resources/top_logtypes.csv")
    savedsearches = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name",
                 'Windows AD Replication Request Initiated from Unsanctioned Location',
                 'ESCU Windows Rapid Authentication On Multiple Hosts Rule']
    
    splunk_tools = SplunkTools(active_saved_searches=savedsearches)
    top_logtypes = top_logtypes[top_logtypes['source'].str.lower().isin(['wineventlog:security', 'wineventlog:system'])]
    top_logtypes = top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:50]
    top_logtypes = [(x[0].lower(), str(x[1])) for x in top_logtypes]
    relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]}))
    # concat top_logtypes and relevant_logtypes, while removing duplicates and keeping order
    top_logtypes = sorted(list(dict.fromkeys(relevant_logtypes + top_logtypes)))       
    log_generator = LogGenerator(top_logtypes)
    wait_time = [10,30, 240]
    # quantities = [100, 200]
    quantities = [1000, 100000, 1000000]
    # diversities = [0, 1]
    diversities = [0, 1, 5]
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
                                                   diversity=int(diversity*31 + 1),
                                                   time_range=time_range)
                write_logs_to_monitor(logs, log_source) # send to splunk
                # wait for 1 minute before executing rules
                logging.info(f'Waiting for 2 minute before executing rules for {rule}')
                sleep(wait_time[i])  # wait for 1 minute for every 1000 logs
                logging.info('Running saved searches')
                results, _ = asyncio.run(splunk_tools.run_saved_searches(time_range, num_measurements=3))
                logging.info(results)
            # clean env
            logging.info('Cleaning environment')
            logging.info(clean_env(splunk_tools, time_range))