import datetime
import logging
import os
import subprocess
import json
import numpy as np
from config import config

# Factory functions to generate paths
def get_system_monitor_path(host):
    base_dir = config.get('paths.splunk_research_dir')
    return f"{base_dir}/monitor_files_{host}/wineventlog:system.txt"

def get_security_monitor_path(host):
    base_dir = config.get('paths.splunk_research_dir')
    return f"{base_dir}/monitor_files_{host}/wineventlog:security.txt"

logger = logging.getLogger(__name__)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def update_running_time(running_time, env_file_path):
    command = f'sed -i "s/RUNNING_TIME=.*/RUNNING_TIME={running_time}/" "{env_file_path}"'
    res = subprocess.run(command, shell=True, capture_output=True, text=True)
    logger.info(res.stdout)
    logger.error(res.stderr)

def choose_random_rules(splunk_tools_instance, num_of_searches, is_get_only_enabled=True):
    logger.info('enable random rules')
    savedsearches = splunk_tools_instance.get_saved_search_names(get_only_enabled=is_get_only_enabled)
    # random_savedsearch = random.sample(savedsearches, num_of_searches)
    random_savedsearch = ['ESCU Network Share Discovery Via Dir Command Rule', 'Detect New Local Admin account',"Known Services Killed by Ransomware", "Non Chrome Process Accessing Chrome Default Dir"]
    for savedsearch in savedsearches:
        if savedsearch not in random_savedsearch:
            splunk_tools_instance.disable_search(savedsearch)
        else:
            splunk_tools_instance.enable_search(savedsearch)
    return random_savedsearch

def update_rules_frequency_and_time_range(splunk_tools_instance, time_range):
    logger.info('update rules frequency')
    splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_cron_expression, f'*/{splunk_tools_instance.rule_frequency} * * * *')
    logger.info('update time range of rules')
    splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_time_range, time_range)   
    
def empty_monitored_files(monitored_file_path):
    logger.info(f'empty the monitored file {monitored_file_path}')
    #check if file exists
    if not os.path.exists(monitored_file_path):
        logger.warning(f'file {monitored_file_path} does not exist')
        return
    with open(monitored_file_path, 'w') as fp:
        fp.write('')
        
def clean_env(splunk_tools_instance, time_range=None,logs_qnt=None, host='host1'):
    if time_range is None:
        time_range = ("04/29/2023:00:00:00","05/30/2023:00:00:00")
        splunk_tools_instance.delete_fake_logs(time_range)
        empty_monitored_files(get_system_monitor_path(host))
        empty_monitored_files(get_security_monitor_path(host))
        return time_range
        
    # add small margin to the time range
    if not type(time_range) is str:
        # Support TimeWindow objects (have start_dt/end_dt), epoch tuples, or string tuples
        if hasattr(time_range, 'start_dt'):
            start_time = time_range.start_dt
            end_time = time_range.end_dt
        elif isinstance(time_range[0], (int, float)):
            start_time = datetime.datetime.fromtimestamp(time_range[0])
            end_time = datetime.datetime.fromtimestamp(time_range[1])
        else:
            start_time = datetime.datetime.strptime(time_range[0], "%m/%d/%Y:%H:%M:%S")
            end_time = datetime.datetime.strptime(time_range[1], "%m/%d/%Y:%H:%M:%S")
        start_time = start_time - datetime.timedelta(minutes=5)
        end_time = end_time + datetime.timedelta(minutes=5)
        # Pass epoch tuple to avoid re-parsing in delete_fake_logs
        time_range = (start_time.timestamp(), end_time.timestamp())
        logger.info(f'update time range to {time_range}')
        empty_monitored_files(get_system_monitor_path(host))
        empty_monitored_files(get_security_monitor_path(host))
    splunk_tools_instance.delete_fake_logs(time_range, logs_qnt=logs_qnt)

    return time_range