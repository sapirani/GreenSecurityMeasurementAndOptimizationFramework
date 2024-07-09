import logging
import subprocess
logger = logging.getLogger(__name__)

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

def update_rules_frequency_and_time_range(splunk_tools_instance, rule_frequency, time_range):
    logger.info('update rules frequency')
    splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_cron_expression, f'*/{rule_frequency} * * * *')
    logger.info('update time range of rules')
    splunk_tools_instance.update_all_searches(splunk_tools_instance.update_search_time_range, time_range)   
    
def empty_monitored_files(monitored_file_path):
    logger.info(f'empty the monitored file {monitored_file_path}')
    with open(monitored_file_path, 'w') as fp:
        fp.write('')
        
def clean_env(splunk_tools_instance, time_range=None):
    if time_range is None:
        time_range = ("04/29/2023:00:00:00","05/30/2023:00:00:00")
        splunk_tools_instance.delete_fake_logs(time_range)
        empty_monitored_files(r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/wineventlog:security.txt")
        empty_monitored_files(r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/wineventlog:system.txt")
        return time_range
    # date = time_range[1].split(':')[0]
    # time_range = (f'{date}:00:00:00', f'{date}:23:59:59')
    splunk_tools_instance.delete_fake_logs(time_range)
    empty_monitored_files(r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/wineventlog:security.txt")
    empty_monitored_files(r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/wineventlog:system.txt")
    return time_range