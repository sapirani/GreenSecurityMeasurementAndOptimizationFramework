import asyncio
from datetime import datetime
import itertools
from time import sleep
import httpx
import json
import logging
from multiprocessing import Pool
import random
import re
import subprocess
from dotenv import load_dotenv
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

load_dotenv('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/.env')
# Precompile the regex pattern
pattern = re.compile(r"'(.*?)' (\D+) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} IST) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \s+(\d+)\s+(\d+\.\d+)")
savedsearches_path = '/opt/splunk/etc/users/shouei/search/local/savedsearches.conf'
APP = 'search'
HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded"
}
PREFIX_PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/'

class SplunkTools:
    def __init__(self, logger):
        self.splunk_host = os.getenv("SPLUNK_HOST")
        self.splunk_port = os.getenv("SPLUNK_PORT")
        self.base_url = f"https://{self.splunk_host}:{self.splunk_port}"
        self.splunk_username = os.getenv("SPLUNK_USERNAME")
        self.splunk_password = os.getenv("SPLUNK_PASSWORD")
        self.index_name = os.getenv("INDEX_NAME")
        self.hec_token1 = os.getenv('HEC_TOKEN1')
        self.hec_token2 = os.getenv('HEC_TOKEN2')
        self.auth = requests.auth.HTTPBasicAuth(self.splunk_username, self.splunk_password)
        self.logger = logger
        self.real_logs_distribution = pd.DataFrame(data=None, columns=['source', 'EventCode', '_time', 'count'])
    
                    
    def write_logs_to_monitor(self, logs, log_source):
        with open(f'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/{log_source}.txt', 'a') as f:
            for log in logs:
                f.write(f'{log}\n\n')
        
        
    def get_saved_search_names(self, get_only_enabled=True, app="search", owner="shouei"):
        query = f"| rest /servicesNS/shouei/search/saved/searches splunk_server=local| search eai:acl.app={app} eai:acl.owner={owner}  | table title, search, cron_schedule, disabled"
        url = f"{self.base_url}/services/search/jobs/export"
        data = {'output_mode': 'json', 'search': query}
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(url,  data=data, auth=self.auth, headers=headers, verify=False)
        if response.status_code != 200:
            self.logger.info(f'Error: {response}')
            return None
        json_objects = response.text.splitlines()
        if 'result' not in json_objects[0]:
            results = []
        else:
            # Parse each line as JSON
            results = [json.loads(obj)['result'] for obj in json_objects]
        if get_only_enabled:
            results = [result for result in results if result['disabled'] == '0']
        return results
    
    def _send_post_request(self, url, data):
        response = requests.post(url, headers=HEADERS, data=data, auth=self.auth, verify=False)
        return response.status_code

    def _update_search(self, saved_search_name, data):
        url = f"{self.base_url}/servicesNS/{self.splunk_username}/{APP}/saved/searches/{saved_search_name}"
        response_code = self._send_post_request(url, data)
        # if response_code == 200:
        #     logging.info(f'Successfully updated saved search "{saved_search_name}".')
        # else:
        if response_code != 200:
            logging.info(f'Failed to update saved search "{saved_search_name}". HTTP status code: {response_code}.')

    def enable_search(self, saved_search_name):
        data = {"disabled": 0}
        self._update_search(saved_search_name, data)
    
    def disable_search(self, saved_search_name):
        data = {"disabled": 1}
        self._update_search(saved_search_name, data)
    
    def update_search_cron_expression(self, saved_search_name, new_schedule):
        data = {"cron_schedule": new_schedule}
        self._update_search(saved_search_name, data)

    def update_search_time_range(self, saved_search_name, time_range):
        earliest_time = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S').timestamp()
        # earliest_time = time_range[0]
        latest_time = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S').timestamp()
        data = {
            "dispatch.earliest_time": earliest_time,
            "dispatch.latest_time": latest_time
        }
        self._update_search(saved_search_name, data)

    def update_all_searches(self, update_func, update_arg):
        searches_names = self.get_saved_search_names()  # Assuming savedsearches_path is defined
        with Pool() as pool:
            pool.starmap(update_func, [(search_name['title'], update_arg) for search_name in searches_names])
            
    # def extract_distribution(self, start_time, end_time):
    #     command = f'/opt/splunk/bin/splunk search "index=main (earliest="{start_time}" latest="{end_time}") | eval is_fake=if(isnotnull(is_fake), is_fake, 0)|stats count by source EventCode is_fake| eventstats sum(count) as totalCount" -maxout 0 -auth shouei:sH231294'
    #     cmd = subprocess.run(command, shell=True, capture_output=True, text=True)
    #     res_dict = {}
    #     if len(cmd.stdout.split('\n')) > 2:
    #         for row in cmd.stdout.split('\n')[2:-1]:
    #             row = row.split()
    #             source = row[0]
    #             event_code = row[1]
    #             is_fake = row[2]
    #             count = row[3]
    #             total_count = row[4]
    #             res_dict[f"{source.lower()} {event_code} {int(is_fake)}"] = int(count)
    #         res_dict['total_count'] = int(total_count)
    #     return res_dict
    def load_real_logs_distribution_bucket(self, start_time, end_time):
        start_time = start_time.timestamp()
        end_time = end_time.timestamp()
        for file in os.listdir(f'{PREFIX_PATH}resources/output_buckets'):
            start_date_file, start_time_file, end_date_file, end_time_file = file.strip(".csv").split('_')[1:]
            start_date_time_file = datetime.strptime(f"{start_date_file} {start_time_file}", '%Y-%m-%d %H-%M-%S').timestamp()
            end_date_time_file = datetime.strptime(f"{end_date_file} {end_time_file}", '%Y-%m-%d %H-%M-%S').timestamp()
            if start_date_time_file <= start_time and end_date_time_file >= end_time:
                self.real_logs_distribution = pd.read_csv(f'{PREFIX_PATH}resources/output_buckets/{file}')
                self.real_logs_distribution['_time'] = pd.to_datetime(self.real_logs_distribution['_time'], format='%Y-%m-%d %H:%M:%S%z')
                self.real_logs_distribution['_time'] = self.real_logs_distribution['_time'].dt.tz_localize(None)
                break                   
    def get_releveant_distribution(self, start_time, end_time):
        # load real logs distribution from csv file with this structure: source, eventcode, _time, count
        date_start_time = datetime.strptime(start_time, '%m/%d/%Y:%H:%M:%S')
        date_end_time = datetime.strptime(end_time, '%m/%d/%Y:%H:%M:%S')
        relevant_logs = self.real_logs_distribution[(self.real_logs_distribution['_time'] >= date_start_time) & (self.real_logs_distribution['_time'] <= date_end_time)]
        if len(relevant_logs) == 0:
            self.logger.info('No relevant logs found in the loaded distribution. Loading the relevant distribution from disk.')
            self.load_real_logs_distribution_bucket(date_start_time, date_end_time)
            relevant_logs = self.real_logs_distribution[(self.real_logs_distribution['_time'] >= date_start_time) & (self.real_logs_distribution['_time'] <= date_end_time)]
        return relevant_logs            
              
    def get_real_distribution(self, start_time, end_time):
        # load real logs distribution from csv file with this structure: source, eventcode, _time, count
        relevant_logs = self.get_releveant_distribution(start_time, end_time)
        relevant_logs = relevant_logs.groupby(['source', 'EventCode']).agg({'count': 'sum'}).reset_index()
        res_dict = {f"{row['source'].lower()} {row['EventCode']}": row['count'] for index, row in relevant_logs.iterrows()}
        return res_dict
        
    def get_search_details(self, search, res_dict, search_endpoint):
        search_id = search
        rule_name = res_dict[search][0].strip()
        time = res_dict[search][1]
        executed_time = res_dict[search][2]
        total_events = res_dict[search][3]
        total_run_time = res_dict[search][4]

        results_response = requests.get(f"{search_endpoint}/{search_id}", params={"output_mode": "json"}, auth=self.auth, verify=False)
        results_data = json.loads(results_response.text)
        try:
            pid = results_data['entry'][0]['content']['pid']
            runDuration = results_data['entry'][0]['content']['runDuration']
            return (rule_name, (search_id, int(pid), executed_time, runDuration, total_events, total_run_time))
        except KeyError as e:
            self.logger.info(f'KeyError - {str(e)}')
        except IndexError as e:
            self.logger.info(f'IndexError - {str(e)}')
        except Exception as e:
            self.logger.info('Unexpected error:', str(e))
        self.logger.info(results_data)

    def get_rules_pids(self, time_range, num_of_searches=0):
        format = "%Y-%m-%d %H:%M:%S"
        format2 = "%m/%d/%Y:%H:%M:%S"
        api_lt = datetime.strptime(time_range[0], format2).timestamp()
        api_et = datetime.strptime(time_range[1], format2).timestamp()
        query = f'index=_audit action=search app=search search_type=scheduled info=completed earliest=-2m api_et={api_lt} api_lt={api_et}\
        | regex search_id=\\"scheduler.*\\"| eval executed_time=strftime(exec_time, \\"{format}\\")\
        | table search_id savedsearch_name _time executed_time event_count total_run_time | sort _time desc | head {num_of_searches}'
        command = f'echo sH231294| sudo -S -E env "PATH"="$PATH" /opt/splunk/bin/splunk search "{query}" -maxout 0 -auth shouei:sH231294'
        cmd = subprocess.run(command, shell=True, capture_output=True, text=True)
        res = cmd.stdout.split('\n')[2:-1]
        res_dict = {re.findall(pattern, line)[0][0]: re.findall(pattern, line)[0][1:] for line in res}
        self.logger.info(f'stderr {cmd.stderr}')
        search_endpoint = f"{self.base_url}/services/search/jobs"
        # Use a multiprocessing pool to get details of all searches in parallel
        with Pool() as pool:
            results = pool.starmap(self.get_search_details, [(search, res_dict, search_endpoint) for search in res_dict])

        pids = {}
        for res in results:
            if res is None:
                continue
            rule_name, details = res
            if rule_name not in pids:
                pids[rule_name] = [details]
            else:
                pids[rule_name].append(details)

        return pids
    
    def get_alert_count(self, sids):
        spl_query = f'search index=_audit action=alert_fired ss_app=search user=shouei earliest=-1h latest=now() | where sid IN {tuple(sids)} |stats count'.replace('\'', '\"')
        # spl_query = 'search index=_internal sourcetype=scheduler thread_id=AlertNotifier* user="shouei"|stats count by savedsearch_name sid'
        url = f"{self.base_url}/services/search/jobs"
        data = {
            "search": spl_query,
            "exec_mode": "oneshot",
            "output_mode": "json"
        }
        response = requests.post(url, headers=HEADERS, data=data, auth=(self.splunk_username, self.splunk_password), verify=False)
        results = json.loads(response.text)
        results = int(results['results'][0]['count'])
        return results
                  
    def extract_logs(self, log_source, time_range=("-24h@h", "now"), eventcode='*', limit=0):
        if  os.path.exists(f'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/logs_to_duplicate_files/{log_source.replace("/", "__")}_{eventcode}.txt'):
            return None        
        # Define your SPL query
        spl_query = f'search index=main source="{log_source}" EventCode={eventcode} earliest="{time_range[0]}" latest="{time_range[1]}"'
        if limit > 0:
            spl_query += f' | head {limit}'
        # Define the REST API endpoint
        url = f"{self.base_url}/services/search/jobs/export"
        data = {'output_mode': 'json', 'search': spl_query}
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(url,  data=data, auth=self.auth, headers=headers, verify=False)
        
        if response.status_code != 200:
            self.logger.info(f'Error: {response}')
            return None
        json_objects = response.text.splitlines()
        if 'result' not in json_objects[0]:
            results = []
        else:
            # Parse each line as JSON
            results = [json.loads(obj)['result']['_raw'] for obj in json_objects]
        self.save_logs(log_source, eventcode, results)
        return results
    

    
    def delete_fake_logs(self, time_range=None):
        url = f"{self.base_url}/services/search/jobs/export"
        if time_range is None:
            time_expression = 'earliest=0'
        else:
            time_expression = f'earliest="{time_range[0]}" latest="{time_range[1]}"'
        data = {
            "search": f'search index=main sourcetype IN ("xmlwineventlog", "wineventlog") is_fake=1 {time_expression} | delete',
            "exec_mode": "oneshot",
            "output_mode": "json"
        }
        response = requests.post(url, headers=HEADERS, data=data, auth=self.auth, verify=False)
        results = response.text
        self.logger.info(results)

        
    def save_logs(self, log_source, eventcode, logs):
        path = f'{PREFIX_PATH}logs_to_duplicate_files'
        if not os.path.exists(path):
            os.makedirs(path)
        # replace / char with '__'
        log_source = log_source.replace('/', '__')
        with open(f'{path}/{log_source}_{eventcode}.txt', 'w') as f:
            for log in logs:
                f.write(f'{log}\n[EOF]\n')
        self.logger.info(f'Saved {len(logs)} logs to {path}/{log_source}_{eventcode}.txt')
                   
    def load_logs_to_duplicate_dict(self, logtypes):
        dir_name = 'logs_resource'
        # load the logs to duplicate from disk
        logs_to_duplicate_dict = {(logtype[0].lower(), logtype[1], istrigger): [] for istrigger,_ in enumerate(['notrigger', 'trigger']) for logtype in logtypes}
        for logtype in logtypes:
            source = logtype[0].lower()
            eventcode = logtype[1]
            for istrigger, istrigger_string in enumerate(['notrigger', 'trigger']):
                path = f'{PREFIX_PATH}{dir_name}/{source.replace("/", "__")}_{eventcode}_{istrigger_string}.txt'
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    text = f.read()
                    results = text.split('\n[EOF]\n')
                    # results = self.split_logs(source, text)   
                    for log in results:
                         if log != '':
                             logs_to_duplicate_dict[(source, eventcode, istrigger)].append(log)
        return logs_to_duplicate_dict   
     
                
    def get_time(self, y, m, d, h, mi, s):
        return datetime(y, m, d, h, mi, s).timestamp()
    
    def sample_log(self, logs, action_value):
        if len(logs) > 0:
            logs = random.sample(logs, min(len(logs), action_value))
            return logs
        else:
            # self.logger.info('No results found or results is not a list.')
            return None  


    async def insert_logs(self, logs, log_source, eventcode, istrigger):
        if len(logs) == 0:
            return
        # select randomly one of the tokens
        hec_tokens = [self.hec_token1, self.hec_token2]
        tasks = []
        for i, token in enumerate(hec_tokens):
            start = i * len(logs) // 2
            end = (i + 1) * len(logs) // 2
            if len(logs) == 1 and i == 0:
                continue                
            task = asyncio.create_task(self._send_logs(logs[start:end], log_source, eventcode, istrigger, token))
            tasks.append(task)
        await asyncio.gather(*tasks)
       
   
    async def _send_logs(self, logs, log_source, eventcode, istrigger, hec_token):
        headers = {
            "Authorization": f"Splunk {hec_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        url = f"http://{self.splunk_host}:8088/services/collector/event"
        events = []
        for i, (log, time) in enumerate(logs):
            events.append(json.dumps({'event': log, 'source': log_source, 'sourcetype': log_source.split(':')[0], 'time': time}))
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, data="\n".join(events))
        if response.status_code == 200:
            self.logger.info(f'Logs successfully sent to Splunk. {len(logs)} logs of source {log_source}_{eventcode}_{istrigger} were sent.')
        else:
            self.logger.info('Failed to send log entry to Splunk.')
            self.logger.info(response.text)
            self.logger.info("\n".join(events))    
            
if __name__ == "__main__":
    logger = logging.getLogger("my_app")
    log_file = 'splunk_tools.log'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    splunk_tools = SplunkTools(logger)
    earliest_time = splunk_tools.get_time(2023, 6, 21, 0, 0, 0)
    latest_time = splunk_tools.get_time(2023, 6, 23, 23, 59, 59)
    # splunk_tools.delete   _fake_logs((earliest_time, latest_time))
    # test generating logs
    # log = splunk_tools.generate_log('xmlwineventlog:microsoft-windows-sysmon/operational', '24')
    # sourcetype = 'xmlwineventlog:microsoft-windows-sysmon/operational'
    # eventcode = '22'
    # logs = []
    # log = splunk_tools.load_logs_to_duplicate_dict([(sourcetype, eventcode)])
    # print(datetime.now())
    # for i in range(20000):
    #     time = datetime.now() - timedelta(days=200)
    #     time = time.timestamp()
    #     logs.append((log[sourcetype, eventcode][0], time))
    # asyncio.run(splunk_tools.insert_logs(logs, sourcetype))
    # print(datetime.now())
    # print(log)
    # print(time)
    # self.logger.info(splunk_tools.get_rules_pids(60))
    # self.logger.info(splunk_tools.extract_logs('WinEventLog:Security', '4624'))
    # test loading logs from disk
    # splunk_tools.load_logs_to_duplicate_dict([('WinEventLog:Security', '2005')])  