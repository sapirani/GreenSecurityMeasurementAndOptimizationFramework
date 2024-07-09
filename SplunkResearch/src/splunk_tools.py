from datetime import datetime
import json
import logging
from multiprocessing import Pool
import re
from dotenv import load_dotenv
import os
import pandas as pd
import requests
from datetime import datetime

load_dotenv('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/.env')
# Precompile the regex pattern
pattern = re.compile(r"'(.*?)' (\D+) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} IDT) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \s+(\d+)\s+(\d+\.\d+)") # IDT and IST are changed when the time is changed
savedsearches_path = '/opt/splunk/etc/users/shouei/search/local/savedsearches.conf'
APP = 'search'
HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded"
}
PREFIX_PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/'
import logging
logger = logging.getLogger(__name__)

class SplunkTools:
    def __init__(self, active_saved_searches=None):
        self.splunk_host = os.getenv("SPLUNK_HOST")
        self.splunk_port = os.getenv("SPLUNK_PORT")
        self.base_url = f"https://{self.splunk_host}:{self.splunk_port}"
        self.splunk_username = os.getenv("SPLUNK_USERNAME")
        self.splunk_password = os.getenv("SPLUNK_PASSWORD")
        self.index_name = os.getenv("INDEX_NAME")
        self.hec_token1 = os.getenv('HEC_TOKEN1')
        self.hec_token2 = os.getenv('HEC_TOKEN2')
        self.auth = requests.auth.HTTPBasicAuth(self.splunk_username, self.splunk_password)
        self.real_logs_distribution = pd.DataFrame(data=None, columns=['source', 'EventCode', '_time', 'count'])
        self.active_saved_searches = self.get_saved_search_names(active_saved_searches)
        
    def query_splunk(self, query, earliest_time, latest_time):
        url = f"{self.base_url}/services/search/jobs/export"
        data = {
            "search": "search "+query,
            "exec_mode": "oneshot",
            "earliest_time": earliest_time,
            "latest_time": latest_time,
            "output_mode": "json"
        }
        headers = {
            "Content-Type": "application/json",
        }
        # measure running time of requests
        time_start = datetime.now()
        response = requests.post(url,  data=data, auth=self.auth, headers=headers, verify=False)
        time_end = datetime.now()
        execution_time = time_end - time_start
        if response.status_code != 200:
            logger.error(f'Error: {response}')
            return None
        json_objects = response.text.splitlines()
        if 'result' not in json.loads(json_objects[0]):
            results = []
        else:
            # Parse each line as JSON
            results = [json.loads(obj)['result'] for obj in json_objects]
        return results, execution_time
    
    def run_saved_searches(self, time_range):
        saved_searches = self.active_saved_searches
        execution_times = []
        results_list = []
        for saved_search in saved_searches:
            results, execution_time = self.run_saved_search(saved_search, time_range)
            execution_times.append(execution_time)
            results_list.append(results)
        return results_list, execution_times
    
    def run_saved_search(self, saved_search, time_range):
        search_name = saved_search['title']
        query = saved_search['search']
        earliest_time = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S').timestamp()
        latest_time = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S').timestamp()
        # insert the time range to the query befor the first pipe
        query = query.split('|')
        query[0] = f'{query[0]} earliest={earliest_time} latest={latest_time}'
        query = '|'.join(query)
        logger.info(f'Running saved search {search_name} with query: {query}')
        results, execution_time = self.query_splunk(query, earliest_time, latest_time)
        return len(results), execution_time.total_seconds()
        
    def make_splunk_request(self, endpoint, method='get', params=None, data=None, headers=None):  
        url = f"{self.base_url}{endpoint}"  
        try:  
            response = requests.request(method, url, auth=self.auth , params=params, data=data, headers=headers, verify=False)  
            response.raise_for_status()  # Raise an exception for HTTP errors  
            if 'application/json' in response.headers.get('Content-Type', ''):  
                return response.json()  # Return JSON response if applicable  
            return response.text  # Return raw response text if not JSON  
        except requests.exceptions.HTTPError as err:  
            logger.error(f'HTTP error occurred: {err}')  
        except Exception as err:  
            logger.error(f'Other error occurred: {err}')  
        return None  
                    
    def write_logs_to_monitor(self, logs, log_source):
        with open(f'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/{log_source}.txt', 'a') as f:
            for log in logs:
                f.write(f'{log}\n\n')        
        
    def get_saved_search_names(self, active_saved_searches, get_only_enabled=True, app="search", owner="shouei"):
        query = f"| rest /servicesNS/shouei/search/saved/searches splunk_server=local| search eai:acl.app={app} eai:acl.owner={owner}  | table title, search, cron_schedule, disabled"
        url = f"{self.base_url}/services/search/jobs/export"
        data = {'output_mode': 'json', 'search': query}
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(url,  data=data, auth=self.auth, headers=headers, verify=False)
        if response.status_code != 200:
            logger.error(f'Error: {response}')
            return None
        json_objects = response.text.splitlines()
        if 'result' not in json_objects[0]:
            results = []
        else:
            # Parse each line as JSON
            results = [json.loads(obj)['result'] for obj in json_objects]
        if get_only_enabled:
            results = [result for result in results if result['title'] in active_saved_searches]
        return sorted(results, key=lambda x: x['title'])
    
    def _send_post_request(self, url, data):
        response = requests.post(url, headers=HEADERS, data=data, auth=self.auth, verify=False)
        return response.status_code

    def _update_search(self, saved_search_name, data):
        url = f"{self.base_url}/servicesNS/{self.splunk_username}/{APP}/saved/searches/{saved_search_name}"
        response_code = self._send_post_request(url, data)
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
        searches_names = self.active_saved_searches
        with Pool() as pool:
            pool.starmap(update_func, [(search_name['title'], update_arg) for search_name in searches_names])
            
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
            logger.info('No relevant logs found in the loaded distribution. Loading the relevant distribution from disk.')
            self.load_real_logs_distribution_bucket(date_start_time, date_end_time)
            relevant_logs = self.real_logs_distribution[(self.real_logs_distribution['_time'] >= date_start_time) & (self.real_logs_distribution['_time'] <= date_end_time)]
        return relevant_logs            
              
    def get_real_distribution(self, start_time, end_time):
        # load real logs distribution from csv file with this structure: source, eventcode, _time, count
        relevant_logs = self.get_releveant_distribution(start_time, end_time)
        relevant_logs = relevant_logs.groupby(['source', 'EventCode']).agg({'count': 'sum'}).reset_index()
        res_dict = {f"{row['source'].lower()} {row['EventCode']}": row['count'] for index, row in relevant_logs.iterrows()}
        return res_dict
        
    def get_search_details(self, search,  is_measure_energy=False):
        search_id = search['search_id'].strip('\'')
        rule_name = search['savedsearch_name']
        executed_time = search['executed_time']
        total_events = search['event_count']
        total_run_time = search['total_run_time']
        if is_measure_energy:
            search_endpoint = f"{self.base_url}/services/search/jobs"
            results_response = requests.get(f"{search_endpoint}/{search_id}", params={"output_mode": "json"}, auth=self.auth, verify=False)
            results_data = json.loads(results_response.text)
            try:
                pid = results_data['entry'][0]['content']['pid']
                runDuration = results_data['entry'][0]['content']['runDuration']
                return (rule_name, (search_id, float(pid), executed_time, float(total_run_time), total_events))
            except KeyError as e:
                logger.info(f'KeyError - {str(e)}')
            except IndexError as e:
                logger.info(f'IndexError - {str(e)}')
            except Exception as e:
                logger.info('Unexpected error:', str(e))
            logger.info(results_data)
        else:
            return (rule_name, (search_id, executed_time, float(total_run_time), total_events))

    def get_rules_pids(self, time_range, num_of_searches, is_measure_energy=False):
        """Get the PIDs of rules that were executed during the given time range."""
        format_str = "%Y-%m-%d %H:%M:%S"
        format_str2 = "%m/%d/%Y:%H:%M:%S"
        api_lt = datetime.strptime(time_range[0], format_str2).timestamp()
        api_et = datetime.strptime(time_range[1], format_str2).timestamp()
        spl_query = (
            f"search index=_audit action=search app=search search_type=scheduled info=completed earliest=-2m search_et={api_lt} search_lt={api_et}"
            f'| regex search_id=\"scheduler.*\"| eval executed_time=strftime(exec_time, \"{format_str}\")'
            f"| table search_id savedsearch_name _time executed_time event_count total_run_time | sort _time desc | head {num_of_searches}"
        )
        url = f"{self.base_url}/services/search/jobs"
        data = {
            "search": spl_query,
            "exec_mode": "oneshot",
            "output_mode": "json"
        }
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(url,  data=data, auth=self.auth, headers=headers, verify=False)
        logger.info(response.text)
        if response.status_code != 200:
            logger.info(f'Error: {response}')
            return None
        results = json.loads(response.text)['results']
        with Pool() as pool:
            results = pool.starmap(self.get_search_details, [(search, is_measure_energy) for search in results])   
        pids = {}
        for res in results:
            if res is None:
                continue
            rule_name, details = res
            pids.setdefault(rule_name, []).append(details)
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
    
    def delete_fake_logs(self, time_range=None):
        url = f"{self.base_url}/services/search/jobs/export"
        if time_range is None:
            time_expression = 'earliest=0'
        else:
            time_expression = f'earliest="{time_range[0]}" latest="{time_range[1]}"'
        data = {
            "search": f'search index=main sourcetype IN ("xmlwineventlog", "wineventlog") host="dt-splunk" {time_expression} | delete',
            "exec_mode": "oneshot",
            "output_mode": "json"
        }
        response = requests.post(url, headers=HEADERS, data=data, auth=self.auth, verify=False)
        results = response.text
        logger.info(results)

    def save_logs(self, log_source, eventcode, logs):
        path = f'{PREFIX_PATH}logs_to_duplicate_files'
        if not os.path.exists(path):
            os.makedirs(path)
        # replace / char with '__'
        log_source = log_source.replace('/', '__')
        with open(f'{path}/{log_source}_{eventcode}.txt', 'w') as f:
            for log in logs:
                f.write(f'{log}\n[EOF]\n')
        logger.info(f'Saved {len(logs)} logs to {path}/{log_source}_{eventcode}.txt')
                   
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
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    # def extract_distribution(self, start_time, end_time):
    #     command = f'/opt/splunk/bin/splunk search "index=main (earliest="{start_time}" latest="{end_time}") | eval is_fake=if(isnotnull(is_fake), is_fake, 0)|stats count by source EventCode is_fake| eventstats sum(count) as totalCount" -maxout 0 -auth shouei:'
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
    # def extract_logs(self, log_source, time_range=("-24h@h", "now"), eventcode='*', limit=0):
    #     if  os.path.exists(f'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/logs_to_duplicate_files/{log_source.replace("/", "__")}_{eventcode}.txt'):
    #         return None        
    #     # Define your SPL query
    #     spl_query = f'search index=main source="{log_source}" EventCode={eventcode} earliest="{time_range[0]}" latest="{time_range[1]}"'
    #     if limit > 0:
    #         spl_query += f' | head {limit}'
    #     # Define the REST API endpoint
    #     url = f"{self.base_url}/services/search/jobs/export"
    #     data = {'output_mode': 'json', 'search': spl_query}
    #     headers = {
    #         "Content-Type": "application/json",
    #     }
    #     response = requests.post(url,  data=data, auth=self.auth, headers=headers, verify=False)
        
    #     if response.status_code != 200:
    #         logger.info(f'Error: {response}')
    #         return None
    #     json_objects = response.text.splitlines()
    #     if 'result' not in json_objects[0]:
    #         results = []
    #     else:
    #         # Parse each line as JSON
    #         results = [json.loads(obj)['result']['_raw'] for obj in json_objects]
    #     self.save_logs(log_source, eventcode, results)
    #     return results
    
    
    # def sample_log(self, logs, action_value):
    #     if len(logs) > 0:
    #         logs = random.sample(logs, min(len(logs), action_value))
    #         return logs
    #     else:
    #         # logger.info('No results found or results is not a list.')
    #         return None  


    # async def insert_logs(self, logs, log_source, eventcode, istrigger):
    #     if len(logs) == 0:
    #         return
    #     # select randomly one of the tokens
    #     hec_tokens = [self.hec_token1, self.hec_token2]
    #     tasks = []
    #     for i, token in enumerate(hec_tokens):
    #         start = i * len(logs) // 2
    #         end = (i + 1) * len(logs) // 2
    #         if len(logs) == 1 and i == 0:
    #             continue                
    #         task = asyncio.create_task(self._send_logs(logs[start:end], log_source, eventcode, istrigger, token))
    #         tasks.append(task)
    #     await asyncio.gather(*tasks)
       
   
    # async def _send_logs(self, logs, log_source, eventcode, istrigger, hec_token):
    #     headers = {
    #         "Authorization": f"Splunk {hec_token}",
    #         "Content-Type": "application/x-www-form-urlencoded",
    #     }
    #     url = f"http://{self.splunk_host}:8088/services/collector/event"
    #     events = []
    #     for i, (log, time) in enumerate(logs):
    #         events.append(json.dumps({'event': log, 'source': log_source, 'sourcetype': log_source.split(':')[0], 'time': time}))
    #     async with httpx.AsyncClient() as client:
    #         response = await client.post(url, headers=headers, data="\n".join(events))
    #     if response.status_code == 200:
    #         logger.info(f'Logs successfully sent to Splunk. {len(logs)} logs of source {log_source}_{eventcode}_{istrigger} were sent.')
    #     else:
    #         logger.info('Failed to send log entry to Splunk.')
    #         logger.info(response.text)
    #         logger.info("\n".join(events))    
            
# if __name__ == "__main__":
   # test run saved searches for 01/05/2023
    # splunk_tools = SplunkTools()
    # splunk_tools.run_saved_searches(['05/01/2023:09:00:00', '05/01/2023:12:10:00'])
    # test get rules pids
    # pids = splunk_tools.get_rules_pids(['01/01/2022:00:00:00', '01/01/2022:00:10:00'], 10)
    # print(pids)
    # test get alert count
    # alert_count = splunk_tools.get_alert_count(['scheduler__admin__search__RMD5d3d4f6f9d2d3c3c_at_1640990400_108'])
    # print(alert_count)
    # test delete fake logs
    # splunk_tools.delete_fake_logs()
    # test save logs
    # splunk_tools.save_logs('WinEventLog:Security', 4624, ['log1', 'log2'])
    # test load logs to duplicate dict
    # logtypes = [('WinEventLog:Security', 4624), ('WinEventLog:Security', 4625)]
    # logs_to_duplicate_dict = splunk_tools.load_logs_to_duplicate_dict(logtypes)
    # print(logs_to_duplicate_dict)
    # test get real distribution
    # real_distribution = splunk_tools.get_real_distribution('01/01/2022:00:00:00', '01/01/2022:00:10:00')
    # print(real_distribution)
    # test get saved search names
    # saved_searches = splunk_tools.get_saved_search_names()
    # print(saved_searches)
    # test run saved search
    # splunk_tools.run_saved_search(saved_searches[0], ['01/01/2022:00:00:00', '01/01/2022:00:10:00'])
    # test enable search
    # splunk_tools.enable_search(saved_searches[0]['title'])
    # test disable search
    # splunk_tools.disable_search(saved_searches[0]['title'])
    # test update search cron expression
    # splunk_tools.update_search_cron_expression(saved_searches[0]['title'], '*/5 * * * *')
    # test update search time range
    # splunk_tools.update_search_time_range(saved_searches[0