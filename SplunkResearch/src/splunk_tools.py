from datetime import datetime
import json
import logging
from multiprocessing import Pool
import re
import time
import splunklib.client as client
import splunklib.results as splunk_results
import psutil

import subprocess
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime

from random import randint

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
class SplunkTools(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SplunkTools, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, active_saved_searches=None, num_of_measurements=1, rule_frequency=1):
        if self._initialized:
            return
        
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
        self.num_of_measurements = num_of_measurements
        self.real_logtypes_counter = {}
        self.rule_frequency = rule_frequency
        self.service = client.connect(
            host=self.splunk_host,
            port=self.splunk_port,
            username=self.splunk_username,
            password=self.splunk_password
        )
        self._initialized = True

    
    def get_num_of_searches(self):
        return len(self.active_saved_searches)
    
    def query_splunk(self, query, earliest_time, latest_time):
        query = f" search {query}"
        k = self.num_of_measurements
        results = []
        execution_times = []
        cpu_integral = []
        io_metrics = []
        interval = 0.1
        for i in range(k):
            # clear cache before each measurement
            # subprocess.run(f'echo 1 > /proc/sys/vm/drop_caches', shell=True)

            while True:
                logger.info(f'Meaurement {i+1}/{k} - Running query: {query}')
                # Create a new search job
                job = self.service.jobs.create(query, earliest_time=earliest_time, latest_time=latest_time)
                process_cpu_percents = []
                io_counters_dict = {"read_chars": 0, "write_chars": 0, "read_count": 0, "write_count": 0, "read_bytes": 0, "write_bytes": 0}
                # A blocking call to wait for the job to finish    
                job.refresh()
                stats = job.content
                pid = stats.get('pid', None)
                if pid is not None:
                    found = False
                    try:
                        job.refresh()                        
                        process = psutil.Process(int(pid))
                        found = True
                        with process.oneshot():
                            while (process.is_running() and not process.status() == 'sleeping') or job.content['isDone'] == '0':
                                logger.debug(f"Process with PID {pid} is {process.status()}.")
                                job.refresh()
                                stats = job.content

                                process = psutil.Process(int(pid))
                                # Extract relevant resource metrics
                                scan_count = stats.get('scanCount', 0)
                                event_count = stats.get('eventCount', 0)
                                result_count = stats.get('resultCount', 0)
                                disk_usage = stats.get('diskUsage', 0)
                                run_duration = stats.get('runDuration', 0)
                                cpu_num = psutil.cpu_count()                                
                                cpu_percent = process.cpu_percent(interval=interval)
                                cpu_times = process.cpu_times()
                                memory_info = process.memory_info()
                                io_counters = process.io_counters()
                                read_chars = io_counters.read_chars
                                write_chars = io_counters.write_chars
                                read_count = io_counters.read_count
                                write_count = io_counters.write_count
                                read_bytes = io_counters.read_bytes
                                write_bytes = io_counters.write_bytes
                                process_cpu_percents.append(cpu_percent)
                                io_counters_dict["read_chars"] += int(read_chars)
                                io_counters_dict["write_chars"] +=int( write_chars)
                                io_counters_dict["read_count"] += int(read_count)
                                io_counters_dict["write_count"] +=int( write_count)
                                io_counters_dict["read_bytes"] += int(read_bytes)
                                io_counters_dict["write_bytes"] +=int( write_bytes)
                                # print(f"Duration: {run_duration}")
                                # print(f"CPU percent: {cpu_percent}")
                                # print(f"CPU times: {cpu_times}")
                                # print(f"Memory info: {memory_info}")
                                # print(f"IO counters: {io_counters}")
                                # print(f"Read chars: {read_chars}")
                                # print(f"Write chars: {write_chars}")
                                # print(f"Read count: {read_count}")
                                # print(f"Write count: {write_count}")
                                # print(f"Read bytes: {read_bytes}")
                                # print(f"Write bytes: {write_bytes}")

                        job.refresh()
                        logger.info(f"Process with PID {pid} is {process.status()} isdone={job.content['isDone']}.")
                        if job.content['isDone'] == '1':
                            break

                                # time.sleep(0.01)
                    except psutil.NoSuchProcess:
                        if found:
                            logger.info(f"process endded")
                            break
                        else:
                            logger.info(f"Process with PID {pid} does not exist.")
                            job = self.service.jobs.create(query, earliest_time=earliest_time, latest_time=latest_time)
                            logger.info(f"Job was recreated.")
                    except psutil.AccessDenied:
                        logger.info(f"Access denied to process with PID {pid}.")
                    
            job.refresh()       
            cpu_auc = np.trapz(process_cpu_percents, dx=interval)
            cpu_integral.append(cpu_auc)
            io_metrics.append(io_counters_dict)
                       
            # Extract the results
            response = job.results(output_mode='json')
            logger.info("Results:")
            reader = splunk_results.JSONResultsReader(response)
            # reader = splunk_results.ResultsReader(job.results())
            results = []
            for result in reader:
                logger.info(result)
                if isinstance(result, dict):
                    results.append(result)
            # Extract the execution time
            execution_times.append(float(run_duration))
            # time.sleep(randint(1, 5)/5)
            logger.info(f"Execution time: {run_duration}")
            logger.info(f"CPU integral: {cpu_auc}")
            logger.info(f"Aletrt count: {len(results)}")
        return results, execution_times, cpu_integral, io_metrics
    
    def run_saved_searches(self, time_range):
        saved_searches = self.active_saved_searches
        mean_execution_times = []
        std_execution_times = []
        cpu= []
        std_cpu = []
        results_list = []
        read_chars = []
        write_chars = []
        read_count = []
        write_count = []
        read_bytes = []
        write_bytes = []
        saved_searches_titles = []
        # make random order of the saved searches
        np.random.shuffle(saved_searches)
        for saved_search in saved_searches:
            results_len, mean_execution_time, std_execution_time, mean_cpu_integral, std_cpu_integrals, sum_read_chars, sum_write_chars, sum_read_count, sum_write_count, sum_read_bytes, sum_write_bytes = self.run_saved_search(saved_search, time_range)
            mean_execution_times.append(mean_execution_time)
            std_execution_times.append(std_execution_time)
            results_list.append(results_len)
            cpu.append(mean_cpu_integral)
            std_cpu.append(std_cpu_integrals)
            read_chars.append(sum_read_chars)
            write_chars.append(sum_write_chars)
            read_count.append(sum_read_count)
            write_count.append(sum_write_count)
            read_bytes.append(sum_read_bytes)
            write_bytes.append(sum_write_bytes)
            saved_searches_titles.append(saved_search['title'])
        return results_list, mean_execution_times, std_execution_times, saved_searches_titles, cpu, std_cpu, read_chars, write_chars, read_count, write_count, read_bytes, write_bytes
    
    def query_metrics_combiner(self, results, execution_times, cpu_integral, io_metrics):
        mean_execution_time = np.mean(execution_times)
        std_execution_time = np.std(execution_times)
        mean_cpu_integral = np.mean(cpu_integral)
        std_cpu_integral = np.std(cpu_integral)
        sum_read_chars = sum([io['read_chars'] for io in io_metrics])
        sum_write_chars = sum([io['write_chars'] for io in io_metrics])
        sum_read_count = sum([io['read_count'] for io in io_metrics])
        sum_write_count = sum([io['write_count'] for io in io_metrics])
        sum_read_bytes = sum([io['read_bytes'] for io in io_metrics])
        sum_write_bytes = sum([io['write_bytes'] for io in io_metrics])
        return len(results), mean_execution_time, std_execution_time, mean_cpu_integral, std_cpu_integral, sum_read_chars, sum_write_chars, sum_read_count, sum_write_count, sum_read_bytes, sum_write_bytes

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
        # clear machine cache
        results, execution_times, cpu_integral, io_metrics = self.query_splunk(query, earliest_time, latest_time)
        return self.query_metrics_combiner(results, execution_times, cpu_integral, io_metrics)
            
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
        while True:
            ts_start_time = start_time.timestamp()
            ts_end_time = end_time.timestamp()
            for file in os.listdir(f'{PREFIX_PATH}resources/output_buckets'):
                start_date_file, start_time_file, end_date_file, end_time_file = file.strip(".csv").split('_')[1:]
                start_date_time_file = datetime.strptime(f"{start_date_file} {start_time_file}", '%Y-%m-%d %H-%M-%S').timestamp()
                end_date_time_file = datetime.strptime(f"{end_date_file} {end_time_file}", '%Y-%m-%d %H-%M-%S').timestamp()
                if start_date_time_file <= ts_start_time and end_date_time_file >= ts_end_time:
                    self.real_logs_distribution = pd.read_csv(f'{PREFIX_PATH}resources/output_buckets/{file}')
                    self.real_logs_distribution['_time'] = pd.to_datetime(self.real_logs_distribution['_time'], format='%Y-%m-%d %H:%M:%S%z')
                    self.real_logs_distribution['_time'] = self.real_logs_distribution['_time'].dt.tz_localize(None)
                    return
            self.create_new_distribution_bucket(ts_start_time, ts_end_time)

    def create_new_distribution_bucket(self, start_time, end_time):
        # round the start and end time to the nearest day
        start_time = datetime.fromtimestamp(start_time)
        end_time = datetime.fromtimestamp(end_time)
        start_time = datetime(start_time.year, start_time.month, start_time.day)
        end_time = start_time + pd.DateOffset(days=1)
        timestamp_start_time = start_time.timestamp()
        timestamp_end_time = end_time.timestamp()
        job = self.service.jobs.create(f'search index=main earliest={timestamp_start_time} latest={timestamp_end_time} |  eval _time=strftime(_time,"%Y-%m-%d %H:%M:%S%z")| stats count by source EventCode _time', earliest_time=timestamp_start_time, latest_time=timestamp_end_time, count=0) 
        while True:
            job.refresh()
            if job.content['isDone'] == '1':
                break
            time.sleep(2)
        # Extract the results
        response = job.results(output_mode='json', count=0)
        reader = splunk_results.JSONResultsReader(response)
        results = []
        start_time = start_time.strftime('%Y-%m-%d_%H-%M-%S')
        end_time = end_time.strftime('%Y-%m-%d_%H-%M-%S')
        file = f"bucket_{start_time}_{end_time}.csv"
        for result in reader:
            if isinstance(result, dict):
                results.append(result)
        self.real_logs_distribution = pd.DataFrame(results, index=None)
        self.real_logs_distribution.to_csv(f'{PREFIX_PATH}resources/output_buckets/{file}')
        self.real_logs_distribution['_time'] = pd.to_datetime(self.real_logs_distribution['_time'], format='%Y-%m-%d %H:%M:%S%z')
        self.real_logs_distribution['_time'] = self.real_logs_distribution['_time'].dt.tz_localize(None)
        
                             
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
        logger.debug(f"current real distribution: {res_dict}")
        
        for logtype in res_dict:
            if logtype in self.real_logtypes_counter:
                self.real_logtypes_counter[logtype] += res_dict[logtype]
            else:
                self.real_logtypes_counter[logtype] = res_dict[logtype]
        return self.real_logtypes_counter
        
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
    
    def get_logs_amount(self, time_range):
        relevant_logs = self.get_releveant_distribution(time_range[0], time_range[1])
        return relevant_logs['count'].sum()
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
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