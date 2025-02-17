from datetime import datetime
import json
import logging
from multiprocessing import Pool
import re
import time
import splunklib.client as client
import splunklib.results as splunk_results
import psutil
import concurrent.futures
from datetime import timezone
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import subprocess
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from datetime import timezone
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
        self.real_logs_distribution['_time'] = pd.to_datetime(self.real_logs_distribution['_time'])
        self.max_workers = 3
        self.total_interval_queue = queue.Queue()
        self.total_cpu_queue = queue.Queue()
        self.total_cpu = 0
        self.stop_cpu_monitor = threading.Event()
        while True:
            try:
                self.active_saved_searches = self.get_saved_search_names(active_saved_searches)
                self.num_of_measurements = num_of_measurements
                self.real_logtypes_counter = {}
                self.rule_frequency = rule_frequency
                self.service = client.connect(
                    host=self.splunk_host,
                    port=self.splunk_port,
                    username=self.splunk_username,
                    password=self.splunk_password,
                    autologin=True
                )
                break
            except Exception as e:
                logger.error(f'Failed to connect to Splunk: {str(e)}')
                time.sleep(5)
        self._initialized = True

    
    def get_num_of_searches(self):
        return len(self.active_saved_searches)
    
    def monitor_total_cpu(self, interval=0.1):
        """Monitor total CPU usage of the machine"""
        # cpu_measurements = []
        # intervals = [] 
        total_cpu_time_start = psutil.cpu_times().user# + psutil.cpu_times().system 
        while not self.stop_cpu_monitor.is_set():
            # start_time = time.time()
            # cpu_percent = psutil.cpu_percent(interval=interval)
            # end_time = time.time()  
            # self.total_cpu_queue.put(cpu_percent)
            # self.total_interval_queue.put(end_time - start_time)
            cpu_time = psutil.cpu_times().user# + psutil.cpu_times().system
            time.sleep(interval)
        self.total_cpu = cpu_time - total_cpu_time_start

    
    def query_splunk(self, query, earliest_time, latest_time):
        query = f" search {query}"
        k = self.num_of_measurements
        results = []
        execution_times = []
        cpu_integral = []
        io_metrics = []
        interval = 0.1
        intervals = []
        
        for i in range(k):
            logger.info(f'Measurement {i+1}/{k} - Running query: {query}')
            job = self.service.jobs.create(query, earliest_time=earliest_time, latest_time=latest_time)
            process_cpu_percents = []
            io_counters_dict = { "read_count": 0,  "write_count": 0, "read_bytes": 0, "write_bytes": 0}
            
            # Wait for job to get PID
            while True:
                job.refresh()
                stats = job.content
                pid = stats.get('pid', None)
                if pid is not None or stats['isDone'] == '1':
                    break
                time.sleep(0.01)

            if pid is not None:
                try:
                    process = psutil.Process(int(pid))
                    process_start_time = time.time()
                    # previous_time = process_start_time
                    process_start_cpu_time = process.cpu_times().user# + process.cpu_times().system
                    while True:
                        try:
                            process_end_cpu_time = process.cpu_times().user# + process.cpu_times().system
                            if not process.is_running():
                                logger.debug(f"Process {pid} has finished running")
                                break
                                
                            job.refresh()
                            if job.content['isDone'] == '1':
                                logger.debug(f"Job is marked as done")
                                break
                            # cpu_percent = process.cpu_percent(interval=interval)
                            # concurrent_time = time.time()
                            # intervals.append(concurrent_time - previous_time)
                            # previous_time = concurrent_time
                            # process_cpu_percents.append(cpu_percent)
                            
                            with process.oneshot():
                                io_counters = process.io_counters()
                                
                                # Update IO metrics
                                for key in io_counters_dict:
                                    io_counters_dict[key] += getattr(io_counters, key)
                                
                            time.sleep(interval)
                            
                        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                            # If the process disappeared but the job isn't done, it might have spawned a new process
                            job.refresh()
                            if job.content['isDone'] == '0':
                                logger.debug(e)
                                logger.debug(f"Process {pid} disappeared but job isn't done. Waiting for job completion.")
                                time.sleep(interval)
                                continue
                            else:
                                logger.debug(f"Process {pid} no longer exists but job is done")
                                break
                            
                        # Add timeout protection
                        if time.time() - process_start_time > 300:  # 5 minute timeout
                            logger.warning(f"Process monitoring timed out after 5 minutes for pid {pid}")
                            break
                            
                except psutil.NoSuchProcess:
                    logger.debug(f"Initial process {pid} not found")
                except Exception as e:
                    logger.error(f"Unexpected error monitoring process {pid}: {e}")
                finally:
                    # Ensure we get the final job status
                    try:
                        job.refresh()
                    except Exception as e:
                        logger.error(f"Error refreshing job status: {e}")
            process_cpu_time = process_end_cpu_time - process_start_cpu_time
            job.refresh()
            stats = job.content
            run_duration = stats.get('runDuration', 0)
            
            # Calculate CPU integral
            # cpu_auc = np.trapz(np.array(process_cpu_percents)/100, intervals)
            cpu_integral.append(process_cpu_time)
            io_metrics.append(io_counters_dict)
            
            # Get results
            response = job.results(output_mode='json')
            results = [result for result in splunk_results.JSONResultsReader(response) if isinstance(result, dict)]
            logger.info(f"Measurement {i+1}/{k} - Query results: {results}")
            execution_times.append(float(run_duration))

        return results, execution_times, cpu_integral, io_metrics

    def run_saved_search_parallel(self, saved_search, time_range):
        search_name = saved_search['title']
        query = saved_search['search']
        earliest_time = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S').timestamp()
        latest_time = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S').timestamp()
        
        # Modify query with time range
        query = query.split('|')
        query[0] = f'{query[0]} earliest={earliest_time} latest={latest_time}'
        query = '|'.join(query)
        
        logger.info(f'Running saved search {search_name} with query: {query}')
        results, execution_times, cpu_integral, io_metrics = self.query_splunk(query, earliest_time, latest_time)
        return self.query_metrics_combiner(results, execution_times, cpu_integral, io_metrics)

    def run_saved_searches_parallel(self, time_range):
        self.stop_cpu_monitor = threading.Event()
        saved_searches = self.active_saved_searches
        results = {
            'mean_execution_times': [], 'std_execution_times': [], 
            'results_list': [], 'cpu': [], 'std_cpu': [],
            'read_chars': [], 'write_chars': [], 'read_count': [],
            'write_count': [], 'read_bytes': [], 'write_bytes': [],
            'saved_searches_titles': [], 'total_cpu_usage': []
        }

        # Start CPU monitoring in a separate thread
        cpu_monitor_thread = threading.Thread(target=self.monitor_total_cpu)
        cpu_monitor_thread.start()

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Change this part: submit all tasks and keep futures in order
                futures = []
                for search in saved_searches:
                    future = executor.submit(self.run_saved_search_parallel, search, time_range)
                    futures.append((future, search['title']))
                
                # Process results in order
                for future, search_title in futures:
                    try:
                        (results_len, mean_exec_time, std_exec_time, 
                        mean_cpu_integral, std_cpu_integrals, 
                        sum_read_count, sum_write_count, 
                        sum_read_bytes, sum_write_bytes) = future.result()

                        # Store results (unchanged)
                        results['mean_execution_times'].append(mean_exec_time)
                        results['std_execution_times'].append(std_exec_time)
                        results['results_list'].append(results_len)
                        results['cpu'].append(mean_cpu_integral)
                        results['std_cpu'].append(std_cpu_integrals)
                        results['read_count'].append(sum_read_count)
                        results['write_count'].append(sum_write_count)
                        results['read_bytes'].append(sum_read_bytes)
                        results['write_bytes'].append(sum_write_bytes)
                        results['saved_searches_titles'].append(search_title)

                    except Exception as e:
                        logger.error(f"Search {search_title} generated an exception: {e}")

        finally:
            # Stop CPU monitoring and get measurements (unchanged)
            self.stop_cpu_monitor.set()
            cpu_monitor_thread.join()
            results['total_cpu_usage'] = [self.total_cpu]

        return (results['results_list'], results['mean_execution_times'], 
                results['std_execution_times'], results['saved_searches_titles'],
                results['cpu'], results['std_cpu'], results['read_count'], results['write_count'],
                results['read_bytes'], results['write_bytes'], results['total_cpu_usage'])
    def query_metrics_combiner(self, results, execution_times, cpu_integral, io_metrics):
        mean_execution_time = np.mean(execution_times)
        std_execution_time = np.std(execution_times)
        mean_cpu_integral = np.mean(cpu_integral)
        std_cpu_integral = np.std(cpu_integral)
        sum_read_count = sum([io['read_count'] for io in io_metrics])
        sum_write_count = sum([io['write_count'] for io in io_metrics])
        sum_read_bytes = sum([io['read_bytes'] for io in io_metrics])
        sum_write_bytes = sum([io['write_bytes'] for io in io_metrics])
        return (len(results), mean_execution_time, std_execution_time, 
                mean_cpu_integral, std_cpu_integral, sum_read_count, sum_write_count, 
                sum_read_bytes, sum_write_bytes)       
           
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
        """
        Load and merge logs from multiple buckets within specified time range.
        start_time and end_time should be timezone-aware datetime objects
        """
        ts_start_time = start_time
        ts_end_time = end_time
        logger.info(f'Loading real logs distribution for time range: {ts_start_time} - {ts_end_time}')
        all_data = []
        current_ts = ts_start_time
        
        while current_ts < ts_end_time:
            bucket_found = False
            
            for file in os.listdir(f'{PREFIX_PATH}resources/output_buckets'):
                start_date_file, start_time_file, end_date_file, end_time_file = file.strip(".csv").split('_')[1:]
                
                start_date_time_file = datetime.strptime(
                    f"{start_date_file} {start_time_file}", 
                    '%Y-%m-%d %H-%M-%S'
                )
                
                end_date_time_file = datetime.strptime(
                    f"{end_date_file} {end_time_file}", 
                    '%Y-%m-%d %H-%M-%S'
                )
                if start_date_time_file <= current_ts and end_date_time_file > current_ts:
                    df = pd.read_csv(f'{PREFIX_PATH}resources/output_buckets/{file}')
                    df['_time'] = pd.to_datetime(df['_time'], format='%Y-%m-%d %H:%M:%S',  errors='coerce')
                    df = df.dropna(subset=['_time'])
                    
                    # Filter data for current time slice
                    mask = (df['_time'] >= current_ts)
                    if current_ts + (end_date_time_file - start_date_time_file) > ts_end_time:
                        mask &= (df['_time'] <= ts_end_time)
                    
                    all_data.append(df[mask])
                    current_ts = end_date_time_file
                    bucket_found = True
                    break
            
            if not bucket_found:
                self.create_new_distribution_bucket(current_ts, ts_end_time)
                continue
        
        if all_data:
            self.real_logs_distribution = pd.concat(all_data, ignore_index=True)
            # include only security and system logs
            self.real_logs_distribution = self.real_logs_distribution[
                self.real_logs_distribution['source'].str.contains('Security|System', case=False, regex=True)
            ]
            return
        

        
    def create_new_distribution_bucket(self, start_time, end_time):
        """
        Create new distribution bucket for given time range.
        start_time and end_time are UTC timestamps
        """
        # Convert timestamps to timezone-aware datetime objects
        # start_time = datetime.fromtimestamp(start_time)
        # end_time = datetime.fromtimestamp(end_time)
        
        # Round to nearest day while preserving timezone
        start_time = datetime(
            start_time.year, start_time.month, start_time.day, 
        )
        end_time = start_time + pd.DateOffset(days=1)
        
        timestamp_start_time = start_time.timestamp()
        timestamp_end_time = end_time.timestamp()
        
        logger.info(f'start_time: {start_time}, end_time: {end_time}')
        logger.info(f'timestamp_start_time: {timestamp_start_time}, timestamp_end_time: {timestamp_end_time}')
        logger.info(f'Creating new distribution bucket for time range: {start_time} - {end_time}')
        
        # Use RFC3339 format for Splunk query
        job = self.service.jobs.create(
            f'search index=main | '
            'eval _time=strftime(_time,"%Y-%m-%d %H:%M:00") | '
            'stats count by source EventCode _time', 
            earliest_time=start_time.strftime('%Y-%m-%d %H:%M:%S'), 
            latest_time=end_time.strftime('%Y-%m-%d %H:%M:%S'),
            time_format='%Y-%m-%d %H:%M:%S',
            count=0
        ) 
        
        while True:
            job.refresh()
            if job.content['isDone'] == '1':
                break
            time.sleep(2)
            
        response = job.results(output_mode='json', count=0)
        reader = splunk_results.JSONResultsReader(response)
        results = []
        
        # Format timestamps for filename while preserving UTC
        start_time_str = start_time.strftime('%Y-%m-%d_%H-%M-%S')
        end_time_str = end_time.strftime('%Y-%m-%d_%H-%M-%S')
        file = f"bucket_{start_time_str}_{end_time_str}.csv"
        
        for result in reader:
            if isinstance(result, dict):
                results.append(result)
                
        self.real_logs_distribution = pd.DataFrame(results, index=None)
        self.real_logs_distribution.to_csv(f'{PREFIX_PATH}resources/output_buckets/{file}')
            
    def get_releveant_distribution(self, start_time, end_time):
        """
        Get relevant logs distribution for a given time range.
        
        Args:
            start_time (str): Start time in format 'MM/DD/YYYY:HH:MM:SS'
            end_time (str): End time in format 'MM/DD/YYYY:HH:MM:SS'
        """
        # # Convert string times to timezone-aware datetime objects
        # date_start_time = datetime.strptime(start_time, '%m/%d/%Y:%H:%M:%S').replace(tzinfo=timezone.utc)
        # date_end_time = datetime.strptime(end_time, '%m/%d/%Y:%H:%M:%S').replace(tzinfo=timezone.utc)
        
        # Convert pandas datetime column to timezone-aware if it isn't already
        # if self.real_logs_distribution['_time'].dt.tz is None:
        #     self.real_logs_distribution['_time'] = self.real_logs_distribution['_time'].dt.tz_localize('UTC')
        
        # Now compare the timezone-aware datetimes
        relevant_logs = self.real_logs_distribution[
            (self.real_logs_distribution['_time'] >= pd.Timestamp(start_time, unit='s') ) & 
            (self.real_logs_distribution['_time'] <= pd.Timestamp(end_time, unit='s') )
        ]
        
        # Check if we have full coverage
        if len(relevant_logs) == 0 or \
        relevant_logs['_time'].min() > start_time or \
        relevant_logs['_time'].max() < end_time:
            logger.info('Loading missing distribution data from disk.')
            self.load_real_logs_distribution_bucket(start_time, end_time)
            relevant_logs = self.real_logs_distribution[
                (self.real_logs_distribution['_time'] >= pd.Timestamp(start_time, unit='s')) & 
                (self.real_logs_distribution['_time'] <= pd.Timestamp(end_time, unit='s'))
            ]
        
        return relevant_logs       
              
    def get_real_distribution(self, start_time, end_time):
        """
        Get real log distribution for a given time range.
        
        Args:
            start_time (str): Start time in format 'MM/DD/YYYY:HH:MM:SS'
            end_time (str): End time in format 'MM/DD/YYYY:HH:MM:SS'
            
        Returns:
            dict: Dictionary of log types and their counts
        """
        # Convert string times to timezone-aware datetime objects
        start_dt = datetime.strptime(start_time, '%m/%d/%Y:%H:%M:%S')
        end_dt = datetime.strptime(end_time, '%m/%d/%Y:%H:%M:%S')
        logger.info(f"start_dt: {start_dt}, end_dt: {end_dt}")
        # Load real logs distribution
        relevant_logs = self.get_releveant_distribution(start_dt, end_dt)
        
        # Group by source and EventCode, summing the counts
        relevant_logs = relevant_logs.groupby(['source', 'EventCode']).agg({'count': 'sum'}).reset_index()
        
        # Create dictionary of log types and counts
        res_dict = {
            (row['source'].lower(), str(row['EventCode'])): row['count'] 
            for index, row in relevant_logs.iterrows()
        }
        logger.debug(f"current real distribution: {res_dict}")
        
        # # Update the real_logtypes_counter
        # for logtype in res_dict:
        #     if logtype in self.real_logtypes_counter:
        #         self.real_logtypes_counter[logtype] += res_dict[logtype]
        #     else:
        #         self.real_logtypes_counter[logtype] = res_dict[logtype]
        
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
           
            "output_mode": "json"
        }
        response = requests.post(url, headers=HEADERS, data=data, auth=self.auth, verify=False)
        results = response.text
        logger.info(time_range)
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
        logs_to_duplicate_dict = {f"{logtype[0].lower()}_{logtype[1]}_{istrigger}": [] for istrigger,_ in enumerate(['notrigger', 'trigger']) for logtype in logtypes}
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
                             logs_to_duplicate_dict[f"{logtype[0].lower()}_{logtype[1]}_{istrigger}"].append(log)
        return logs_to_duplicate_dict   
     
                
    def get_time(self, y, m, d, h, mi, s):
        return datetime(y, m, d, h, mi, s).timestamp()
    
    def get_logs_amount(self, time_range):
        start_dt = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S')
        end_dt = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S')
        relevant_logs = self.get_releveant_distribution(start_dt, end_dt)
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