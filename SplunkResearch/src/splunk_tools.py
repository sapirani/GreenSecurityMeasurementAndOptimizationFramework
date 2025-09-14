import asyncio
from datetime import datetime
from enum import Enum
import json
import logging
from multiprocessing import Pool
import re
import time
import splunklib.client as client
import splunklib.results as splunk_results
import splunklib.binding
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
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from utils.general_consts import LoggerName
from application_logging.handlers.elastic_handler import get_elastic_logging_handler

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

from application_logging.logging_utils import get_measurement_logger
ES_URL = "http://127.0.0.1:9200"
ES_USER = "elastic"
ES_PASS = "SwmQNU7y"
PULL_INTERVAL_SECONDS = 2  # seconds
es_logger = get_measurement_logger(
    logger_name=LoggerName.METRICS_AGGREGATIONS,
    logger_handler=get_elastic_logging_handler(ES_USER, ES_PASS, ES_URL, "sid"),
)

@dataclass
class ProcessMetrics:
    cpu_time: float = 0.0
    read_count: int = 0
    write_count: int = 0
    read_bytes: int = 0
    write_bytes: int = 0

@dataclass
class QueryMetrics:
    search_name: str
    results_count: int
    execution_time: float
    cpu: float
    io_metrics: Dict[str, int]
    start_time: float = 0.0
    end_time: float = 0.0



class Mode(Enum):
    PROFILE = 'profile'
    ATTACK = 'attack'

class SplunkTools(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SplunkTools, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, active_saved_searches=None, rule_frequency=1, mode=Mode.ATTACK):
        if self._initialized:
            return
        self.mode = mode
        self.pids = []
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
        self.total_cpu_time = 0
        self.stop_cpu_monitor = threading.Event()
        while True:
            try:
                self.active_saved_searches = self.get_saved_search_names(active_saved_searches)
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
                time.sleep(120)#TODO: change to 5
        self._initialized = True


    def check_license_usage(self):
        """
        Get the remaining license quota for the Splunk instance.
        """
        query = """search index=_internal source=*license_usage.log type=Usage 
                | eval used_mb = b/1024/1024 
                | stats sum(used_mb) as used_mb_today by pool 
                | join type=left pool 
                    [| rest /services/licenser/pools 
                    | fields title quota_used quota 
                    | rename title as pool 
                    | eval quota_mb = quota/1024/1024 
                    | eval quota_used_mb = quota_used/1024/1024] 
                | eval remaining_mb = quota_mb - used_mb_today 
                | table pool quota_mb used_mb_today remaining_mb"""
        self.service.jobs.create(query, earliest_time='-1d', latest_time='now')
        job = self.service.jobs.create(query, earliest_time='-1d', latest_time='now')
        while True:
            job.refresh()
            if job.content['isDone'] == '1':
                break
            time.sleep(2)
        response = job.results(output_mode='json')
        reader = splunk_results.JSONResultsReader(response)
        results = []
        for result in reader:
            if isinstance(result, dict):
                results.append(result)
        if len(results) > 0:
            results = results[0]
            return {
                'pool': results['pool'],
                'quota_mb': float(results['quota_mb']),
                'used_mb_today': float(results['used_mb_today']),
                'remaining_mb': float(results['remaining_mb'])
            }
    
    
    def get_num_of_searches(self):
        return len(self.active_saved_searches)
    
    def start_cpu_monitoring(self):
        """Start CPU monitoring in a separate thread"""
        self.stop_cpu_monitor.clear()
        self._cpu_monitor_thread = threading.Thread(target=self.monitor_total_cpu)
        self._cpu_monitor_thread.start()

    def stop_cpu_monitoring(self):
        """Stop CPU monitoring and wait for thread to finish"""
        if self._cpu_monitor_thread is not None:
            self.stop_cpu_monitor.set()
            self._cpu_monitor_thread.join()
            self._cpu_monitor_thread = None

      

    async def execute_query(self, search_name: str, query: str, earliest_time: float, latest_time: float) -> QueryMetrics:
        query = f" search {query}"
        # loop = asyncio.get_event_loop()

        job = self.service.jobs.create(query, earliest_time=earliest_time, latest_time=latest_time)
        io_counters_dict = { "read_count": 0,  "write_count": 0, "read_bytes": 0, "write_bytes": 0}
        
        # Wait for job to get PID
        while True:
            job.refresh()  # Make job.refresh()() non-blocking
            stats = job.content
            pid = stats.get('pid', None)
            if pid is not None:
                break
            if stats['isDone'] == '1':
                job = self.service.jobs.create(query, earliest_time=earliest_time, latest_time=latest_time)
            await asyncio.sleep(0.01)  # Use asyncio.sleep instead of time.sleep

        process_end_cpu_time = 0
        process_start_cpu_time = 0
        logger.info(f"Monitoring process {pid}")
        if pid is not None:
            try:
                self.pids.append(pid)
                sid = job.content.get('sid', None)
                if self.mode == Mode.PROFILE:
                    es_logger.info(f"PID SID MAPPING", extra={
                        'pid': pid,
                        'sid': sid,
                        'search_name': search_name
                    }
                    )
                process = psutil.Process(int(pid))
                process_start_time = time.time()
                # Get initial CPU times in a non-blocking way
                process_start_cpu_time =  process.cpu_times().user
                # process_start_cpu_time = await loop.run_in_executor(
                #     None, lambda: process.cpu_times().user  # + process.cpu_times().system
                # )
                
                while True:
                    try:
                        # Get CPU times non-blockingly
                        process_end_cpu_time =  process.cpu_times().user
                        # Check if process is running in a non-blocking way
                        is_running = process.is_running
                        if not is_running:
                            logger.debug(f"Process {pid} has finished running")
                            break
                            
                        # Refresh job status non-blockingly
                        job.refresh()
                        if job.content['isDone'] == '1':
                            logger.debug(f"Job is marked as done")
                            break

                        # Get IO counters non-blockingly
                        try:
                            # Run the oneshot operation in a thread
                            def get_io_counters():
                                with process.oneshot():
                                    return process.io_counters()
                                    
                            io_counters = get_io_counters()
                            
                            # Update IO metrics
                            for key in io_counters_dict:
                                io_counters_dict[key] += getattr(io_counters, key)
                                
                        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                            logger.debug(f"Error getting IO counters: {e}")
                        
                        await asyncio.sleep(0.1)  # Non-blocking sleep
                        
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        # If the process disappeared but the job isn't done, it might have spawned a new process
                        job.refresh()
                        if job.content['isDone'] == '0':
                            logger.error(e)
                            logger.error(f"Process {pid} disappeared but job isn't done. Waiting for job completion.")
                            await asyncio.sleep(0.1)
                            continue
                        else:
                            logger.debug(f"Process {pid} no longer exists but job is done")
                            break
                        
                    # Add timeout protection
                    if time.time() - process_start_time > 600:  # 5 minute timeout
                        logger.warning(f"Process monitoring timed out after 10 minutes for pid {pid}")
                        job.cancel()
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
        
        # Get final CPU time
        process_cpu_time = process_end_cpu_time - process_start_cpu_time
        
        # Get final job status
        job.refresh()
        stats = job.content
        run_duration = float(stats.get('runDuration', 0))
        
        # Get results 
        response = job.results(output_mode='json')
        
        # Parse results non-blockingly

        results =  [result for result in splunk_results.JSONResultsReader(response) if isinstance(result, dict)]
        
        # Get results count
        results_count = len(results)
        
        metric = QueryMetrics(
            search_name=search_name,
            results_count=results_count,
            execution_time=run_duration,
            cpu=process_cpu_time,
            io_metrics=io_counters_dict
        )
        
        return metric, results
        

    
    async def run_saved_search(self, saved_search: str, time_range: Tuple[str, str]) -> QueryMetrics:
        """Run a saved search and collect metrics"""
        search_name = saved_search
        query = self.active_saved_searches[saved_search]['search']
        
        # Parse time range
        earliest_time = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S').timestamp()
        latest_time = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S').timestamp()
        
        logger.info(f'Running saved search {search_name}')
        return await self.execute_query(search_name, query, earliest_time, latest_time)

    async def run_saved_searches(
        self, 
        time_range: Tuple[str, str], 
        running_plan: Dict[str, int]=None,
        num_measurements: int = 1        
    ) -> Tuple[List[QueryMetrics], float]:
        """
        FIXED: Already properly parallelized with asyncio.gather
        Just needed the execute_query to be truly async
        """
        # Start CPU monitoring
        # self.start_cpu_monitoring()
        
        all_tasks = []
        if running_plan is None:
            running_plan = {search: num_measurements for search in self.active_saved_searches}
        
        for search_name, num_measurements in running_plan.items():
            for i in range(num_measurements):
                # Create a task for each measurement
                task = asyncio.create_task(self.run_saved_search(
                    search_name, time_range))
                all_tasks.append(task)
        # run a thread that check if all pids founded and then enrich elasic index
        if self.mode == Mode.PROFILE:
            pids_thread = threading.Thread(
                target=self.enrich_elastic_index_with_pids,
            )
            pids_thread.start()
        # Run all tasks concurrently
        results = await asyncio.gather(*all_tasks)
        if self.mode == Mode.PROFILE:        
            # Wait for the pids thread to finish
            pids_thread.join()
            self.pids = []
        
        # Filter out any failed measurements
        valid_results = [m for m, r in results if isinstance(m, QueryMetrics)]
        for result in valid_results:
            result.start_time = time_range[0]
            result.end_time = time_range[1]
        
        if len(valid_results) < len(results):
            logger.warning(f"Some measurements failed: {len(results) - len(valid_results)} failures")
        # log each res in line
        for result in valid_results:
            logger.info(f"Search: {result.search_name}, "
                        f"Results: {result.results_count}, "
                        f"Execution Time: {result.execution_time:.2f}s, "
                        f"CPU: {result.cpu:.2f}s, ")
                    #   f"IO Metrics: {result.io_metrics}")
        return valid_results, self.total_cpu_time
            
        # finally:
        #     # Stop CPU monitoring
        #     self.stop_cpu_monitoring()
        
    def enrich_elastic_index_with_pids(self):
        while len(self.pids)!= len(self.active_saved_searches):
            time.sleep(0.1)
        logger.info(f"Enriching Elastic index with PIDs: {self.pids}")
        # Enrich the Elastic index with the PIDs
        # # Execute the enrich policy using 'target' and provide authentication explicitly
        response = es_logger.handlers[0].es.transport.perform_request(
            method="POST",
            target="/_enrich/policy/search_name/_execute",  # 'target' is the correct parameter here
            headers={
                "Authorization": "Basic ZWxhc3RpYzpTd21RTlU3eQ=="  # base64 encoded "elastic:SVR4mUZl"
            }
        )
        
    def monitor_total_cpu(self):
        """Monitor total CPU usage across all Splunk processes"""
        start_time = time.time()
        start_cpu = psutil.cpu_times().user

        while not self.stop_cpu_monitor.is_set():
            time.sleep(0.1)

        self.total_cpu_time = psutil.cpu_times().user - start_cpu
        
    
    def mock_run_saved_searches(self, time_range: Tuple[str, str]) -> List[QueryMetrics]:
        """
        Mock function to simulate running multiple saved searches and collecting metrics.
        This is a placeholder and should be replaced with actual implementation.
        """
        metrics = []
        for saved_search in self.active_saved_searches:
            metric = self.mock_run_saved_search(saved_search, time_range)
            metrics.append(metric)
        return metrics
    
    def mock_run_saved_search(self, saved_search: str, time_range: Tuple[str, str]) -> QueryMetrics:
        """
        Mock function to simulate running a saved search and collecting metrics.
        This is a placeholder and should be replaced with actual implementation.
        """
        search_name = saved_search
        query = self.active_saved_searches[saved_search]['search']
        
        # Simulate CPU usage and IO metrics
        cpu_time = 0
        io_metrics = {
            "read_count": 0,
            "write_count": 0,
            "read_bytes": 0,
            "write_bytes": 0
        }
        
        return QueryMetrics(
            search_name=search_name,
            results_count=0,
            execution_time=0,
            cpu=cpu_time,
            io_metrics=io_metrics,
            start_time=time_range[0],
            end_time=time_range[1]
        )

        
        
        
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
            results = {result['title']: result for result in results if result['title'] in active_saved_searches}
        return results
    
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
            pool.starmap(update_func, [(search_name, update_arg) for search_name in searches_names])
            
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
                    for col in df.select_dtypes(include=['float64']).columns:
                        df[col] = df[col].astype('float32')
                    for col in df.select_dtypes(include=['int64']).columns:
                        df[col] = df[col].astype('int32')
                    # Filter data for current time slice
                    # mask = (df['_time'] >= current_ts)
                    # if current_ts + (end_date_time_file - start_date_time_file) > ts_end_time:
                        # mask &= (df['_time'] <= ts_end_time)
                    
                    all_data.append(df)
                    # all_data.append(df[mask])
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
            self.real_logs_distribution = self.real_logs_distribution.set_index('_time').sort_index()

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
        # Then your lookup becomes:
        start_ts = pd.Timestamp(start_time, unit='s')
        end_ts = pd.Timestamp(end_time, unit='s')
        relevant_logs = self.real_logs_distribution.loc[start_ts:end_ts]
        
        # Check if we have full coverage
        # if len(relevant_logs) == 0 or \
        # relevant_logs.index.min() > start_time or \
        # relevant_logs.index.max() < end_time:
            # logger.info('Loading missing distribution data from disk.')
            # self.load_real_logs_distribution_bucket(start_time, end_time)
            # new_logs = self.real_logs_distribution.loc[
            #     pd.Timestamp(start_time, unit='s'):pd.Timestamp(end_time, unit='s')
            # ]

            # # Concatenate with existing
        
            # relevant_logs = pd.concat([relevant_logs, new_logs])
            # logger.info('No relevant logs found for the given time range.')
            # logger.info(f'Start time: {start_time}, End time: {end_time}')
            # logger.info(f'Relevant logs shape: {relevant_logs.shape}')
            # logger.info(f'Relevant logs index range: {relevant_logs.index.min()} - {relevant_logs.index.max()}')
        relevant_logs.loc[:, 'count'] = relevant_logs['count'].astype(int)
        
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
        # Use RFC3339 format for Splunk query
        if time_range is None:
            time_range = ("0", datetime.now())
        if type(time_range) is str:
            start_time = f"-{time_range}"

            job = self.service.jobs.create(
                f'search index=main host="dt-splunk" earliest={start_time}| delete', 
                earliest_time=start_time, 
                time_format='%Y-%m-%d %H:%M:%S',
                count=0
            )
        else:
            start_time = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            job = self.service.jobs.create(
                f'search index=main host="dt-splunk" | delete', 
                earliest_time=start_time, 
                latest_time=end_time,
                time_format='%Y-%m-%d %H:%M:%S',
                count=0
            )
        while True:
            job.refresh()
            if job.content['isDone'] == '1':
                break
            time.sleep(2)
        
        # Get the results properly
        results = job.results(output_mode='json')
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

                
    def get_time(self, y, m, d, h, mi, s):
        return datetime(y, m, d, h, mi, s).timestamp()
    
    def get_logs_amount(self, time_range):
        start_dt = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S')
        end_dt = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S')
        relevant_logs = self.get_releveant_distribution(start_dt, end_dt)
        return relevant_logs['count'].sum()
  
  
  
  
  
  
  
  
  
  
  