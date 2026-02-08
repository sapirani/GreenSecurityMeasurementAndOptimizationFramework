import asyncio
from asyncio import events
from datetime import datetime
from enum import Enum
import json
import logging
from multiprocessing import Pool
from pyexpat import model
import random
import re
import time
from xxlimited import Str
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
from config import config
from datetime import datetime
from datetime import timezone
from random import randint
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional
from utils.general_consts import LoggerName
from application_logging.handlers.elastic_handler import get_elastic_logging_handler
import traceback

# Import configuration
sys_path = os.path.dirname(os.path.abspath(__file__))
if sys_path not in os.sys.path:
    os.sys.path.insert(0, sys_path)
from config import config

CLK_TCK = os.sysconf(os.sysconf_names['SC_CLK_TCK'])

# Load .env from config base directory
base_dir = config.get('paths.base_dir', '/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch')
env_path = os.path.join(base_dir, 'src/.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

# Precompile the regex pattern
pattern = re.compile(r"'(.*?)' (\D+) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} IDT) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \s+(\d+)\s+(\d+\.\d+)") # IDT and IST are changed when the time is changed
savedsearches_path = '/opt/splunk/etc/users/shouei/search/local/savedsearches.conf'
APP = 'search'
HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded"
}
PREFIX_PATH = base_dir + '/'
import logging
logger = logging.getLogger(__name__)

from application_logging.logging_utils import get_measurement_logger

# Load Elasticsearch configuration from config (SECURITY: No hardcoded credentials)
ES_PORT = config.get('elasticsearch.port', 9200)
ES_URL = f"http://127.0.0.1:{ES_PORT}"
ES_USER = config.get('elasticsearch.username', 'elastic')
ES_PASS = config.get('elasticsearch.password')  # MUST be in secrets.yaml
if ES_PASS is None:
    logger.warning("Elasticsearch password not configured in secrets.yaml")
    ES_PASS = ""  # Fallback to empty string to avoid crashes

PULL_INTERVAL_SECONDS = config.get('splunk.pull_interval_seconds', 2)

es_logger = get_measurement_logger(
    logger_name=LoggerName.PROCESS_METRICS,
    logger_handler=get_elastic_logging_handler(ES_USER, ES_PASS, ES_URL, "sid"),
)

BASIC_QUERIES = {"Windows Event For Service Disabled":"`wineventlog_system` EventCode=7040",
"Detect New Local Admin Account":"`wineventlog_security` EventCode=4732",
"ESCU Network Share Discovery Via Dir Command Rule":"`wineventlog_security` EventCode=5140",
"Known Services Killed by Ransomware":"`wineventlog_system` EventCode=7036",
"Non Chrome Process Accessing Chrome Default Dir":"`wineventlog_security` EventCode=4663",
"Kerberoasting SPN Request With RC4 Encryption":"`wineventlog_security` EventCode=4769 ",
"Clop Ransomware Known Service Name":"`wineventlog_system` EventCode=7045",
'Windows AD Replication Request Initiated from Unsanctioned Location':"`wineventlog_security` EventCode=4662",
'ESCU Windows Rapid Authentication On Multiple Hosts Rule':"`wineventlog_security` EventCode=4624",}
@dataclass
class QueryMetrics:
    timestamp: str
    pid: str
    events_count: int
    search_name: str
    results_count: int
    execution_time: float
    cpu: float
    io_metrics: Dict[str, int]
    memory_mb: float
    start_time: float = 0.0
    end_time: float = 0.0

# ----------------------
# Linux process metrics
# ----------------------
def read_proc_metrics(pid: int):
    try:
        metrics = {}
        # CPU, memory, threads, page faults
        with open(f"/proc/{pid}/stat") as f:
            data = f.read().split()
            utime = int(data[13])
            stime = int(data[14])
            tcount = int(data[19])
            rss = int(data[23]) * os.sysconf('SC_PAGE_SIZE') / 1024**2
            majflt = int(data[11])
            minflt = int(data[10])
            metrics.update({
                "cpu_seconds": (utime + stime) / CLK_TCK,
                "threads": tcount,
                "mem_used_mb": rss,
                "page_faults_minor": minflt,
                "page_faults_major": majflt,
            })
        # IO stats
        with open(f"/proc/{pid}/io") as f:
            io_data = f.read().splitlines()
            io_metrics = {}
            for line in io_data:
                k, v = line.split(":")
                io_metrics[k.strip()] = int(v.strip())
            metrics.update({
                "read_bytes": io_metrics.get("read_bytes", 0),
                "write_bytes": io_metrics.get("write_bytes", 0),
                "read_count": io_metrics.get("syscr", 0),
                "write_count": io_metrics.get("syscw", 0),
            })
        # FD count
        metrics["fd_used"] = len(os.listdir(f"/proc/{pid}/fd"))
        return metrics
    except FileNotFoundError:
        return None

async def monitor_process(pid: int, interval=0.2):
    """Async generator yielding snapshots of all metrics until process ends."""
    while True:
        metrics = read_proc_metrics(pid)
        if metrics is None:
            break
        yield metrics, datetime.now()
        await asyncio.sleep(interval)

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

    def __init__(self, active_saved_searches=None, rule_frequency=1, mode=Mode.ATTACK, ip=1):
        if self._initialized:
            return
        self.mode = mode
        self.pids = []
        # SPLUNK_HOST_{ip} is dynamic per IP, so it remains in .env
        self.splunk_host = os.getenv(f"SPLUNK_HOST_{ip}")
        # Static credentials come from config/secrets.yaml
        self.splunk_port = config.get('splunk.port', '8089')
        self.base_url = f"https://{self.splunk_host}:{self.splunk_port}"
        self.splunk_username = config.get('splunk.username')
        self.splunk_password = config.get('splunk.password')
        self.index_name = config.get('splunk.index_name', 'main')
        self.hec_token1 = config.get('splunk.hec_token1')
        self.hec_token2 = config.get('splunk.hec_token2')
        self.auth = requests.auth.HTTPBasicAuth(self.splunk_username, self.splunk_password)
        self.real_logs_distribution = pd.DataFrame(data=None, columns=['source', 'EventCode', '_time', "host", 'count'])
        self.real_logs_distribution['_time'] = pd.to_datetime(self.real_logs_distribution['_time'])

        self.max_workers = 8
        self.total_interval_queue = queue.Queue()
        self.total_cpu_queue = queue.Queue()
        self.total_cpu_time = 0
        self.stop_cpu_monitor = threading.Event()
        while True:
            try:
                self.active_saved_searches = self.get_saved_search_names(active_saved_searches, owner=self.splunk_username)
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
                sleep_time = config.get('splunk.operation_sleep_time', 120)
                time.sleep(sleep_time)
        self._initialized = True
        self.log_file_prefix = f'{base_dir}/monitor_files_{self.splunk_host}'
        self.subset = {}
        self.subset_real_logs_distribution = pd.DataFrame()
        self.full_hosts_list = {}
        # csv for profile mode results
        if self.mode == Mode.PROFILE:
            self.profile_results_file_path = f'profiling_results_{self.splunk_host}.csv'
            self.profile_results_df = pd.DataFrame(columns=[
                'timestamp', 'pid', 'events_count', 'search_name', 'results_count',
                'execution_time', 'cpu', 'read_bytes', 'write_bytes',
                'read_count', 'write_count',  'memory_mb', 'start_time', 'end_time'
            ])
        # if self.mode == Mode.PROFILE:
        #         response = es_logger.handlers[0].es.delete_by_query(
        #             index="sid",
        #             body={
        #                 "query": {"match_all": {}}
        #             },
        #             refresh=True  # ensure immediate refresh
        #         )
        #         logger.info(f"Deleted {response['deleted']} documents from index sid")

    async def run_saved_search(self, search_name: str, start_time: Optional[float] = None, end_time: Optional[float] = None, interval: float = 0.2) -> QueryMetrics:
        """
        Replaces the old logic. Calls the remote endpoint to do the heavy lifting.
        start_time/end_time should be Epoch strings or relative time strings (e.g. "-15m").
        """
        
        # 1. Prepare Payload
        payload = {
            "search_name": search_name,
            "index_name": self.index_name,
            "earliest_time": start_time,
            "latest_time": end_time,
            "app": "search",
            "owner": self.splunk_username,
        }

        # 2. Call the Remote API (The "Black Box")
        # We use standard requests, but you can wrap this in aiohttp if you want async IO
        try:
            # Using verify=False because self-signed certs are common in dev
            response = requests.post(
                self.base_url + "/services/profiler_api",
                auth=self.auth, 
                json=payload, 
                verify=False,
                params={'output_mode': 'json'}
            )
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Profiling data for {search_name}: {data}")
        except Exception as e:
            # Handle error gracefully
            logger.error(f"Error profiling {search_name}: {e}")
            try:
                error_data = response.json()
                logger.error(f"Error Message: {error_data.get('error')}")
                logger.error("-" * 40)
                logger.error("REMOTE TRACEBACK:")
                logger.error(error_data.get('traceback'))
                logger.error("-" * 40)
            except ValueError:
                # If server didn't return JSON (e.g., plain HTML error page)
                logger.error(f"Raw Response Text: {response.text}")
            return None
        events_count = 0
        if self.mode == Mode.PROFILE:
            # Check events count via BASIC_QUERIES
            eventcode_search = BASIC_QUERIES.get(search_name, None)
            if eventcode_search:
                secondary_host = config.get('hosts.secondary', '132.72.81.150')
                default_host = config.get('splunk.default_host', 'dt-splunk')
                eventcode_query = f'search index={self.index_name} {eventcode_search} host IN ({secondary_host}, "{default_host}", {self.splunk_host})  | stats count'
                eventcode_job = self.service.jobs.create(eventcode_query, earliest_time=start_time, latest_time=end_time)
                while True:
                    eventcode_job.refresh()
                    if eventcode_job.content['isDone'] == '1':
                        break
                    await asyncio.sleep(0.1)
                events_count = int(eventcode_job["eventCount"]) if "eventCount" in eventcode_job else 0
        
        # 3. Map Response to QueryMetrics (The "Original Flow")
        qmetric = QueryMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            pid=int(data.get("pid", 0)),
            search_name=search_name,
            results_count=data.get("results_count", 0),
            execution_time=data.get("execution_time", 0.0),
            
            # Now this is the REAL /proc value from the server
            cpu=data.get("cpu_seconds", 0.0), 
            events_count= events_count,
            memory_mb=data.get("memory_mb", 0.0),
            io_metrics={
                "read_bytes": data.get("read_bytes", 0),    
                "write_bytes": data.get("write_bytes", 0),
                "read_count": data.get("read_count", 0),
                "write_count": data.get("write_count", 0),
                }
        )
        if self.mode == Mode.PROFILE:
            # Append to profile results DataFrame
            new_row = asdict(qmetric)
            # append new_row to csv
            new_row_df = pd.DataFrame([new_row])
            new_row_df.to_csv(self.profile_results_file_path, mode='a', header=not os.path.exists(self.profile_results_file_path), index=False)
        return qmetric

    def clear_os_cache(self):
        try:
            # Run the command exactly as requested
            subprocess.run(
                ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], 
                check=True
            )
            print("[+] OS Cache Cleared Successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Failed to clear cache: {e}")
        
    async def run_saved_searches(self,time_range = None, max_parallel: int = 10) -> List[QueryMetrics]:
        # self.clear_os_cache()
        sem = asyncio.Semaphore(max_parallel)
        earliest_time = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S').timestamp()
        latest_time = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S').timestamp()
        async def wrapped(name):
            async with sem:
                return await self.run_saved_search(name, start_time=earliest_time, end_time=latest_time)

        tasks = [asyncio.create_task(wrapped(name)) for name in self.active_saved_searches]
        results = await asyncio.gather(*tasks)
        # Update results with time_range
        for result in results:
            result.start_time = time_range[0]
            result.end_time = time_range[1]
            logger.info(f'Finished running saved search: {result.search_name}')
        return results
    
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


            df  = pd.read_csv("/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch/resources/all_dist_by_host.csv")
 
            df['_time'] = pd.to_datetime(df['_time'], format='%Y-%m-%d %H:%M:%S',  errors='coerce')
            df = df.dropna(subset=['_time'])
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype('float32')
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = df[col].astype('int32')
            self.real_logs_distribution  = df
            self.real_logs_distribution['source'].str.contains('Security|System', case=False, regex=True)
            
            self.real_logs_distribution = self.real_logs_distribution.set_index('_time').sort_index()  
            self.full_hosts_list = self.real_logs_distribution['host'].unique().tolist()
    # def load_real_logs_distribution_bucket(self, start_time, end_time):
    #         """
    #         Load and merge logs from multiple buckets within specified time range.
    #         start_time and end_time should be timezone-aware datetime objects
    #         """
    #         ts_start_time = start_time
    #         ts_end_time = end_time
    #         logger.info(f'Loading real logs distribution for time range: {ts_start_time} - {ts_end_time}')
    #         all_data = []
    #         current_ts = ts_start_time

    #         logger.info('Loading sampled real logs distribution from splunk_results.csv')
    #         df  = pd.read_csv("/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch/src/splunk_results_random.csv")
    #         # df  = pd.read_csv("/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/splunk_results.csv")
    #         # df['_time'] = pd.to_datetime(df['_time'], format="%Y-%m-%dT%H:%M:%S.000%z",  errors='coerce')
    #         # # convert _time format from 2023-09-01T00:00:00 to 2023-09-01 00:00:00
    #         # df['_time'] = df['_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    #         df['_time'] = pd.to_datetime(df['_time'], format='%Y-%m-%d %H:%M:%S',  errors='coerce')
    #         df = df.dropna(subset=['_time'])
    #         for col in df.select_dtypes(include=['float64']).columns:
    #             df[col] = df[col].astype('float32')
    #         for col in df.select_dtypes(include=['int64']).columns:
    #             df[col] = df[col].astype('int32')
    #         self.sampled_real_logs_distribution  = df
    #         self.sampled_real_logs_distribution['source'].str.contains('Security|System', case=False, regex=True)
            
    #         self.sampled_real_logs_distribution = self.sampled_real_logs_distribution.set_index('_time').sort_index()

    #         logger.info('Loading real logs distribution from output_buckets folder')
    #         while current_ts < ts_end_time:
    #             bucket_found = False
    
    #             for file in os.listdir(f'{PREFIX_PATH}resources/output_buckets'):
    #                     start_date_file, start_time_file, end_date_file, end_time_file = file.strip(".csv").split('_')[1:]
                        
    #                     start_date_time_file = datetime.strptime(
    #                         f"{start_date_file} {start_time_file}", 
    #                         '%Y-%m-%d %H-%M-%S'
    #                     )
                        
    #                     end_date_time_file = datetime.strptime(
    #                         f"{end_date_file} {end_time_file}", 
    #                         '%Y-%m-%d %H-%M-%S'
    #                     )
    #                     if start_date_time_file <= current_ts and end_date_time_file > current_ts:
    #                         df = pd.read_csv(f'{PREFIX_PATH}resources/output_buckets/{file}')
    #                         df['_time'] = pd.to_datetime(df['_time'], format='%Y-%m-%d %H:%M:%S',  errors='coerce')
    #                         df = df.dropna(subset=['_time'])
    #                         for col in df.select_dtypes(include=['float64']).columns:
    #                             df[col] = df[col].astype('float32')
    #                         for col in df.select_dtypes(include=['int64']).columns:
    #                             df[col] = df[col].astype('int32')
    #                         # Filter data for current time slice
    #                         # mask = (df['_time'] >= current_ts)
    #                         # if current_ts + (end_date_time_file - start_date_time_file) > ts_end_time:
    #                             # mask &= (df['_time'] <= ts_end_time)
                            
    #                         all_data.append(df)
    #                         # all_data.append(df[mask])
    #                         current_ts = end_date_time_file
    #                         bucket_found = True
    #                         break
                    
    #             if not bucket_found:
    #                 self.create_new_distribution_bucket(current_ts, ts_end_time)
    #                 continue
                    
            
    #         if all_data:
    #             self.real_logs_distribution = pd.concat(all_data, ignore_index=True)
    #             # include only security and system logs
    #             self.real_logs_distribution = self.real_logs_distribution[
    #                 self.real_logs_distribution['source'].str.contains('Security|System', case=False, regex=True)
    #             ]
    #             self.real_logs_distribution = self.real_logs_distribution.set_index('_time').sort_index()

    #             return
        
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
            timestamp=datetime.now(timezone.utc).isoformat(),
            pid="0",
            events_count=0,
            search_name=search_name,
            results_count=0,
            execution_time=0,
            cpu=cpu_time,
            io_metrics=io_metrics,
            start_time=time_range[0],
            end_time=time_range[1],
            memory_mb=0
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
        if not os.path.exists(self.log_file_prefix):
            os.makedirs(self.log_file_prefix)
        with open(f'{self.log_file_prefix}/{log_source}.txt', 'a') as f:
            for log in logs:
                f.write(f'{log}\n\n')        
        
    def get_saved_search_names(self, active_saved_searches, get_only_enabled=True, app="search", owner="shouei"):
        query = f"| rest /servicesNS/{owner}/search/saved/searches splunk_server=local| search eai:acl.app={app} eai:acl.owner={owner}  | table title, search, cron_schedule, disabled"
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
            
        

            
    def create_new_distribution_bucket(self, start_time, end_time):
            """
            Create new distribution bucket for given time range.
            Writes to CSV incrementally to handle millions of records.
            """
            # Round to nearest day while preserving timezone
            start_time = datetime(
                start_time.year, start_time.month, start_time.day, 
            )
            end_time = start_time + pd.DateOffset(days=10)
            
            # Logging setup
            logger.info(f'Creating new distribution bucket for: {start_time} - {end_time}')
            
            # 1. Create the Search Job
            job = self.service.jobs.create(
                f'search index={self.index_name} | '
                'eval _time=strftime(_time,"%Y-%m-%d %H:%M:00") | '
                'stats count by source EventCode _time', 
                earliest_time=start_time.strftime('%Y-%m-%d %H:%M:%S'), 
                latest_time=end_time.strftime('%Y-%m-%d %H:%M:%S'),
                time_format='%Y-%m-%d %H:%M:%S',
            ) 
            
            # 2. Wait for Job Completion
            while True:
                job.refresh()
                if job.content['isDone'] == '1':
                    break
                time.sleep(2)
                
            # 3. Setup File Path
            start_time_str = start_time.strftime('%Y-%m-%d_%H-%M-%S')
            end_time_str = end_time.strftime('%Y-%m-%d_%H-%M-%S')
            file_name = f"bucket_{start_time_str}_{end_time_str}.csv"
            full_path = f'{PREFIX_PATH}resources/output_buckets/{file_name}'
            
            # 4. Pagination & Incremental Write
            offset = 0
            page_size = 50000
            header_written = False
            total_records = 0
            
            logger.info(f"Starting fetch for job {job.sid}...")

            while True:
                # Fetch a page (chunk) of results
                response = job.results(
                    output_mode='json', 
                    count=page_size, 
                    offset=offset
                )
                
                reader = splunk_results.JSONResultsReader(response)
                batch_data = []
                
                # Extract valid rows from this chunk
                for result in reader:
                    if isinstance(result, dict):
                        batch_data.append(result)
                
                # If batch is empty, we are done
                if not batch_data:
                    break
                    
                # Convert batch to DataFrame
                df_batch = pd.DataFrame(batch_data)
                
                # Append to CSV
                # mode='a' appends to the file
                # header=not header_written ensures header is only written in the first pass
                df_batch.to_csv(full_path, mode='a', header=not header_written, index=False)
                
                # Update state
                records_in_batch = len(batch_data)
                offset += records_in_batch
                total_records += records_in_batch
                header_written = True
                
                logger.info(f"Written {records_in_batch} records. Total so far: {total_records}")

                # Safety break: if we got fewer records than requested, we are at the end
                if records_in_batch < page_size:
                    break

            logger.info(f"Completed. Total records exported: {total_records}")
            
    def get_releveant_distribution(self, start_time, end_time, hosts_percentage=100):
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
        if hosts_percentage >= 100:
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
        else:
            # check if subset is empty or time range is outside current subset
            if not self.subset:
                # get random hosts
                random_hosts = random.sample(self.full_hosts_list, 
                                            int(len(self.full_hosts_list) * hosts_percentage / 100))
                self.subset = set(random_hosts)
                self.subset_real_logs_distribution = self.real_logs_distribution[
                    self.real_logs_distribution['host'].isin(self.subset)
                ]
                
            relevant_logs = self.subset_real_logs_distribution.loc[start_ts:end_ts]
            relevant_logs.loc[:, 'count'] = relevant_logs['count'].astype(int)
        
        return relevant_logs       
              
    def get_real_distribution(self, start_time, end_time, hosts_percentage=100):
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
        logger.debug(f"start_dt: {start_dt}, end_dt: {end_dt}")
        # Load real logs distribution
        relevant_logs = self.get_releveant_distribution(start_dt, end_dt, hosts_percentage)
        
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

    
    def run_search(self, query, earliest_time, latest_time):
        print(f'Running search: {query} from {earliest_time} to {latest_time}')
        job = self.service.jobs.create(
            f" search {query}", 
            earliest_time=earliest_time, 
            latest_time=latest_time
        )
        
        while True:
            job.refresh()
            if job.content['isDone'] == '1':
                break
            time.sleep(2)
        
        response = job.results(output_mode='json', count=0)
        reader = splunk_results.JSONResultsReader(response)
        results = []
        
        for result in reader:
            if isinstance(result, dict):
                results.append(result)
        
        return results
    
    
    def delete_fake_logs(self, time_range=None, condition=None, logs_qnt=None, max_attempts=5):
        """
        Deletes logs from Splunk and retries if not all logs are deleted.
        Stops after `max_attempts` attempts to avoid infinite recursion.
        """

        def run_delete_query(time_range, condition):
            """Runs the delete query once and returns number of deleted logs."""

            # 1. Build Query
            # Note: Ensure IP is quoted in SPL if it causes issues, though usually fine.
            secondary_host = config.get('hosts.secondary', '132.72.81.150')
            base_query = f'search index={self.index_name} host={secondary_host}'
            if condition:
                query = f'{base_query} {condition} | delete'
            else:
                query = f'{base_query} | delete'

            # 2. Handle Time (Convert to Epoch)
            kwargs = {
                "count": 0,
                "exec_mode": "blocking" # Recommended for delete operations to ensure completion
            }

            if time_range is None:
                # Default to all time or specific window if needed
                kwargs["earliest_time"] = "0" 
            elif isinstance(time_range, str):
                # Relative time (e.g., "15m") is safe to pass as string
                kwargs["earliest_time"] = f"-{time_range}"
                kwargs["latest_time"] = "now"
            else:
                # Absolute time: Convert to Epoch (Float)
                t_start = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S')
                t_end = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S')
                
                # .timestamp() converts to epoch based on local machine time
                kwargs["earliest_time"] = t_start.timestamp()
                kwargs["latest_time"] = t_end.timestamp()

            # 3. Create Job
            # We removed 'time_format' as it's unnecessary with Epoch
            job = self.service.jobs.create(query, **kwargs)

            # 4. Parse Results Safely
            # With exec_mode='blocking', no need to poll loop
            reader = job.results(output_mode='json')
            results_json = json.load(reader)
            
            deleted_count = 0
            if 'results' in results_json and len(results_json['results']) > 0:
                # | delete usually returns one row with field "deleted"
                # We iterate just in case, or safely access index 0
                for row in results_json['results']:
                    if 'deleted' in row:
                        deleted_count += int(row['deleted'])
            
            return deleted_count
        # --- Main loop with stop condition ---
        total_deleted = 0
        attempts = 0

        while attempts < max_attempts:
            deleted = run_delete_query(time_range, condition)
            total_deleted += deleted
            attempts += 1

            logger.info(f"Attempt {attempts}: deleted {deleted} logs (total {total_deleted})")

            if logs_qnt is not None:
                remaining = logs_qnt - total_deleted
                if remaining <= 0:
                    logger.info(f"Deleted all {logs_qnt} logs successfully.")
                    break
                elif remaining <= 50:
                    logger.warning(f"{remaining} logs remain — within tolerance, stopping.")
                    break
                else:
                    logger.warning(f"{remaining} logs remain, retrying...")
            else:
                # No expected quantity, run only once
                break

        if attempts == max_attempts:
            logger.error(f"Reached max attempts ({max_attempts}) with incomplete deletion.")

        return total_deleted

  
  
  
  
  
  
  