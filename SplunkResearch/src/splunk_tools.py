import asyncio
from datetime import datetime
import itertools
import httpx
import json
import logging
from multiprocessing import Pool
import random
import re
import subprocess
from dotenv import load_dotenv
import os
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
        
    def get_saved_search_names(self, get_only_enabled=True):
        names = []
        with open(savedsearches_path, 'r') as f:
            content = f.read()
            search_names = re.findall(r'^\[(.*?)\]', content, flags=re.MULTILINE)
            for name in search_names:
                match = re.search(rf"^\[{name}\][^\[]*?disabled = (0|1)", content, re.MULTILINE | re.DOTALL)
                if match is not None and get_only_enabled:
                    # logging.info(f'Saved search "{name}" is disabled. Skipping.')
                    continue
                names.append(name)
        return names
    
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
            pool.starmap(update_func, [(search_name, update_arg) for search_name in searches_names])


    async def _send_logs(self, logs, log_source, hec_token):
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
            self.logger.info(f'Logs successfully sent to Splunk. {len(logs)} logs of source {log_source} were sent.')
        else:
            self.logger.info('Failed to send log entry to Splunk.')
            self.logger.info(response.text)
            self.logger.info("\n".join(events))

    async def insert_logs(self, logs, log_source):
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
            task = asyncio.create_task(self._send_logs(logs[start:end], log_source, token))
            tasks.append(task)
        await asyncio.gather(*tasks)
            
            
    # def insert_log(self, log_entry, log_source):
    #     # BUG: Splunk split this log: b'06/13/2023 05:53:49 PM\nLogName=Application\nEventCode=16384\nEventType=4\nComputerName=LB-111-4.auth.ad.bgu.ac.il\nSourceName=Microsoft-Windows-Security-SPP\nType=Information\nRecordNumber=700003\nKeywords=Classic\nTaskCategory=Logoff\nOpCode=None\nMessage=Successfully scheduled Software Protection service for re-start at 2023-06-18T06:23:05Z. Reason: RulesEngine.\n\nIsFakeLog=True'
    #     source = log_source
    #     sourcetype = log_source.split(':')[0]
    #     # Splunk REST API endpoint
    #     url = f"{self.base_url}/services/receivers/simple"
    #     if log_entry is None:
    #         self.logger.info('Log entry is None. Skipping.')
    #         return
    #     # Send the log entry to Splunk
    #     response = requests.post(f"{url}?sourcetype={sourcetype}&source={source}&index={self.index_name}", data=log_entry.encode('utf-8'), headers=HEADERS, auth=(self.splunk_username, self.splunk_password), verify=False)
    #     # Check the response status
    #     if response.status_code == 200:
    #         return 'Log entry successfully sent to Splunk.'
    #     else:
    #         return 'Failed to send log entry to Splunk.'
            
    def extract_distribution(self, start_time, end_time, fake=False):
        # Placeholder for your Splunk extraction script
        # This should be replaced with your existing script
        # fake_flag = 'host="132.72.81.150:8088"' if fake else 'host!="132.72.81.150:8088"'            
        # command = f'/opt/splunk/bin/splunk search "index=main (earliest="{start_time}" latest="{end_time}") {fake_flag} |stats count by source EventCode | eventstats sum(count) as totalCount" -maxout 0 -auth shouei:sH231294'
        command = f'/opt/splunk/bin/splunk search "index=main (earliest="{start_time}" latest="{end_time}") | eval is_fake=if(host=\\"{"132.72.81.150:8088"}\\", 1, 0) |stats count by source EventCode is_fake| eventstats sum(count) as totalCount" -maxout 0 -auth shouei:sH231294'
        cmd = subprocess.run(command, shell=True, capture_output=True, text=True)
        res_dict = {}
        if len(cmd.stdout.split('\n')) > 2:
            for row in cmd.stdout.split('\n')[2:-1]:
                row = row.split()
                source = row[0]
                event_code = row[1]
                is_fake = row[2]
                count = row[3]
                total_count = row[4]
                res_dict[f"{source.lower()} {event_code} {int(is_fake)}"] = int(count)
            res_dict['total_count'] = int(total_count)
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

    def get_rules_pids(self, time):
        format = "%Y-%m-%d %H:%M:%S"
        query = f'index=_audit action=search app=search search_type=scheduled info=completed  earliest=-{time}m@m latest=now | regex search_id=\\"scheduler.*\\"| eval executed_time=strftime(exec_time, \\"{format}\\")\
        | table search_id savedsearch_name _time executed_time event_count total_run_time'
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
    
    def get_alert_count(self, time_range):
        spl_query = f'search index=_internal sourcetype=scheduler thread_id=AlertNotifier* user="shouei" earliest={time_range[0]} latest={time_range[1]}|stats count'
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
        
    # def split_logs(self, log_source, logs):
    #     # # Split the response by lines and parse each line as a separate JSON object
    #     # if log_source.split(':')[0] == 'wineventlog':
    #     #     return re.split(r'(?m)^(?=\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} (?:AM|PM))', logs)
    #     # else:
    #     #     pattern_start = re.compile("<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'>")
    #     #     pattern_end = re.compile('/Event>')
    #     #     # Split by '<Event'
    #     #     parts = pattern_start.split(logs)
    #     #     # Further split each part by '</Event>'
    #     #     parts = [pattern_end.split(part) for part in parts]
    #     #     # Flatten the list
    #     #     parts = list(itertools.chain(*parts))
    #     #     # Remove empty strings
    #     #     parts = [part for part in parts if part]
    #     #     # Add the '<Event' and '</Event>' tags to the corresponding parts
    #     #     parts = [f'{pattern_start.pattern}{part}{pattern_end.pattern}' for i, part in enumerate(parts) if (i-1) % 3 == 0 ]
    #     #     return parts
    #     return logs.split('[EOF]')


    # Function to fetch message from the provided website
    def fetch_message(self, event_code):
        url = f"https://www.ultimatewindowssecurity.com/securitylog/encyclopedia/event.aspx?eventid={event_code}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            example_div = soup.find('div', class_='block')
            if example_div:
                if event_code.startswith('9'):
                    message = example_div.find_all('p')[1].text.strip().replace("Event Xml:\r\n", '')
                else:
                    # Example structure 1
                    h2 = example_div.find('h2')
                    if h2 and "Examples of" in h2.text:
                        # Extract the paragraphs excluding the last two with links
                        paragraphs = example_div.find_all('p')[:-2]
                        message_parts = [paragraph.text.strip() for paragraph in paragraphs]
                        message = '\n'.join(message_parts)
                    else:
                        message = f"Unable to extract message for event code {event_code}"
                return message
        else:
            return f"Unable to fetch message for event code {event_code}"

    # Function to generate synthetic logs with the general structure
    def generate_log(self, log_source, event_code):
        if  os.path.exists(f'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/logs_to_duplicate_files/{log_source.replace("/", "__")}_{event_code}.txt'):
            return None  
        if log_source.split(':')[0] == 'wineventlog':
            log_name = log_source.split(':')[1]
            log_message = self.fetch_message(event_code)
            
            log = f"08/06/2023 12:43:05 PM\nLevel=Information\n"
            log += f"LogName={log_name}\n"
            log += f"EventCode={event_code}\n"
            log += f"Source=Microsoft-Windows-Security-Auditing\n"
            log += f"User/Account=user123\n"
            log += f"ComputerName=MyComputer\n"
            log += f"TaskCategory=General\n"
            log += f"Message={log_message}\n" 
            self.save_logs(log_source, event_code, [log])       
            return log
        else:
            log_message = self.fetch_message(f"9{'0'*(4-len(event_code))}{event_code}")
            self.save_logs(log_source, event_code, [log_message])       
            return log_message  
                  
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
        # if len(results) == 0:
        #     return None
        # results = self.split_logs(log_source, response.text)
        # Remove any empty lines
        # results = [line for line in results if line.strip()]
        # create the directory if it doesn't exist
        self.save_logs(log_source, eventcode, results)
        return results
        
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
        # load the logs to duplicate from disk
        logs_to_duplicate_dict = {(logtype[0].lower(), logtype[1]): [] for logtype in logtypes}
        for logtype in logtypes:
            source = logtype[0].lower()
            eventcode = logtype[1]
            if not os.path.exists(f'{PREFIX_PATH}logs_to_duplicate_files/{source.replace("/", "__")}_{eventcode}.txt'):
                continue
            with open(f'{PREFIX_PATH}logs_to_duplicate_files/{source.replace("/", "__")}_{eventcode}.txt', 'r') as f:
                text = f.read()
                results = text.split('\n[EOF]\n')
                # results = self.split_logs(source, text)                
                for log in results:
                     if log != '':
                         logs_to_duplicate_dict[(source, eventcode)].append(log)  
        return logs_to_duplicate_dict   
     
    
    def sample_log(self, logs, action_value):
        if len(logs) > 0:
            logs = random.sample(logs, min(len(logs), action_value))
            return logs
        else:
            # self.logger.info('No results found or results is not a list.')
            return None  
        
    def get_time(self, y, m, d, h, mi, s):
        return datetime(y, m, d, h, mi, s).timestamp()
    
    def delete_fake_logs(self, time_range=None):
        url = f"{self.base_url}/services/search/jobs/export"
        if time_range is None:
            time_expression = 'earliest=0'
        else:
            time_expression = f'earliest="{time_range[0]}" latest="{time_range[1]}"'
        data = {
            "search": f'search index=main host=\"{"132.72.81.150:8088"}\"{time_expression} | delete',
            "exec_mode": "oneshot",
            "output_mode": "json"
        }
        response = requests.post(url, headers=HEADERS, data=data, auth=self.auth, verify=False)
        results = response.text
        self.logger.info(results)

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