import json
from multiprocessing import Pool
import re
import subprocess
from dotenv import load_dotenv
import os
import requests

load_dotenv('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/.env')
# Precompile the regex pattern
pattern = re.compile(r"'(.*?)' (\D+) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} IDT) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \s+(\d+)\s+(\d+\.\d+)")

# Create a session for HTTP requests
session = requests.Session()
session.verify = False

class SplunkTools:
    def __init__(self):
        self.splunk_host = os.getenv("SPLUNK_HOST")
        self.splunk_port = os.getenv("SPLUNK_PORT")
        self.base_url = f"https://{self.splunk_host}:{self.splunk_port}"
        self.splunk_username = os.getenv("SPLUNK_USERNAME")
        self.splunk_password = os.getenv("SPLUNK_PASSWORD")
        self.index_name = os.getenv("INDEX_NAME")
        self.auth = requests.auth.HTTPBasicAuth(self.splunk_username, self.splunk_password)
    
    def insert_log(self, log_entry, log_source):
        # Splunk REST API endpoint
        url = f"{self.base_url}/services/receivers/simple"
        # Define the headers for the HTTP request
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        # Send the log entry to Splunk
        response = requests.post(f"{url}?sourcetype={log_source}&index={self.index_name}", data=log_entry, headers=headers, auth=(self.splunk_username, self.splunk_password), verify=False)
        # Check the response status
        if response.status_code == 200:
            print('Log entry successfully sent to Splunk.')
        else:
            print('Failed to send log entry to Splunk.')
            
    def extract_distribution(self, start_time, end_time):
        # Placeholder for your Splunk extraction script
        # This should be replaced with your existing script
        command = f'/opt/splunk/bin/splunk search "index=main (earliest="{start_time}" latest="{end_time}")|stats count by source EventCode | eventstats sum(count) as totalCount" -maxout 0 -auth shouei:sH231294'
        print(command)
        cmd = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        res_dict = {}
        if len(cmd.stdout.split('\n')) > 2:
            for row in cmd.stdout.split('\n')[2:-1]:
                row = row.split()
                source = row[0]
                event_code = row[1]
                count = row[2]
                total_count = row[3]
                res_dict[f"{source} {event_code}"] = int(count)
            res_dict['total_count'] = int(total_count)
        return res_dict
    
    
    def get_search_details(self, search, res_dict, search_endpoint):
        search_id = search
        rule_name = res_dict[search][0].strip()
        time = res_dict[search][1]
        total_events = res_dict[search][2]
        total_run_time = res_dict[search][3]

        results_response = session.get(f"{search_endpoint}/{search_id}", params={"output_mode": "json"}, auth=self.auth)
        results_data = json.loads(results_response.text)
        try:
            pid = results_data['entry'][0]['content']['pid']
            runDuration = results_data['entry'][0]['content']['runDuration']
            return (rule_name, (search_id, int(pid), time, runDuration, total_events, total_run_time))
        except KeyError as e:
            print(f'KeyError - {str(e)}')
        except IndexError as e:
            print(f'IndexError - {str(e)}')
        except Exception as e:
            print('Unexpected error:', str(e))

    def get_searches_and_jobs_info(self, time):
        format = "%Y-%m-%d %H:%M:%S"
        query = f'index=_audit action=search app=search search_type=scheduled info=completed  earliest=-{time}m@m latest=now | regex search_id=\\"scheduler.*\\"| eval executed_time=strftime(exec_time, \\"{format}\\") | table search_id savedsearch_name _time executed_time event_count total_run_time'
        command = f'echo sH231294| sudo -S -E env "PATH"="$PATH" /opt/splunk/bin/splunk search "{query}" -maxout 0 -auth shouei:sH231294'
        cmd = subprocess.run(command, shell=True, capture_output=True, text=True)
        res = cmd.stdout.split('\n')[2:-1]
        res_dict = {re.findall(pattern, line)[0][0]: re.findall(pattern, line)[0][1:] for line in res}
        print('stderr ', cmd.stderr)

        
        search_endpoint = f"{self.base_url}/services/search/jobs"

        # Use a multiprocessing pool to get details of all searches in parallel
        with Pool() as pool:
            results = pool.starmap(self.get_search_details, [(search, res_dict, search_endpoint) for search in res_dict])

        pids = {}
        for rule_name, details in results:
            if rule_name not in pids:
                pids[rule_name] = [details]
            else:
                pids[rule_name].append(details)

        return pids
    def extract_alerts(self):
        # Define your SPL query
        spl_query = 'search index=_internal sourcetype=scheduler thread_id=AlertNotifier* user="shouei"|stats count by savedsearch_name sid'

        # Define the REST API endpoint
        url = f"{self.base_url}/services/search/jobs"

        # Define the headers for the HTTP request
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        # Define the data for the HTTP request
        data = {
            "search": spl_query,
            "exec_mode": "oneshot",
            "output_mode": "json"
        }

        # Send the HTTP request to the REST API endpoint
        response = session.post(url, headers=headers, data=data, auth=(self.splunk_username, self.splunk_password))

        # Parse the JSON response
        results = json.loads(response.text)
        return results