import requests
import json
from typing import List
from time import sleep
import json
import os
import urllib3
import subprocess
import datetime
import re
urllib3.disable_warnings()


def get_searches_info(time):
    format = "%Y-%m-%d %H:%M:%S"
    print(time)
    query = f'index=_audit action=search app=search search_type=scheduled info=completed  earliest=-{time}m@m latest=now | regex search_id=\\"scheduler.*\\"| eval executed_time=strftime(exec_time, \\"{format}\\") | table search_id savedsearch_name _time executed_time event_count total_run_time'
    print(query)
    command = f'echo sH231294| sudo -S -E env "PATH"="$PATH" splunk search "{query}" -maxout 0 -auth shouei:sH231294'
    print(command)
    cmd = subprocess.run(command, shell=True, capture_output=True, text=True)
    res = cmd.stdout.split('\n')[2:-1]
    res_dict = {}
    for line in res:
        pattern = r"'(.*?)' (\D+) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} IDT) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \s+(\d+)\s+(\d+\.\d+)"
        matches = re.findall(pattern, line)
        for match in matches:
            res_dict[match[0]] = match[1:]
    print(cmd.stderr)
    return res_dict


def get_jobs_info(res_dict, base_url, username, password):
    pids={}
    auth = requests.auth.HTTPBasicAuth(username, password)
    search_endpoint = f"{base_url}/services/search/jobs"
    results_params = {
        "output_mode": "json"
    }
    for search in res_dict:
        search_id = search
        rule_name = res_dict[search][0].strip()
        time = res_dict[search][1]
        total_events = res_dict[search][2]
        total_run_time = res_dict[search][3]
        
        results_response = requests.get(f"{search_endpoint}/{search_id}", params=results_params, auth=auth, verify=False)
        results_data = json.loads(results_response.text)
        try:
            pid = results_data['entry'][0]['content']['pid']
            runDuration = results_data['entry'][0]['content']['runDuration']
            if rule_name not in pids:
                pids[rule_name] = [(search_id, int(pid), time, runDuration, total_events, total_run_time)]
            else:
                pids[rule_name].append((search_id, int(pid), time, runDuration, total_events, total_run_time))
        except Exception as e:
            print('error', e)
            print('result', results_data)
            print('search_id', search_id)
            print('time', time)
            continue
    return pids

def execute(time, base_url, username, password):
    res_dict = get_searches_info(time)
    pids = get_jobs_info(res_dict, base_url, username, password)
    print(pids)
    return pids

if __name__ == '__main__':
    newest_dir = os.sys.argv[1]
    time = f"{os.sys.argv[2]}"
    base_url = "https://132.72.81.150:8089"
    username = 'shouei'
    password = 'sH231294'
    pids = execute(time, base_url, username, password)
    # Write the dictionary to a file in the newest directory
    with open(os.path.join("", newest_dir, 'pids.json'), 'w') as f:
        json.dump(pids, f)
    print(pids)
    print('Done')