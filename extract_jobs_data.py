import requests
import json
from typing import List
from time import sleep
import json
import os
import urllib3
urllib3.disable_warnings()
def authenticate(base_url: str, username: str, password: str) -> requests.auth.HTTPBasicAuth:
    """Returns an HTTPBasicAuth object for the given username and password."""
    return requests.auth.HTTPBasicAuth(username, password)

def start_search(base_url: str, auth: requests.auth.HTTPBasicAuth, query: str) -> str:
    """Starts a search job and returns the search ID."""
    search_endpoint = f"{base_url}/services/search/jobs"
    search_params = {
        "search": query,
        "output_mode": "json",
        "max_count": 10000
    }
    response = requests.post(search_endpoint, data=search_params, auth=auth, verify=False)
    sleep(2)
    response_data = json.loads(response.text)
    return response_data["sid"]

def get_search_results(base_url: str, auth: requests.auth.HTTPBasicAuth, search_id: str) -> List[str]:
    """Retrieves the PIDs for the search job with the given search ID."""
    search_endpoint = f"{base_url}/services/search/jobs"
    results_params = {
        "output_mode": "json"
    }
    results_response = requests.get(f"{search_endpoint}/{search_id}/results", params=results_params, auth=auth, verify=False)
    # print(results_response)
    while results_response.status_code == 204:
            results_response = requests.get(f"{search_endpoint}/{search_id}/results", params=results_params, auth=auth, verify=False)
            # print(results_response)
    results_data = json.loads(results_response.text)
    pids = {}
    for result in results_data["results"]:
        search_id = result['search_id'].strip('\'')
        search_name = result['savedsearch_name'].strip('\'')
        time = result['_time'].strip('\'')
        # print(search_id)
        results_response = requests.get(f"{search_endpoint}/{search_id}", params=results_params, auth=auth, verify=False)
        results_data = json.loads(results_response.text)
        # print(results_data)
        try:
            pid = results_data['entry'][0]['content']['pid']
            runDuration = results_data['entry'][0]['content']['runDuration']
            # createTime = results_data['entry'][0]['content']['createTime']
            pids[search_name] = (search_id, int(pid), time, runDuration)
        except Exception as e:
            print('error', e)
            print('result', results_data)
            continue
    return pids

# Define a function to extract the serial number from a directory name
def get_serial_number(dirname):
    return int(dirname.split(' ')[-1])

# make a main function instead of running the script directly
# def main(start_time, end_time, newest_dir):
#     base_url = "https://132.72.81.150:8089"
#     username = 'shouei'
#     password = 'sH231294'
#     query = f'search index=_audit action=search app=search | regex search_id="rt.*" | where _time >= relative_time("{end_time}","{start_time}") | table search_id savedsearch_name _time'
#     auth = authenticate(base_url, username, password)
#     search_id = start_search(base_url, auth, query)
#     pids = get_search_results(base_url, auth, search_id)
#     print(newest_dir)
#     # Construct the target path
#     path = ""
#     # Write the dictionary to a file in the newest directory
#     with open(os.path.join(path, newest_dir, 'pids.json'), 'w') as f:
#         json.dump(pids, f)
#     print(pids)
#     print('Done')

if __name__ == '__main__':
    base_url = "https://132.72.81.150:8089"
    username = 'shouei'
    password = 'sH231294'
    time = "-10m@m"
    # query = f'search index=_audit action=search app=search | regex search_id="rt.*" | where _time >= relative_time(now(),"{time}") | table search_id savedsearch_name _time'
    query = f'search index=_audit action=search app=search | regex search_id="scheduler.*" | where _time >= relative_time(now(),"{time}") | table search_id savedsearch_name _time'
    auth = authenticate(base_url, username, password)
    search_id = start_search(base_url, auth, query)
    pids = get_search_results(base_url, auth, search_id)
    # pids = [x for x in map(int, pids)]
    print(len(pids.values()))
   # Construct the target path
    path = ""
    
    # # Get the list of directories in the target path
    # dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    # # Sort the directories by serial number in descending order
    # dirs.sort(key=get_serial_number, reverse=True)

    # # Get the newest directory
    # newest_dir = dirs[0]
    # get the newest directory name from the sys arg of the program
    newest_dir = os.sys.argv[1]
    print(newest_dir)
    # Write the dictionary to a file in the newest directory
    with open(os.path.join(path, newest_dir, 'pids.json'), 'w') as f:
        json.dump(pids, f)
    print(pids)
    print('Done')