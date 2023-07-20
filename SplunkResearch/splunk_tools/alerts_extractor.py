
import requests
import json



def execute(base_url, username, password):
    # Define your SPL query
    spl_query = 'search index=_internal sourcetype=scheduler thread_id=AlertNotifier* user="shouei"|stats count by savedsearch_name sid'

    # Define the REST API endpoint
    url = f"{base_url}/services/search/jobs"

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
    response = requests.post(url, headers=headers, data=data, auth=(username, password), verify=False)

    # Parse the JSON response
    results = json.loads(response.text)
    return results



if __name__ == '__main__':
    # Define your Splunk instance details
    base_url = "https://132.72.81.150:8089"
    username = "shouei"
    password = "sH231294"
    print(execute(base_url, username, password))