
import requests
import json

# Define your Splunk instance details
splunk_host = "132.72.81.150"
splunk_port = 8089
username = "shouei"
password = "sH231294"

# Define your SPL query
spl_query = 'search index=_internal sourcetype=scheduler thread_id=AlertNotifier* user="shouei"|stats count by savedsearch_name sid'

# Define the REST API endpoint
url = f"https://{splunk_host}:{splunk_port}/services/search/jobs"

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

# Print the results
for result in results['results']:
    print(result)

