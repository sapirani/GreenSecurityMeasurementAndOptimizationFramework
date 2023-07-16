import requests
import json
# import rules from rules.py
# from rules_gpt import rules
from rules_cheet_sheet import rules

import requests
from requests.auth import HTTPBasicAuth

# Your Splunk server settings
SPLUNK_SERVER = "localhost"
USERNAME = "shouei"
PASSWORD = "sH231294"
APP = "search"



# Base URL for the REST API
base_url = f"https://{SPLUNK_SERVER}:8089/servicesNS/{USERNAME}/{APP}/saved/searches"

# Headers
headers = {
    "Content-Type": "application/x-www-form-urlencoded"
}

# Iterate over each rule in the rules list
for rule in rules:
    # The URL to update the saved search
    url = f"{base_url}/{rule['name']}"

    # The data to update
    data = {
        # "search": rule["search"],
        "dispatch.ttl": "18p",
        # Add any other parameters you want to update here
    }

    # Send the POST request
    response = requests.post(url, headers=headers, data=data, verify=False, auth=HTTPBasicAuth(USERNAME, PASSWORD))

    # Check if the request was successful
    if response.status_code == 200:
        print(f'Successfully updated saved search "{rule["name"]}".')
    else:
        print(f'Failed to update saved search "{rule["name"]}". HTTP status code: {response.status_code}.')

print("Update completed.")



# url = "https://localhost:8089/servicesNS/shouei/search/saved/searches"
# auth = ("shouei", "sH231294")
# headers = {
#     "Content-Type": "application/x-www-form-urlencoded"
# }



# for rule in rules:
#     data = {
#         "name": rule["name"],
#         "search": rule["search"],
#         "cron_schedule": rule["cron_schedule"],
#         "dispatch.earliest_time": "-15m@m",
#         "dispatch.latest_time": "now",
#         "alert.expires": "90m",
#         "is_scheduled": "1",
#         "alert_type": "number of events",
#         "alert_comparator": "greater than",
#         "alert_threshold": "0",
#         "alert.suppress": "0",
#         "alert.severity": rule["severity"],
#         "alert.track": "1",
#         "description": rule["description"]
#         "dispatch.ttl": "18p",
#     }

#     response = requests.post(url, headers=headers, data=data, auth=auth, verify=False)

#     if response.status_code == 201:
#         print(f'Alert "{rule["name"]}" was created successfully.')
#     else:
#         print(f'Failed to create alert "{rule["name"]}". Status code: {response.status_code}, Response: {response.text}')

