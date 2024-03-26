import requests
import datetime

def extract_events(index_name, start_date, end_date, batch_size, output_dir):
    # Set up Splunk credentials
    splunk_url = "https://132.72.81.150:8089"
    username = 'shouei'
    password = 'sH231294'

    HEADERS = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # Construct search query
    search_query = f'search index={index_name} earliest="{start_date}" latest="{end_date}"'

    # Perform search
    search_params = {
        'exec_mode': 'blocking',
        'search': search_query,
        'count': batch_size,
        'output_mode': 'json'
    }
    search_url = splunk_url + '/services/search/jobs/export'
    auth = requests.auth.HTTPBasicAuth(username, password)
    response = requests.post(search_url, headers=HEADERS, data=search_params, auth=auth, verify=False)
    # Check response status
    if response.status_code != 200:
        print(f"Error: Failed to retrieve events. Status code: {response.status_code}")
        return


     # Process results and save them to files
    results_json = response.text
    file_count = 1
    for result in results_json.split('\n'):
        if result.strip():
            with open(f"{output_dir}/events_{file_count}.txt", "a") as file:
                file.write(result.strip() + "\n")
            if file.tell() >= 10 * 1024 * 1024:  # 10 MB limit per file
                file_count += 1

# Example usage
if __name__ == "__main__":
    index_name = "test1"
    start_date = "05/01/2023:00:00:00"
    end_date = "05/14/2024:23:59:59"
    batch_size = 10000
    output_dir = "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/output"
    extract_events(index_name, start_date, end_date, batch_size, output_dir)