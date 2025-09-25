import re

def extract_info_from_log(log_content):
    info = {
        'time_ranges': [],
        'alert_values': [],
        'duration_values': []
    }

    # Extract all time ranges
    time_range_matches = re.finditer(r"new time_range: \('(.+)', '(.+)'\)", log_content)
    for match in time_range_matches:
        info['time_ranges'].append(tuple((match.group(1), match.group(2))))

    # Extract alert and duration values
    alert_pattern = r"(?:no agent )?alert value: (\d+(?:\.\d+)?)"
    duration_pattern = r"(?:no agent )?duration value: (\d+(?:\.\d+)?)"

    alert_matches = re.finditer(alert_pattern, log_content)
    duration_matches = re.finditer(duration_pattern, log_content)

    alert_values = [float(match.group(1)) for match in alert_matches]
    duration_values = [float(match.group(1)) for match in duration_matches]

    # Pair alert and duration values
    for i in range(0, len(alert_values), 2):
        if i + 1 < len(alert_values):
            info['alert_values'].append([alert_values[i], alert_values[i+1]])
        else:
            info['alert_values'].append([alert_values[i]])

    for i in range(0, len(duration_values), 2):
        if i + 1 < len(duration_values):
            info['duration_values'].append([duration_values[i], duration_values[i+1]])
        else:
            info['duration_values'].append([duration_values[i]])

    return info

# Read the log file
with open('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments__/splunk-v12/no_agent_1200_04/26/2024:13:00:00/log.txt', 'r') as file:
    log_content = file.read()

# Extract information
extracted_info = extract_info_from_log(log_content)

# Print the extracted information
print(extracted_info)
import json
# dump the extracted information to a file in splunk-v12 directory as a json file
with open('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments__/splunk-v12/no_agent_1200_04/26/2024:13:00:00/no_agent_values.json', 'w') as file:
    json.dump(extracted_info, file)
    