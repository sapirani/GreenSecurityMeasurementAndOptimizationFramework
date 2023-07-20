# some_file.py
import subprocess
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment')
from scanner import main as scanner
from faker import Faker
import requests
import json
import random
from datetime import datetime
from config import wineventlog_log, sysmon_log, fields_wineventlog, replacement_values_wineventlog, fields_sysmon, replacement_values_sysmon
import re
from xml.etree import ElementTree as ET

class LogGenerator:
    def __init__(self, replacements):
        self.replacements = replacements

    def replace_fields_in_log(self, log, log_source, time_range):
        # random time from time_range
        start_date, end_date=time_range
        start_date = datetime.strptime(start_date_string, '%m/%d/%Y:%H:%M:%S') 
        end_date = datetime.strptime(end_date_string, '%m/%d/%Y:%H:%M:%S') 
        time = Faker().date_time_between(start_date, end_date, tzinfo=None)      
                        
        if log_source == 'WinEventLog':
            new_log = re.sub(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [APM]{2}", time.strftime("%m/%d/%Y %I:%M:%S %p"), log, flags=re.MULTILINE)
            new_log += '\nIsFakeLog=True'
            for field, new_value in self.replacements.items():
                new_log = re.sub(f"{field}=\S+", f"{field}={new_value}", new_log, flags=re.MULTILINE)
                print(new_log)
        else:
            xml = ET.fromstring(log)
            for field, new_value in self.replacements.items():
                for elem in xml.iter():
                    if elem.attrib.get('Name') == field:
                        elem.text = new_value
            for elem in xml.iter():
                if elem.attrib.get('Name') == 'UtcTime':
                    elem.text = time.isoformat() 
                    print(elem.text)  
            # Find the 'TimeCreated' element and set the 'SystemTime' attribute
            time_created_elem = xml.find('{http://schemas.microsoft.com/win/2004/08/events/event}System')
            if time_created_elem is not None:
                time_created_elem.set('SystemTime', time.isoformat())  # 'Z' is added to indicate UTC time
       
            # Register the namespace with a prefix
            ET.register_namespace('', 'http://schemas.microsoft.com/win/2004/08/events/event')
            # Use the registered prefix in your XPath query
            event_data_elem = xml.find('{http://schemas.microsoft.com/win/2004/08/events/event}EventData')
            if event_data_elem is not None:
                ET.SubElement(event_data_elem, 'Data', {'Name': 'IsFakeLog'}).text = 'True'
                print('added')
            new_log = ET.tostring(xml, encoding='unicode')
        return new_log


class AttackModel:
    def __init__(self):
        self.measurement = scanner
        self.log_generator = LogGenerator
        self.wanted_distribution = None

    def generate_log(self, log_to_copy, log_source, replacements_dict, time_range):
        # Placeholder for your log generation function
        # This function should take a log and modify some details
        log_generator = LogGenerator(replacements_dict)
        new_log = log_generator.replace_fields_in_log(log_to_copy, log_source, time_range)
        return new_log
        


    
    def insert_log(self, log_entry, log_source):
        # Placeholder for your log generation function
        # This function should take a log and modify some details


        # Splunk server information
        splunk_host = '132.72.81.150'
        splunk_port = '8089'
        splunk_username = 'shouei'
        splunk_password = 'sH231294'
        index_name = 'main'
        sourcetype = log_source
        # Splunk REST API endpoint
        url = f'https://{splunk_host}:{splunk_port}/services/receivers/simple'

        # Define the headers for the HTTP request
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        # Send the log entry to Splunk
        response = requests.post(f"{url}?sourcetype={sourcetype}&index={index_name}", data=log_entry, headers=headers, auth=(splunk_username, splunk_password), verify=False)
        # Check the response status
        if response.status_code == 200:
            print('Log entry successfully sent to Splunk.')
        else:
            print('Failed to send log entry to Splunk.')


    
    def measure(self):
        # Placeholder for your measurement system
        # This should be replaced with your existing Python script
        self.measurement()

    def extract_distribution(self, start_time, end_time):
        # Placeholder for your Splunk extraction script
        # This should be replaced with your existing script
        command = f'/opt/splunk/bin/splunk search "index=main (earliest="{start_time}" latest="{end_time}")|stats count by source EventCode | eventstats sum(count) as totalCount" -maxout 0 -auth shouei:sH231294'
        print(command)
        cmd = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = cmd.stdout
        
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

    def compare_distributions(self, dist1, dist2):
        # Placeholder for your distribution comparison function
        # This could use a metric like KL divergence
        pass
    
    def execute(self, log, log_type, replacement_dict, time_range):
        # Placeholder for your execution function
        # This should be replaced with your existing script
        log_entry_wineventlog = self.generate_log(log, log_type, replacement_dict, time_range)
        # distribution = self.extract_distribution(*time_range)
        # print(distribution)
        # reward = self.compare_distributions(model.wanted_distribution, distribution)
        self.insert_log(log_entry_wineventlog, log_type)


if __name__ == "__main__":
    model = AttackModel()
    # model.measure()    
    # Generate a variant replacement dictionary
    replacement_dict_wineventlog = {field: random.choice(replacement_values_wineventlog[field]) for field in fields_wineventlog}
    replacement_dict_sysmon = {field: random.choice(replacement_values_sysmon[field]) for field in fields_sysmon}
    
    start_date_string = '06/14/2023:00:00:00'
    end_date_string = '06/15/2023:00:00:00'

    time_range = (start_date_string, end_date_string)
    
    model.execute(wineventlog_log, 'WinEventLog', replacement_dict_wineventlog, time_range)
    model.execute(sysmon_log, 'xmlwineventlog', replacement_dict_sysmon, time_range)
