# some_file.py
import subprocess
import pandas as pd
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/splunk_tools')
from alerts_extractor import execute as alerts_extractor
from jobs_info_extractor import execute as jobs_info_extractor
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
import os
from dotenv import load_dotenv
load_dotenv('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/.env')

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
        self.splunk_host = os.getenv("SPLUNK_HOST")
        self.splunk_port = os.getenv("SPLUNK_PORT")
        self.base_url = f"https://{self.splunk_host}:{self.splunk_port}"
        self.splunk_username = os.getenv("SPLUNK_USERNAME")
        self.splunk_password = os.getenv("SPLUNK_PASSWORD")
        self.index_name = os.getenv("INDEX_NAME")

    def generate_log(self, log_to_copy, log_source, replacements_dict, time_range):
        # Placeholder for your log generation function
        # This function should take a log and modify some details
        log_generator = LogGenerator(replacements_dict)
        new_log = log_generator.replace_fields_in_log(log_to_copy, log_source, time_range)
        return new_log

    
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


    
    def measure(self, path, measurement_num, time_delta=60):
        # Placeholder for your measurement system
        # This should be replaced with your existing Python script
        self.measurement()
        pids_energy_file_path = os.path.join(path,f'Measurement {measurement_num}', 'processes_data.csv')
        pids_energy_df = pd.read_csv(pids_energy_file_path)
        pids_energy_df['Time(sec)'] = pd.to_datetime(pids_energy_df['Time(sec)'], unit='s').dt.to_pydatetime()
        # TODO problem with the sudo. doesnt recognize splunk command
        rules_pids = jobs_info_extractor(time_delta, self.base_url, self.splunk_username, self.splunk_password)

        rules_energy_df = self.find_rules_energy(rules_pids, pids_energy_df)
        return rules_energy_df
    
    def find_rules_energy(self, rules_pids, pids_energy_df):
        # Placeholder for your distribution comparison function
        # This could use a metric like KL divergence
        data = []
        print(rules_pids)
        for name, rules in rules_pids.items():
            for e in rules:
                sid, pid, time, run_duration = e[:4]
                data.append((name, sid, pid, time, run_duration)) 
        rules_pids_df = pd.DataFrame(data, columns=['name', 'sid', 'pid', 'time', 'run_duration'])
        rules_pids_df.time = pd.to_datetime(rules_pids_df.time)
        rules_pids_df.sort_values('time', inplace=True)
        print(rules_pids_df)
        print(pids_energy_df)
        rules_energy_df = pids_energy_df[pids_energy_df['PID'].isin(rules_pids_df.pid.values)].sort_values('Time(sec)') 
        return rules_energy_df
    
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

    def compare_distributions(self, dist1, dist2):
        # Placeholder for your distribution comparison function
        # This could use a metric like KL divergence
        pass
    
    def get_reward(self, alerts_status, rules_energy_df, dist_distance):
        pass
    
    def perform_action(self, action, log, log_type, replacement_dict, time_range):
        # TODO according to the action, generate logs and insert them to splunk
        pass

    
    def execute(self, log, log_type, replacement_dict, time_range, path, measurement_num):
        # STATE
        rules_energy_df = model.measure(path, measurement_num)
        # print(rules_energy_df)
        alerts_status = alerts_extractor(self.base_url, self.splunk_username, self.splunk_password)
        # print(alerts_status)
        current_distribution = self.extract_distribution(*time_range)
        # print(current_distribution)
        # TODO send to agent
        
        # REWARD
        dist_distance = self.compare_distributions(model.wanted_distribution, current_distribution)
        reward = self.get_reward(alerts_status, rules_energy_df, dist_distance)
        # TODO send to agent
        
        # ACTION
        fake_log = self.generate_log(log, log_type, replacement_dict, time_range)  
        # print(fake_log)      
        self.insert_log(fake_log, log_type)
        
        


if __name__ == "__main__":
    model = AttackModel()
    path = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.88.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'
    measurement_num = 151
    
    
    
    
    # Generate a variant replacement dictionary
    replacement_dict_wineventlog = {field: random.choice(replacement_values_wineventlog[field]) for field in fields_wineventlog}
    replacement_dict_sysmon = {field: random.choice(replacement_values_sysmon[field]) for field in fields_sysmon}
    start_date_string = '06/15/2023:00:00:00'
    end_date_string = '06/15/2023:23:59:59'
    time_range = (start_date_string, end_date_string)
    
    model.execute(wineventlog_log, 'WinEventLog', replacement_dict_wineventlog, time_range, path, measurement_num)
    # model.execute(sysmon_log, 'xmlwineventlog', replacement_dict_sysmon, time_range, path, measurement_num)
