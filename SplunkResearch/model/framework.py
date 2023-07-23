import subprocess
import pandas as pd
import sys
import urllib3
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/splunk_tools')
from splunk_tools import SplunkTools
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment')
from Scanner.scanner import main as scanner
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
urllib3.disable_warnings()
from log_generator import LogGenerator


class FrameWork:
    def __init__(self):
        self.measurement = scanner
        self.log_generator = LogGenerator()
        self.wanted_distribution = None
        self.splunk_tools = SplunkTools()
    
    def measure(self, path, measurement_num, time_delta=60):
        # Placeholder for your measurement system
        # This should be replaced with your existing Python script
        self.measurement()
        pids_energy_file_path = os.path.join(path,f'Measurement {measurement_num}', 'processes_data.csv')
        pids_energy_df = pd.read_csv(pids_energy_file_path)
        pids_energy_df['Time(sec)'] = pd.to_datetime(pids_energy_df['Time(sec)'], unit='s').dt.to_pydatetime()
        rules_pids = self.splunk_tools.get_searches_and_jobs_info(time_delta)
        rules_energy_df = self.find_rules_energy(rules_pids, pids_energy_df)
        return rules_energy_df
    
    def find_rules_energy(self, rules_pids, pids_energy_df):
        # Placeholder for your distribution comparison function
        # This could use a metric like KL divergence
        data = []
        for name, rules in rules_pids.items():
            for e in rules:
                sid, pid, time, run_duration = e[:4]
                data.append((name, sid, pid, time, run_duration)) 
        rules_pids_df = pd.DataFrame(data, columns=['name', 'sid', 'pid', 'time', 'run_duration'])
        rules_pids_df.time = pd.to_datetime(rules_pids_df.time)
        rules_pids_df.sort_values('time', inplace=True)
        rules_energy_df = pids_energy_df[pids_energy_df['PID'].isin(rules_pids_df.pid.values)].sort_values('Time(sec)') 
        return rules_energy_df
    

    
    def execute(self, log, log_type, replacement_dict, time_range, path, measurement_num):
        # STATE
        rules_energy_df = self.measure(path, measurement_num)
        alerts_status = self.splunk_tools.extract_alerts()
        current_distribution = self.splunk_tools.extract_distribution(*time_range)
        # REWARD
        dist_distance = self.log_generator.compare_distributions(self.wanted_distribution, current_distribution)
        reward = self.log_generator.get_reward(alerts_status, rules_energy_df, dist_distance)        
        # ACTION
        fake_log = self.log_generator.replace_fields_in_log(log, log_type, time_range, replacement_dict)
        self.splunk_tools.insert_log(fake_log, log_type)
        
        


if __name__ == "__main__":
    framework = FrameWork()
    path = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.88.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'
    measurement_num = 151
    
    
    
    
    # Generate a variant replacement dictionary
    replacement_dict_wineventlog = {field: random.choice(replacement_values_wineventlog[field]) for field in fields_wineventlog}
    replacement_dict_sysmon = {field: random.choice(replacement_values_sysmon[field]) for field in fields_sysmon}
    start_date_string = '06/15/2023:00:00:00'
    end_date_string = '06/15/2023:23:59:59'
    time_range = (start_date_string, end_date_string)
    
    
    # BUG: Splunk doesnt parse all the logs
    # TODO: create a loop that will generate many logs and insert them to splunk
    framework.execute(wineventlog_log, 'WinEventLog', replacement_dict_wineventlog, time_range, path, measurement_num)
    # model.execute(sysmon_log, 'xmlwineventlog', replacement_dict_sysmon, time_range, path, measurement_num)
