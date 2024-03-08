import multiprocessing
import subprocess
import time
import pandas as pd
from time import sleep
import os
import sys
import urllib3
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/Scanner')
from scanner_class import Scanner
urllib3.disable_warnings()


PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.92.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'
CPU_TDP = 200

class Measurement:
    def __init__(self, logger, splunk_tools, num_of_searches, measure_energy):
        self.logger = logger
        self.splunk_tools = splunk_tools
        self.num_of_searches = num_of_searches
        self.current_measurement_path = None
        self.measure_energy = measure_energy
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        
        
    def energy_equation(self, processes_data):
        processes_data['CPU(W)'] = processes_data['CPU(%)'] * CPU_TDP / 100
        processes_data['CPU(J)'] = processes_data['CPU(W)'] * processes_data['delta_time']

    def merge_energy_and_rule_data(self, pids_energy_df, rules_pids_df):
        splunk_pids_energy_df = pids_energy_df[pids_energy_df['PID'].isin(rules_pids_df.pid.values)].sort_values('Time(sec)') 
        if len(splunk_pids_energy_df) == 0:
            raise Exception('No matching PIDs')
        rules_energy_df = pd.merge(splunk_pids_energy_df, rules_pids_df, left_on='PID', right_on='pid')
        if len(rules_energy_df) == 0:
            raise Exception('Problem with merging the dataframes')
        rules_energy_df['Time(sec)'] = pd.to_datetime(rules_energy_df['Time(sec)'])
        rules_energy_df = rules_energy_df.sort_values(by=['name', 'Time(sec)'])
        return rules_energy_df

    def get_rules_energy(self):
        processes_data = pd.read_csv(os.path.join(self.current_measurement_path, 'processes_data.csv'))
        time_differences = processes_data['Time(sec)'].diff().fillna(0)
        processes_data['delta_time'] = time_differences
        processes_data['Time(sec)'] = pd.to_datetime(processes_data['Time(sec)'], unit='s')
        self.energy_equation(processes_data)
        return processes_data
    
    def get_rules_metadata(self, time_range, num_of_searches):
        while True:
            rules_pids = self.splunk_tools.get_rules_pids(time_range, num_of_searches)
            data = []
            for name, rules in rules_pids.items():
                for e in rules:
                    sid, pid, time, run_duration, total_events, total_run_time = e 
                    data.append((name, sid, pid, time, run_duration, total_events, total_run_time)) 
            rules_pids_df = pd.DataFrame(data, columns=['name', 'sid', 'pid', 'time', 'run_duration', 'total_events', 'total_run_time'])
            if len(rules_pids_df.name.unique()) == num_of_searches and len(rules_pids_df) == num_of_searches:
                break
            rules_pids_df = None
            sleep(1)
        rules_pids_df.time = pd.to_datetime(rules_pids_df.time)
        rules_pids_df.sort_values('time', inplace=True)
        num_of_rules = len(rules_pids_df['name'].unique())
        self.logger.info(f"num of extracted rules data: {num_of_rules}")
        return rules_pids_df, num_of_rules
    

    def measure(self, time_range, time_delta=60):
        self.logger.info('measuring')
        if self.measure_energy:
            process = self.start_measurement_process()     
        while True:
            rules_pids_df, num_of_rules = self.get_rules_metadata(time_range, self.num_of_searches)
            if self.measure_energy:
                measurement_num = max([int(folder.split(' ')[1]) for folder in os.listdir(PATH) if folder.startswith('Measurement')])
                self.current_measurement_path = os.path.join(PATH, f'Measurement {measurement_num}')
                try:
                    if self.parent_conn.poll():
                        message = self.parent_conn.recv()
                        self.logger.info(f"received message: {message}")
                        if "error" in message:
                            raise Exception(f'Scanner process failed {message}')
                    pids_energy_df = self.get_rules_energy()
                    try:
                        rules_energy_df = self.merge_energy_and_rule_data(pids_energy_df, rules_pids_df)
                    except Exception as e:
                        self.logger.error(e)
                        continue
                    
                    rule_total_energy = rules_energy_df.groupby('name').agg({'CPU(J)': 'sum', 'run_duration': 'first', 'sid': 'first'}).reset_index()
                    rule_total_energy.to_csv(os.path.join(self.current_measurement_path, 'grouped_rules_energy.csv'), index=False)
                    rule_total_energy_dict = rule_total_energy[['name', 'CPU(J)', 'run_duration', 'sid']].to_dict('records')
                    if len(rule_total_energy_dict) == num_of_rules:
                        self.logger.info('Finished measuring - breaking loop')
                        self.parent_conn.send('stop')
                        process.join()
                        process.terminate()
                        break
                except Exception as e:
                    self.logger.error(f'Error in measuring: {e}')
                    # check if the scanner process is still running
                    if process.is_alive():
                        self.logger.info('Scanner process is still running - terminating it')
                        process.terminate()
                    else:
                        self.logger.info('Scanner process is not running')
                    
                    process = self.start_measurement_process()
                    continue
            else:
                return rules_pids_df.groupby('name').agg({'run_duration': 'first', 'sid': 'first'}).reset_index().to_dict('records')

        
        # Optionally, you can wait for the scanner process to complete
        # scanner_process.wait()

        # Retrieve the stdout and stderr from the scanner process
        # scanner_stdout, scanner_stderr = scanner_process.communicate()

        # Log the output
        # self.logger.info(f"Scanner stdout: {scanner_stdout}")
        # self.logger.info(f"Scanner stderr: {scanner_stderr}")        
        return rule_total_energy_dict

    def start_measurement_process(self):
        scanner = Scanner(self.logger)  
        process = multiprocessing.Process(target=scanner.main, args=(self.child_conn,))
        process.start()
        sleep(5)
        return process

           
