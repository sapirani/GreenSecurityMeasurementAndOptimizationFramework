import multiprocessing
import subprocess
import time
import pandas as pd
from time import sleep
import os
import sys
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/Scanner')
from scanner_class import Scanner


PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.92.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'
CPU_TDP = 200

class Measurement:
    def __init__(self, logger, splunk_tools, num_of_searches):
        self.logger = logger
        self.splunk_tools = splunk_tools
        self.num_of_searches = num_of_searches
        self.current_measurement_path = None
        
        
    def energy_equation(self, rules_energy_df):
        rules_energy_df['CPU(W)'] = rules_energy_df['CPU(%)'] * CPU_TDP / 100
        rules_energy_df['CPU(J)'] = rules_energy_df['CPU(W)'] * rules_energy_df['delta_time']

    def merge_energy_and_rule_data(self, pids_energy_df, rules_pids_df):
        splunk_pids_energy_df = pids_energy_df[pids_energy_df['PID'].isin(rules_pids_df.pid.values)].sort_values('Time(sec)') 
        if len(splunk_pids_energy_df) == 0:
            print('No matching PIDs')
            return
        rules_energy_df = pd.merge(splunk_pids_energy_df, rules_pids_df, left_on='PID', right_on='pid')
        if len(rules_energy_df) == 0:
            print('Problem with the merge')
            return
        rules_energy_df['Time(sec)'] = pd.to_datetime(rules_energy_df['Time(sec)'])
        rules_energy_df = rules_energy_df.sort_values(by=['name', 'Time(sec)'])
        return rules_energy_df

    def fetch_energy_data(self):
        processes_data = pd.read_csv(os.path.join(self.current_measurement_path, 'processes_data.csv'))
        time_differences = processes_data['Time(sec)'].diff().fillna(0)
        processes_data['delta_time'] = time_differences
        processes_data['Time(sec)'] = pd.to_datetime(processes_data['Time(sec)'], unit='s')
        return processes_data

    def measure(self, time_range, time_delta=60):
        self.logger.info('measuring')
        # scanner_process = subprocess.Popen(['python', '../Scanner/scanner.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # start main function in a separate process
        # from Scanner.scanner import main as scanner
        scanner = Scanner(self.logger)
        
        process = multiprocessing.Process(target=scanner.main)
        process.start()     
        sleep(10)
        # self.logger.info(scanner_process.pid)
        while True:
            # Start the scanner.py script in a separate process
            # Now, start the get_rules_data function
            rules_pids_df, num_of_rules = self.splunk_tools.get_rules_data(time_range, self.num_of_searches)
            measurement_num = max([int(folder.split(' ')[1]) for folder in os.listdir(PATH) if folder.startswith('Measurement')])
            self.current_measurement_path = os.path.join(PATH, f'Measurement {measurement_num}')
            try:
                pids_energy_df = self.fetch_energy_data()
                rules_energy_df = self.merge_energy_and_rule_data(pids_energy_df, rules_pids_df)
                if rules_energy_df is None:
                    print('No matching PIDs')
                    continue
                self.energy_equation(rules_energy_df)
                rule_total_energy = rules_energy_df.groupby('name').agg({'CPU(J)': 'sum', 'run_duration': 'first', 'sid': 'first'}).reset_index()
                rule_total_energy.to_csv(os.path.join(self.current_measurement_path, 'grouped_rules_energy.csv'), index=False)
                rule_total_energy_dict = rule_total_energy[['name', 'CPU(J)', 'run_duration', 'sid']].to_dict('records')
                if len(rule_total_energy_dict) == num_of_rules:
                    self.logger.info('Finished measuring - breaking loop')
                    break
            except Exception as e:
                self.logger.error(f'Error in measuring: {e}')
                continue
        with open("/home/shouei/GreenSecurity-FirstExperiment/should_scan.txt", "w") as f:
            f.write('finished')
        #TODO מםם' I dont need the should_scan.txt file
        process.join()
        process.terminate()
        
        # Optionally, you can wait for the scanner process to complete
        # scanner_process.wait()

        # Retrieve the stdout and stderr from the scanner process
        # scanner_stdout, scanner_stderr = scanner_process.communicate()

        # Log the output
        # self.logger.info(f"Scanner stdout: {scanner_stdout}")
        # self.logger.info(f"Scanner stderr: {scanner_stderr}")        
        return rule_total_energy_dict

           
