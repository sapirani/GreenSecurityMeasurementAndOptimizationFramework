import datetime
import os
import subprocess
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import pandas as pd
CPU_TDP=200
PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.92.1.el7.x86_64/Splunk Enterprise SIEM/Power Saver Plan/One Scan/'

class RewardCalc:
    def __init__(self, relevant_logtypes, dt_manager, logger, splunk_tools, rule_frequency, num_of_searches):
        # self.previous_energy = 1
        # self.previous_alert = 0
        self.average_energy = 0
        self.average_alert = 0
        self.average_duration = 0
        self.epsilon = 0
        self.reward_dict = {'energy': [0], 'alerts': [0], 'distributions': [], 'fraction': [], 'total': []}
        self.reward_values_dict = {'energy': [], 'alerts': [], 'distributions': [], 'fraction': [], 'duration': [], "num_of_rules":[]}
        self.relevant_logtypes = relevant_logtypes
        self.dt_manager = dt_manager
        self.logger = logger
        self.splunk_tools = splunk_tools
        self.rule_frequency = rule_frequency
        self.time_rules_energy = []
        self.current_measurement_path = ''
        self.num_of_searches = num_of_searches
      
    def get_previous_full_reward(self):
        return self.reward_dict['alerts'][-1], self.reward_dict['energy'][-1]

    def get_partial_reward(self, real_distribution, current_state):
        fraction_val, distributions_val = self.get_partial_reward_values(real_distribution, current_state)
        if distributions_val > 0.2:
            return -distributions_val
        if distributions_val <= 0.2:
            return 1/distributions_val
        
        # if fraction_val > 1:
        #     return -100
        # if distributions_val > 0.2:
        #     return -distributions_val
        # if distributions_val < 0.2:
        #     return 1-distributions_val


            
        # if fraction_val > 1:
        #     fraction_reward = -100*fraction_val
        # elif fraction_val == 0:
        #     fraction_reward = -200
        # # elif fraction_val == 1:
        # #     fraction_reward = 100
        # else:
        #     fraction_reward = fraction_val*100
        # if distributions_val > 0.2:
        #     distributions_reward = -100*distributions_val
        # else:
        #     distributions_reward = (1-distributions_val)*100

        # self.reward_dict['distributions'].append(distributions_reward)
        # self.reward_dict['fraction'].append(fraction_reward)
        # self.logger.info(f"distributions reward: {distributions_reward}")
        # self.logger.info(f"fraction reward: {fraction_reward}")

        # return fraction_reward,distributions_reward
      
    def get_full_reward(self, time_range, real_distribution, current_state):
        action_upper_bound = 1
        is_limit_learner = current_state[-1]
        fraction_val, distributions_val = self.get_partial_reward_values(real_distribution, current_state)
        if fraction_val > action_upper_bound:
            return -(fraction_val-action_upper_bound)*10000
        if is_limit_learner:
            return 1
        # elif fraction_val < 100:
        #     return (fraction_val-100)*10
        alert_val, energy_val, energy_increase, duration_val, duration_increase = self.get_full_reward_values(time_range=time_range)
        self.update_average_values()
        if distributions_val > 0.2:
            return -(abs(duration_val)**2)
        elif distributions_val <= 0.2:
            return duration_val**2
        # return duration_val/distributions_val #max(distributions_val, 0.1)
        # if fraction_val <= action_upper_bound:
        #     if duration_increase > 0.2:
        #         return duration_increase*100
        #     else:
        #         return duration_increase*10
        
            # return energy_val*100
            # if fraction_val == 100:
            #     return energy_val*10
            # else:
            #     return energy_val
        
        # if distributions_val >= 0.2 or fraction_val > 1:
        #     return -(abs(energy_increase)**3)
        # elif distributions_val < 0.2 and fraction_val <= 1:
        #     return energy_increase**3
        # formulation of reward function that maximize the energy while minimizing the difference between the distributions
        # FIX: problem with fraction value
        # if fraction_val > 100:
        #     return -(fraction_val-100)*1000
        # if energy_increase > 0.5:
        #     return energy_val*energy_increase*1000
        # return energy_val*1000
        # if distributions_val >= 0.5:
        #     return -distributions_val*100
        # if energy_increase > 0:
        #     return (energy_increase*100)**2
        # else:
        #     return -((energy_increase*100)**2)
                    
        # if  (distributions_val >= 0.2 and energy_increase < 0) or (distributions_val < 0.2):
        #     return energy_increase**3

        # if alert_val > 0 or fraction_val > 1:
        #     return -100
        # if energy_increase > 0 and distributions_val <= 0.1:
        #     return energy_increase**3
        # elif energy_increase > 0 and distributions_val > 0.1:
        #     return energy_increase
        # elif energy_increase <= 0 and distributions_val <= 0.1:
        #     return energy_increase
        # elif energy_increase < 0 and distributions_val > 0.1:
        #     return energy_increase**3
        # else:
        #     return 0
        ###########
        # if alert_val:
        #     alert_reward = 0
        # else:
        #     alert_reward = 1
        # if energy_increase > 0.4:
        #     energy_reward = 1000*energy_increase
        # else:   
        #     energy_reward = energy_increase*100
            
        # self.reward_dict['energy'].append(energy_reward)
        # self.reward_dict['alerts'].append(alert_reward)
        # self.logger.info(f"energy reward: {energy_reward}")
        # self.logger.info(f"alert reward: {alert_reward}")
        # return alert_reward, energy_reward
    def update_average_values(self):
        self.average_energy = sum(self.reward_values_dict['energy'])/len(self.reward_values_dict['energy'])
        self.average_alert = sum(self.reward_values_dict['alerts'])/len(self.reward_values_dict['alerts'])
        self.average_duration = sum(self.reward_values_dict['duration'])/len(self.reward_values_dict['duration'])
        self.logger.info(f"average energy: {self.average_energy}")
        self.logger.info(f"average alert: {self.average_alert}")
        self.logger.info(f"average duration: {self.average_duration}")
        
    def get_full_reward_values(self, time_range):
        self.logger.info('wait til next rule frequency')
        self.dt_manager.wait_til_next_rule_frequency(self.rule_frequency)
        duration_val, energy_val = self.measure(time_range=time_range, time_delta=self.rule_frequency)
        if self.reward_values_dict['num_of_rules'][-1] < self.num_of_searches:
            duration_val = self.average_duration
            energy_val = self.average_energy
        alert_val = 0#self.splunk_tools.get_alert_count(time_range) #TODO: change to get_alert_count
        if self.average_energy == 0:
            energy_increase = 0
        else:
            energy_increase = (energy_val - self.average_energy)/self.average_energy
        if self.average_duration == 0:
            duration_increase = 0
        else:
            duration_increase = (duration_val - self.average_duration)/self.average_duration
        # self.previous_energy = energy_val        
        self.reward_values_dict['alerts'].append(alert_val)
        self.logger.info(f"incease in energy: {energy_increase}")
        self.logger.info(f"incease in duration: {duration_increase}")
        self.logger.info(f"alert value: {alert_val}")
        return alert_val, energy_val, energy_increase, duration_val, duration_increase
    

    def get_partial_reward_values(self, real_distribution, current_state):
        action_upper_bound = 1
        fraction_val = action_upper_bound - float(current_state[-2])
        distributions_val = self.compare_distributions(real_distribution, current_state[:len(self.relevant_logtypes)])     
        distributions_val = distributions_val + 0.000000000001
        self.reward_values_dict['distributions'].append(distributions_val)
        self.reward_values_dict['fraction'].append(fraction_val)
        self.logger.info(f"distributions value: {distributions_val}")
        self.logger.info(f"fraction value: {fraction_val}")
        return fraction_val,distributions_val
        
    def measure(self, time_range, time_delta=60): #tool
        # self.measurement()
        # run the measurement script with subprocess
        self.logger.info('measuring')
        cmd = subprocess.run('python ../Scanner/scanner.py', shell=True, capture_output=True, text=True)
        self.logger.info(cmd.stdout)
        self.logger.info(cmd.stderr)
        # find the latest measurement folder
        measurement_num = max([int(folder.split(' ')[1]) for folder in os.listdir(PATH) if folder.startswith('Measurement')])
        self.current_measurement_path = os.path.join(PATH,f'Measurement {measurement_num}')
        rules_enegry_df = self.get_rule_total_energy(time_delta, time_range)
        current_energy = rules_enegry_df['CPU(J)'].sum()
        duration = rules_enegry_df['run_duration'].sum()
        self.reward_values_dict['energy'].append(current_energy)
        self.reward_values_dict['duration'].append(duration)
        self.logger.info(f"energy value: {current_energy}")
        return duration, current_energy
    
  
    
    def get_rule_total_energy(self, time_delta, time_range):
        # exec_time = time_range[-1]
        # exec_time_timestamp = datetime.datetime.strptime(exec_time, '%m/%d/%Y:%H:%M:%S').timestamp()
        rules_energy_df = self.extract_energy_per_rule(time_delta) 
        rule_duration = rules_energy_df.groupby('name')['run_duration'].mean()      
        self.energy_equation(rules_energy_df)
        rule_total_energy = rules_energy_df.groupby('name')[['CPU(J)']].sum().reset_index()
        rule_total_energy = pd.merge(rule_total_energy, rule_duration, left_on='name', right_on='name')
        rule_total_energy.to_csv(os.path.join(self.current_measurement_path, 'grouped_rules_energy.csv'))
        rule_total_energy_dict = rule_total_energy[['name', 'CPU(J)', 'run_duration']].to_dict('records')
        # rule_total_energy_dict['time_range'] = str(time_range)
        self.time_rules_energy.append({'time_range':str(time_range), 'rules':rule_total_energy_dict})
        return rule_total_energy


    def energy_equation(self, rules_energy_df):
        # rules_energy_df['delta_time'] = rules_energy_df.groupby('name')["Time(sec)"].diff().dt.total_seconds().fillna(0)
        rules_energy_df['CPU(W)'] = rules_energy_df['CPU(%)'] * CPU_TDP / 100
        rules_energy_df['CPU(J)'] = rules_energy_df['CPU(W)'] * rules_energy_df['delta_time']
    
    def extract_energy_per_rule(self, time_delta): 
        pids_energy_df = self.fetch_energy_data()
        rules_pids_df = self.get_rules_data(time_delta)
        rules_energy_df = self.merge_energy_and_rule_data(pids_energy_df, rules_pids_df)
        num_of_rules = len(rules_energy_df['name'].unique())
        self.reward_values_dict['num_of_rules'].append(num_of_rules)
        self.logger.info(f"num of extracted rules data: {num_of_rules}")
        rules_energy_df.to_csv(os.path.join(self.current_measurement_path, 'rules_energy.csv'))
        return rules_energy_df

    def merge_energy_and_rule_data(self, pids_energy_df, rules_pids_df):
        splunk_pids_energy_df = pids_energy_df[pids_energy_df['PID'].isin(rules_pids_df.pid.values)].sort_values('Time(sec)') 
        rules_energy_df = pd.merge(splunk_pids_energy_df, rules_pids_df, left_on='PID', right_on='pid')
        # create a new column for the time interval
        rules_energy_df["Time(sec)"] = pd.to_datetime(rules_energy_df["Time(sec)"])
        rules_energy_df = rules_energy_df.sort_values(by=['name', "Time(sec)"])
        return rules_energy_df

    def get_rules_data(self, time_delta):
        rules_pids = self.splunk_tools.get_rules_pids(time_delta)
        data = []
        for name, rules in rules_pids.items():
            for e in rules:
                sid, pid, time, run_duration, total_events, total_run_time = e 
                data.append((name, sid, pid, time, run_duration, total_events, total_run_time)) 
        rules_pids_df = pd.DataFrame(data, columns=['name', 'sid', 'pid', 'time', 'run_duration', 'total_events', 'total_run_time'])
        rules_pids_df.time = pd.to_datetime(rules_pids_df.time)
        rules_pids_df.sort_values('time', inplace=True)
        return rules_pids_df

    def fetch_energy_data(self):
        processes_data = pd.read_csv(os.path.join(self.current_measurement_path, 'processes_data.csv'))
        
        time_differences = []
        previous_time = None
        for time in processes_data['Time(sec)'].unique():
            if previous_time is None:
                previous_time = time
            for i in processes_data[processes_data['Time(sec)'] == time].index:
                time_differences.append(time - previous_time)
            previous_time = time
        processes_data['delta_time'] = time_differences
        
        processes_data['Time(sec)'] = pd.to_datetime(processes_data['Time(sec)'], unit='s').dt.to_pydatetime()
        return processes_data

             
    def compare_distributions(self, dist1, dist2):#tool
        # Placeholder for your distribution comparison function
        # This could use a metric like KL divergence
        self.logger.info(f"dist1: {dist1}")
        self.logger.info(f"dist2: {dist2}")
        epsilon = 1e-10  # or any small positive value
        dist1 = [max(prob, epsilon) for prob in dist1]
        dist2 = [max(prob, epsilon) for prob in dist2]
        return entropy(dist1, dist2)
        
        # return wasserstein_distance(dist1, dist2)