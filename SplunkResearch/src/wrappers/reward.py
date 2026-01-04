from ast import Tuple
import asyncio
from curses import raw
from os import path
import pickle
import random
from time import sleep
import gymnasium as gym
from gymnasium.core import RewardWrapper
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import sys

from traitlets import Bool
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src')

from env_utils import *
import logging
logger = logging.getLogger(__name__)
from time_manager import TimeWindow
std = 13

class AlertPredictor:
    """Separate class to handle alert prediction logic"""
    
    def __init__(self, expected_alerts: Dict, epsilon: float = 1):
        self.expected_alerts = expected_alerts
        self.epsilon = epsilon
        self.current_alerts = {}
        
    def predict_alert_reward(self, rule_name: str, baseline_alert: float, 
                           diversity_logs: Dict, section_logtypes: Dict, 
                           is_mock: bool = False) -> float:
        """Predict alert reward for a specific rule"""
        relevant_log = section_logtypes.get(rule_name, None)
        if relevant_log:
            relevant_log = "_".join(relevant_log[0]) + "_1"
            diversity = diversity_logs.get(relevant_log, 0)
            
            if rule_name in ['ESCU Windows Rapid Authentication On Multiple Hosts Rule']:
                predicted_current = baseline_alert
            else:
                predicted_current = baseline_alert + diversity
        else:
            predicted_current = baseline_alert

        
        self.current_alerts[rule_name] = predicted_current
        expected = self.expected_alerts.get(rule_name, 0)
        gap = max(0, predicted_current - expected)
        reward = -(gap) / (expected + self.epsilon)
        
        return reward
    
    def predict_overall_alert_reward(self, baseline_metrics: Dict, diversity_logs: Dict,
                                   section_logtypes: Dict, is_mock: bool = False, normalized_distribution: Optional[np.ndarray] = np.array([0.0])) -> float:
        """Predict overall alert reward for all rules"""
        rewards = []
        baseline_alerts = {rule: baseline_metrics.get(rule, {}).get('alert', 0) for rule in self.expected_alerts}
        for rule_name, expected in self.expected_alerts.items():
            baseline_alert = baseline_metrics.get(rule_name, {}).get('alert', 0)
            reward = self.predict_alert_reward(
                rule_name, baseline_alert, diversity_logs, 
                section_logtypes, is_mock
            )
            rewards.append(reward)
            
        # return np.mean(rewards) if rewards else 0
        # return sum(self.current_alerts.values())
        normal_alert_rate = sum(baseline_alerts.values())
        return min(-(sum(self.current_alerts.values()) - normal_alert_rate  + self.epsilon)/ (normal_alert_rate + self.epsilon), 0)
        # return min(-(sum(self.current_alerts.values()) - 7 )/ (7 + self.epsilon), 0)
        # return min(-(sum(self.current_alerts.values()) - sum(self.expected_alerts.values()))/ (sum(self.expected_alerts.values()) + self.epsilon), 0)


class BaseRuleExecutionWrapperWithPrediction(RewardWrapper):
    """Enhanced base wrapper with alert prediction capability"""
    
    def __init__(self, env, is_mock: bool = False,
                 enable_prediction: bool = True, alert_threshold: float = -0.5,
                 skip_on_low_alert: bool = True, use_energy: bool = True,use_alert: bool = True, is_eval: bool = False, is_train: bool = False, beta: float = 0.33, gamma: float = 0.33):
        super().__init__(env)

        self.is_mock = is_mock
        
        # Prediction configuration
        self.enable_prediction = enable_prediction
        self.alert_threshold = alert_threshold
        self.skip_on_low_alert = skip_on_low_alert
        

        self.use_energy = use_energy
        self.use_alert = use_alert
        self.energy_models = {}
        self.beta = beta
        self.gamma = gamma
        self.injection_id = 0
        self.baseline_measured = False
        # if is_mock:
        #     for rule in self.unwrapped.splunk_tools.active_saved_searches:
        #         self.energy_models[rule] = pickle.load(open(f"/home/shouei/GreenSecurity-FirstExperiment/baseline_splunk_train-v32_2880_cpu_regressor_results/RandomForestRegressor_{rule}_with alert = 0.pkl", "rb"))
        
        # Initialize alert predictor
        self.expected_alerts = {
            'ESCU Windows Rapid Authentication On Multiple Hosts Rule': 0,
            'Windows AD Replication Request Initiated from Unsanctioned Location': 0,
            'Windows Event For Service Disabled': 2.5,
            'Detect New Local Admin Account': 0.7,
            'ESCU Network Share Discovery Via Dir Command Rule': 0,
            'Known Services Killed by Ransomware': 6,
            'Non Chrome Process Accessing Chrome Default Dir': 0,
            'Kerberoasting SPN Request With RC4 Encryption': 0,
            'Clop Ransomware Known Service Name': 0
        }
        self.measuring = False
        self.alert_predictor = AlertPredictor(self.expected_alerts)
        self.execution_decisions = []
        self.is_eval = is_eval
        self.is_train = is_train
        # lode joblib models for energy consumption for each rule
        # for rule in self.expected_alerts:
        model_path = f"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/mlp_models_all_rules_cpu.joblib"
        # model_path = f"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/models_all_rules_cpu.joblib"
        # model_path = f"/home/shouei/GreenSecurity-FirstExperiment/model_{rule}.joblib"
        if path.exists(model_path):
            self.energy_models['all'] = joblib.load(model_path)
            # self.energy_models[rule] = joblib.load(model_path)
        self.distributions = []
        self.alerts = []
        self.epsilon = .00000001


  
    def get_baseline_data(self, time_range: TimeWindow, rerun=False) -> Dict:
        """Get baseline data - execute after cleaning if needed"""
        num_of_measurements = self.unwrapped.config.baseline_num_of_measurements
        relevant_rows = self.unwrapped.baseline_df[(self.unwrapped.baseline_df['start_time'] == time_range[0]) & (self.unwrapped.baseline_df['end_time'] == time_range[1])]
        actual_num_of_measurements = relevant_rows.groupby(['start_time', 'end_time', 'search_name']).size().values[0] if not relevant_rows.empty else 0
        needed_measurements = num_of_measurements - actual_num_of_measurements
        
        # check if some search is missing
        running_dict = {}
        existing_searches = set()
        if len(relevant_rows) != 0:
            existing_searches = set(relevant_rows.reset_index()['search_name'].values)
            running_dict.update({search: needed_measurements for search in existing_searches})
        missing_searches = set([search for search in self.unwrapped.splunk_tools.active_saved_searches]) - existing_searches
        running_dict.update({search: num_of_measurements for search in missing_searches})
        if rerun:
            running_dict = {search: num_of_measurements for search in self.unwrapped.splunk_tools.active_saved_searches}

        if self.use_energy or self.use_alert :
            if sum(running_dict.values()) > 0:
                logger.info(f"Need to run baseline for {running_dict}")
                empty_monitored_files(SYSTEM_MONITOR_FILE_PATH)
                empty_monitored_files(SECURITY_MONITOR_FILE_PATH)
                logger.info('Cleaning the environment')
                clean_env(self.unwrapped.splunk_tools, time_range)
                logger.info('Measure no agent reward values')
                logger.info('wait for the environment to be cleaned')
                sleep(3)
                                
                logger.info(f"Running {running_dict}")
                # Execute rules and get metrics
                # if self.is_mock:
                #     rules_metrics = self.mock_rules_metrics(time_range)
                # else:
                rules_metrics = asyncio.run(self.unwrapped.splunk_tools.run_saved_searches(time_range))
                new_lines = self.convert_metrics(time_range, rules_metrics)
                if len(new_lines) != 0:
                    if rerun:
                        # remove existing rows for the time range
                        self.unwrapped.baseline_df = self.unwrapped.baseline_df[~((self.unwrapped.baseline_df['start_time'] == time_range[0]) & (self.unwrapped.baseline_df['end_time'] == time_range[1]))]
                    self.unwrapped.baseline_df = pd.concat([self.unwrapped.baseline_df, pd.DataFrame(new_lines)])
                self.baseline_measured = True            

                
        random_val = np.random.randint(0, 10)
        if random_val % 3 == 0 and not self.is_mock:
            self.unwrapped.baseline_df.to_csv(self.unwrapped.baseline_path, index=False)
            
        relevant_rows = self.unwrapped.baseline_df[(self.unwrapped.baseline_df['start_time'] == time_range[0]) & (self.unwrapped.baseline_df['end_time'] == time_range[1])]
        
        return relevant_rows
    
    def predict_and_decide_execution(self, time_range: TimeWindow, diversity_logs: Dict, distribution_value: float) -> Tuple[bool, float, Dict]:
        """Predict alert reward and decide whether to execute rules"""
        # First get baseline data (this might trigger cleaning and baseline execution)
        baseline_data = self.get_baseline_data(time_range)
        
        if baseline_data.empty:
            # No baseline data, must execute
            return True, None, {}
            
        # Process baseline metrics
        grouped_baseline = baseline_data.groupby('search_name')
        raw_baseline_metrics, _ = self.process_metrics(grouped_baseline)
        
        # Predict alert reward
        predicted_reward = self.alert_predictor.predict_overall_alert_reward(
            raw_baseline_metrics, 
            diversity_logs,
            self.unwrapped.section_logtypes,
            self.is_mock, 
        )
        
        # Decide whether to execute
        should_execute = True
        if self.enable_prediction and self.skip_on_low_alert:
            # should_execute =  ((predicted_reward <= (sum(self.expected_alerts.values()) + (self.alert_threshold*std))) and (distribution_value < self.env.distribution_threshold) and self.use_energy) or self.is_eval
            should_execute = ((predicted_reward >= self.alert_threshold) and (distribution_value < self.env.distribution_threshold) and self.use_energy) or self.is_eval or True # TRY!!!!!!!
            
            # self.unwrapped.should_delete = should_execute and not self.is_mock and not self.measuring 
            
        # Log decision
        self.execution_decisions.append({
            'time_range': time_range,
            'predicted_reward': predicted_reward,
            'should_execute': should_execute,
            'threshold': self.alert_threshold
        })
        
        logger.info(f"Alert prediction: reward={predicted_reward:.3f}, " f"threshold={self.alert_threshold}, execute={should_execute}")
        
        return should_execute, predicted_reward, raw_baseline_metrics
    
    def mock_rules_metrics(self, time_range: TimeWindow) -> Dict:
        """Mock rules metrics for the given time range"""
        rules_metrics = self.unwrapped.splunk_tools.mock_run_saved_searches(time_range)
        return rules_metrics
    
    def get_current_reward_values(self, time_range: TimeWindow, should_execute: Bool) -> Tuple[pd.DataFrame, Dict]:
        """Get current reward values"""
        if (self.is_mock and not self.measuring) or not should_execute or not self.use_energy or not self.use_alert:
            rules_metrics = self.mock_rules_metrics(time_range)
        else:
            # rules_metrics, total_cpu = self.unwrapped.splunk_tools.run_saved_searches(
            #     time_range, None, self.unwrapped.config.num_of_measurements)
            rules_metrics = asyncio.run(self.unwrapped.splunk_tools.run_saved_searches(
                time_range))
        relevant_rows = self.convert_metrics(time_range, rules_metrics)
        relevant_rows = pd.DataFrame(relevant_rows)
        grouped = relevant_rows.groupby('search_name')
        return self.process_metrics(grouped)
    
    def get_baseline_reward_values(self, time_range: TimeWindow) -> Tuple[pd.DataFrame, Dict]:
        """Get baseline reward values"""
        relevant_rows = self.get_baseline_data(time_range)
        grouped = relevant_rows.groupby('search_name')
        return self.process_metrics(grouped)
    
    def convert_metrics(self, time_range, rules_metrics):
        # logger.info(f"rules_metrics: {rules_metrics}")
        return [{
            'search_name': metric.search_name,
            'alert': metric.results_count,
            'duration': metric.execution_time,
            'cpu': metric.cpu,
            'start_time': metric.start_time,
            'end_time': metric.end_time,
            'read_count': metric.io_metrics['read_count'],
            'write_count': metric.io_metrics['write_count'],
            'read_bytes': metric.io_metrics['read_bytes'],
            'write_bytes': metric.io_metrics['write_bytes'],
            'memory_mb': metric.memory_mb
        } for metric in rules_metrics]

    def process_metrics(self, grouped):
        # print(grouped)
        raw_metrics = {}
        for search_name, group in grouped:
            raw_metrics[search_name] = {
                'duration': group['duration'].mean(),
                'cpu': group['cpu'].mean(),
                'read_count': group['read_count'].mean(),
                'write_count': group['write_count'].mean(),
                'read_bytes': group['read_bytes'].mean(),
                'write_bytes': group['write_bytes'].mean(),
                'memory_mb': group['memory_mb'].mean(),
                'alert': group['alert'].mean()}
                
            if raw_metrics[search_name]['alert'] != round(raw_metrics[search_name]['alert']):
                logger.info(f"Alert value is not an integer: {search_name}, {raw_metrics[search_name]['alert']}, {group['alert']}")
                # choose the measurement with the highest alert value
                max_alert = group['alert'].max()
                max_alert_index = group['alert'].idxmax()
                raw_metrics[search_name]['alert'] = max_alert
                raw_metrics[search_name]['duration'] = group['duration'].loc[max_alert_index]
                raw_metrics[search_name]['cpu'] = group['cpu'].loc[max_alert_index]
                raw_metrics[search_name]['read_count'] = group['read_count'].loc[max_alert_index]
                raw_metrics[search_name]['write_count'] = group['write_count'].loc[max_alert_index]
                raw_metrics[search_name]['read_bytes'] = group['read_bytes'].loc[max_alert_index]
                raw_metrics[search_name]['write_bytes'] = group['write_bytes'].loc[max_alert_index]

        combined_metrics = {
            'duration': sum([metric['duration'] for metric in raw_metrics.values()]),
            'cpu': sum([metric['cpu'] for metric in raw_metrics.values()]),
            'read_count': sum([metric['read_count'] for metric in raw_metrics.values()]),
            'write_count': sum([metric['write_count'] for metric in raw_metrics.values()]),
            'read_bytes': sum([metric['read_bytes'] for metric in raw_metrics.values()]),
            'write_bytes': sum([metric['write_bytes'] for metric in raw_metrics.values()]),
            'memory_mb': sum([metric['memory_mb'] for metric in raw_metrics.values()]),
            'alert': sum([metric['alert'] for metric in raw_metrics.values()])
        }

        
        return raw_metrics, combined_metrics
       
    def reward(self, reward: float) -> float:
        return reward
      
    def step(self, action):
        """Override step to properly handle info updates with prediction"""
        obs, reward, terminated, truncated, info = super().step(action)
        if info.get('done', True) and info.get('distribution_reward') == 0:
            reward = 0

        if info.get('done', True):
            self.baseline_measured = False
            inserted_logs = info.get('inserted_logs', 0)
            diversity_logs = info.get('diversity_episode_logs', {})
            
            # Predict and decide whether to execute
            should_execute, predicted_alert_reward, raw_baseline_metrics = self.predict_and_decide_execution(
                info['current_window'], 
                diversity_logs,
                info.get('ac_distribution_value', 0)
            )
            # # dump baseline metrics to all data (once)
            # if len(self.unwrapped.all_baseline_data) < len(self.unwrapped.baseline_df.time_range.unique()):
            #     self.unwrapped.all_baseline_data.append({
            #         'time_range': info['current_window'],
            #         'ac_real_distribution': self.unwrapped.ac_real_distribution,
            #         'raw_metrics':raw_baseline_metrics,
            #     })
            # else:
            #     # dump and empty the all_data to csv in path if exists
            #     all_baseline_data_df = pd.json_normalize(self.unwrapped.all_baseline_data, sep='_')
            #     all_baseline_data_df.to_csv(self.unwrapped.all_baseline_data_path, index=False, mode='w')          
            #     self.unwrapped.all_baseline_data = []
            
            # Store prediction info
            info['predicted_alert_reward'] = predicted_alert_reward
            info['execution_skipped'] = not should_execute and False # TRY!!!!!!!
            if  ((random.randint(0, 100000) < 10   or self.is_eval) and should_execute and not self.is_mock):
                logger.info(f"Measuring")
                self.measuring = True
            else:
                self.measuring = False
                
            self.unwrapped.should_delete = should_execute and (not self.is_mock or self.measuring)
                
            # self.unwrapped.is_mock = (not should_execute or not self.use_energy or not self.use_alert) 
            time_range = info['current_window']
            time_range_date = (datetime.datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S'), datetime.datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S'))
            # inject logs if not is_mock
            if (not self.is_mock  or self.measuring) and should_execute and self.use_energy and self.use_alert:
                if self.baseline_measured:
                    logger.info(f"Empty deletion dict of log generator and relvant fake_splunk_state")
                    self.unwrapped.log_generator.logs_to_delete = {}
                    for t_r in self.unwrapped.log_generator.fake_splunk_state:
                        t_r_datetime = (datetime.datetime.strptime(t_r[0], '%m/%d/%Y:%H:%M:%S'), datetime.datetime.strptime(t_r[1], '%m/%d/%Y:%H:%M:%S'))
                        if t_r_datetime[0] >= time_range_date[0] and t_r_datetime[1] <= time_range_date[1]:
                            self.unwrapped.log_generator.fake_splunk_state[t_r] = {}
                self.env.env.inject_episodic_logs(self.injection_id) # access to action wrapper0
                self.injection_id += 1
                # wait for the logs to be injected
                # sleep(4)
            should_run = True
            attempt = 0
            stop_loop = False
            # while should_run and not stop_loop:
            #     rerun = False
            #     if attempt > 4:
            #         logger.info(f"Re-running due to mismatch in alerts difference")
            rerun = False
                    
            #     attempt += 1
                # Normal execution flow
                
            raw_metrics, combined_metrics = self.get_current_reward_values(info['current_window'], should_execute)
            # logger.info(f"Raw metrics: {raw_metrics}")
            should_run = False
            baseline_raw_metrics, combined_baseline_metrics = self.process_metrics(
                self.get_baseline_data(info['current_window'], rerun).groupby('search_name')
            )
                # if rerun or self.is_mock:
                #     stop_loop = True

            # find the difference of alerts between raw_metrics and baseline_raw_metrics
            alerts_diff = {rule: raw_metrics[rule]['alert'] - baseline_raw_metrics.get(rule, {}).get('alert', 0) for rule in self.expected_alerts}
            
            # check compatibility of alerts_diff with diversity info
            for rule in self.expected_alerts:
                if rule == 'ESCU Windows Rapid Authentication On Multiple Hosts Rule':
                    continue
                relevant_log = self.unwrapped.section_logtypes.get(rule, None)
                if relevant_log:
                    log_type = "_".join(relevant_log[0]) + "_1"
                    diversity_value = diversity_logs.get(log_type, 0)
                    if alerts_diff[rule] != diversity_value:
                        
                        logger.warning(f"Alert difference mismatch for {rule}: alerts_diff={alerts_diff[rule]}, diversity_value={diversity_value}")
                        # get the field real_ts of the events in the results of the query to find the mismatch
                        def get_event_times(rule_name, time_range):
                            # query splunk for the events in the time range
                            
                            query = f'index={self.unwrapped.splunk_tools.index_name} host IN ("dt-splunk", 132.72.81.150) EventCode={relevant_log[0][1]}  | stats count by real_ts var_id'
                            print(time_range_date)
                            results = self.unwrapped.splunk_tools.run_search(query, *time_range)
                            formatted_log = "\n".join([json.dumps(record) for record in results])
                            logger.info(f"Event times for {rule_name} in {time_range}: {formatted_log}")
                        get_event_times(rule, (time_range_date[0].timestamp(), time_range_date[1].timestamp()))
                    else:
                        logger.info(f"Alert difference match for {rule}: alerts_diff={alerts_diff[rule]}, diversity_value={diversity_value}")
                        
            if self.is_mock and self.measuring and should_execute:  
                combined_metrics['real_cpu'] = combined_metrics['cpu'] 

            info['combined_metrics'] = combined_metrics
            info['combined_baseline_metrics'] = combined_baseline_metrics
            if self.is_mock and (self.use_alert or self.use_energy):
                current_alerts = self.alert_predictor.current_alerts
                # baseline_alerts = {rule: raw_baseline_metrics[rule]['alert'] for rule in self.expected_alerts}
                for rule in self.expected_alerts:
                    raw_metrics[rule]['alert'] = current_alerts[rule]
                    
                    # raw_baseline_metrics[rule]['cpu'] = 0
                    # raw_baseline_metrics[rule]['duration'] = 0
                info['combined_metrics']['alert'] = sum(current_alerts.values())# + sum(baseline_alerts.values())
                # sleep(2)

                
            if should_execute:
                self.unwrapped.all_data.append({
                    'time_range': info['current_window'],
                    'ac_fake_distribution': self.unwrapped.ac_fake_distribution,
                    'raw_metrics':raw_metrics,
                })
                if random.randint(0, 1000) % 10 == 0:
                    # dump and empty the all_data to csv in path if exists
                    if self.unwrapped.all_data:
                        all_data_df = pd.json_normalize(self.unwrapped.all_data, sep='_')
                        all_data_df.to_csv(self.unwrapped.all_data_path, index=False, mode='a', header=not path.exists(self.unwrapped.all_data_path))
                    
                    self.unwrapped.all_data = []
            # Set a penalty reward for skipping
            # info['alert_reward'] = predicted_alert_reward
            if not self.use_alert:
                predicted_alert_reward = 0
                
            ############## Distribution reward ##############    
            # info['ac_distribution_reward'] = -200*(info['ac_distribution_value'] ** 2)
            self.distributions.append(info['ac_distribution_value'])
            self.alerts.append(-predicted_alert_reward)

            
            info['ac_distribution_reward'] = -np.tanh(info['ac_distribution_value']*2)
            
            # info['ac_distribution_reward'] = dist_reward
            # info['ac_distribution_reward'] = 30*info['ac_distribution_value']
            
            ############## Alert reward ##############
            # info['alert_reward'] = ((predicted_alert_reward - sum(self.expected_alerts.values()))/std)
            # self.mean_alert = (self.unwrapped.all_steps_counter//self.unwrapped.total_steps - 1)*self.mean_alert + sum(raw_baseline_metrics[rule]['alert'] for rule in self.expected_alerts)/(self.unwrapped.all_steps_counter//self.unwrapped.total_steps)
            info['alert_reward'] = predicted_alert_reward
            info['norm_alert_reward'] = -np.tanh(-predicted_alert_reward/30)
            ############### Total reward ##############
            reward = self.gamma*info['ac_distribution_reward']
            reward += self.beta*info['norm_alert_reward']
            # reward = 0
            # if info.get('ac_distribution_value', 0) > self.env.distribution_threshold or predicted_alert_reward < self.alert_threshold:
            #     if info.get('ac_distribution_value', 0) > self.env.distribution_threshold:
            #         reward += 0.2*info['ac_distribution_reward']
            #         # reward += -(2**info['ac_distribution_reward'])
                        
            #     if predicted_alert_reward < self.alert_threshold:
            #         reward += 0.2*info['alert_reward']
            #         # reward += -(2**info['alert_reward'])
            # else:
            #     reward = 1

            
            # Store in info for other wrappers to use
            info['raw_metrics'] = raw_metrics
            info['raw_baseline_metrics'] = raw_baseline_metrics
            
        return obs, reward, terminated, truncated, info
    
class EnergyRewardWrapper(RewardWrapper):
    ENERGY_CHANGE_TARGET = 1  # Target energy change for normalization
    """Wrapper for energy consumption rewards"""
    def __init__(self, env: gym.Env, alpha: float = 0.5,is_mock: bool = False):
        super().__init__(env)
        self.alpha = alpha
        self.is_mock = is_mock
        self.energies = []
        self.epsilon = 1e-8
    
    def estimate_energy_consumption(self, fake_dist, rules_alerts):
        # Placeholder for energy consumption estimation logic
        # This should return the estimated energy consumption for the current state
        # Use the energy model to estimate energy consumption
        model = self.env.energy_models["all"]
        # normalize the fake distribution and rule alert
        if not isinstance(fake_dist, np.ndarray):
            fake_dist = np.array(list(fake_dist)[:-1])
        rules_alerts_array = np.array(list(rules_alerts.values()))
        if fake_dist.ndim == 1:
            fake_dist = fake_dist.reshape(-1, 1)
        if rules_alerts_array.ndim == 1:
            rules_alerts_array = rules_alerts_array.reshape(-1, 1)
        # print(fake_dist, rules_alerts[rule])
        
        fake_dist = (fake_dist) /237158
        rules_alerts_array = (rules_alerts_array - 0.0) / 38
        X = np.concatenate((fake_dist, rules_alerts_array)).flatten().reshape(1, -1)
        
        estimated_energy = model.predict(X)[0]
        return estimated_energy

    
    def step(self, action):
        """Override step to properly handle info updates"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get('done', True):
            if self.is_mock and not info.get('execution_skipped', False):  
                rules_alerts = {rule: info['raw_metrics'][rule]['alert'] for rule in self.unwrapped.splunk_tools.active_saved_searches}                                         
                estimated_energy = self.estimate_energy_consumption(
                    self.unwrapped.ac_fake_distribution.values(),
                    rules_alerts
                )
                # for rule in self.unwrapped.splunk_tools.active_saved_searches:
                #     info['raw_metrics'][rule]['cpu'] = cpu_dict[rule]
                info['combined_metrics']['cpu'] = estimated_energy
            current = info['combined_metrics']['cpu']

            baseline = info['combined_baseline_metrics']['cpu']

            
            energy_reward = (current  - baseline) / (baseline + self.epsilon)
            self.energies.append(energy_reward)

            # if energy_reward <= 0.1:
            #     energy_reward = 0
            # energy_reward = np.clip(energy_reward, 0, 1) # Normalize to [0, 1]
            info['energy_reward'] = energy_reward
            # mean_energy = np.mean(self.energies) if len(self.energies) > 0 else 0
            # std_energy = np.std(self.energies) if len(self.energies) > 0 else 0
            
            info['norm_energy_reward'] = np.clip(energy_reward, 0, 1)
            # info['norm_energy_reward'] = np.clip(energy_reward/EnergyRewardWrapper.ENERGY_CHANGE_TARGET, 0, 1)
            logger.info(f"Energy reward: {energy_reward:.3f}, current: {current:.3f}, baseline: {baseline:.3f}")
            # reward +=  energy_reward
            # reward += self.alpha*energy_reward
            # if reward != 1:
            #     return obs, reward, terminated, truncated, info

            reward += self.alpha*info['norm_energy_reward']
            # reward += 0.6*max(energy_reward, 0)
    

        # if info.get('done', True):
        #     if self.is_mock and not info.get('execution_skipped', False):  
        #         rules_alerts = {rule: info['raw_metrics'][rule]['alert'] for rule in self.unwrapped.splunk_tools.active_saved_searches}                                         
        #         estimated_energy = self.estimate_energy_consumption(
        #             self.unwrapped.ac_fake_distribution.values(),
        #             rules_alerts
        #         )
        #         # for rule in self.unwrapped.splunk_tools.active_saved_searches:
        #         #     info['raw_metrics'][rule]['cpu'] = cpu_dict[rule]
        #         info['combined_metrics']['cpu'] = estimated_energy
        #     current = info['combined_metrics']['cpu']

        #     baseline = info['combined_baseline_metrics']['cpu']

            
        #     energy_reward = (current  - baseline) / baseline
        #     # if energy_reward <= 0.1:
        #     #     energy_reward = 0
        #     # energy_reward = np.clip(energy_reward, 0, 1) # Normalize to [0, 1]
        #     info['energy_reward'] = energy_reward
        #     logger.info(f"Energy reward: {energy_reward:.3f}, current: {current:.3f}, baseline: {baseline:.3f}")
        #     # reward +=  energy_reward
        #     # reward += self.alpha*energy_reward
        #     if reward != 1:
        #         return obs, reward, terminated, truncated, info
        #     reward = max(energy_reward, 0) * 100
        #     if energy_reward > 0.6:
        #         reward **= 3
        #     elif energy_reward > 0.4:
        #         reward **= 2
            
            # reward += self.unwrapped.total_steps*self.alpha*energy_reward
        # reward = energy_reward/(reward + self.epsilon)
            # reward += self.alpha * energy_reward
            
            
        return obs, reward, terminated, truncated, info

class DistributionRewardWrapper(RewardWrapper):
    """Wrapper for distribution similarity rewards"""
    def __init__(self, env: gym.Env, gamma: float = 0.2, epsilon: float = 1e-8, distribution_freq: int = 3, distribution_threshold: float = 0.22):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.distribution_reward_freq = distribution_freq
        self.distribution_threshold = distribution_threshold #0.18 #0.22
        
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)    
        # self.update_fake_distribution(self.episode_logs)

        step_counter = info.get('step', 0)
        # if  random.randint(0, 10) % self.distribution_reward_freq == 0:
        #     info['distribution_value'] = dist_value
        #     dist_reward = self._calculate_distribution_reward(dist_value)
        #     # dist_reward /= 0.6 # NOrmalize the reward
        #     info['distribution_reward'] = dist_reward
        #     # reward += dist_reward
        #     reward += dist_reward

                # reward += self.gamma * dist_reward
        # if info.get('done', True):
        dist_value = self._calculate_distribution_value(
            self.unwrapped.ac_real_state,
            self.unwrapped.ac_fake_state
        )
        info['ac_distribution_value'] = dist_value
        # if not info.get('done', True):
            # dist_value = self._calculate_distribution_value(
            #     self.unwrapped.real_state,
            #     self.unwrapped.fake_state
            # )
            # reward = -self.unwrapped.step_counter*(dist_value ** 2)
            # dist_reward = self._calculate_distribution_reward(dist_value)
            # dist_reward /= 0.6 # NOrmalize the reward
        #     info['ac_distribution_reward'] = dist_reward
        #     # reward += self.gamma*dist_reward
        #     if reward < -1:
        #         return obs, reward, terminated, truncated, info
        #     if dist_value > self.distribution_threshold:
        #         reward = -dist_value*100
        #         reward = dist_reward
        #     else:
        #         reward = dist_reward

        # # since this is the last wrapper, we can consider it as final reward
        # logger.info(f"Reward: {reward}")  
        return obs, reward, terminated, truncated, info
    
    def _calculate_distribution_reward(self, distribution_value: float) -> float:
        # return 0.1 * np.log(distribution_value+self.epsilon) - distribution_value**2
        # d_target = 0.2
        # if distribution_value > 1.5*d_target:
        #     self.gamma *= 1.5
        # elif distribution_value < 0.5*d_target:
        #     self.gamma *= 0.5
        # return -self.gamma * distribution_value
        return -distribution_value
        # return -distribution_value/0.26
        # return -(distribution_value*100)*3
        # return np.log(1-distribution_value)
     
    def _calculate_distribution_value(self, real_dist, fake_dist):
        # Add epsilon and normalize
        real_dist = (real_dist + self.epsilon) / np.sum(real_dist + self.epsilon)
        fake_dist = (fake_dist + self.epsilon) / np.sum(fake_dist + self.epsilon)
        
        # Calculate JSD
        # m = (real_dist + fake_dist) / 2
        # jsd = (self._kl_divergence(real_dist, m) + 
        #        self._kl_divergence(fake_dist, m)) / 2
        # return jsd
        # Calculate KL divergence
        return self._kl_divergence(real_dist , fake_dist)
        # return self.chi_square(fake_dist, real_dist)
        
    def _kl_divergence(self, p, q):
        return np.sum(p * np.log(p / q))
    
    def chi_square(self, p, q):
        return np.sum((p - q) ** 2 / (p + q + self.epsilon))

