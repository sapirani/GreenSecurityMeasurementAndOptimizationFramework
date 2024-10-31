from abc import ABC, abstractmethod
import gym
from gym import spaces
import numpy as np
import logging
logger = logging.getLogger(__name__)

class ActionStrategy(ABC):
    @abstractmethod
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator):
        self.relevant_logtypes = relevant_logtypes
        self.action_upper_bound = action_upper_bound
        self.step_size = step_size
        self.action_duration = action_duration
        self.splunk_tools_instance = splunk_tools_instance
        self.log_generator = log_generator
        self.current_episode_accumulated_action = np.zeros(((len(self.relevant_logtypes)-1)*2+1,))

    @abstractmethod
    def create_action_space(self):
        pass

    @abstractmethod
    def preprocess_action(self, action):
        pass

    @abstractmethod
    def perform_action(self, env, action):
        pass 
    
    def perform_act(self, time_range, i, istrigger, absolute_act):
        logtype = self.relevant_logtypes[i]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        self.current_episode_accumulated_action[i*2+istrigger] += absolute_act
        fake_logs = self.log_generator.generate_logs(logsource, eventcode, istrigger,time_range, absolute_act)
        self.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
        logger.debug(f"inserted {len(fake_logs)} logs of type {logsource} {eventcode} {istrigger}")
    
    @abstractmethod
    def reset(self):
        pass
    
    def get_step_size(self):
        return self.step_size

    def get_action_duration(self):
        return self.action_duration
    
    def check_done(self):
        pass
    
    def reset(self):
        self.current_episode_accumulated_action = np.zeros(((len(self.relevant_logtypes)-1)*2+1,))
        
class ActionStrategy0(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator)


    def create_action_space(self):
        return spaces.Box(low=0, high=self.action_upper_bound, shape=((len(self.relevant_logtypes)-1)*2+1,), dtype=np.float64)
    
    def preprocess_action(self, action):
        return action
    
    def perform_action(self, action, time_range):
        logger.debug(f"performing action {action}")
        logger.debug(f"Sum of action: {sum(action)}")
        super().perform_action(action, time_range)
        for i, logtype in enumerate(self.relevant_logtypes):
            for istrigger in range(2):
                act = action[i*2+istrigger]
                absoulte_act = int(act*self.step_size)
                if act:
                    self.perform_act(time_range, i, istrigger, absoulte_act)
                if i == len(self.relevant_logtypes)-1:
                    break



class ActionStrategy1(ActionStrategy0):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator)

    def preprocess_action(self, action):
        action_norm_factor = sum(action)
        if action_norm_factor > 0:
            action /= action_norm_factor
        return action

    def perform_action(self, action, time_range):
        logger.debug(f"performing action {action}")
        action = self.preprocess_action(action)
        super().perform_action(action, time_range)

class ActionStrategy2(ActionStrategy0):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator)
    
    def preprocess_action(self, action):
        pass
    
    def perform_action(self, action, time_range):
        logger.debug(f"performing action {action}")
        super().perform_action(action, time_range) 
        if sum(action) > 1:
            return
        super().perform_action(action, time_range)
        
class ActionStrategy3(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator)


    def create_action_space(self):
        return spaces.MultiDiscrete([self.step_size, (len(self.relevant_logtypes)-1)*2+1])
    
    def preprocess_action(self, action):
        return action
    
    def perform_action(self, action, time_range):
        logger.debug(f"performing action {action}")
        logger.debug(f"Sum of action: {sum(action)}")
        super().perform_action(action, time_range)
        act = action[0]
        i = action[1]//2
        istrigger = action[1]%2
        self.perform_act(time_range, i, istrigger, act)
        
class ActionStrategy4(ActionStrategy0):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator)

    def create_action_space(self):
        return spaces.MultiBinary((len(self.relevant_logtypes)-1)*2+1)
    
    def preprocess_action(self, action):
        return action
    
    def perform_action(self, action, time_range):
        logger.debug(f"performing action {action}")
        logger.debug(f"Sum of action: {sum(action)}")
        super().perform_action(action, time_range)
        for i, logtype in enumerate(self.relevant_logtypes):
            for istrigger in range(2):
                act = action[i*2+istrigger]
                if act:
                    self.perform_act(time_range, i, istrigger, self.step_size)
                if i == len(self.relevant_logtypes)-1:
                    break