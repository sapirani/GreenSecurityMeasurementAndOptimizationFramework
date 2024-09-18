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

    @abstractmethod
    def create_action_space(self):
        pass

    @abstractmethod
    def preprocess_action(self, action):
        pass

    @abstractmethod
    def perform_action(self, env, action):
        pass
    
    @abstractmethod
    def perform_act(self, time_range, i, istrigger, act):
        pass

class ActionStrategy1(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator)
        self.current_episode_accumulated_action = np.zeros(((len(self.relevant_logtypes)-1)*2+1,))


    def create_action_space(self):
        return spaces.Box(low=0, high=self.action_upper_bound, shape=((len(self.relevant_logtypes)-1)*2+1,), dtype=np.float64)

    def preprocess_action(self, action):
        action_norm_factor = sum(action)
        if action_norm_factor > 0:
            action /= action_norm_factor
        return action

    def perform_action(self, action, time_range):
        action = self.preprocess_action(action)
        for i, logtype in enumerate(self.relevant_logtypes):
            for istrigger in range(2):
                act = action[i*2+istrigger]
                if act:
                    self.perform_act(time_range, i, istrigger, act)
                if i == len(self.relevant_logtypes)-1:
                    break
    
    def perform_act(self, time_range, i, istrigger, act):
        logtype = self.relevant_logtypes[i]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        absolute_act = int(act*self.step_size)
        self.current_episode_accumulated_action[i*2+istrigger] += absolute_act
        fake_logs = self.log_generator.generate_logs(logsource, eventcode, istrigger,time_range, absolute_act)
        self.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
        logger.debug(f"inserted {len(fake_logs)} logs of type {logsource} {eventcode} {istrigger}")


    def get_step_size(self):
        return self.step_size

    def get_action_duration(self):
        return self.action_duration
    
class ActionStrategy2(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator)
        self.current_episode_accumulated_action = np.zeros(((len(self.relevant_logtypes)-1)*2+1,))
    
    def create_action_space(self):
        # action is a tuple of (logtype, istrigger, act)
        return spaces.Tuple([spaces.Discrete(len(self.relevant_logtypes)), spaces.Discrete(2), spaces.Box(low=0, high=self.action_upper_bound, shape=(1,), dtype=np.float64)])
    
    def perform_action(self, action, time_range):
        logtype, istrigger, act = action
        if logtype == len(self.relevant_logtypes)-1 and istrigger == 1:
            return
        logsource = self.relevant_logtypes[logtype][0].lower()
        eventcode = self.relevant_logtypes[logtype][1]
        absolute_act = int(act*self.step_size)
        self.current_episode_accumulated_action[logtype*2+istrigger] += absolute_act
        fake_logs = self.log_generator.generate_logs(logsource, eventcode, istrigger, time_range, absolute_act)
        self.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
        logger.debug(f"inserted {len(fake_logs)} logs of type {logsource} {eventcode} {istrigger}")