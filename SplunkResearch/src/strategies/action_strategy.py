from abc import ABC, abstractmethod
import time
import gym
from gym import spaces
import numpy as np
import logging
logger = logging.getLogger(__name__)

class ActionStrategy(ABC):
    @abstractmethod
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator, remaining_quota):
        self.relevant_logtypes = relevant_logtypes
        self.action_upper_bound = action_upper_bound
        self.step_size = step_size
        self.action_duration = action_duration
        self.splunk_tools_instance = splunk_tools_instance
        self.log_generator = log_generator
        self.current_episode_accumulated_action = np.zeros(((len(self.relevant_logtypes)-1)*2+1,))
        self.quota = remaining_quota
        self.remaining_quota = remaining_quota
        self.should_delete = False
        
    @abstractmethod
    def create_action_space(self):
        pass

    @abstractmethod
    def preprocess_action(self, action):
        pass

    @abstractmethod
    def perform_action(self, env, action):
        pass 
    
    def perform_act(self, time_range, i, istrigger, absoulte_act):
        logtype = self.relevant_logtypes[i]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        self.current_episode_accumulated_action[i*2+istrigger] += absoulte_act
        fake_logs = self.log_generator.generate_logs(logsource, eventcode, istrigger,time_range, absoulte_act)
        self.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
        logger.info(f"inserted {len(fake_logs)} logs of type {logsource} {eventcode} {istrigger}")
        
    
    def get_step_size(self):
        return self.step_size

    def get_action_duration(self):
        return self.action_duration
    
    
    def reset(self):
        self.current_episode_accumulated_action = np.zeros(((len(self.relevant_logtypes)-1)*2+1,))
        self.remaining_quota = self.quota
        self.should_delete = False
        
        
class ActionStrategy0(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota)


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
                    return 0



class ActionStrategy1(ActionStrategy0):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator, remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator, remaining_quota)

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
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota)
    
    def preprocess_action(self, action):
        pass
    
    def perform_action(self, action, time_range):
        logger.debug(f"performing action {action}")
        if sum(action) > 1:
            return 1
        super().perform_action(action, time_range)
        
class ActionStrategy3(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator, remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota)


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
        return 0
    
class ActionStrategy4(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota)

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
                    return 0

class ActionStrategy5(ActionStrategy0):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota)
    
    def preprocess_action(self, action):
        return action
    
    def perform_act(self, time_range, i, istrigger, absoulte_act):
        logtype = self.relevant_logtypes[i]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        if self.remaining_quota >= 0:
            fake_logs = self.log_generator.generate_logs(logsource, eventcode, istrigger,time_range, absoulte_act)
            self.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
            logger.info(f"inserted {len(fake_logs)} logs of type {logsource} {eventcode} {istrigger}")
            self.should_delete = True
        
    def perform_action(self, action, time_range):
        logger.info(f"performing action {action}")
        logger.debug(f"Sum of action: {sum(action)}")
        # if self.remaining_quota - self.quota * sum(action) < 0:
        #     self.remaining_quota -= self.quota * sum(action)
        #     return 1
        for i, logtype in enumerate(self.relevant_logtypes):
            for istrigger in range(2):
                act = action[i*2+istrigger]
                absoulte_act = int(act*self.quota)
                self.remaining_quota -= absoulte_act
                self.current_episode_accumulated_action[i*2+istrigger] += absoulte_act
                
                if act and sum(action) <= 1 and self.remaining_quota >= 0:
                    self.perform_act(time_range, i, istrigger, absoulte_act)
                if i == len(self.relevant_logtypes)-1:
                    if sum(action) <= 1 and self.remaining_quota >= 0:
                        return 0
                    else:
                        break
        if self.remaining_quota < 0 or sum(action) > 1:
            return 1



class ActionStrategy6(ActionStrategy0):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota)
    
    def preprocess_action(self, action):
        return action 
    
    def perform_act(self, time_range, i, istrigger, absoulte_act):
        logtype = self.relevant_logtypes[i]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        fake_logs = self.log_generator.generate_logs(logsource, eventcode, istrigger,time_range, absoulte_act)
        self.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
        logger.info(f"inserted {len(fake_logs)} logs of type {logsource} {eventcode} {istrigger}")
        self.should_delete = True
        
    def record_action(self, action):
        action = self.preprocess_action(action)
        for i, logtype in enumerate(self.relevant_logtypes):
            for istrigger in range(2):
                act = action[i*2+istrigger]
                absoulte_act = int(act*self.quota)
                self.remaining_quota -= absoulte_act
                self.current_episode_accumulated_action[i*2+istrigger] += absoulte_act
                if i == len(self.relevant_logtypes)-1:
                    return   
    
    def perform_action(self, action, time_range):
        action = self.preprocess_action(action)
        logger.info(f"performing action {action}")
        logger.debug(f"Sum of action: {sum(action)}")
        for i, logtype in enumerate(self.relevant_logtypes):
            for istrigger in range(2):
                act = action[i*2+istrigger]
                absoulte_act = int(act*self.quota)
                if act:
                    self.perform_act(time_range, i, istrigger, absoulte_act)
                    time.sleep(.1)
                if i == len(self.relevant_logtypes)-1:
                    return 0
 
        


class ActionStrategy7(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota)
        self.action_quotas =[]
    def create_action_space(self):
        return spaces.Box(low=0, high=self.action_upper_bound, shape=((len(self.relevant_logtypes))*2,), dtype=np.float64)
    
    def preprocess_action(self, action):
        return np.concatenate((action[0:1], action[1:]/sum(action[1:]) if sum(action[1:]) > 0 else action[1:]))
    
    def perform_act(self, time_range, i, istrigger, absoulte_act):
        logtype = self.relevant_logtypes[i]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        fake_logs = self.log_generator.generate_logs(logsource, eventcode, istrigger,time_range, absoulte_act)
        self.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
        logger.info(f"inserted {len(fake_logs)} logs of type {logsource} {eventcode} {istrigger}")
        self.should_delete = True
        
    def record_action(self, action):
        action = self.preprocess_action(action)
        self.remaining_quota = action[0]
        current_quota = action[0]*self.quota
        self.action_quotas.append(current_quota)
        action = action[1:]
        for i, logtype in enumerate(self.relevant_logtypes):
            for istrigger in range(2):
                act = action[i*2+istrigger]
                absoulte_act = int(act*current_quota)
                # self.remaining_quota -= absoulte_act
                self.current_episode_accumulated_action[i*2+istrigger] += absoulte_act
                if i == len(self.relevant_logtypes)-1:
                    return   
    
    def perform_action(self, action, time_range):
        action = self.preprocess_action(action)
        self.remaining_quota = action[0]
        current_quota = action[0]*self.quota
        action = action[1:]
        logger.info(f"performing action {action} with quota {current_quota}")
        logger.debug(f"Sum of action: {sum(action)}")
        for i, logtype in enumerate(self.relevant_logtypes):
            for istrigger in range(2):
                act = action[i*2+istrigger]
                absoulte_act = int(act*current_quota)
                if act:
                    self.perform_act(time_range, i, istrigger, absoulte_act)
                    time.sleep(.1)
                if i == len(self.relevant_logtypes)-1:
                    return 0
 
        

