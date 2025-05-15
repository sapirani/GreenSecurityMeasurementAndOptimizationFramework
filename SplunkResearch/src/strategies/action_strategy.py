from abc import ABC, abstractmethod
import time
from gymnasium import spaces
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
    
    def reset_step(self):
        pass   
        
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
        if current_quota > 0:
            for i, logtype in enumerate(self.relevant_logtypes):
                for istrigger in range(2):
                    act = action[i*2+istrigger]
                    absoulte_act = int(act*current_quota)
                    if act:
                        self.perform_act(time_range, i, istrigger, absoulte_act)
                        time.sleep(.1)
                    if i == len(self.relevant_logtypes)-1:
                        return 0
 
class ActionStrategy8(ActionStrategy7): # goes with reward_strategy 44
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota)
        self.action_quotas =[]
        self.action_shape = (len(self.relevant_logtypes)+1,)
        self.current_episode_accumulated_action = np.zeros(self.action_shape)
        
    def create_action_space(self):
        return spaces.Box(low=0, high=self.action_upper_bound, shape=self.action_shape, dtype=np.float64)
    
    def record_action(self, action):
        action = self.preprocess_action(action)
        self.remaining_quota = action[0]
        current_quota = action[0]*self.quota
        self.action_quotas.append(current_quota)
        action = action[1:]
        for i, logtype in enumerate(self.relevant_logtypes):
            act = action[i]
            absoulte_act = int(act*current_quota)
            # self.remaining_quota -= absoulte_act
            self.current_episode_accumulated_action[i] += absoulte_act
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
            istrigger = 0   
            act = action[i]
            absoulte_act = int(act*current_quota)
            if act:
                self.perform_act(time_range, i, istrigger, absoulte_act)
                time.sleep(.1)
            if i == len(self.relevant_logtypes)-1:
                return 0

    
    def reset(self):
        self.current_episode_accumulated_action = np.zeros(self.action_shape)
        self.remaining_quota = self.quota
        self.should_delete = False
 
class ActionStrategy9(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota)
        self.action_quotas =[]
        self.action_shape = (len(self.relevant_logtypes)*2-1)
        self.current_episode_accumulated_action = np.zeros(self.action_shape)
        
    def preprocess_action(self, action):
        return action
    
    def create_action_space(self):
        return spaces.MultiBinary(self.action_shape)
    
    def record_action(self, action):
        action = self.preprocess_action(action)
        self.remaining_quota = 1
        current_quota = self.remaining_quota*self.quota
        self.action_quotas.append(current_quota)
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
        self.remaining_quota = 1
        if sum(action) > 0:
            current_quota = self.remaining_quota*self.quota/len(action)
        else:
            current_quota = 0
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

    
    def reset(self):
        self.current_episode_accumulated_action = np.zeros(self.action_shape)
        self.remaining_quota = self.quota
        self.should_delete = False
 
class ActionStrategy10(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota)
        self.action_quotas =[]
        self.action_shape = (len(self.relevant_logtypes)*2)
        self.current_episode_accumulated_action = np.zeros(self.action_shape)
        self.current_step_action = np.zeros(self.action_shape)
        
    def preprocess_action(self, action):
        # if action[1] == len(self.relevant_logtypes)-1:
        #     action[2] = 0
        return action
    
    def create_action_space(self):
        return spaces.MultiDiscrete([100, len(self.relevant_logtypes), 2, 5])
    
    def record_action(self, action):
        action = self.preprocess_action(action)
        self.remaining_quota = action[0]/100
        current_quota = int(self.remaining_quota*self.quota)
        self.action_quotas[-1] += current_quota
        i = action[1]
        istrigger = action[2]
        absoulte_act = current_quota
        # self.remaining_quota -= absoulte_act
        self.current_episode_accumulated_action[i*2+istrigger] += absoulte_act
        self.current_step_action[i*2+istrigger] = absoulte_act

    def perform_action(self, action, time_range):
        action = self.preprocess_action(action)
        self.remaining_quota = action[0]/100
        current_quota = int(self.remaining_quota*self.quota)
        i = action[1]
        istrigger = action[2]
        diversity = action[3]
        absoulte_act = current_quota
        if absoulte_act:
            self.perform_act(time_range, i, istrigger, absoulte_act, diversity)
            # time.sleep(.1)

    def perform_act(self, time_range, i, istrigger, absoulte_act, diversity):
        logtype = self.relevant_logtypes[i]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        fake_logs = self.log_generator.generate_logs(logsource, eventcode, istrigger,time_range, absoulte_act, diversity)
        self.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
        logger.info(f"inserted {len(fake_logs)} logs of type {logsource} {eventcode} {istrigger} with diversity {diversity}")
        self.should_delete = True
    
    def reset(self):
        self.current_episode_accumulated_action = np.zeros(self.action_shape)
        self.remaining_quota = self.quota
        self.should_delete = False
        self.action_quotas.append(0)
        
    def reset_step(self):
        self.current_step_action = np.zeros(self.action_shape)
 
class ActionStrategy11(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, splunk_tools_instance, log_generator,remaining_quota)
        self.action_quotas =[]
        self.action_shape = (len(self.relevant_logtypes)*2-1)
        self.current_episode_accumulated_action = np.zeros(self.action_shape)
        self.current_step_action = np.zeros(self.action_shape)
        
    def preprocess_action(self, action):
        if action[1] == len(self.relevant_logtypes)-1:
            action[2] = 0
        return action
    
    def create_action_space(self):
        return spaces.MultiDiscrete([100, len(self.relevant_logtypes), 6])
    
    def record_action(self, action):
        action = self.preprocess_action(action)
        self.remaining_quota = action[0]/100
        current_quota = int(self.remaining_quota*self.quota)
        self.action_quotas.append(current_quota)
        i = action[1]
        istrigger = min(action[2],1)
        
        absoulte_act = current_quota
        # self.remaining_quota -= absoulte_act
        self.current_episode_accumulated_action[i*2+istrigger] += absoulte_act
        self.current_step_action[i*2+istrigger] = absoulte_act

    def perform_action(self, action, time_range):
        action = self.preprocess_action(action)
        self.remaining_quota = action[0]/100
        current_quota = int(self.remaining_quota*self.quota)
        i = action[1]
        istrigger = min(action[2],1)
        diversity = action[2]
        absoulte_act = current_quota
        if absoulte_act:
            self.perform_act(time_range, i, istrigger, absoulte_act, diversity)
            # time.sleep(.1)

    def perform_act(self, time_range, i, istrigger, absoulte_act, diversity):
        logtype = self.relevant_logtypes[i]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        fake_logs = self.log_generator.generate_logs(logsource, eventcode, istrigger,time_range, absoulte_act, diversity)
        self.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
        logger.info(f"inserted {len(fake_logs)} logs of type {logsource} {eventcode} {istrigger} with diversity {diversity}")
        self.should_delete = True
    
    def reset(self):
        self.current_episode_accumulated_action = np.zeros(self.action_shape)
        self.remaining_quota = self.quota
        self.should_delete = False
        
    def reset_step(self):
        self.current_step_action = np.zeros(self.action_shape)
        
import numpy as np
from gymnasium import spaces
import logging


class ActionStrategy12(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, 
                 splunk_tools_instance, log_generator, remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, 
                        splunk_tools_instance, log_generator, remaining_quota)
        self.action_quotas = []
        self.action_shape = (4,)  # (q_t, e_t, τ_t, d_t)
        self.current_episode_accumulated_action = np.zeros((len(self.relevant_logtypes)*2,))
        self.current_step_action = np.zeros((len(self.relevant_logtypes)*2,))
        
    def create_action_space(self):
        # Create a Box space for (q_t, e_t, τ_t, d_t)
        # q_t ∈ [0,100], e_t ∈ [0,k-1], τ_t ∈ [0,1], d_t ∈ [1,5]
        low = np.array([0, 0, 0, 1])
        high = np.array([1, len(self.relevant_logtypes)-1, 1, 5])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space
    
    def preprocess_action(self, action):
        # Ensure actions stay within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Convert e_t and d_t to integers
        action[1] = round(action[1])
        action[3] = round(action[3])
        # Convert τ_t to binary
        action[2] = round(float(action[2]))
        return action
    
    def record_action(self, action):
        action = self.preprocess_action(action)
        self.remaining_quota = action[0]
        current_quota = int(self.remaining_quota*self.quota)
        self.action_quotas[-1] += current_quota
        
        i = int(action[1])  # event type index
        is_trigger = int(action[2])  # binary trigger flag
        absolute_act = current_quota
        
        self.current_episode_accumulated_action[i*2+is_trigger] += absolute_act
        self.current_step_action[i*2+is_trigger] = absolute_act

    def perform_action(self, action, time_range):
        action = self.preprocess_action(action)
        self.remaining_quota = action[0]
        current_quota = int(self.remaining_quota*self.quota)
        
        i = int(action[1])  # event type index
        is_trigger = int(action[2])  # binary trigger flag
        diversity = int(action[3])  # log diversity level
        absolute_act = current_quota
        
        if absolute_act:
            self.perform_act(time_range, i, is_trigger, absolute_act, diversity)

    def perform_act(self, time_range, i, is_trigger, absolute_act, diversity):
        logtype = self.relevant_logtypes[i]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        fake_logs = self.log_generator.generate_logs(
            logsource, eventcode, is_trigger, 
            time_range, absolute_act, diversity
        )
        self.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
        logger.info(
            f"inserted {len(fake_logs)} logs of type {logsource} "
            f"{eventcode} {is_trigger} with diversity {diversity}"
        )
        self.should_delete = True
    
    def reset(self):
        self.current_episode_accumulated_action = np.zeros((len(self.relevant_logtypes)*2,))
        self.remaining_quota = self.quota
        self.should_delete = False
        self.action_quotas.append(0)
        
    def reset_step(self):
        self.current_step_action = np.zeros((len(self.relevant_logtypes)*2,))
        
class ActionStrategy13(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, 
                 splunk_tools_instance, log_generator, remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, 
                        splunk_tools_instance, log_generator, remaining_quota)
        self.action_quotas = []
        self.action_shape = (4,)  # (q_t, e_t, p_t, d_t) where p_t is trigger probability
        self.current_episode_accumulated_action = np.zeros((len(self.relevant_logtypes)*2,))
        self.current_step_action = np.zeros((len(self.relevant_logtypes)*2,))
        
    def create_action_space(self):
        # Create a Box space for (q_t, e_t, p_t, d_t)
        # q_t ∈ [0,1] - quota fraction
        # e_t ∈ [0,k-1] - event type
        # p_t ∈ [0,1] - trigger probability
        # d_t ∈ [1,5] - diversity level
        low = np.array([0, 0, 0, 1])
        high = np.array([1, len(self.relevant_logtypes)-1, 1, 5])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space
    
    def preprocess_action(self, action):
        # Ensure actions stay within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Convert e_t and d_t to integers, keep p_t as float
        processed_action = action.copy()
        processed_action[1] = round(float(action[1]))  # event type
        processed_action[3] = round(float(action[3]))  # diversity
        
        return processed_action
    
    def record_action(self, action):
        action = self.preprocess_action(action)
        self.remaining_quota = action[0]
        current_quota = int(self.remaining_quota * self.quota)
        self.action_quotas[-1] += current_quota
        
        i = int(action[1])  # event type index
        trigger_prob = action[2]  # trigger probability
        
        # Split quota based on trigger probability
        trigger_quota = int(current_quota * trigger_prob)
        non_trigger_quota = current_quota - trigger_quota
        
        # Record both triggering and non-triggering actions
        self.current_episode_accumulated_action[i*2] += non_trigger_quota
        self.current_episode_accumulated_action[i*2 + 1] += trigger_quota
        self.current_step_action[i*2] = non_trigger_quota
        self.current_step_action[i*2 + 1] = trigger_quota

    def perform_action(self, action, time_range):
        action = self.preprocess_action(action)
        self.remaining_quota = action[0]
        current_quota = int(self.remaining_quota * self.quota)
        
        if current_quota <= 0:
            return
            
        i = int(action[1])  # event type index
        trigger_prob = action[2]  # trigger probability
        diversity = int(action[3])  # diversity level
        
        # Split quota based on trigger probability
        trigger_quota = int(current_quota * trigger_prob)
        non_trigger_quota = current_quota - trigger_quota
        
        # Generate non-triggering logs
        if non_trigger_quota > 0:
            self.perform_act(time_range, i, False, non_trigger_quota, diversity)
            
        # Generate triggering logs
        if trigger_quota > 0:
            self.perform_act(time_range, i, True, trigger_quota, diversity)

    def perform_act(self, time_range, i, is_trigger, absolute_act, diversity):
        logtype = self.relevant_logtypes[i]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        fake_logs = self.log_generator.generate_logs(
            logsource, eventcode, is_trigger, 
            time_range, absolute_act, diversity
        )
        self.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
        logger.info(
            f"inserted {len(fake_logs)} logs of type {logsource} "
            f"{eventcode} {is_trigger} with diversity {diversity}"
        )
        self.should_delete = True
    
    def reset(self):
        self.current_episode_accumulated_action = np.zeros((len(self.relevant_logtypes)*2,))
        self.remaining_quota = self.quota
        self.should_delete = False
        self.action_quotas.append(0)
        
    def reset_step(self):
        self.current_step_action = np.zeros((len(self.relevant_logtypes)*2,))


class ActionStrategy14(ActionStrategy):
    def __init__(self, relevant_logtypes, action_upper_bound, step_size, action_duration, 
                 splunk_tools_instance, log_generator, remaining_quota):
        super().__init__(relevant_logtypes, action_upper_bound, step_size, action_duration, 
                        splunk_tools_instance, log_generator, remaining_quota)
        self.action_quotas = []
        self.num_logtypes = len(self.relevant_logtypes)
        # Each log type needs quota, trigger, diversity (3 values)
        self.action_shape = (self.num_logtypes * 3,)
        self.current_episode_accumulated_action = np.zeros((len(self.relevant_logtypes)*2,))
        self.current_step_action = np.zeros((len(self.relevant_logtypes)*2,))
        
    def create_action_space(self):
        # For each log type we need:
        # - quota proportion [0,1]
        # - trigger probability [0,1]
        # - diversity level [1,5]
        # total quota used is sum of all quotas
        action_dim = self.num_logtypes * 3 + 1
        
        low = np.zeros(action_dim)
        high = np.ones(action_dim)
        
        # Set diversity bounds (every third position)
        for i in range(self.num_logtypes):
            low[i*3 + 2] = 1
            high[i*3 + 2] = 5
            
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space
    
    def preprocess_action(self, action):
        processed_action = action.copy()
        
        # normalize all log types quota to sum to 1
        quota_sum = np.sum(processed_action[::3])
        if quota_sum > 0:
            for i in range(self.num_logtypes):
                processed_action[i*3] /= quota_sum
        logger.info(f"processed action: {processed_action}")
        # Ensure actions stay within bounds
        processed_action = np.clip(processed_action, self.action_space.low, self.action_space.high)
        
        # Round diversity levels to integers
        for i in range(self.num_logtypes):
            processed_action[i*3 + 2] = round(float(action[i*3 + 2]))

        return processed_action
    
    def record_action(self, action):
        action = self.preprocess_action(action)
        total_quota_used = 0
        current_total_quota = int(action[-1] * self.quota)
        # Process each log type
        for i in range(self.num_logtypes):
            quota = action[i*3]  # Quota proportion for this type
            trigger_prob = action[i*3 + 1]  # Trigger probability
            
            # Calculate quota for this log type
            type_quota = int(quota * current_total_quota)
            total_quota_used += type_quota
            
            # Split by trigger probability
            trigger_quota = int(type_quota * trigger_prob)
            non_trigger_quota = type_quota - trigger_quota
            
            # Record both quotas
            self.current_episode_accumulated_action[i*2] += non_trigger_quota
            self.current_episode_accumulated_action[i*2 + 1] += trigger_quota
            self.current_step_action[i*2] = non_trigger_quota
            self.current_step_action[i*2 + 1] = trigger_quota
            
        self.action_quotas[-1] += total_quota_used
        self.remaining_quota = 0

    def perform_action(self, action, time_range):
        action = self.preprocess_action(action)
        
        # Process each log type
        for i in range(self.num_logtypes):
            quota = action[i*3]  # Quota proportion
            trigger_prob = action[i*3 + 1]  # Trigger probability
            diversity = int(action[i*3 + 2])  # Diversity level
            current_total_quota = int(action[-1] * self.quota)
            
            # Calculate quota for this log type
            type_quota = int(quota * current_total_quota)
            
            if type_quota > 0:
                # Split into triggering and non-triggering
                trigger_quota = int(type_quota * trigger_prob)
                non_trigger_quota = type_quota - trigger_quota
                
                # Generate non-triggering logs
                if non_trigger_quota > 0:
                    self.perform_act(time_range, i, False, non_trigger_quota, diversity)
                
                # Generate triggering logs
                if trigger_quota > 0:
                    self.perform_act(time_range, i, True, trigger_quota, diversity)

    def perform_act(self, time_range, i, is_trigger, absolute_act, diversity):
        logtype = self.relevant_logtypes[i]
        logsource = logtype[0].lower()
        eventcode = logtype[1]
        fake_logs = self.log_generator.generate_logs(
            logsource, eventcode, is_trigger, 
            time_range, absolute_act, diversity
        )
        self.splunk_tools_instance.write_logs_to_monitor(fake_logs, logsource)
        logger.info(
            f"inserted {len(fake_logs)} logs of type {logsource} "
            f"{eventcode} {is_trigger} with diversity {diversity}"
        )
        self.should_delete = True
    
    def reset(self):
        self.current_episode_accumulated_action = np.zeros((len(self.relevant_logtypes)*2,))
        self.remaining_quota = 0
        self.should_delete = False
        self.action_quotas.append(0)
        
    def reset_step(self):
        self.current_step_action = np.zeros((len(self.relevant_logtypes)*2,))