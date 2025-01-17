from abc import ABC, abstractmethod

from gym import spaces
import numpy as np
import logging
logger = logging.getLogger(__name__)


class StateStrategy(ABC):
    def __init__(self, top_logtypes):
        self.observation_spaces = None
        self.state = None

        self.top_logtypes = top_logtypes
        self.real_logtype_distribution = {logtype: 0 for logtype in top_logtypes}
        self.fake_logtype_distribution = {logtype: 0 for logtype in top_logtypes}
        self.step_real_logtype_distribution = {logtype: 0 for logtype in top_logtypes}
        self.step_fake_logtype_distribution = {logtype: 0 for logtype in top_logtypes}
        self.real_state = None
        self.fake_state = None
        self.step_real_state = None
        self.step_fake_state = None

        self.diff_state = None
        self.abs_real = None
        self.abs_fake = None
        self.step_abs_real = None
        self.step_abs_fake = None
        self.remainig_quota = 0
    
    @abstractmethod
    def create_state(self):
        pass
    
    def update_state(self):
        real_state, fake_state, step_real_state, step_fake_state = self.get_abs_states()
        logger.debug(f"Real state: {real_state}")
        logger.debug(f"Fake state: {fake_state}")
        logger.debug(f"Step Real state: {step_real_state}")
        logger.debug(f"Step Fake state: {step_fake_state}")
        self.abs_real = real_state
        self.abs_fake = fake_state
        self.step_abs_real = step_real_state
        self.step_abs_fake = step_fake_state
        diff_state = [x-y for x,y in zip(fake_state, real_state)]
        
        diff_state = self.normalize_state(diff_state)
        real_state = self.normalize_state(real_state)
        fake_state = self.normalize_state(fake_state)
        
        step_real_state = self.normalize_state(step_real_state)
        step_fake_state = self.normalize_state(step_fake_state)
        
        self.real_state = real_state
        self.fake_state = fake_state
        self.diff_state = diff_state

        self.step_real_state = step_real_state
        self.step_fake_state = step_fake_state
        return real_state,fake_state, diff_state

    def normalize_state(self, state):
        state = np.array(state)
        state = state/np.sum(state) if np.sum(state) != 0 else np.ones(len(state))/len(state)
        return state
    
    def get_abs_states(self):
        real_state = []
        fake_state = []
        step_real_state = []
        step_fake_state = []
        for i, logtype in enumerate(self.top_logtypes):
            # create state vector
            if logtype in self.step_real_logtype_distribution:
                step_real_count = self.step_real_logtype_distribution[logtype]
            else:
                step_real_count = 0
            step_real_state.append(step_real_count)
            step_fake_state.append(step_real_count)
            if logtype in self.step_fake_logtype_distribution:
                step_fake_state[i] += self.step_fake_logtype_distribution[logtype]
            
            real_state.append(self.real_logtype_distribution[logtype])
            fake_state.append(self.fake_logtype_distribution[logtype])
        return real_state,fake_state, step_real_state, step_fake_state
        # return np.array(real_state),np.array(fake_state)
    
    def update_distributions(self, real_distribution_dict, fake_logtypes_counter):
        self.step_real_logtype_distribution = real_distribution_dict
        self.step_fake_logtype_distribution = fake_logtypes_counter
        for logtype in real_distribution_dict:
            if logtype in self.top_logtypes:
                self.real_logtype_distribution[logtype] += real_distribution_dict[logtype]
                self.fake_logtype_distribution[logtype] += real_distribution_dict[logtype]
            if logtype in fake_logtypes_counter:
                self.fake_logtype_distribution[logtype] += fake_logtypes_counter[logtype]
    
    def update_quota(self, quota):
        self.remainig_quota = quota
    
    def reset(self):
        self.real_state = None
        self.fake_state = None
        self.diff_state = None
        self.real_logtype_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.fake_logtype_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.step_real_logtype_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.step_fake_logtype_distribution = {logtype: 0 for logtype in self.top_logtypes}
        self.remainig_quota = 0
        return self.update_state()
    
    
class StateStrategy1(StateStrategy):
    def __init__(self, top_logtypes):
        super().__init__(top_logtypes)
        
    
    def create_state(self):
        self.observation_spaces = spaces.Box(low=0,high=1,shape=(len(self.top_logtypes),),dtype=np.float64)
        return self.observation_spaces
    
    def update_state(self):
        real_state, fake_state, diff_state = super().update_state()
        return np.array(fake_state)
    
class StateStrategy2(StateStrategy):
    def __init__(self, top_logtypes):
        super().__init__(top_logtypes)
        
    
    def create_state(self):
        self.observation_spaces = spaces.Box(low=0,high=1,shape=(len(self.top_logtypes),),dtype=np.float64)
        return self.observation_spaces
    
    def update_state(self):
        real_state, fake_state, diff_state = super().update_state()
        return np.array(diff_state)
    
class StateStrategy3(StateStrategy):
    def __init__(self, top_logtypes):
        super().__init__(top_logtypes)
    
    def create_state(self):
        self.observation_spaces = spaces.Box(low=0,high=1,shape=(len(self.top_logtypes),),dtype=np.float64)
        return self.observation_spaces
    
    def update_state(self):
        return np.zeros(len(self.top_logtypes))
    
class StateStrategy4(StateStrategy):
    def __init__(self, top_logtypes):
        super().__init__(top_logtypes)
    
    def create_state(self):
        self.observation_spaces = spaces.Box(low=0,high=1,shape=(len(self.top_logtypes)*2,),dtype=np.float64)
        return self.observation_spaces
    
    def update_state(self):
        real_state, fake_state, diff_state = super().update_state()
        return np.array(real_state + fake_state)

class StateStrategy5(StateStrategy):
    def __init__(self, top_logtypes):
        super().__init__(top_logtypes)
    
    def create_state(self):
        self.observation_spaces = spaces.Box(low=-np.inf,high=np.inf,shape=(2*len(self.top_logtypes)+1,),dtype=np.float64)
        return self.observation_spaces
    
    def update_state(self):
        real_state, fake_state, diff_state = super().update_state()
        return np.array(real_state + fake_state + [self.remainig_quota])

class StateStrategy6(StateStrategy):
    def __init__(self, top_logtypes,relevant_logtypes):
        super().__init__(top_logtypes)
        self.week_day = 0
        self.hour = 0
        self.relevant_logtypes = relevant_logtypes
        self.episodic_action = np.zeros(2*len(relevant_logtypes)-1)
    
    def create_state(self):
        self.observation_spaces = spaces.Box(low=-np.inf,high=np.inf,shape=(2*len(self.top_logtypes)+2*len(self.relevant_logtypes)-1+2,),dtype=np.float64)
        return self.observation_spaces
    
    def update_time(self, week_day, hour):
        self.week_day = week_day
        self.hour = hour
    
    def update_episodic_action(self, action):
        self.episodic_action = action
        
    def update_state(self):
        real_state, fake_state, diff_state = super().update_state()
        return np.array(real_state + fake_state + self.episodic_action.tolist() + [self.week_day] + [self.hour])

class StateStrategy7(StateStrategy):
    def __init__(self, top_logtypes, rules):
        super().__init__(top_logtypes)
        self.rules = rules
        self.rules_alerts = np.zeros(len(rules), dtype=int)
        self.week_day = 0
        self.hour = 0
        
    def create_state(self):
        self.observation_spaces = spaces.Box(low=0,high=1,shape=(len(self.top_logtypes)*2 + len(self.rules)+2,),dtype=np.float64)
        return self.observation_spaces
    
    def update_rules_alerts(self, rules_alerts):

        self.rules_alerts = np.logical_or(self.rules_alerts, rules_alerts)
    
    def update_state(self):
        real_state, fake_state, diff_state = super().update_state()
        return np.array(real_state + fake_state + self.rules_alerts.tolist() + [self.week_day] + [self.hour])
    
    def update_time(self, week_day, hour):
        self.week_day = week_day
        self.hour = hour
    
    def reset(self):
        self.rules_alerts = np.zeros(len(self.rules), dtype=int)
        return super().reset()

class StateStrategy8(StateStrategy):
    def __init__(self, top_logtypes):
        super().__init__(top_logtypes)
        
        self.week_day = 0
        self.hour = 0
        
    def create_state(self):
        self.observation_spaces = spaces.Box(low=0,high=np.inf,shape=(len(self.top_logtypes),),dtype=np.float64)
        return self.observation_spaces

    
    def update_state(self):
        real_state, fake_state, diff_state = super().update_state()
        abs_diff = np.array(self.abs_fake) - np.array(self.abs_real)
        return np.array((abs_diff/(np.array(self.abs_real)+1)).tolist())

class StateStrategy9(StateStrategy):
    def __init__(self, top_logtypes):
        super().__init__(top_logtypes)
        
        self.week_day = 0
        self.hour = 0
        
    def create_state(self):
        self.observation_spaces = spaces.Box(low=0,high=np.inf,shape=(len(self.top_logtypes)*2,),dtype=np.float64)
        return self.observation_spaces

    
    def update_state(self):
        real_state, fake_state, diff_state = super().update_state()
        abs_diff = np.array(self.abs_fake) - np.array(self.abs_real)
        return np.array((abs_diff/(np.array(self.abs_real)+1)).tolist() + abs_diff.tolist())

class StateStrategy10(StateStrategy):
    def __init__(self, top_logtypes):
        super().__init__(top_logtypes)

        self.log_number = 0
    
    def update_log_number(self, log_number):
        self.log_number = log_number
    
    def create_state(self):
        self.observation_spaces = spaces.Box(low=0,high=np.inf,shape=(len(self.top_logtypes)+1,),dtype=np.float64)
        return self.observation_spaces

    
    def update_state(self):
        real_state, fake_state, diff_state = super().update_state()
        abs_diff = np.array(self.abs_fake) - np.array(self.abs_real)
        return np.array((abs_diff/(np.array(self.abs_real)+1)).tolist()+[self.log_number])
    
class StateStrategy11(StateStrategy):
    def __init__(self, top_logtypes):
        super().__init__(top_logtypes)
        self.log_number = 0
    
    def create_state(self):
        self.observation_spaces = spaces.Box(low=0,high=1,shape=(len(self.top_logtypes)*2,),dtype=np.float64)
        return self.observation_spaces
    
    def update_log_number(self, log_number):
        self.log_number = log_number
    
    def update_state(self):
        real_state, fake_state, diff_state = super().update_state()
        return np.array(real_state + fake_state + [self.log_number])