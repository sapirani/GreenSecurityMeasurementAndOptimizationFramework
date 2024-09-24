from abc import ABC, abstractmethod

from gym import spaces
import numpy as np


class StateStrategy(ABC):
    def __init__(self, top_logtypes):
        self.observation_spaces = None
        self.state = None
        self.real_logtype_distribution = None
        self.fake_logtype_distribution = None
        self.top_logtypes = top_logtypes
        self.real_state = None
        self.fake_state = None
        self.diff_state = None
    
    @abstractmethod
    def create_state(self):
        pass
    
    def update_state(self):
        real_state = []
        fake_state = []
        for i, logtype in enumerate(self.top_logtypes):
            # create state vector
            logtype = ' '.join(logtype)
            if logtype in self.real_logtype_distribution:
                real_count = self.real_logtype_distribution[logtype]
                real_state.append(real_count)
                fake_state.append(real_count)
            else:
                real_state.append(0)
                fake_state.append(0)
            if logtype in self.fake_logtype_distribution:
                fake_state[i] += self.fake_logtype_distribution[logtype]

        real_total_sum = sum(real_state)
        fake_total_sum = sum(fake_state)
        diff_state = [x-y for x,y in zip(fake_state, real_state)]
        diff_state = [x/sum(diff_state) if sum(diff_state) != 0 else 1/len(diff_state) for x in diff_state]
        real_state = [x/real_total_sum if real_total_sum!= 0 else 1/len(real_state) for x in real_state]
        fake_state = [x/fake_total_sum if fake_total_sum != 0 else 1/len(fake_state) for x in fake_state]
        self.real_state = real_state
        self.fake_state = fake_state
        self.diff_state = diff_state
        return real_state,fake_state, diff_state
    
    def update_distributions(self, real_distribution_dict, fake_logtypes_counter):
        self.real_logtype_distribution = real_distribution_dict
        self.fake_logtype_distribution = fake_logtypes_counter
    
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