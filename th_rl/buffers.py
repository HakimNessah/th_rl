import numpy as np
import torch
from collections import namedtuple, deque

class ReplayBuffer():
    def __init__(self, capacity, experience):
        '''
        Experience = namedtuple('Experience', field_names=['state', 'action', 'reward','done', 'new_state'])
        '''
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.experience = experience

    def __len__(self):
        return len(self.buffer)

    def append(self, *args):        
        self.buffer.append(self.experience(*args))

    def sample(self, batch_size, cast=None):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        output = zip(*[self.buffer[idx] for idx in indices])
        if cast:
            output = (torch.tensor(t, dtype=dt) for t,dt in zip(output,cast))              
        return output
    
    def replay(self, cast=None):
        indices = np.arange(0, len(self.buffer))
        output = zip(*[self.buffer[idx] for idx in indices])
        if cast:
            output = (torch.tensor(t, dtype=dt) for t,dt in zip(output,cast))         
        return output     

    def empty(self):
        self.buffer = deque(maxlen=self.capacity)