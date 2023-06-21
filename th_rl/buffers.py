from collections import deque
import numpy
import torch


class ReplayBuffer:
    def __init__(self, capacity, experience):
        """
        Experience = namedtuple('Experience', field_names=['state', 'action', 'reward','done', 'new_state'])
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.experience = experience

    def __len__(self):
        return len(self.buffer)

    def append(self, *args):
        self.buffer.append(self.experience(*args))

    def sample(self, batch_size):
        indices = numpy.random.choice(len(self.buffer), batch_size, replace=False)
        output = zip(*[self.buffer[idx] for idx in indices])
        return output

    def replay(self, replay_size=0):
        if replay_size == 0:
            indices = numpy.arange(0, len(self.buffer))
        else:
            indices = numpy.arange(len(self.buffer) - replay_size, len(self.buffer))
        output = zip(*[self.buffer[idx] for idx in indices])
        return output

    def empty(self):
        self.buffer = deque(maxlen=self.capacity)
