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

    def sample(self, batch_size, cast=None):
        indices = numpy.random.choice(len(self.buffer), batch_size, replace=False)
        output = zip(*[self.buffer[idx] for idx in indices])
        if cast:
            output = (torch.tensor(t, dtype=dt) for t, dt in zip(output, cast))
        return output

    def replay(self, cast=None, replay_size=0):
        if replay_size == 0:
            indices = numpy.arange(0, len(self.buffer))
        else:
            indices = numpy.arange(len(self.buffer) - replay_size, len(self.buffer))
        output = zip(*[self.buffer[idx] for idx in indices])
        if cast:
            output = (
                torch.tensor(numpy.array(t), dtype=dt) for t, dt in zip(output, cast)
            )
        return output

    def empty(self):
        self.buffer = deque(maxlen=self.capacity)
