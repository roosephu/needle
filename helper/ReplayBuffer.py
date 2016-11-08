import numpy as np
import random
from SegmentTree import SegmentTree

class ReplayBuffer:
    def __init__(self, size): # default use priority
        self.queue = SegmentTree(size)
        self.size = size
        self.index = 0
        self.length = 0

    def add(self, experience, priority=1.):
        # if len(self.queue) <= self.index:
        #     self.queue.append(experience)
        # else:
        #     self.queue[self.index] = experience
        self.queue.update(self.index, experience, priority)
        self.index = (self.index + 1) % self.size
        self.length += 1

    def __len__(self):
        return min(self.length, self.size)

    def sample(self, batch_size):
        samplings = []
        for i in range(batch_size):
            samplings.append(self.queue.sample()) # wouldn't use choice(, size=batch_size)

        ret = []
        for i in range(len(samplings[0])):
            ret.append(np.concatenate([s[i] for s in samplings]))
        return ret