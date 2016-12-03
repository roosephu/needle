import gflags
import numpy as np

from needle.helper.buffer.SegmentTree import SegmentTree

gflags.DEFINE_integer("replay_buffer_size", 10000, "size of experience replay buffer")


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

    @staticmethod
    def reshape(values):
        shape = np.max([v.shape for v in values], axis=0)
        zeros = shape * 0
        values = [np.pad(v, zip(zeros, shape - v.shape), 'constant') for v in values]
        return np.concatenate(values)

    @staticmethod
    def finalize(samplings):
        ret = []
        for i in range(len(samplings[0])):
            ret.append(ReplayBuffer.reshape([s[i] for s in samplings]))
        return ret

    def sample(self, batch_size):
        samplings = []
        for i in range(batch_size):
            samplings.append(self.queue.sample()) # wouldn't use choice(, size=batch_size)
        return ReplayBuffer.finalize(samplings)

    def latest(self, batch_size):
        samplings = []
        base = (self.index - batch_size + self.size) % self.size
        for i in range(batch_size):
            samplings.append(self.queue.find((base + i) % self.size))
        return ReplayBuffer.finalize(samplings)

def main():
    print ReplayBuffer.reshape([np.array([[1], [2]]), np.array([[2, 2], [3, 3], [4, 5]])])

if __name__ == "__main__":
    main()