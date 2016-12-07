import numpy as np
import gflags

gflags.DEFINE_integer("batch_size", 10, "configure batch size")


class BatchBuffer:
    def __init__(self): # default use priority
        self.queue = []

    def add(self, experience):
        self.queue.append(experience)

    def __len__(self):
        # return len(self.queue)
        return sum([x[0][0] for x in self.queue])

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
            ret.append(BatchBuffer.reshape([s[i] for s in samplings]))
        return ret

    def get(self):
        ret = BatchBuffer.finalize(self.queue)
        self.queue = []
        return ret
