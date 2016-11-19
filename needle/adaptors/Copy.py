import numpy as np

class CopyAdaptor(object):
    def __init__(self, env):
        self.env = env
        self.input_dim = 6
        self.output_dim = 20

    def state(self, state):
        return 0. + (np.arange(6) == state).reshape(1, -1)

    def to_env(self, action):
        return [action[0] % 2, action[0] // 2 % 2, action[0] // 4]

