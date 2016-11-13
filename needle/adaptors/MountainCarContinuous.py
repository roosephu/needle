import numpy as np

class MountainCarContinuousAdaptor(object):
    def __init__(self, env):
        self.env = env
        self.input_dim = 2
        self.output_dim = 1

    def state(self, state):
        return np.array([state])

    def to_env(self, action):
        return action[0]

    def to_agent(self, action):
        return np.array([action])