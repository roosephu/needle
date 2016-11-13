import numpy as np

class CartPoleAdaptor(object):
    def __init__(self, env):
        self.env = env
        self.input_dim = 4
        self.output_dim = 2

    def state(self, state):
        return np.array([state])

    def to_env(self, action):
        return action[0]

    def to_agent(self, action):
        return np.array([action])
