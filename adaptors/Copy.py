import numpy as np

class CopyAdaptor:
    def __init__(self, env):
        self.env = env
        self.input_dim = 1
        self.output_dim = 20

    def state(self, state):
        return np.array([[state]])

    def to_env(self, action):
        return [action[0] % 2, action[0] // 2 % 2, action[0] // 4]

    def to_agent(self, action):
        return np.array([action[0] + action[1] * 2 + action[2] * 4])