from needle.adaptors import register_adaptor
import numpy as np


@register_adaptor("MountainCar-v0")
class Adaptor(object):
    def __init__(self, env):
        self.env = env
        self.input_dim = 2
        self.output_dim = 3

    def state(self, state):
        return np.array([state])

    def to_env(self, action):
        return action[0]

    def reset(self):
        pass
