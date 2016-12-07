from needle.adaptors import register_adaptor
import numpy as np
import logging


@register_adaptor("DuplicatedInput-v0")
class Adaptor(object):
    def __init__(self, env):
        self.env = env
        self.input_dim = 32
        self.output_dim = 20

    def state(self, input):
        current_input = (np.arange(6) == input).reshape(1, -1)
        state = np.hstack([current_input, self.last_input, self.last_action])
        logging.debug("input = %s, last input = %s, last action = %s" % (input, np.sum(self.last_input * np.arange(6)), np.sum(self.last_action * np.arange(20))))
        self.last_input = current_input
        return state

    def to_env(self, action):
        action = action[0]
        self.last_action = ((np.arange(20) == action)).reshape(1, -1)
        logging.debug("action = %s" % ([action % 2, action // 2 % 2, action // 4],))
        return [action % 2, action // 2 % 2, action // 4]

    def reset(self):
        self.last_input = np.zeros((1, 6))
        self.last_action = np.zeros((1, 20))
