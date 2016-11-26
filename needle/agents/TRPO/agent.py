import tensorflow as tf
import numpy as np
import logging
import gflags
from needle.agents import BasicAgent, register_agent
from needle.agents.TRPO.model import Model

FLAGS = gflags.FLAGS


def softmax(x):
    x -= np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


@register_agent("TRPO")
class Agent(BasicAgent):
    def __init__(self, input_dim, output_dim):
        self.model = Model()
        self.buffer = []

    def feedback(self, state, action, reward, done, new_state):
        self.buffer.append((state, action, reward, done, new_state))

        if done:
            self.update()

    def update(self):
        pass

    def action(self, inputs, show=False):
        logits = self.model.infer(np.array([inputs]))[0][0]
        actions = softmax(logits)
        # actions = (actions + 0.01) / (self.output_dim * 0.01 + 1)
        logging.debug("logits = %s" % (logits - max(logits),))
        return np.array([np.random.choice(len(actions), p=actions)])

    def reset(self):
        pass

