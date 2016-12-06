import gflags
import logging
import numpy as np

from needle.agents import BasicAgent, register_agent
from needle.agents.REINFORCE.net import Net
from needle.helper.OU_process import OUProcess
from needle.helper.softmax_sampler import SoftmaxSampler
from needle.helper.batcher import Batcher

FLAGS = gflags.FLAGS



@register_agent("REINFORCE")
class Agent(SoftmaxSampler, Batcher, BasicAgent):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.counter = 0

        self.net = Net(input_dim, output_dim)
        self.net.build_infer()
        self.net.build_train()

        self.baseline = 20

    def train_batch(self, lengths, mask, states, choices, rewards, new_states):
        # advantages = rewards * (np.expand_dims(lengths, 1) - self.baseline)
        advantages = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1] - self.baseline
        # logging.info("advantages = %s, rewards = %s" % (advantages[0], rewards[0]))
        feed_dict = self.net.get_dict(lengths, mask, states, choices, advantages)

        self.baseline = self.baseline * 0.9 + np.mean(lengths) * 0.1
        self.net.train(feed_dict)

    def action(self, inputs):
        return self.softmax_action(
            self.net.infer(np.array([inputs])),
            noise=self.noise,
        )

    def reset(self):
        self.noise = OUProcess()
        self.net.reset()
        self.counter = 0
