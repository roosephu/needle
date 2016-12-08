import gflags
import logging
import numpy as np

from needle.agents import BasicAgent, register_agent
from needle.agents.TNPG.net import Net
from needle.helper.conjugate_gradient import conjugate_gradient
from needle.helper.OU_process import OUProcess
from needle.helper.softmax_sampler import SoftmaxSampler
from needle.helper.batcher import Batcher
from needle.helper.utils import softmax

# if program encounters NaN, decrease this value
gflags.DEFINE_float("delta_KL", 0.01, "KL divergence between two sets of parameters")
FLAGS = gflags.FLAGS

line_search_decay = 0.5


def get_matrix(model, states, choices, advantages, num_parameters):
    func = lambda direction: model.fisher_vector_product(direction, [states], [choices], [advantages])
    A = np.zeros((num_parameters, num_parameters))
    k = np.zeros(num_parameters)
    for i in range(num_parameters):
        k[i] = 1.
        A[:, i] = func(k)
        k[i] = 0.
    # s, v, d = np.linalg.svd(A)
    # logging.debug("singular values of A = %s" % (v,))
    # logging.debug("A = %s" % (A,))
    return A


@register_agent("TNPG")
class Agent(SoftmaxSampler, Batcher, BasicAgent):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.counter = 0

        self.net = Net(input_dim, output_dim)
        self.net.build_infer()
        self.net.build_train()

        self.batch_mode = "step"
        self.baseline = 20

    def train_batch(self, lengths, mask, states, choices, rewards, new_states):
        logging.info("========== train iteration ==========")

        advantages = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
        old_logits = self.net.infer(states)
        old_actions = softmax(old_logits)
        feed_dict = self.net.get_dict(lengths, mask, states, choices, advantages, old_logits)

        gradient = self.net.gradient(feed_dict)

        natural_gradient, dot_prod = conjugate_gradient(
            lambda direction: self.net.fisher_vector_product(direction, feed_dict),
            gradient,
        )
        natural_gradient *= np.sqrt(2 * FLAGS.delta_KL / (dot_prod + 1e-8))
        variables = self.net.get_variables()

        old_loss, old_KL = self.net.test(feed_dict)
        logging.info("old loss = %s, old KL = %s" % (old_loss, np.mean(old_KL)))

        while True:
            self.net.apply_var(variables - natural_gradient)
            new_loss, new_KL = self.net.test(feed_dict)
            logging.info("new loss = %s, new KL = %s" % (new_loss, np.mean(new_KL)))
            if new_KL - old_KL <= FLAGS.delta_KL and new_loss <= old_loss:
                break
            natural_gradient *= line_search_decay

    def action(self, inputs):
        return self.softmax_action(
            self.net.infer(np.array([inputs])),
            noise=self.noise,
        )

    def reset(self):
        self.noise = OUProcess()
        self.net.reset()
        self.counter = 0
