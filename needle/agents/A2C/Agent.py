import tensorflow as tf
import numpy as np
import gflags
import logging
from needle.agents.A2C.Model import Model
from needle.agents.Agent import BasicAgent
from needle.helper.ReplayBuffer import ReplayBuffer
from needle.helper.OUProcess import OUProcess

gflags.DEFINE_integer("num_units", 100, "# hidden units for LSTM")
gflags.DEFINE_float("GAE_decay", 0.96, "TD(lambda)")
gflags.DEFINE_integer("t_max", 5, "how many steps each actor learner performs")
gflags.DEFINE_float("noise_weight", 0., "OU noise applied on actions")
FLAGS = gflags.FLAGS


def softmax(x):
    x -= np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


class Agent(BasicAgent):
    def __init__(self, input_dim, output_dim):
        # we don't need ShadowNet
        # ShadowNet(lambda: Model(input_dim, output_dim), FLAGS.tau, "A2C")
        self.model = Model(input_dim, output_dim)
        self.model.build_infer()
        self.model.build_train()

        self.buffer = ReplayBuffer(FLAGS.batch_size)
        self.output_dim = output_dim
        self.counter = 0

    def init(self):
        tf.get_default_session().run(tf.initialize_all_variables())
        # tf.get_default_session().run(self.model.op_shadow_init)

    def reset(self):
        self.model.reset()
        self.noise = OUProcess(shape=self.output_dim)
        self.last_state = self.model.current_state

    def action(self, inputs, show=False):
        logits = self.model.infer(np.array([inputs]))[0][0]
        noise = self.noise.next() * FLAGS.noise_weight
        # logging.debug("logits = %s" % (logits))
        actions = softmax(logits + noise)
        return np.array([np.random.choice(len(actions), p=actions)])

    def feedback(self, inputs, action, reward, done, new_inputs):
        self.counter += 1
        reward = np.array([reward])

        experience = inputs, action, reward, new_inputs
        self.buffer.add(experience)

        if done or self.counter == FLAGS.t_max:
            self.update(done)

    def update(self, done):
        inputs, actions, rewards, new_inputs = self.buffer.latest(self.counter)
        # last_cell, last_state = self.last_state
        # logging.info("inputs = %s, states = %s, rewards = %s, actions = %s, last cell = %s, last state = %s" %
        #              (inputs.shape, states.shape, rewards.shape, actions.shape, last_cell.shape, last_state.shape))

        all_inputs = np.vstack([[inputs[0]], new_inputs])
        # logging.info("inputs = %s, new_inputs = %s" % (inputs, new_inputs))
        values = self.model.values(self.last_state, [all_inputs])[0]
        # logging.info("values = %s" % (values.shape, ))

        if done:
            value = 0
        else:
            value = values[-1]

        advantages = []
        for i in reversed(range(len(values) - 1)):
            value = rewards[i] + value * FLAGS.gamma
            advantages.append(value - values[i])
        advantages = np.array(list(reversed(advantages)))
        logging.debug("a = %s, v = %s" % (advantages, values))

        # logging.info("advantages = %s, values = %s, sum = %s" % (advantages[:3], values[:3], sum(advantages)))
        lengths = np.array([self.counter])
        self.model.train(self.last_state, [inputs], [advantages], [actions], [lengths])

        # logging.info(self.last_state.h[0, :3])
        self.last_state = self.model.current_state
        # logging.info(self.last_state.h[0, :3])
        self.counter = 0

