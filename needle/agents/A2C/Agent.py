import tensorflow as tf
import numpy as np
import gflags
import logging
from needle.agents.A2C.Model import Model
from needle.agents.Agent import BasicAgent
from needle.helper.ReplayBuffer import ReplayBuffer
from needle.helper.ShadowNet import ShadowNet
from needle.helper.OUProcess import OUProcess

gflags.DEFINE_integer("num_units", 100, "# hidden units for LSTM")
gflags.DEFINE_float("GAE_decay", 0.96, "TD(lambda)")
FLAGS = gflags.FLAGS


def softmax(x):
    x -= np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


class Agent(BasicAgent):
    def __init__(self, input_dim, output_dim):
        self.model = ShadowNet(lambda: Model(input_dim, output_dim), FLAGS.tau, "A2C")
        self.buffer = []
        self.replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size)
        self.output_dim = output_dim

    def init(self):
        tf.get_default_session().run(tf.initialize_all_variables())
        tf.get_default_session().run(self.model.op_shadow_init)

    def reset(self):
        self.model.origin.reset()
        if len(self.buffer) != 0:
            self.add_to_replay_buffer()
        self.buffer = []
        self.noise = OUProcess(shape=self.output_dim)

    def action(self, state, show=False):
        logits = self.model.origin.infer(np.array([state]))[0][0]
        noise = self.noise.next() * 0.
        actions = softmax(logits + noise)
        return np.array([np.random.choice(len(actions), p=actions)])

    def feedback(self, state, action, reward, done, new_state):
        reward = np.array([reward])

        experience = state, action, reward, new_state
        self.buffer.append(experience)

    def add_to_replay_buffer(self):
        data = []
        num_timesteps = len(self.buffer)
        for i in range(len(self.buffer[0])):
            data.append(np.concatenate([s[i] for s in self.buffer]))

        # values = np.concatenate(self.values)
        states, actions, rewards, _ = data
        episode = tuple(map(lambda x: np.array([x]), (states, actions, rewards, np.array([num_timesteps]))))
        self.replay_buffer.add(episode)

    def train(self):
        # self.add_to_replay_buffer()
        # states, actions, rewards, dones = self.replay_buffer.sample(args.batch_size)

        if len(self.replay_buffer) >= FLAGS.batch_size:
            states, actions, rewards, lengths = self.replay_buffer.sample(FLAGS.batch_size)
            # logging.info("%s %s %s %s" % (states.shape, lengths.shape, actions.shape, rewards.shape))
            # print states.shape, actions.shape, rewards.shape, dones.shape

            # advantages = rewards + args.gamma * values[1:] - values[:1]
            num_timesteps = states.shape[1]

            mask = np.tile(np.arange(num_timesteps).reshape((1, -1)), (FLAGS.batch_size, 1)) < lengths
            values = self.model.origin.values(states) * mask
            logging.info(values[0])
            advantages = rewards + 0. # FLAGS.gamma * np.pad(values[:, 1:], [(0, 0), (0, 1)], "constant") - values
            for i in reversed(range(num_timesteps - 1)): # TODO should infer from shadow net
                # Generalized Advantage Estimator
                advantages[:, i] += advantages[:, i + 1] * FLAGS.gamma #  * FLAGS.GAE_decay
            advantages -= values
            # print advantages[0, :3], values[0, :3]

            # logging.info("advantages = %s, rewards = %s, values = %s" % (advantages, rewards, values))
            # logging.info("total advantages = %s", advantages[0])
            # print states.shape, advantages.shape, actions.shape, dones.shape
            self.model.origin.train(states, advantages, actions, lengths)
            tf.get_default_session().run(self.model.op_shadow_train)
