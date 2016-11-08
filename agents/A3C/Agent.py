import tensorflow as tf
import logging
import numpy as np
from arguments import args
from Model import Model

def decode(action_space, index):
    if len(action_space) == 1:
        return [index]
    return [index % action_space[0]] + decode(action_space[1:], index / action_space[0])

def encode(index):
    # return index[0] + index[1] * 2 + index[2] * 4
    return index

class Agent:
    def __init__(self, env):
        self.model = Model(4, 2)
        self.saver = tf.train.Saver()

        if args.mode == "train" and args.init:
            logging.warning("Initializing variables...")
            tf.get_default_session().run(tf.initialize_all_variables())
        else:
            logging.warning("Restore variables...")
            self.saver.restore(tf.get_default_session(), args.model_dir)

        self.env = env
        self.action_space = (2, ) # (2, 2, 5)
        self.buffer = []
        self.values = []

    def reset(self):
        self.model.reset()
        self.buffer = []
        self.values = []

    def action(self, state, show=False):
        action, value = self.model.infer(np.array([state]))
        self.values.append(value[0])
        # print action, value
        return decode(self.action_space, np.random.choice(len(action[0][0]), p=action[0][0]))

    def feedback(self, state, action, reward, done, new_state):
        reward = np.array([reward])
        done = np.array([int(done)])
        # print encode(action[0]), action[0], decode(self.action_space, encode(action[0]))
        action = np.array([encode(action[0])]) # ad-hoc

        experience = state, action, reward, done, new_state
        self.buffer.append(experience)

    def train(self):
        data = []
        for i in range(len(self.buffer[0])):
            data.append(np.concatenate([s[i] for s in self.buffer]))

        states, actions, rewards, dones, _ = data
        values = np.concatenate(self.values + [[0]])
        advantages = rewards + args.gamma * values[1:] - values[:1]
        for i in reversed(range(len(advantages) - 1)):
            advantages[i] += advantages[i + 1] * args.gamma * args.GAE_decay
        # logging.warning("advantages = %s, rewards = %s, values = %s" % (advantages, rewards, values))
        # logging.warning("total advantages = %s", advantages[0])
        # print states.shape, advantages.shape, actions.shape, dones.shape
        self.model.train([states], [advantages], [actions], [dones]) # single batch
