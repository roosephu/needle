import tensorflow as tf
import logging
import numpy as np
from arguments import args
from Model import Model
from helper.ReplayBuffer import ReplayBuffer
from helper.ShadowNet2 import ShadowNet

def decode(action_space, index):
    if len(action_space) == 1:
        return [index]
    return [index % action_space[0]] + decode(action_space[1:], index / action_space[0])

def encode(action_space, index):
    # if len(action_space) == 1:
    #     return index[0]
    # return index[0] + encode(action_space[1:], index[1:]) * action_space[0]
    # return index[0] + index[1] * 2 + index[2] * 4
    return index

class Agent:
    def __init__(self, env):
        self.model = ShadowNet(lambda: Model(4, 2), args.tau, "A2C")
        self.saver = tf.train.Saver()

        if args.mode == "train" and args.init:
            logging.info("Initializing variables...")
            tf.get_default_session().run(tf.initialize_all_variables())
            tf.get_default_session().run(self.model.op_shadow_init)
        else:
            logging.info("Restore variables...")
            self.saver.restore(tf.get_default_session(), args.model_dir)

        self.env = env
        self.action_space = (2, 2, 5)
        self.buffer = []
        # self.values = []

        self.replay_buffer = ReplayBuffer(args.replay_buffer_size)

    def reset(self, save=False):
        self.model.origin.reset()
        if len(self.buffer) != 0:
            self.add_to_replay_buffer()
        self.buffer = []
        if save:
            self.saver.save(tf.get_default_session(), args.model_dir)
        # self.values = []

    def action(self, state, show=False):
        action = self.model.origin.infer(np.array([state]))
        # self.values.append(value[0])
        # print np.max(action[0][0])
        return decode(self.action_space, np.random.choice(len(action[0][0]), p=action[0][0]))

    def feedback(self, state, action, reward, done, new_state):
        reward = np.array([reward])
        # print encode(action[0]), action[0], decode(self.action_space, encode(action[0]))
        action = np.array([encode(self.action_space, action[0])]) # ad-hoc

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

        if len(self.replay_buffer) >= args.batch_size:
            states, actions, rewards, lengths = self.replay_buffer.sample(args.batch_size)
            # print values.shape, dones.shape, actions.shape, states.shape
            # print states.shape, actions.shape, rewards.shape, dones.shape

            # advantages = rewards + args.gamma * values[1:] - values[:1]
            num_timesteps = states.shape[1]

            mask = np.tile(np.arange(num_timesteps).reshape((1, -1)), (args.batch_size, 1)) < lengths
            values = self.model.shadow.values(states) * mask
            advantages = rewards + args.gamma * np.pad(values[:, 1:], [(0, 0), (0, 1)], "constant") - values
            for i in reversed(range(num_timesteps - 1)): # TODO should infer from shadow net
                # Geenralized Advantage Estimator
                advantages[:, i] += advantages[:, i + 1] * args.gamma * args.GAE_decay
            # print advantages[0, :3], values[0, :3]

            # logging.info("advantages = %s, rewards = %s, values = %s" % (advantages, rewards, values))
            # logging.info("total advantages = %s", advantages[0])
            # print states.shape, advantages.shape, actions.shape, dones.shape
            self.model.origin.train(states, advantages, actions, lengths)
            tf.get_default_session().run(self.model.op_shadow_train)
