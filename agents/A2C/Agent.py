import tensorflow as tf
import logging
import numpy as np
from arguments import args
from Model import Model
from agents.Agent import BasicAgent
from helper.ReplayBuffer import ReplayBuffer
from helper.ShadowNet2 import ShadowNet

class Agent(BasicAgent):
    def __init__(self, input_dim, output_dim):
        self.model = ShadowNet(lambda: Model(input_dim, output_dim), args.tau, "A2C")
        self.saver = tf.train.Saver()

        if args.mode == "train" and args.init:
            logging.info("Initializing variables...")
            tf.get_default_session().run(tf.initialize_all_variables())
            tf.get_default_session().run(self.model.op_shadow_init)
        else:
            logging.info("Restore variables...")
            self.saver.restore(tf.get_default_session(), args.model_dir)

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

    def action(self, state, show=False):
        '''
            Input:
                @state: of shape (1, input_dim)
            Output:
                @action: of shape (1, )
        '''
        action = self.model.origin.infer(np.array([state]))
        # self.values.append(value[0])
        # print np.max(action[0][0])
        return np.array([np.random.choice(len(action[0][0]), p=action[0][0])])

    def feedback(self, state, action, reward, done, new_state):
        '''
            Input:
                @state: of shape (1, input_dim)
                @action: of shape (1, )
                @reward: integer
                @done: boolean
                @new_state: of shape (1, input_dim)
        '''
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

    def train_episode(self):
        # self.add_to_replay_buffer()
        # states, actions, rewards, dones = self.replay_buffer.sample(args.batch_size)

        if len(self.replay_buffer) >= args.batch_size:
            states, actions, rewards, lengths = self.replay_buffer.sample(args.batch_size)
            # logging.info("%s %s %s %s" % (states.shape, lengths.shape, actions.shape, rewards.shape))
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
