import tensorflow as tf
import logging
import numpy as np
from ..DDPG.ShadowNet import ShadowNet
from ..DDPG.ReplayBuffer import ReplayBuffer
from ..DDPG.OUProcess import OUProcess
from Value import Value
from arguments import args

class Agent:
    def __init__(self, env):
        self.sess = tf.Session()
        self.value = ShadowNet(self.sess, lambda: Value(self.sess, 4, 2, 1e-2), args.tau, "value")
        self.saver = tf.train.Saver()

        if args.mode == "train" and args.init:
            logging.warning("Initialize variables...")
            self.sess.run(tf.initialize_all_variables())
            self.sess.run(self.value.shadow.op_init)
        else:
            logging.warning("Restore variables...")
            self.saver.restore(self.sess, args.model_dir)

        self.replay_buffer = ReplayBuffer(1000000)
        self.epsilon = args.epsilon
        self.env = env
        logging.warning("epsilon = %s" % (self.epsilon))

    def reset(self):
        self.saver.save(self.sess, args.model_dir)
        self.noise = OUProcess()

    def action(self, state, show=False):
        if np.random.rand() < self.epsilon:
            return [self.env.action_space.sample()]
        values = self.value.origin.infer(state)
        # if show:
        action = np.argmax(values)
        # logging.warning("values = %s, action = %s" % (values, action))
        return [action]

    def feedback(self, state, action, reward, done, new_state):
        reward = np.array([reward])
        done = np.array([int(done)])

        # logging.warning("action = %s, done = %s, reward = %s" % (action, done, reward))

        experience = state, action, reward, done, new_state
        self.replay_buffer.add(experience)
        if len(self.replay_buffer.queue) >= args.batch_size:
            states, actions, rewards, dones, new_states = self.replay_buffer.sample(args.batch_size)

            optimal_actions = np.argmax(self.value.origin.infer(new_states), axis=1)
            values = rewards + args.gamma * (1 - dones) * self.value.shadow.infer(new_states)[np.arange(args.batch_size), optimal_actions]

            self.value.origin.train(states, actions, values)
            self.sess.run(self.value.shadow.op_train)
