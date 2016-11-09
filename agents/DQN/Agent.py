import tensorflow as tf
import logging
import numpy as np
from helper.ShadowNet import ShadowNet
from helper.ReplayBuffer import ReplayBuffer
from agents.DDPG.OUProcess import OUProcess
from Value import Value
from arguments import args

class Agent:
    def __init__(self, env):
        self.value = ShadowNet(lambda: Value(4, 2, 1e-2), args.tau, "value")
        self.value.origin._finish_origin()
        self.saver = tf.train.Saver()

        if args.mode == "train" and args.init:
            logging.info("Initialize variables...")
            tf.get_default_session().run(tf.initialize_all_variables())
            tf.get_default_session().run(self.value.op_shadow_init)
        else:
            logging.info("Restore variables...")
            self.saver.restore(tf.get_default_session(), args.model_dir)

        self.replay_buffer = ReplayBuffer(1000000)
        self.epsilon = args.epsilon
        self.env = env
        logging.info("epsilon = %s" % (self.epsilon))

    def reset(self):
        self.saver.save(tf.get_default_session(), args.model_dir)
        self.noise = OUProcess()

    def action(self, state, show=False):
        if np.random.rand() < self.epsilon:
            return [self.env.action_space.sample()]
        values = self.value.origin.infer(state)
        # if show:
        action = np.argmax(values)
        # logging.info("values = %s, action = %s" % (values, action))
        return [action]

    def feedback(self, state, action, reward, done, new_state):
        reward = np.array([reward])
        done = np.array([int(done)])
        action = np.array(action)

        # logging.info("action = %s, done = %s, reward = %s" % (action, done, reward))

        experience = state, action, reward, done, new_state
        self.replay_buffer.add(experience)
        if len(self.replay_buffer) >= args.batch_size:
            states, actions, rewards, dones, new_states = self.replay_buffer.sample(args.batch_size)
            # logging.info("%s %s %s %s" % (states.shape, actions.shape, rewards.shape, dones.shape))

            optimal_actions = np.argmax(self.value.origin.infer(new_states), axis=1)
            values = rewards + args.gamma * (1 - dones) * self.value.shadow.infer(new_states)[np.arange(args.batch_size), optimal_actions]

            self.value.origin.train(states, actions, values)
            tf.get_default_session().run(self.value.op_shadow_train)
