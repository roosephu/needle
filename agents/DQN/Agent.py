import tensorflow as tf
import logging
import numpy as np
from helper.ShadowNet import ShadowNet
from helper.ReplayBuffer import ReplayBuffer
from agents.Agent import BasicAgent
from Value import Value
from arguments import args

class Agent(BasicAgent):
    def __init__(self, input_dim, output_dim):
        self.output_dim = output_dim
        self.value = ShadowNet(lambda: Value(input_dim, output_dim, 1e-2), args.tau, "value")
        self.value.origin._finish_origin()
        self.saver = tf.train.Saver()

        self.replay_buffer = ReplayBuffer(args.replay_buffer_size)
        self.epsilon = args.epsilon
        logging.info("epsilon = %s" % (self.epsilon))

    def init(self):
        tf.get_default_session().run(tf.initialize_all_variables())
        tf.get_default_session().run(self.value.op_shadow_init)

    def action(self, state, show=False):
        if np.random.rand() < self.epsilon:
            return np.array([np.random.randint(self.output_dim)])
        values = self.value.origin.infer(state)
        # if show:
        action = np.argmax(values)
        # logging.info("values = %s, action = %s" % (values, action))
        return np.array([action])

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
