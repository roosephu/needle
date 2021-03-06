import gflags
import numpy as np
import tensorflow as tf

from needle.agents import BasicAgent, register_agent
from needle.agents.DQN.value import Value
from needle.helper.buffer.ReplayBuffer import ReplayBuffer
from needle.helper.shadow_net import ShadowNet

gflags.DEFINE_float("epsilon", 0.05, "eps-greedy to explore")
FLAGS = gflags.FLAGS


@register_agent("DQN")
class Agent(BasicAgent):
    def __init__(self, input_dim, output_dim):
        super(Agent, self).__init__(input_dim, output_dim)
        # self.input_dim = input_dim
        # self.output_dim = output_dim
        self.value = ShadowNet(lambda: Value(input_dim, output_dim, 1e-2), FLAGS.tau, "value")
        self.saver = tf.train.Saver()

        self.replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size)
        self.epsilon = FLAGS.epsilon

    def init(self):
        tf.get_default_session().run(tf.initialize_all_variables())
        tf.get_default_session().run(self.value.op_shadow_init)

    def action(self, state, show=False):
        if np.random.rand() < self.epsilon:
            return np.array([np.random.randint(self.output_dim)])
        values = self.value.origin.infer(state)
        action = np.argmax(values)
        # logging.debug("values = %s, action = %s" % (values, action))
        return np.array([action])

    def feedback(self, state, action, reward, done, new_state):
        reward = np.array([reward])
        done = np.array([int(done)])
        action = np.array(action)

        # logging.info("action = %s, done = %s, reward = %s" % (action, done, reward))

        experience = state, action, reward, done, new_state
        self.replay_buffer.add(experience)
        if len(self.replay_buffer) >= FLAGS.batch_size:
            states, actions, rewards, dones, new_states = self.replay_buffer.sample(FLAGS.batch_size)
            # logging.info("%s %s %s %s" % (states.shape, actions.shape, rewards.shape, dones.shape))

            optimal_actions = np.argmax(self.value.origin.infer(new_states), axis=1)
            values = rewards + FLAGS.gamma * (1 - dones) * \
                self.value.shadow.infer(new_states)[np.arange(FLAGS.batch_size), optimal_actions]

            self.value.origin.train(states, actions, values)
            tf.get_default_session().run(self.value.op_shadow_train)
