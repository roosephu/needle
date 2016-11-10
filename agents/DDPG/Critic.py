from helper.ShadowNet import Sunlit
import tensorflow as tf
import numpy as np

class Critic(Sunlit):
    def __init__(self, state_dim, action_dim, learning_rate):
        self.learning_rate = learning_rate
        self.action_dim = action_dim
        self.state_dim = state_dim

    def build_infer(self):
        self.op_states = tf.placeholder(tf.float32, [None, self.state_dim])
        self.op_actions = tf.placeholder(tf.float32, [None, self.action_dim])
        self.op_inputs = tf.concat(1, [self.op_states, self.op_actions])

        h1 = tf.contrib.layers.fully_connected(
            inputs=self.op_inputs,
            num_outputs=30,
            biases_initializer=tf.random_normal_initializer(stddev=0.01),
            # normalizer_fn=tf.contrib.layers.batch_norm,
            activation_fn=tf.nn.relu,
        )
        h2 = tf.contrib.layers.fully_connected(
            inputs=h1,
            num_outputs=30,
            biases_initializer=tf.random_normal_initializer(stddev=0.01),
            # normalizer_fn=tf.contrib.layers.batch_norm,
            activation_fn=tf.nn.relu,
        )
        self.op_critic = tf.reshape(tf.contrib.layers.fully_connected(
            inputs=h2,
            num_outputs=1,
            biases_initializer=tf.random_normal_initializer(stddev=0.01),
            activation_fn=None,
        ), [-1])

    def build_train(self):
        self.op_rewards = tf.placeholder(tf.float32, [None])

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        regularization = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(0.01),
            self.variables,
        )
        self.op_loss = tf.reduce_mean((self.op_rewards - self.op_critic)**2) + regularization
        self.op_summary = tf.merge_summary([
            tf.scalar_summary("critic loss", self.op_loss),
            tf.histogram_summary("critic", self.op_critic),
        ])

        self.op_grad_actions = tf.gradients(self.op_critic, self.op_actions)[0]
        self.op_train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.op_loss)

    def train(self, states, actions, rewards):
        _ = tf.get_default_session().run(
            self.op_train,
            feed_dict={
                self.op_states: states,
                self.op_actions: actions,
                self.op_rewards: rewards,
            }
        )

    def infer(self, states, actions):
        critic = tf.get_default_session().run(
            self.op_critic,
            feed_dict={
                self.op_states: states,
                self.op_actions: actions,
            }
        )
        return critic

    def grad(self, states, actions):
        grad = tf.get_default_session().run(
            self.op_grad_actions,
            feed_dict={
                self.op_states: states,
                self.op_actions: actions,
            }
        )
        return grad
