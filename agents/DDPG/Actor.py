from helper.ShadowNet import Sunlit
import tensorflow as tf
import numpy as np


class Actor(Sunlit):
    def __init__(self, state_dim, action_dim, learning_rate):
        self.learning_rate = learning_rate
        # build critic network
        self.op_states = tf.placeholder(tf.float32, [None, state_dim])

        h1 = tf.contrib.layers.fully_connected(
            inputs=self.op_states,
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
        self.op_actions = tf.contrib.layers.fully_connected(
            inputs=h2,
            num_outputs=action_dim,
            biases_initializer=tf.random_normal_initializer(stddev=0.01),
            # normalizer_fn=tf.contrib.layers.batch_norm,
            activation_fn=tf.nn.tanh,
        )

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        regularization = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(0.01),
            self.variables,
        )

        self.op_grad_actions = tf.placeholder(tf.float32, [None, action_dim])
        self.op_loss = tf.reduce_sum(-self.op_grad_actions * self.op_actions) # + regularization
        self.op_summary = tf.merge_summary([
            tf.scalar_summary("actor loss", self.op_loss),
            tf.histogram_summary("actor", self.op_actions),
        ])

    # def get_op_train(self):
    #     self.op_grads = tf.gradients(self.op_actions, self.variables, -self.op_grad_actions)
    #     self.op_grads2 = tf.gradients(self.op_loss, self.variables)
    #     return tf.train.AdamOptimizer(1e-4).apply_gradients(zip(self.op_grads2, self.variables))

    def train(self, states, grad_actions):
        _, summary = tf.get_default_session().run(
            [self.op_train, self.op_summary],
            feed_dict={
                self.op_states: states,
                self.op_grad_actions: grad_actions,
            }
        )
        return summary

    def infer(self, states):
        actions = tf.get_default_session().run(
            self.op_actions,
            feed_dict={
                self.op_states: states,
            }
        )
        return actions