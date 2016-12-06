import tensorflow as tf
import numpy as np
import logging
import gflags
from needle.agents.A2C.actor import Actor
from needle.agents.A2C.critic import Critic
from needle.helper.shadow_net import Sunlit

gflags.DEFINE_float("entropy_penalty", 0.01, "entropy penalty for policy")
FLAGS = gflags.FLAGS


class Model(Sunlit):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.current_state = None

    def build_infer(self):
        self.lstm = tf.nn.rnn_cell.LSTMCell(FLAGS.num_units)
        self.op_inputs = tf.placeholder(tf.float32, [None, None, self.state_dim])

        self.batch_size = tf.shape(self.op_inputs)[0]
        self.num_timesteps = tf.shape(self.op_inputs)[1]
        self.initial_state = self.lstm.zero_state(self.batch_size, tf.float32)
        # logging.info(self.op_inputs[:, 0, :].get_shape())

        # logging.info("shape of inputs: %s" % (self.op_inputs))
        # self.op_outputs, self.op_states = tf.nn.dynamic_rnn(self.lstm, self.op_inputs, initial_state=self.initial_state)

        h = tf.contrib.layers.fully_connected(
            inputs=self.op_inputs,
            num_outputs=FLAGS.num_units,
            biases_initializer=tf.constant_initializer(),
            activation_fn=tf.nn.relu,
        )
        self.op_outputs = tf.contrib.layers.fully_connected(
            inputs=h,
            num_outputs=FLAGS.num_units,
            biases_initializer=tf.constant_initializer(),
            activation_fn=tf.nn.relu,
        )
        # self,op_outputs = tf.Print(self.op_outputs, [tf.reduce_sum(self.op_outputs)], message="sum")
        self.op_states = self.initial_state

        self.critic = Critic(FLAGS.num_units)
        self.op_values = self.critic.values(self.op_outputs)

    def build_train(self):

        self.actor = Actor(self.action_dim)
        self.op_logits = self.actor.actions(self.op_outputs)
        self.op_actions = tf.nn.softmax(self.op_logits)

        self.learning_rate = FLAGS.learning_rate
        self.op_lengths = tf.placeholder(tf.int32, shape=[None, 1])
        # self.op_rewards = tf.placeholder(tf.float32, shape=[None, None])
        self.op_advantages = tf.placeholder(tf.float32, shape=[None, None])
        advantages = self.op_advantages * \
            tf.to_float(tf.tile(tf.expand_dims(tf.range(self.num_timesteps), 0), (self.batch_size, 1)) < self.op_lengths)
        self.op_choices = tf.placeholder(tf.int32, shape=[None, None])

        # self.op_advantages = (self.op_rewards - self.op_values) * (1 - self.op_dones)
        self.op_critic_loss = tf.reduce_sum(-advantages * self.op_values)

        # self.op_logits = tf.Print(
        #     self.op_logits,
        #     [self.op_logits],
        #     message="logits"
        # )

        op_actions_log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(self.op_logits, self.op_choices)

        # op_actions_prob = tf.Print(
        #     op_actions_prob,
        #     [tf.shape(op_actions_prob), tf.shape(self.op_advantages), op_actions_prob, num_timesteps],
        #     message="advantage",
        #     first_n=-1,
        # )

        self.op_entropy_penalty = tf.nn.log_softmax(self.op_logits) * self.op_actions * FLAGS.entropy_penalty

        self.op_actor_loss = tf.reduce_sum(-advantages * op_actions_log_prob)

        regularization = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(0.01),
            # tf.all_variables(), # TODO all trainable variables?
            tf.global_variables(),
        )

        # for var in tf.all_variables():
        #     logging.info("var = %s" % (var.name,))

        self.op_loss = self.op_actor_loss * 0.1 + self.op_critic_loss + self.op_entropy_penalty # #  + regularization
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # grads_and_vars = optimizer.compute_gradients(self.op_loss)
        # clipped_grads_and_vars = zip(
        #     tf.clip_by_global_norm([grad for grad, _ in grads_and_vars], 10)[0],
        #     [var for _, var in grads_and_vars],
        # )
        # print grads_and_vars, clipped_grads_and_vars
        # self.op_train = optimizer.apply_gradients(clipped_grads_and_vars)
        self.op_train = optimizer.minimize(self.op_loss)

    def reset(self):
        self.current_state = tf.get_default_session().run(
            self.initial_state,
            feed_dict={
                self.op_inputs: np.zeros((1, 1, self.state_dim)),
            }
        )

    def infer(self, inputs):
        new_state, logits = tf.get_default_session().run(
            [self.op_states, self.op_logits],
            feed_dict={
                self.op_inputs: inputs,
                self.initial_state: self.current_state,
            }
        )
        self.current_state = new_state
        return logits

    def values(self, state, inputs):
        values = tf.get_default_session().run(
            self.op_values,
            feed_dict={
                self.op_inputs: inputs,
                self.initial_state: state,
            }
        )
        return values

    def train(self, initial_state, inputs, advantages, choices, lengths):
        tf.get_default_session().run(
            self.op_train,
            feed_dict={
                self.initial_state: initial_state,
                self.op_choices: choices,
                self.op_inputs: inputs,
                self.op_advantages: advantages,
                self.op_lengths: lengths,
            }
        )

def main():
    with tf.Session():
        model = Model(2, 10)

if __name__ == "__main__":
    main()
