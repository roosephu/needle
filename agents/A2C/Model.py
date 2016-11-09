import tensorflow as tf
import numpy as np
import logging
from arguments import args
from Actor import Actor
from Critic import Critic
from helper.ShadowNet2 import Sunlit

class Model(Sunlit):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def build_infer(self):
        self.lstm = tf.nn.rnn_cell.LSTMCell(args.num_units)
        self.op_inputs = tf.placeholder(tf.float32, [None, None, self.state_dim])

        self.batch_size = tf.shape(self.op_inputs)[0]
        self.initial_state = self.lstm.zero_state(self.batch_size, tf.float32)

        # logging.info("shape of inputs: %s" % (self.op_inputs))
        self.op_outputs, self.op_states = tf.nn.dynamic_rnn(self.lstm, self.op_inputs, initial_state=self.initial_state)
        # self.op_outputs = tf.reshape(tf.contrib.layers.fully_connected(
        #     inputs=tf.reshape(self.op_inputs, [-1, state_dim]),
        #     num_outputs=args.num_units,
        #     # biases_initializer=tf.constant_initializer(),
        #     activation_fn=None,
        # ), [batch_size, num_timesteps, args.num_units])
        # self.op_states = self.initial_state

        self.critic = Critic(args.num_units)
        self.op_values = self.critic.values(self.op_outputs)

    def build_train(self):
        self.num_timesteps = tf.shape(self.op_inputs)[1]

        self.actor = Actor(self.action_dim)
        self.op_logits = self.actor.actions(self.op_outputs)
        self.op_actions = tf.nn.softmax(self.op_logits)

        self.learning_rate = args.learning_rate
        self.op_lengths = tf.placeholder(tf.int32, shape=[None, 1])
        # self.op_rewards = tf.placeholder(tf.float32, shape=[None, None])
        self.op_advantages = tf.placeholder(tf.float32, shape=[None, None])
        advantages = self.op_advantages * \
            tf.to_float(tf.tile(tf.expand_dims(tf.range(self.num_timesteps), 0), (self.batch_size, 1)) < self.op_lengths)
        self.op_choices = tf.placeholder(tf.int32, shape=[None, None])

        # self.op_advantages = (self.op_rewards - self.op_values) * (1 - self.op_dones)
        self.op_critic_loss = tf.reduce_mean(-advantages * self.op_values)

        # self.op_logits = tf.Print(
        #     self.op_logits,
        #     [self.op_logits],
        #     message="logits"
        # )

        all_index = tf.range(self.batch_size * self.num_timesteps)
        choice_index = all_index // self.num_timesteps * self.num_timesteps * self.action_dim \
            + all_index % self.num_timesteps * self.action_dim \
            + tf.reshape(self.op_choices, [-1])

        op_actions_log_prob = tf.reshape(
            tf.gather(tf.reshape(tf.nn.log_softmax(self.op_logits), [-1]), choice_index),
            [self.batch_size, self.num_timesteps]
        )

        # op_actions_prob = tf.Print(
        #     op_actions_prob,
        #     [tf.shape(op_actions_prob), tf.shape(self.op_advantages), op_actions_prob, num_timesteps],
        #     message="advantage",
        #     first_n=-1,
        # )

        self.op_actor_loss = tf.reduce_mean(-advantages * op_actions_log_prob)

        regularization = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(0.01),
            tf.all_variables(), # TODO all trainable variables?
        )

        self.op_loss = self.op_actor_loss + self.op_critic_loss #  + regularization
        self.op_train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.op_loss, name="train")

    def reset(self):
        self.current_state = tf.get_default_session().run(
            self.initial_state,
            feed_dict={
                self.op_inputs: np.zeros((1, 1, self.state_dim)),
            }
        )

    def infer(self, inputs):
        new_state, action = tf.get_default_session().run(
            [self.op_states, self.op_actions],
            feed_dict={
                self.op_inputs: inputs,
                self.initial_state: self.current_state,
            }
        )
        self.current_state = new_state
        return action

    def values(self, inputs):
        values = tf.get_default_session().run(
            self.op_values,
            feed_dict={
                self.op_inputs: inputs,
            }
        )
        return values

    def train(self, inputs, advantages, choices, lengths):
        tf.get_default_session().run(
            self.op_train,
            feed_dict={
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
