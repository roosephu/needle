import tensorflow as tf
import numpy as np
import logging
from arguments import args
from Actor import Actor
from Critic import Critic
from agents.DDPG.ShadowNet import ShadowNet

class Model:
    def __init__(self, state_dim, action_dim):
        self.lstm = tf.nn.rnn_cell.LSTMCell(args.num_units)
        self.state_dim = state_dim
        self.actor = Actor(action_dim)
        # self.critic = ShadowNet(tf.get_default_session(), lambda: Critic(), args.tau, "critic")
        self.critic = Critic()

        self.op_inputs = tf.placeholder(tf.float32, [None, None, state_dim])
        # logging.warning("shape of inputs: %s" % (self.op_inputs))
        batch_size = tf.shape(self.op_inputs)[0]
        num_timesteps = tf.shape(self.op_inputs)[1]
        self.initial_state = self.lstm.zero_state(batch_size, tf.float32)
        self.op_dones = tf.placeholder(tf.float32, shape=[None, None])
        self.op_rewards = tf.placeholder(tf.float32, shape=[None, None])
        self.op_choices = tf.placeholder(tf.int32, shape=[None, None])

        # self.op_outputs, self.op_states = tf.nn.dynamic_rnn(self.lstm, self.op_inputs, initial_state=self.initial_state)
        self.op_outputs = tf.reshape(tf.contrib.layers.fully_connected(
            inputs=tf.reshape(self.op_inputs, [-1, state_dim]),
            num_outputs=args.num_units,
            biases_initializer=tf.constant_initializer(),
            activation_fn=None,
        ), [batch_size, num_timesteps, args.num_units])
        self.op_states = self.initial_state

        batched_outputs = tf.reshape(self.op_outputs, [-1, args.num_units])

        # batched_outputs = tf.Print(batched_outputs, [tf.shape(self.op_outputs), tf.shape(self.op_states)], message="op_inputs.shape op_states.shape")

        batched_values = self.critic.values(batched_outputs)
        self.op_values = tf.reshape(batched_values, [batch_size, num_timesteps])
        op_advantages = self.op_rewards - self.op_values
        self.op_critic_loss = tf.reduce_mean(op_advantages**2)

        batched_actions = self.actor.actions(batched_outputs)
        self.op_actions = tf.reshape(batched_actions, [batch_size, num_timesteps, action_dim])

        all_index = tf.range(batch_size * num_timesteps)
        choice_index = all_index // num_timesteps * num_timesteps * action_dim \
            + all_index % num_timesteps * action_dim \
            + self.op_choices
        op_actions_prob = tf.gather(tf.reshape(self.op_actions, [-1]), choice_index)
        self.op_actor_loss = tf.reduce_mean(tf.stop_gradient(-op_advantages) * tf.log(op_actions_prob))

        regularization = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(0.01),
            tf.all_variables(),
        )

        self.op_loss = self.op_actor_loss + self.op_critic_loss + regularization
        self.op_train = tf.train.AdamOptimizer(1e-2).minimize(self.op_loss)

        # loss = []
        # for i in range(max_depth):
        #     op_input = tf.placeholder([batch_size, input_dim])
        #     op_reward = tf.placeholder([batch_size])
        #     op_done = tf.placeholder([batch_size])
        #     op_choices = tf.placeholder([batch_size])

        #     state = self.lstm(op_input, state)
        #     op_action = self.actor.infer(state)
        #     op_value = self.critic.infer(state)
        #     op_advantage = op_reward - op_value

        #     op_action_prob = tf.gather(tf.reshape(op_action, [-1]), tf.range(batch_size) * action_dim + op_action)
        #     loss.append(tf.done * (op_advantage**2 + tf.log(op_action_prob) * tf.stop_gradient(op_advantage)))
        #     self.recurrent.append((state, op_input, op_done, op_action, op_value))
        # self.op_loss = tf.add_n(*loss)
        # self.op_train = tf.train.AdamOptimizer().minimize(self.op_loss)

    def reset(self):
        self.current_state = tf.get_default_session().run(
            self.initial_state,
            feed_dict={
                self.op_inputs: np.zeros((1, 1, self.state_dim)),
            }
        )

    def infer(self, inputs):
        new_state, action, value = tf.get_default_session().run(
            [self.op_states, self.op_actions, self.op_values],
            feed_dict={
                self.op_inputs: inputs,
                self.initial_state: self.current_state,
            }
        )
        self.current_state = new_state
        return action, value

    def train(self, inputs, rewards, choices, dones):
        tf.get_default_session().run(
            [self.op_train],
            feed_dict={
                self.op_choices: choices,
                self.op_inputs: inputs,
                self.op_rewards: rewards,
                self.op_dones: dones,
            }
        )

def main():
    with tf.Session():
        model = Model(2, 10)

if __name__ == "__main__":
    main()
