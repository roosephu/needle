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

        self.op_inputs = tf.placeholder(tf.float32, [None, None, state_dim])
        # logging.warning("shape of inputs: %s" % (self.op_inputs))
        batch_size = tf.shape(self.op_inputs)[0]
        num_timesteps = tf.shape(self.op_inputs)[1]
        self.initial_state = self.lstm.zero_state(batch_size, tf.float32)
        self.op_lengths = tf.placeholder(tf.int32, shape=[None, 1])
        # self.op_rewards = tf.placeholder(tf.float32, shape=[None, None])
        self.op_advantages = tf.placeholder(tf.float32, shape=[None, None])
        advantages = self.op_advantages * \
            tf.to_float(tf.tile(tf.expand_dims(tf.range(num_timesteps), 0), (batch_size, 1)) < self.op_lengths)
        self.op_choices = tf.placeholder(tf.int32, shape=[None, None])

        self.actor = Actor(action_dim)
        # self.critic = ShadowNet(tf.get_default_session(), lambda: Critic(), args.tau, "critic")
        self.critic = ShadowNet(lambda: Critic(args.num_units, self.op_advantages), args.tau, "critic")

        self.op_outputs, self.op_states = tf.nn.dynamic_rnn(self.lstm, self.op_inputs, initial_state=self.initial_state)
        # self.op_outputs = tf.reshape(tf.contrib.layers.fully_connected(
        #     inputs=tf.reshape(self.op_inputs, [-1, state_dim]),
        #     num_outputs=args.num_units,
        #     # biases_initializer=tf.constant_initializer(),
        #     activation_fn=None,
        # ), [batch_size, num_timesteps, args.num_units])
        # self.op_states = self.initial_state

        self.op_values = self.critic.origin.values(self.op_outputs)
        self.op_shadow_values = self.critic.shadow.values(self.op_outputs)
        # self.op_advantages = (self.op_rewards - self.op_values) * (1 - self.op_dones)
        self.op_critic_loss = tf.reduce_mean(-advantages * self.op_values)

        self.op_logits = self.actor.actions(self.op_outputs)

        # self.op_logits = tf.Print(
        #     self.op_logits,
        #     [self.op_logits],
        #     message="logits"
        # )

        self.op_actions = tf.nn.softmax(self.op_logits)

        all_index = tf.range(batch_size * num_timesteps)
        choice_index = all_index // num_timesteps * num_timesteps * action_dim \
            + all_index % num_timesteps * action_dim \
            + tf.reshape(self.op_choices, [-1])

        op_actions_log_prob = tf.reshape(tf.gather(tf.reshape(tf.nn.log_softmax(self.op_logits), [-1]), choice_index), [batch_size, num_timesteps])

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

        # don't need op_critic_loss any more due to self.critic
        self.op_loss = self.op_actor_loss + self.op_critic_loss #  + regularization
        self.op_train = tf.train.AdamOptimizer(1e-3).minimize(self.op_loss)

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
        # print rewards
        tf.get_default_session().run(
            [self.op_train, self.critic.shadow.op_train],
            feed_dict={
                self.op_choices: choices,
                self.op_inputs: inputs,
                # self.critic.origin.op_states: inputs,
                # self.op_rewards: rewards,
                self.op_advantages: advantages,
                self.op_lengths: lengths,
            }
        )

def main():
    with tf.Session():
        model = Model(2, 10)

if __name__ == "__main__":
    main()
