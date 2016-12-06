import tensorflow as tf
import gflags
from needle.helper.variables_list import VariableList

FLAGS = gflags.FLAGS


class Net(VariableList):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = FLAGS.learning_rate

    def build_infer(self):
        self.op_inputs = tf.placeholder(tf.float32, [None, None, self.state_dim])

        self.batch_size = tf.shape(self.op_inputs)[0]
        # logging.info(self.op_inputs[:, 0, :].get_shape())

        # h = tf.contrib.layers.fully_connected(
        #     inputs=self.op_inputs,
        #     num_outputs=FLAGS.num_units,
        #     # biases_initializer=None,
        #     activation_fn=tf.nn.relu,
        # )
        self.op_logits = tf.contrib.layers.fully_connected(
            inputs=self.op_inputs,
            num_outputs=self.action_dim,
            # biases_initializer=None,
            activation_fn=None,
        )
        self.op_actions = tf.nn.softmax(self.op_logits)

    def build_train(self):

        self.op_advantages = tf.placeholder(tf.float32)
        self.op_choices = tf.placeholder(tf.int32)
        self.op_mask = tf.placeholder(tf.float32)
        self.op_length = tf.placeholder(tf.float32)

        op_actions_log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(self.op_logits, self.op_choices)

        self.op_loss = tf.reduce_sum(-self.op_advantages * op_actions_log_prob * self.op_mask) / \
                       tf.to_float(self.batch_size)
        self.op_train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.op_loss)

    def get_dict(self, lengths, mask, inputs, choices, advantages):
        return {
            self.op_mask: mask,
            self.op_inputs: inputs,
            self.op_length: lengths,
            self.op_choices: choices,
            self.op_advantages: advantages,
        }

    def infer(self, inputs):
        logits = tf.get_default_session().run(
            self.op_logits,
            feed_dict={
                self.op_inputs: inputs,
            }
        )
        return logits

    def reset(self):
        pass

    def train(self, feed_dict):
        tf.get_default_session().run(
            self.op_train,
            feed_dict=feed_dict
        )
