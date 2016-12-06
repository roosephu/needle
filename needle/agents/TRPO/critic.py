import tensorflow as tf
import logging

from needle.helper.assign_gradient import AssignGradient
from needle.helper.utils import declare_variables


class Critic(AssignGradient):
    def __init__(self, input_dim):
        self.scope = "critic"
        self.input_dim = input_dim
        self.num_hidden_units = 100

    def build_infer(self):
        self.op_inputs = tf.placeholder(tf.float32, [None, None, self.input_dim])

        self.op_values = self.values(self.op_inputs)

    def build_train(self):
        self.op_loss = tf.reduce_mean(self.op_values) # only one step here
        for v in self.variables:
            logging.info("variables = %s" % (v.name,))
        self.op_grad = self.flatten_gradient(self.op_loss)

    @declare_variables
    def values(self, inputs):
        h = tf.contrib.layers.fully_connected(
            inputs=inputs,
            num_outputs=self.num_hidden_units,
        )
        values = tf.reshape(tf.contrib.layers.fully_connected(
            inputs=h,
            num_outputs=1,
            activation_fn=None,
        ), tf.shape(inputs)[:-1])
        return values

    def infer(self, inputs):
        return tf.get_default_session().run(
            self.op_values,
            feed_dict={
                self.op_inputs: inputs,
            }
        )

    def gradient(self, feed_dict):
        return self.get_flat_gradient(
            self.op_grad,
            feed_dict=feed_dict,
        )

