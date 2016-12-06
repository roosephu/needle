import tensorflow as tf
import numpy as np
from cached_property import cached_property
import logging


class VariableList(object):
    @cached_property
    def variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    @cached_property
    def num_parameters(self):
        num_parameters = 0
        for var in self.variables:
            num_parameters += int(np.prod(var.get_shape()))
        return num_parameters

    @cached_property
    def op_variables(self):
        flattened = []
        for var in self.variables:
            flattened.append(tf.reshape(var, [-1]))
            logging.warning("var = %s, shape = %s" % (var.name, var.get_shape()))
        return tf.concat(0, flattened)

    def get_variables(self):
        return tf.get_default_session().run(
            self.variables
        )

    def flatten_gradient(self, op_loss):
        flattened = []
        for grad in tf.gradients(op_loss, self.variables):
            flattened.append(tf.reshape(grad, [-1]))
        # logging.debug(flattened)
        return tf.concat(0, flattened)

    def get_flat_gradient(self, grad, feed_dict):
        return tf.get_default_session().run(
            grad,
            feed_dict=feed_dict,
        )
