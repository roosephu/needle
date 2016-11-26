import tensorflow as tf
import numpy as np
import logging
import gflags

FLAGS = gflags.FLAGS


class Model:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = FLAGS.learning_rate

    def build_infer(self):
        self.op_inputs = tf.placeholder(tf.float32, [None, None, self.state_dim])

        self.batch_size = tf.shape(self.op_inputs)[0]
        self.num_timesteps = tf.shape(self.op_inputs)[1]
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

    def flatten_gradient(self, op_loss):
        flattened = []
        for grad in tf.gradients(op_loss, self.variables):
            flattened.append(tf.reshape(grad, [-1]))
        # logging.debug(flattened)
        return tf.concat(0, flattened)

    def flatten_variables(self):
        flattened = []
        for var in self.variables:
            flattened.append(tf.reshape(var, [-1]))
            logging.warning("var = %s, shape = %s" % (var.name, var.get_shape()))
        return tf.concat(0,flattened)

    def unpack(self, grad):
        grads = []
        index = 0
        for var in self.variables:
            shape = var.get_shape()
            num_elements = int(np.prod(shape))
            # logging.debug("num elements = %s, shape = %s" % (num_elements, var.get_shape()))
            grads.append(tf.reshape(grad[index:index + num_elements], shape))
            index += num_elements
        return grads

    def build_train(self):
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

        self.op_advantages = tf.placeholder(tf.float32)
        self.op_choices = tf.placeholder(tf.int32)

        all_index = tf.range(self.batch_size * self.num_timesteps)
        choice_index = all_index // self.num_timesteps * self.num_timesteps * self.action_dim \
            + all_index % self.num_timesteps * self.action_dim \
            + tf.reshape(self.op_choices, [-1])

        op_actions_log_prob = tf.reshape(
            tf.gather(tf.reshape(tf.nn.log_softmax(self.op_logits), [-1]), choice_index),
            [self.batch_size, self.num_timesteps],
        )

        self.op_sur_loss = tf.reduce_sum(-self.op_advantages * op_actions_log_prob) / tf.to_float(self.batch_size)
        self.op_sur_grad = self.flatten_gradient(self.op_sur_loss)

        # TRUE KL divergence should be the following. However, constants are ignored
        # self.kl_divergence = self.old_distribution * tf.log(self.old_distribution / tf.nn.softmax(self.logits))

        self.op_old_actions = tf.identity(self.op_actions)  # IMPORTANT tf.identity
        self.op_kl_divergence = -tf.reduce_sum(tf.stop_gradient(self.op_old_actions) * tf.nn.log_softmax(self.op_logits)) \
            / tf.to_float(self.batch_size * self.num_timesteps)

        # Hessian-vector product
        self.op_flat_gradient = self.flatten_gradient(self.op_kl_divergence)  # partial KL divergence / partial theta

        num_parameters = self.op_flat_gradient.get_shape()[0]

        self.op_direction = tf.placeholder(tf.float32, [num_parameters])  # compute F v for direction v
        self.op_product = self.flatten_gradient(tf.reduce_sum(self.op_direction * self.op_flat_gradient))

        # setting up how to update parameters
        self.op_natural_gradient = tf.placeholder(tf.float32, [num_parameters])
        natural_grads = zip(self.unpack(self.op_natural_gradient), self.variables)
        self.op_train = tf.train.GradientDescentOptimizer(1).apply_gradients(natural_grads)
        # self.op_train = tf.train.GradientDescentOptimizer(0.1).minimize(self.op_sur_loss)

        self.op_variables = self.flatten_variables()
        self.op_delta = tf.placeholder(tf.float32, [num_parameters])

        assigns = []
        for var, delta in zip(self.variables, self.unpack(self.op_delta)):
            assigns.append(tf.assign_sub(var, delta))
        self.op_apply_delta = tf.group(*assigns)

    def fisher_vector_product(self, vec, inputs, choices, advantages):
        product = tf.get_default_session().run(
            self.op_product,
            feed_dict={
                self.op_direction: vec,
                self.op_inputs: inputs,
                self.op_choices: choices,
                self.op_advantages: advantages,
            }
        )
        return product

    def infer(self, inputs):
        logits = tf.get_default_session().run(
            self.op_logits,
            feed_dict={
                self.op_inputs: inputs,
            }
        )
        return logits

    def gradient(self, inputs, choices, advantages):
        gradient = tf.get_default_session().run(
            self.op_sur_grad,
            feed_dict={
                self.op_inputs: inputs,
                self.op_choices: choices,
                self.op_advantages: advantages,
            }
        )
        return gradient

    def train(self, natural_gradient):
        tf.get_default_session().run(
            self.op_train,
            feed_dict={
                self.op_natural_gradient: natural_gradient,
            }
        )

    def reset(self):
        pass

    # def get_loss(self, inputs, choices, advantages):
    #     return tf.get_default_session().run(
    #         self.op_sur_loss,
    #         feed_dict={
    #             self.op_inputs: inputs,
    #             self.op_choices: choices,
    #             self.op_advantages: advantages,
    #         }
    #     )

    def apply_delta(self, delta):
        tf.get_default_session().run(
            self.op_apply_delta,
            feed_dict={
                self.op_delta: delta,
            }
        )

    def test(self, inputs, choices, advantages, old_actions=None):
        feed_dict = {
            self.op_inputs: inputs,
            self.op_choices: choices,
            self.op_advantages: advantages,
        }
        if old_actions is not None:
            feed_dict[self.op_old_actions] = old_actions

        # index = 0
        # for var in self.variables:
        #     shape = var.get_shape()
        #     num_elements = int(np.prod(shape))
        #     feed_dict[var] = variables[index:index + num_elements].reshape(shape)
        #     index += num_elements

        return tf.get_default_session().run(
            [self.op_sur_loss, self.op_kl_divergence, self.op_actions, self.op_variables],
            feed_dict=feed_dict,
        )

    def get_variables(self):
        return tf.get_default_session().run(
            self.op_variables
        )
