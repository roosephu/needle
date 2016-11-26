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

        h = tf.contrib.layers.fully_connected(
            inputs=self.op_inputs,
            num_outputs=FLAGS.num_units,
            biases_initializer=tf.constant_initializer(),
            activation_fn=tf.nn.relu,
        )
        self.op_logits = tf.contrib.layers.fully_connected(
            inputs=h,
            num_outputs=self.action_dim,
            biases_initializer=tf.constant_initializer(),
            activation_fn=tf.nn.relu,
        )
        self.op_actions = tf.nn.softmax(self.op_logits)

    def flatten_gradient(self, loss):
        flattened = []
        for grad in tf.gradients(loss, self.variables):
            flattened.append(tf.reshape(grad, [-1]))
        return tf.concat(flattened, 0)

    def unpack(self, grad):
        grads = []
        index = 0
        for var in self.variables:
            num_elements = np.prod(tf.shape(var))
            grads.append(grad[index:index + num_elements])
            index += num_elements
        return grads

    def build_train(self):
        self.op_advantages = tf.placeholder(tf.float32)
        self.op_choices = tf.placeholder(tf.int32)
        self.op_sur_loss = 0 # TODO: wrong tf.reduce_sum(self.op_actions / tf.stop_gradient(self.op_actions) * self.op_advantages)
        self.op_sur_grad = self.flatten_gradient(self.op_sur_loss)

        # TRUE KL divergence should be the following. However, constants are ignored
        # self.kl_divergence = self.old_distribution * tf.log(self.old_distribution / tf.nn.softmax(self.logits))
        self.op_kl_divergence = tf.reduce_sum(
            tf.stop_gradient(self.op_actions) * tf.nn.log_softmax(self.op_logits)) / self.num_timesteps

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

        # partial L/partial theta
        self.op_flat_gradient = self.flatten_gradient(self.op_kl_divergence)

        num_parameters = tf.shape(self.op_flat_gradient)[0]

        self.op_direction = tf.placeholder(tf.float32, [num_parameters]) # compute F v for direction v
        self.op_product = self.flatten_gradient(tf.reduce_sum(self.op_direction * self.op_flat_gradient))

        self.op_natural_gradient = tf.placeholder(tf.float32, [num_parameters])
        natural_grads = zip(self.unpack(self.op_natural_gradient), self.variables)
        self.op_train = tf.train.GradientDescentOptimizer(1).apply_gradients(natural_grads)

    def fisher_vector_product(self, vec):
        product = tf.get_default_session().run(
            self.op_product,
            feed_dict={
                self.op_direction: vec,
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
            self.op_flat_gradient,
            feed_dict={
                self.op_inputs: inputs,
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
