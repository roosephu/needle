import tensorflow as tf

class Actor(object):
    def __init__(self, action_dim, num_units=100):
        self.scope = "actor"
        self.num_units = num_units
        self.action_dim = action_dim

    def actions(self, states):
        with tf.variable_scope(self.scope, reuse=type(self.scope) != str) as self.scope:
            h1 = tf.contrib.layers.fully_connected(
                inputs=states,
                num_outputs=self.num_units,
                biases_initializer=tf.constant_initializer(),
                # weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                activation_fn=tf.nn.relu,
            )
            h2 = tf.contrib.layers.fully_connected(
                inputs=h1,
                num_outputs=self.num_units,
                biases_initializer=tf.constant_initializer(),
                # weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                activation_fn=tf.nn.relu,
            )
            logits = tf.contrib.layers.fully_connected(
                inputs=states,
                num_outputs=self.action_dim,
                # weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                biases_initializer=tf.constant_initializer(),
                activation_fn=None,
            )
        return logits
