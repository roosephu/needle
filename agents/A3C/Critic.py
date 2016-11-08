import tensorflow as tf

class Critic:
    def __init__(self, num_units=100):
        self.scope = "critic"
        self.num_units = num_units

    def values(self, states):
        with tf.variable_scope(self.scope, reuse=type(self.scope) != str) as tf.scope:
            h = tf.contrib.layers.fully_connected(
                inputs=states,
                num_outputs=self.num_units,
                biases_initializer=tf.constant_initializer(),
                activation_fn=tf.nn.relu,
            )
            v = tf.reshape(tf.contrib.layers.fully_connected(
                inputs=h,
                num_outputs=1,
                biases_initializer=tf.constant_initializer(),
                activation_fn=None,
            ), [-1])
        return v