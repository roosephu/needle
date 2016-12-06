import tensorflow as tf

class Critic(object):
    def __init__(self, state_dim, num_units=100):
        self.scope = "critic"
        self.num_units = num_units
        self.op_states = tf.placeholder(tf.float32, shape=[None, None, state_dim])
        self.op_values = self.values(self.op_states)
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        self.learning_rate = 1e-3

    def values(self, states):
        with tf.variable_scope(self.scope, reuse=type(self.scope) != str) as self.scope:
            h1 = tf.contrib.layers.fully_connected(
                inputs=states,
                num_outputs=self.num_units,
            )
            h2 = tf.contrib.layers.fully_connected(
                inputs=h1,
                num_outputs=self.num_units,
            )
            op_values = tf.reshape(tf.contrib.layers.fully_connected(
                inputs=states,
                num_outputs=1,
                activation_fn=None,
            ), tf.shape(states)[:-1])

        return op_values

    def infer(self, states):
        with tf.variable_scope(self.scope, reuse=True):
            return tf.get_default_session().run(
                self.op_values,
                feed_dict={
                    self.op_states: states,
                }
            )