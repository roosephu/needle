import tensorflow as tf
from helper.ShadowNet import Sunlit

class Value(Sunlit):
    def __init__(self, state_dim, action_dim, learning_rate):
        self.learning_rate = learning_rate
        self.state_dim = state_dim
        self.action_dim = action_dim

    def build_infer(self):
        self.op_states = tf.placeholder(tf.float32, [None, self.state_dim])
        h1 = tf.contrib.layers.fully_connected(
            inputs=self.op_states,
            num_outputs=30,
            biases_initializer=tf.random_normal_initializer(stddev=0.01),
            # normalizer_fn=tf.contrib.layers.batch_norm,
            activation_fn=tf.nn.relu,
        )
        h2 = tf.contrib.layers.fully_connected(
            inputs=h1,
            num_outputs=30,
            biases_initializer=tf.random_normal_initializer(stddev=0.01),
            # normalizer_fn=tf.contrib.layers.batch_norm,
            activation_fn=tf.nn.relu,
        )
        self.op_values = tf.contrib.layers.fully_connected(
            inputs=h2,
            num_outputs=self.action_dim,
            biases_initializer=tf.random_normal_initializer(stddev=0.01),
            # normalizer_fn=tf.contrib.layers.batch_norm,
            activation_fn=None,
        )

    def build_train(self):
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        regularization = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(0.01),
            self.variables,
        )

        self.op_actions = tf.placeholder(tf.int32, [None])
        self.op_rewards = tf.placeholder(tf.float32, [None])
        n = tf.shape(self.op_actions)[0]
        self.op_computed_values = tf.gather(tf.reshape(self.op_values, [-1]), tf.range(n) * self.action_dim + self.op_actions)
        self.op_loss = tf.reduce_sum((self.op_rewards - self.op_computed_values)**2)
        self.op_train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.op_loss)

    def train(self, states, actions, values):
        tf.get_default_session().run(
            self.op_train,
            feed_dict={
                self.op_states: states,
                self.op_actions: actions,
                self.op_rewards: values,
            }
        )

    def infer(self, states):
        values = tf.get_default_session().run(
            self.op_values,
            feed_dict={
                self.op_states: states,
            }
        )
        return values