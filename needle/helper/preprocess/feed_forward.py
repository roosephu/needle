import tensorflow as tf


class FeedForwardPreprocessor(object):
    def __init__(self, input_dim, output_dim, **options):
        self.scope = "preprocess"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_units = options.get("num_units", 100)

    def process(self, inputs):
        with tf.variable_scope(self.scope, reuse=type(self.scope) != str) as self.scope:
            h = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=self.num_units,
            )
            features = tf.contrib.layers.fully_connected(
                inputs=h,
                num_outputs=self.output_dim,
            )
        return features

    def reset(self):
        pass

    def feed_dict(self):
        return {}

