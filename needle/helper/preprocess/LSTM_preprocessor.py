import tensorflow as tf
import numpy as np


class LSTMPreprocessor(object):
    def __init__(self, input_dim, output_dim, **options):
        self.scope = "preprocess"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_units = options.get("num_units", 100)

        self.current_state = None

    def preprocess(self, op_inputs):
        self.lstm = tf.nn.rnn_cell.LSTMCell(self.num_units)
        self.op_inputs = op_inputs
        batch_size = tf.shape(op_inputs)[0]
        self.initial_state = self.lstm.zero_state(batch_size, tf.float32)
        self.op_outputs, self.op_states = tf.nn.dynamic_rnn(self.lstm, op_inputs, initial_state=self.initial_state)
        return self.op_outputs, self.op_states

    def reset(self):
        self.current_state = tf.get_default_session().run(
            self.initial_state,
            feed_dict={
                self.op_inputs: np.zeros((1, 1, self.input_dim)),
            }
        )
