import tensorflow as tf
import numpy as np
from needle.helper.VariablesList import VariableList
from cached_property import cached_property


class AssignGradient(VariableList):

    @cached_property
    def _op_delta(self):
        return tf.placeholder(tf.float32, [self.num_parameters])

    @cached_property
    def _op_apply_delta(self):
        assigns = []
        for var, delta in zip(self.variables, self._unpack(self._op_delta)):
            assigns.append(tf.assign_sub(var, delta))
        return tf.group(*assigns)

    def _unpack(self, grad):
        grads = []
        index = 0
        for var in self.variables:
            shape = var.get_shape()
            num_elements = int(np.prod(shape))
            grads.append(tf.reshape(grad[index:index + num_elements], shape))
            index += num_elements
        return grads

    def apply_grad(self, grad):
        tf.get_default_session().run(
            self._op_apply_delta,
            feed_dict={
                self._op_delta: grad,
            }
        )



