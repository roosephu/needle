import tensorflow as tf
from needle.helper.assign_gradient import AssignGradient
from cached_property import cached_property
from needle.helper.utils import merge_dict


class FisherVectorProduct(AssignGradient):

    # compute F v for direction v
    @cached_property
    def _op_direction(self):
        return tf.placeholder(tf.float32, [self.num_parameters])

    def build_fisher_vector_product(self, op_kl_divergence):
        op_flat_gradient = self.flatten_gradient(op_kl_divergence)  # partial KL divergence / partial theta
        self._op_product = self.flatten_gradient(tf.reduce_sum(self._op_direction * op_flat_gradient))

    def _infer_fisher_vector_product(self, vec, feed_dict):
        product = tf.get_default_session().run(
            self._op_product,
            feed_dict=merge_dict(feed_dict, {
                self._op_direction: vec,
            })
        )
        return product
