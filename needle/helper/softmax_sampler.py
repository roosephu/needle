import numpy as np
import gflags
import logging
from needle.helper.utils import softmax

FLAGS = gflags.FLAGS


class SoftmaxSampler(object):
    def softmax_action(self, logits, noise=None):
        # noise = self.noise.next() * FLAGS.noise_weight
        logits = logits[0][0]
        if noise is not None:
            logits = logits + noise.next()
        actions = softmax(logits)
        # actions = (actions + 0.01) / (self.output_dim * 0.01 + 1)
        # logging.debug("logits = %s" % (logits - max(logits),))
        return np.array([np.random.choice(len(actions), p=actions)])
