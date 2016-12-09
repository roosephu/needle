import numpy as np
import gflags
import logging
from cached_property import cached_property
from needle.helper.buffer.BatchBuffer import BatchBuffer

FLAGS = gflags.FLAGS


class Batcher(object):
    @cached_property
    def batch_mode(self):
        return "episode"

    @cached_property
    def _episode_count(self):
        return 0

    @cached_property
    def _buffer(self):
        return BatchBuffer()

    @cached_property
    def _episode_buffer(self):
        return BatchBuffer()

    def feedback(self, state, action, reward, done, new_state):
        reward = np.array([reward])

        experience = state, action, reward, new_state
        self._episode_buffer.add(experience)

    def train_batch(self, length, mask, states, actions, rewards, new_states):
        raise RuntimeError("not implemented")

    def train(self, done):
        states, actions, rewards, new_states = self._episode_buffer.get()
        length = len(states)
        if self.batch_mode == "step":
            self._episode_count += length
        elif self.batch_mode == "episode":
            self._episode_count += 1

        mask = np.ones(length)

        episode = np.array([length]), np.array([mask]), np.array([states]), np.array([actions]), \
                  np.array([rewards]), np.array([new_states])
        self._buffer.add(episode)
        # logging.info("buffer = %s" % (self._episode_count))
        if self._episode_count >= FLAGS.batch_size:
            logging.info("========== train iteration ==========")

            self._episode_count = 0
            data = self._buffer.get()
            self.train_batch(*data)
