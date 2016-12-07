import gflags
import logging
import tensorflow as tf

gflags.DEFINE_string("agent", "", "which agent to play")
FLAGS = gflags.FLAGS

registered_agents = {}


def register_agent(name):
    def add(agent):
        if name in registered_agents:
            raise RuntimeError("Duplicated agent registered: %s" % (name,))
        registered_agents[name] = agent
        logging.info("Register agent %s" % (name,))
    return add


def find_agent():
    if FLAGS.agent not in registered_agents:
        raise RuntimeError("No Agent registered found.")

    return registered_agents[FLAGS.agent]


class BasicAgent(object):
    def __init__(self, input_dim, output_dim, noise_gen=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_gen = noise_gen
        self.noise = None

        self.counter = 0

    def init(self):
        tf.get_default_session().run(tf.global_variables_initializer())

    def reset(self):
        if hasattr(self, "noise_gen") and self.noise_gen is not None:
            self.noise = self.noise_gen()
        self.counter = 0

    def action(self, state):
        '''
            Input:
                @state: of shape (1, input_dim)
            Output:
                @action: of shape (1, )
        '''

    def feedback(self, state, action, reward, done, new_state):
        '''
            Input:
                @state: of shape (1, input_dim)
                @action: of shape (1, )
                @reward: integer
                @done: boolean
                @new_state: of shape (1, input_dim)
        '''
        self.counter += 1

    def train(self, finished):
        pass