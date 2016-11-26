import numpy as np
import tensorflow as tf
import logging
import gflags
from needle.agents import BasicAgent, register_agent
from needle.agents.TNPG.model import Model
from needle.helper.utils import softmax
from needle.helper.OUProcess import OUProcess
from needle.helper.ReplayBuffer import ReplayBuffer

# if program encounters NaN, decrease this value
gflags.DEFINE_float("delta_KL", 0.001, "KL divergence between two sets of parameters")
FLAGS = gflags.FLAGS

line_search_decay = 0.5


def conjugate_gradient(mat_vec_prod, y, iterations=10):  # solve A x = y, given function x -> A x
    # k = np.zeros(y.shape)
    # A = np.zeros((10, 10))
    # for i in range(10):
    #     k[i] = 1.
    #     # logging.debug("i = %s, Ay = %s" % (i, mat_vec_prod(y)))
    #     A[:, i] = mat_vec_prod(k)
    #     k[i] = 0.
    # logging.debug("A = %s, rank(A) = %s" % (A, np.linalg.matrix_rank(A, 1e-4)))

    r = y
    l = r.dot(r)
    b = r
    x = np.zeros(y.shape)
    eps = 1e-8

    # regularization term Ax = y => (A + delta I) x = y, too large delta will do harm to FIM.
    # Too small delta results in NaN
    delta = 1e-4

    limit = y.shape[0] * 1e-4 # early stop in the case A is not full rank
    for k in range(iterations):
        Ab = mat_vec_prod(b) + b * delta
        bAb = b.dot(Ab)
        alpha = l / (bAb + eps)
        x = x + alpha * b
        r = r - alpha * Ab
        # logging.debug("Ab = %s, x = %s, b = %s" % (Ab, x, b))

        new_l = r.dot(r)
        # logging.debug("new l = %s, alpha = %s, bAb = %s, x = %s" % (new_l, alpha, bAb, x))
        if new_l <= limit:
            break
        beta = new_l / (l + eps)
        b = r + beta * b
        l = new_l
    # logging.debug("Ax - y = %s" % (mat_vec_prod(x) - y,))

    return x, x.dot(y - r)

def get_matrix(model, states, choices, advantages, num_paramters):
    func = lambda direction: model.fisher_vector_product(direction, [states], [choices], [advantages])
    A = np.zeros((num_paramters, num_paramters))
    k = np.zeros(num_paramters)
    for i in range(num_paramters):
        k[i] = 1.
        A[:, i] = func(k)
        k[i] = 0.
    # s, v, d = np.linalg.svd(A)
    # logging.debug("singular values of A = %s" % (v,))
    # logging.debug("A = %s" % (A,))
    return A


@register_agent("TNPG")
class Agent(BasicAgent):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.counter = 0

        self.model = Model(input_dim, output_dim)
        self.model.build_infer()
        self.model.build_train()

        self.buffer = ReplayBuffer(FLAGS.replay_buffer_size)
        self.baseline = 20

    def feedback(self, state, action, reward, done, new_state):
        self.counter += 1
        reward = np.array([reward])

        experience = state, action, reward
        self.buffer.add(experience)

        if done or self.counter == 200:
            self.update()

    def update(self):
        states, choices, rewards = self.buffer.latest(self.counter)

        # old_logits = self.model.infer([states])
        # logging.debug("old logits = %s" % (old_logits,))
        # weight = old_var[:8].reshape([4, 2])
        # bias = old_var[8:].reshape([2])
        #
        # logging.info("flat grad = %s" % tf.get_default_session().run(self.model.op_flat_gradient, feed_dict={
        #     self.model.op_inputs: [states],
        # }))
        # g = np.zeros((4, 2))
        # for i in range(len(rewards)):
        #     pi = softmax(states[i].dot(weight) + bias)
        #     logging.info(pi)
        #     g += states[i].reshape(4, 1).dot((pi * (1 - pi)).reshape(1, 2))
        # logging.info("computed = %s" % (g))

        advantages = rewards[:]
        num_timesteps = len(advantages)
        # for i in reversed(range(num_timesteps - 1)):
        #     advantages[i] += advantages[i + 1] * FLAGS.gamma
        advantages *= num_timesteps - self.baseline

        # very important! which value to be chosen remains, however, requires more experiments.
        self.baseline = self.baseline * 0.9 + num_timesteps * 0.1

        gradient = self.model.gradient([states], [choices], [advantages])
        # old_loss = self.model.get_loss([states], [choices], [advantages])
        # logging.info("old loss = %s" % (old_loss,))

        # T = np.zeros((10, 10))
        # for i in range(num_timesteps):
        #     T += get_matrix(self.model, states[i:i + 1], choices[i:i + 1], advantages[i:i + 1], 10)
        #     s, v, d = np.linalg.svd(T)
        #     logging.debug("v  = %s" % (v / (i + 1),))
        # T /= num_timesteps

        # T = get_matrix(self.model, states, choices, advantages, 10)
        # logging.debug("T = %s" % (np.linalg.inv(T),))

        natural_gradient, dot_prod = conjugate_gradient(
            lambda direction: self.model.fisher_vector_product(direction, [states], [choices], [advantages]),
            gradient,
        )
        natural_gradient *= np.sqrt(2 * FLAGS.delta_KL / dot_prod)
        variables = self.model.get_variables()

        # logging.debug("gradient = %s" % (gradient,))
        # natural_gradient *= 0.1
        # natural_gradient = gradient * 0.01

        # logging.info("step size = %s" % (step_size,))
        # new_logits = self.model.infer([states])
        # logging.debug("new logits = %s" % (new_logits,))

        # logging.debug("xAx = %s, natgrad dot grad = %s" % (dot_prod, natural_gradient.dot(gradient)))
        # logging.debug("gradient  = %s" % (gradient,))
        # logging.debug("natgrad   = %s" % (natural_gradient,))
        # logging.info("variables = %s" % (variables,))

        old_loss, old_KL, old_actions, _ = self.model.test([states], [choices], [advantages])
        # logging.info("old loss = %s, old KL = %s" % (old_loss, old_KL))

        self.model.apply_delta(natural_gradient)

        while True:
            new_loss, new_KL, new_actions, var = self.model.test([states], [choices], [advantages], old_actions)
            # logging.debug("new variables %s" % (var,))
            # KL_divergence = np.mean(np.sum(old_actions * np.log(old_actions / new_actions), axis=2))
            # logging.debug("    variables %s" % (variables - natural_gradient,))
            # logging.debug("old_actions = %s" % (old_actions[0].T))
            # logging.debug("new_actions = %s" % (new_actions[0].T))
            # logging.debug("shape = %s" % (np.sum(old_actions * np.log(old_actions / new_actions), axis=2).shape,))

            # logging.info("new loss = %s, KL divergence = %s" % (new_loss, new_KL - old_KL))
            if new_KL - old_KL <= FLAGS.delta_KL and new_loss <= old_loss:
                break
            self.model.apply_delta(natural_gradient * (line_search_decay - 1))
            natural_gradient *= line_search_decay

        # self.model.apply_delta(-natural_gradient)
        # self.model.train(natural_gradient)  # TODO: check if it is SGD

        # old_dist = softmax(old_logits)
        # new_dist = softmax(new_logits)
        # logging.info("KL divergence = %s" % (np.mean(np.sum(old_dist * np.log(old_dist / new_dist), axis=-1))))

    def action(self, inputs):
        logits = self.model.infer(np.array([inputs]))[0][0]
        noise = self.noise.next() * FLAGS.noise_weight
        actions = softmax(logits + noise)
        # actions = (actions + 0.01) / (self.output_dim * 0.01 + 1)
        # logging.debug("logits = %s" % (logits - max(logits),))
        return np.array([np.random.choice(len(actions), p=actions)])

    def reset(self):
        self.noise = OUProcess()
        self.model.reset()
        self.counter = 0
