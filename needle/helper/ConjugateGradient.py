import numpy as np
import gflags

gflags.DEFINE_float("CG_damping", 1e-4, "damping coefficient when applying CG")
gflags.DEFINE_float("CG_norm_limit", 1e-2, "stops when the L2 norm of residual is smaller than this")
FLAGS = gflags.FLAGS


# TODO: extend to non-linear conjugate gradient methods
class ConjugateGradientSolver(object):
    def __init__(self, iterations=10):
        self.damping = FLAGS.CG_damping
        self.norm_limit = FLAGS.CG_norm_limit
        self.iterations = iterations

    def solve(self, mat_vec_prod, y):
        r = y
        l = r.dot(r)
        b = r
        x = np.zeros(y.shape)
        eps = 1e-8

        # regularization term Ax = y => (A + delta I) x = y, too large delta will do harm to FIM.
        # Too small delta results in NaN
        # one idea originally from Levenberg - Marquardt algorithm
        # This idea can even be broadened into diag(A).
        delta = self.damping

        limit = y.shape[0] * self.norm_limit ** 2  # early stop in the case A is not full rank
        for k in range(self.iterations):
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

