import tensorflow as tf
import numpy as np
import logging

class Sunlit:
    def __init__(self):
        pass

    def get_op_train(self):
         return tf.train.AdamOptimizer(self.learning_rate).minimize(self.op_loss, name="train")

    def init_operators(self, op_train=None, op_init=None):
        if op_train != None:
            self.op_train = op_train
        else:
            self.op_train = self.get_op_train()

        if op_init != None:
            self.op_init = op_init
        else:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            self.op_init = tf.initialize_variables(variables, name="init")

class ShadowNet:
    def __init__(self, sess, create_model, tau, name):
        scope_name = "shadow_net_" + name
        with tf.variable_scope(scope_name):
            with tf.variable_scope("origin") as origin_scope:
                origin_model = create_model()
                origin_model.init_operators()

            # for v in variables:
            #     logging.warning(v.name)

            with tf.variable_scope("shadow") as shadow_scope:
                shadow_model = create_model()
                all_variables = {var.name: var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}

                updates, inits = [], []
                for origin_var in origin_model.variables:
                    # logging.warning("%s" % (origin_var.name))
                    suffix = origin_var.name[len(origin_scope.name):]
                    shadow_var = all_variables[origin_var.name[:len(origin_scope.name) - len("origin")] + "shadow" + suffix]
                    logging.info("%s %s" % (origin_var.name, shadow_var.name))
                    updates.append(tf.assign(shadow_var, shadow_var * (1 - tau) + tau * origin_var))
                    inits.append(tf.assign(shadow_var, origin_var))

                op_train = tf.group(*updates, name="train")
                op_init  = tf.group(*inits  , name="init")
                shadow_model.init_operators(op_init=op_init, op_train=op_train)

        self.origin = origin_model
        self.shadow = shadow_model
        self.sess = sess
