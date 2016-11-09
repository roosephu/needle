import tensorflow as tf
import numpy as np
import logging

class Sunlit:
    def __init__(self):
        pass

    def build_infer(self):
        raise UserWarning("No Inference Implemented")

    def build_train(self):
        raise UserWarning("No Training Implemented")


class ShadowNet:
    def __init__(self, create_model, tau, name):
        scope_name = "shadow_net_" + name
        with tf.variable_scope(scope_name):
            with tf.variable_scope("origin") as origin_scope:
                self.origin = create_model()
                self.origin.build_infer()
                self.origin.build_train()

            with tf.variable_scope("shadow") as shadow_scope:
                self.shadow = create_model()
                self.shadow.build_infer()
                all_variables = {var.name: var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}
                shadow_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

                updates, inits = [], []
                for shadow_var in shadow_variables:
                    # logging.info("%s" % (origin_var.name))
                    suffix = shadow_var.name[len(origin_scope.name):]
                    origin_var = all_variables[shadow_var.name[:len(origin_scope.name) - len("shadow")] + "origin" + suffix]
                    logging.debug("%s %s" % (origin_var.name, shadow_var.name))
                    updates.append(tf.assign(shadow_var, shadow_var * (1 - tau) + tau * origin_var))
                    inits.append(tf.assign(shadow_var, origin_var))

                self.op_shadow_train = tf.group(*updates, name="train")
                self.op_shadow_init  = tf.group(*inits  , name="init")

