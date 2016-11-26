import gflags
import logging

gflags.DEFINE_string("env", "", "environment name")
FLAGS = gflags.FLAGS

registered_adaptors = {}


def register_adaptor(name):
    def add(agent):
        if name in registered_adaptors:
            raise RuntimeError("Duplicated adaptor registered: %s" % (name,))
        registered_adaptors[name] = agent
        logging.info("Register adaptor %s" % (name,))
    return add


def find_adaptor():
    if FLAGS.env not in registered_adaptors:
        raise RuntimeError("No environment registered found.")

    return registered_adaptors[FLAGS.env]
