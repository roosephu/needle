from needle.adaptors.CartPole import CartPoleAdaptor
from needle.adaptors.Copy import CopyAdaptor
from needle.adaptors.MountainCar import MountainCarAdaptor
from needle.adaptors.MountainCarContinuous import MountainCarContinuousAdaptor
import gflags

gflags.DEFINE_string("env", "", "environment name")
FLAGS = gflags.FLAGS


def find_adaptor():
    if FLAGS.env == "CartPole-v0":
        adaptor = CartPoleAdaptor
    elif FLAGS.env == "MountainCar-v0":
        adaptor = MountainCarAdaptor
    elif FLAGS.env == "Copy-v0":
        adaptor = CopyAdaptor
    elif FLAGS.env == "MountainCarContinuous-v0":
        adaptor = MountainCarContinuousAdaptor
    else:
        raise RuntimeError("No Adaptor Found.")
    return adaptor
