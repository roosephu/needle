from arguments import args
from CartPole import CartPoleAdaptor
from Copy import CopyAdaptor
from MountainCar import MountainCarAdaptor
from MountainCarContinuous import MountainCarContinuousAdaptor

def find_adaptor():
    if args.env == "CartPole-v0":
        adaptor = CartPoleAdaptor
    elif args.env == "MountainCar-v0":
        adaptor = MountainCarAdaptor
    elif args.env == "Copy-v0":
        adaptor = CopyAdaptor
    elif args.env == "MountainCarContinuous-v0":
        adaptor = MountainCarContinuousAdaptor
    else:
        raise RuntimeError("No Adaptor Found.")
    return adaptor