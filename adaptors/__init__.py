from arguments import args
from CartPole import CartPoleAdaptor
from Copy import CopyAdaptor
from MountainCar import MountainCarAdaptor

def find_adaptor():
    if args.env == "CartPole-v0":
        adaptor = CartPoleAdaptor
    elif args.env == "MountainCar-v0":
        adaptor = MountainCarAdaptor
    elif args.env == "Copy-v0":
        adaptor = CopyAdaptor
    else:
        raise RuntimeError("No Adaptor Found.")
    return adaptor