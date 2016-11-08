import argparse

parser = argparse.ArgumentParser(description="Experiments on Reinforcement Learning")
parser.add_argument("--env", help="environment name", required=True, type=str)
parser.add_argument("--batch-size", help="configure batch size (default 10)", default=10, type=int)
parser.add_argument("--mode", help="inference or training (default infer)", default="infer")
parser.add_argument("--gamma", help="value discount per step (default 0.9)", default=0.9, type=float)
parser.add_argument("--model-dir", help="directory to save models", default=None)
parser.add_argument("--log-dir", help="directory to save logs", default=None)
parser.add_argument("--no-init", dest="init", help="initialize all variables", action="store_false")
parser.add_argument("--tau", help="learning rate for update shadow nets", default=0.001, type=float)
parser.add_argument("--monitor", help="path to save recordings", default="", type=str)
parser.add_argument("--epsilon", help="eps-greedy to explore", default=0.05, type=float)
parser.add_argument("--iterations", help="# iterations to run", default=1000, type=int)
parser.add_argument("--num-units", help="# hidden units for LSTM", default=100, type=int)
parser.add_argument("--GAE-decay", help="TD(lambda)", default=0.98, type=float)
parser.set_defaults(init=True)
args = parser.parse_args()

if args.model_dir == None:
    args.model_dir = "./models/" + args.env + '/model'
if args.log_dir == None:
    args.log_dir = "./models/" + args.env + '/log'
