import gflags
from needle.agents.A2C.Agent import Agent as A2CAgent
from needle.agents.DDPG.Agent import Agent as DDPGAgent
from needle.agents.DQN.Agent import Agent as DQNAgent

gflags.DEFINE_string("agent", "", "which agent to play")
FLAGS = gflags.FLAGS


def find_agent():
    if FLAGS.agent == "A2C":
        agent = A2CAgent
    elif FLAGS.agent == "DDPG":
        agent = DDPGAgent
    elif FLAGS.agent == "DQN":
        agent = DQNAgent
    else:
        raise RuntimeError("No Agent Found.")

    return agent
