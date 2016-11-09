from arguments import args

def find_agent():
    if args.agent == "A2C":
        from A2C.Agent import Agent
    elif args.agent == "DDPG":
        from DDPG.Agent import Agent
    elif args.agent == "DQN":
        from DQN.Agent import Agent
    else:
        raise RuntimeError("No Agent Found.")

    return Agent