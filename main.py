import logging
import tensorflow as tf
import gym
from arguments import args
from agents import find_agent
from adaptors import find_adaptor

def main():
    logging.root.setLevel(logging.INFO)

    env = gym.make(args.env)
    if args.monitor != "":
        env.monitor.start(args.monitor)
    # logging.warning("action space: %s, %s, %s" % (env.action_space, env.action_space.high, env.action_space.low))

    logging.warning("Making new agent: %s" % (args.agent))
    adaptor = find_adaptor()(env)
    agent = find_agent()(adaptor.input_dim, adaptor.output_dim)

    saver = tf.train.Saver()
    if args.mode == "train" and args.init:
        logging.info("Initializing variables...")
        agent.init()
    else:
        logging.info("Restore variables...")
        saver.restore(tf.get_default_session(), args.model_dir)

    for iterations in range(args.iterations):
        if iterations % 10 == 0:
            logging.root.setLevel(logging.DEBUG)
        agent.reset()
        state = env.reset()

        done = False
        total_rewards = 0
        steps = 0

        while not done and steps < env.spec.timestep_limit:
            steps += 1

            action = adaptor.to_env(agent.action(adaptor.state(state)))
            # logging.warning("action = %s" % (action))
            # if steps % 100 == 0:
            #     logging.warning(action[0])
            new_state, reward, done, info = env.step(action)
            if steps == env.spec.timestep_limit:
                done = False
            agent.feedback(adaptor.state(state), adaptor.to_agent(action), reward, done, adaptor.state(new_state))
            state = new_state

            total_rewards += reward
            # if iterations % 10 == 0 and steps % 1 == 0 and args.mode == "infer":
            #     env.render()
            #     logging.warning("step: #%d, action = %.3f, reward = %.3f, iteration = %d" % (steps, action[0], reward, iterations))
            # if episode == 0:
            #     print observation, action, info

        # if iterations % args.batch_size == 0:
        if args.mode == "train":
            agent.train()
        logging.info("iteration #%d: total rewards = %.3f, steps = %d" % (iterations, total_rewards, steps))
        if iterations % 10 == 0:
            logging.root.setLevel(logging.INFO)

        if iterations % 50 == 0:
            saver.save(tf.get_default_session(), args.model_dir)

    if args.monitor != "":
        env.monitor.close()

if __name__ == "__main__":
    with tf.Session().as_default():
        main()
