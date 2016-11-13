import logging
import tensorflow as tf
import gym
import gflags
import sys
from needle.agents import find_agent
from needle.adaptors import find_adaptor

gflags.DEFINE_string("mode", "infer", "inference or training (default infer)")
gflags.DEFINE_integer("batch_size", 10, "configure batch size (default 10)")
gflags.DEFINE_float("gamma", 0.99, "value discount per step")
gflags.DEFINE_string("model_dir", "", "directory to save models")
gflags.DEFINE_string("log_dir", "", "directory to save logs")
gflags.DEFINE_boolean("train_without_init", False, "initialize all variables when training")
gflags.DEFINE_string("monitor", "", "path to save recordings")
gflags.DEFINE_integer("iterations", 10000, "# iterations to run")
gflags.DEFINE_float("learning_rate", 1e-3, "learning rate")
FLAGS = gflags.FLAGS


def main():
    logging.root.setLevel(logging.INFO)

    env = gym.make(FLAGS.env)
    if FLAGS.monitor != "":
        env.monitor.start(FLAGS.monitor)
    # logging.warning("action space: %s, %s, %s" % (env.action_space, env.action_space.high, env.action_space.low))

    logging.warning("Making new agent: %s" % (FLAGS.agent,))
    adaptor = find_adaptor()(env)
    agent = find_agent()(adaptor.input_dim, adaptor.output_dim)

    saver = tf.train.Saver()
    if (FLAGS.mode == "train" and FLAGS.train_without_init) or FLAGS.model_dir == "":
        logging.info("Initializing variables...")
        agent.init()
    else:
        logging.info("Restore variables...")
        saver.restore(tf.get_default_session(), FLAGS.model_dir)

    for iterations in range(FLAGS.iterations):
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
        if FLAGS.mode == "train":
            agent.train()
        logging.info("iteration #%d: total rewards = %.3f, steps = %d" % (iterations, total_rewards, steps))
        if iterations % 10 == 0:
            logging.root.setLevel(logging.INFO)

        if iterations % 50 == 0 and FLAGS.model_dir != "":
            saver.save(tf.get_default_session(), FLAGS.model_dir)

    if FLAGS.monitor != "":
        env.monitor.close()

if __name__ == "__main__":
    FLAGS(sys.argv)
    with tf.Session().as_default():
        main()
