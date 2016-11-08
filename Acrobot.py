#%%
import tensorflow as tf
import numpy as np
import gym
import argparse
import logging

ENV_NAME = "Acrobot-v1"

parser = argparse.ArgumentParser(description="Experiments on Reinforcement Learning")
parser.add_argument("--batch-size", help="configure batch size (default 10)", default=10, type=int)
parser.add_argument("--method", help="inference or training (default infer)", default="infer")
parser.add_argument("--gamma", help="value discount per step (default 0.9)", default=0.9, type=float)
parser.add_argument("--model-dir", help="directory to save models", default=ENV_NAME + '/model')
parser.add_argument("--no-init", dest="init", help="initialize all variables", action="store_false")
parser.set_defaults(init=True)
args = parser.parse_args()

def resblock(x):
    n = int(x.get_shape()[-1])
    h = tf.contrib.layers.fully_connected(
        inputs=x,
        num_outputs=n,
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.contrib.layers.batch_norm,
    )
    y = tf.contrib.layers.fully_connected(
        inputs=h,
        num_outputs=n,
        activation_fn=None,
        normalizer_fn=tf.contrib.layers.batch_norm,
    )
    return tf.nn.relu(y + x)

def transform(x, d):
    return tf.contrib.layers.fully_connected(
        inputs=x,
        num_outputs=d,
        biases_initializer=tf.random_normal_initializer(stddev=0.5),
        activation_fn=None
    )

def resnet(x, units, output_dim, depth):
    if x.get_shape()[-1] != units:
        x = transform(x, units)
    for i in range(depth):
        x = resblock(x)
    if x.get_shape()[-1] != output_dim:
        x = transform(x, output_dim)
    return x

#%%
class Model:
    def __init__(self, input_dim, output_dim):
        op_global_step = tf.Variable(0, name="global_step", trainable=False)
        op_inputs = tf.placeholder(tf.float32, [None, input_dim], name="inputs")
        op_labels = tf.placeholder(tf.int32, [None], name="labels")
        op_rewards = tf.placeholder(tf.float32, [None], name="rewards")
        op_logits = tf.identity(transform(op_inputs, output_dim), name="logits")

        # h = tf.contrib.layers.fully_connected(
        #     inputs=op_inputs,
        #     num_outputs=4,
        #     normalizer_fn=tf.contrib.layers.batch_norm,
        # )
        # h = tf.contrib.layers.fully_connected(
        #     inputs=h,
        #     num_outputs=1,
        #     activation_fn=None,
        # )
        op_baseline = tf.reshape(transform(op_inputs, 1), [-1], name="baseline")
        op_actions = tf.reshape(tf.multinomial(op_logits, 1), [-1], name="actions")

        n = tf.shape(op_labels)[0]
        action_log_prob = tf.gather(tf.reshape(tf.nn.log_softmax(op_logits), [-1]), op_labels + output_dim * tf.range(n))

        reward_loss = tf.reduce_mean(-(op_rewards - op_baseline) * action_log_prob, name="reward_loss")
        baseline_loss = tf.reduce_mean((op_baseline - op_rewards)**2, name="baseline_loss")
        self.op_loss = tf.add(reward_loss, baseline_loss * 1, name="loss")

        op_train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.op_loss, global_step=op_global_step, name="train")

        self.inputs = []
        self.labels = []

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        if args.method == "train" and args.init:
            logging.warning("Initialize variables...")
            self.sess.run(tf.initialize_all_variables())
        else:
            logging.warning("Restore variables...")
            self.saver.restore(self.sess, args.model_dir)

    def train(self, rewards):
        _, loss, reward_loss, baseline_loss = self.sess.run(
            ["train", "loss:0", "reward_loss:0", "baseline_loss:0"],
            feed_dict={
                "inputs:0": np.vstack(self.inputs),
                "labels:0": np.hstack(self.labels),
                "rewards:0": rewards,
            }
        )

        self.inputs = []
        self.labels = []

        if args.method == "train":
            self.saver.save(self.sess, args.model_dir)
        return loss, reward_loss, baseline_loss

    def infer(self, inputs):
        actions, baseline = self.sess.run(
            ["actions:0", "baseline:0"],
            feed_dict={
                "inputs:0": inputs,
            }
        )

        self.inputs += [inputs]
        self.labels += [actions]

        return actions, baseline

    def debug(self):
        x = self.sess.run("fully_connected/weights:0")
        print x[:, 0] - x[:, 1]
        # print self.sess.run(tf.all_variables())

#%%
def main():
    env = gym.make(ENV_NAME)
    print env.action_space, env.observation_space
    model = Model(6, 3)
    baseline = 0

    for iterations in range(1000):
        rewards = []
        episode_rewards = []

        for episode in range(args.batch_size):
            observation = env.reset()
            done = False

            episode_reward = []
            steps = 0

            while not done and steps < 2000:
                action, baseline = model.infer(np.array([observation]))
                action, baseline = action[0], baseline[0]

                last_state = observation
                observation, reward, done, info = env.step(action)

                episode_reward += [reward]
                steps += 1
                if episode == 0 and iterations % 1 == 0 and steps % 1 == 0 and args.method == "infer":
                    env.render()
                    logging.warning("step: #%d, baseline = %.3f" % (steps, baseline))
                # if episode == 0:
                #     print observation, action, info

            # if not done:
            #     episode_reward[steps - 1] -= 10
            for i in range(steps - 1):
                episode_reward[steps - 2 - i] += args.gamma * episode_reward[steps - 1 - i]
            rewards.append(np.array(episode_reward))
            episode_rewards.append(episode_reward[0])

        rewards = np.hstack(rewards)
        loss, reward_loss, baseline_loss = 0, 0, 0
        if args.method == "train":
            loss, reward_loss, baseline_loss = model.train(rewards)

        performance = sum(episode_rewards) / args.batch_size
        # baseline = 0.9 * baseline + 0.1 * performance
        logging.warning("iteration #%d: %.3f, loss = %.3f, reward loss = %.3f, baseline loss = %.3f" % \
            (iterations, performance, loss, reward_loss, baseline_loss))

        # model.debug()
        # for x in tf.all_variables():
            # print x.name

if __name__ == "__main__":
    main()
