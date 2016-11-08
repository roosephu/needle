#%%
import tensorflow as tf
import numpy as np
import gym

#%%
class Model:
    def __init__(self, input_dim, output_dim):
        self.op_inputs = tf.placeholder(tf.float32, [None, input_dim])
        self.op_labels = tf.placeholder(tf.int32, [None])
        self.op_rewards = tf.placeholder(tf.float32, [None])
        # h = tf.contrib.layers.fully_connected(
        #     inputs=self.op_inputs,
        #     num_outputs=5,
        #     activation_fn=tf.sigmoid,
        #     normalizer_fn=tf.contrib.layers.batch_norm
        # )
        self.op_logits = tf.contrib.layers.fully_connected(
            inputs=self.op_inputs,
            num_outputs=output_dim,
            # biases_initializer=tf.random_normal_initializer(stddev=0.5),
            activation_fn=None,
        )
        self.op_actions = tf.multinomial(self.op_logits, 1)

        n = tf.shape(self.op_labels)[0]
        action_log_prob = tf.gather(tf.reshape(tf.nn.log_softmax(self.op_logits), [-1]), self.op_labels + output_dim * tf.range(n))
        self.op_loss = tf.reduce_mean(
            # self.op_rewards * tf.nn.sparse_softmax_cross_entropy_with_logits(self.op_outputs, self.op_labels))
            -self.op_rewards * action_log_prob)

        # optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        self.op_train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.op_loss)

        self.inputs = []
        self.labels = []

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def train(self, rewards):
        self.sess.run(
            self.op_train,
            feed_dict={
                self.op_inputs: np.vstack(self.inputs),
                self.op_labels: np.hstack(self.labels),
                self.op_rewards: rewards,
            }
        )
        self.inputs = []
        self.labels = []

    def infer(self, inputs):
        actions = self.sess.run(
            self.op_actions,
            feed_dict={
                self.op_inputs: inputs,
            }
        )
        actions = actions[:, 0]

        self.inputs += [inputs]
        self.labels += [actions]

        return actions

    def debug(self):
        x = self.sess.run("fully_connected/weights:0")
        print x[:, 0] - x[:, 1]
        # print self.sess.run(tf.all_variables())

#%%
def main():
    config = {
        'batch_size': 10
    }
    env = gym.make('CartPole-v1')
    model = Model(4, 2)
    baseline = 0

    for iterations in range(1000):
        rewards = []
        episode_rewards = []

        for episode in range(config['batch_size']):
            observation = env.reset()
            done = False

            episode_reward = 0
            steps = 0
            while not done:
                # if episode == 0:
                #     env.render()
                action = model.infer(np.array([observation]))[0]
                observation, reward, done, info = env.step(action)
                episode_reward += reward
                steps += 1
                # if episode == 0:
                #     print observation, action, info

            rewards.append(np.arange(steps, 0, -1))
            episode_rewards.append(episode_reward)

        rewards = np.hstack(rewards) - baseline
        model.train(rewards)

        performance = sum(episode_rewards) / config['batch_size']
        baseline = 0.9 * baseline + 0.1 * performance
        print "iteration #%d: %.3f" % (iterations, performance)

        # model.debug()
        # for x in tf.all_variables():
            # print x.name

if __name__ == "__main__":
    main()
