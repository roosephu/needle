import tensorflow as tf
import numpy as np
import logging
from ReplayBuffer import ReplayBuffer
from ShadowNet import ShadowNet
from Actor import Actor
from Critic import Critic
from OUProcess import OUProcess
from arguments import args

class Agent:
    def __init__(self):
        self.sess = tf.Session()
        self.actor = ShadowNet(self.sess, lambda: Actor(self.sess, 2, 1, 1e-4), args.tau, "actor")
        self.critic = ShadowNet(self.sess, lambda: Critic(self.sess, 2, 1, 1e-3), args.tau, "critic")
        self.summary_writer = tf.train.SummaryWriter(args.log_dir)

        self.saver = tf.train.Saver()
        if args.mode == "train" and args.init:
            logging.warning("Initialize variables...")
            self.sess.run(tf.initialize_all_variables())
            self.sess.run([self.actor.shadow.op_init, self.critic.shadow.op_init])
        else:
            logging.warning("Restore variables...")
            self.saver.restore(self.sess, args.model_dir)

        self.replay_buffer = ReplayBuffer(1000000)
        self.noise_decay = 1
        self.noise_count = 0

    def reset(self):
        self.noise = OUProcess()
        self.noise_count += 1
        if self.noise_count % 50 == 0:
            self.noise_decay *= 0.8
        self.saver.save(self.sess, args.model_dir)

    def feedback(self, state, action, reward, done, new_state):
        reward = np.array([reward])
        done = np.array([int(done)])

        experience = state, action, reward, done, new_state
        self.replay_buffer.add(experience)

        if len(self.replay_buffer.queue) >= 10000:
            states, actions, rewards, dones, new_states = self.replay_buffer.sample(args.batch_size)

            q_rewards = rewards + args.gamma * (1 - dones) * self.critic.shadow.infer(new_states, self.actor.shadow.infer(new_states))
            # print q_rewards.T[:10]
            # if steps % 100 == 0:
            #     print states[:1], self.actor.origin.infer(states[:1]).T
                # print state, self.actor.origin.infer(state).T

            self.critic.origin.train(states, actions, q_rewards)

            actor_actions = self.actor.origin.infer(states)
            grad_actions = self.critic.origin.grad(states, actor_actions)
            # print grad_actions.T
            self.actor.origin.train(states, grad_actions)

            # grads = self.actor.origin.sess.run(
            #     self.actor.origin.op_grads + self.actor.origin.op_grads2,
            #     feed_dict={
            #         self.actor.origin.op_states: states,
            #         self.actor.origin.op_grad_actions: actor_actions,
            #     }
            # )
            # l = len(grads)
            # for i in range(l // 2):
            #     print grads[i] - grads[i + l // 2]
            # exit()

            self.sess.run([self.actor.shadow.op_train, self.critic.shadow.op_train])

    def action(self, state, show=False):
        # if len(self.replay_buffer.queue) >= 10:
        #     states, _, _, _, _ = self.replay_buffer.sample(31)
        #     state = np.vstack([state, states])
        action = self.actor.origin.infer(state)
        if show:
            logging.warning("action = %s, state = %s" % (action, state))
        return action + self.noise.next() * self.noise_decay
