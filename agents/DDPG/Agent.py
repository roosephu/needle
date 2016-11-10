import tensorflow as tf
import numpy as np
import logging
from helper.ReplayBuffer import ReplayBuffer
from helper.ShadowNet import ShadowNet
from agents.Agent import BasicAgent
from Actor import Actor
from Critic import Critic
from OUProcess import OUProcess
from arguments import args

class Agent(BasicAgent):
    def __init__(self, input_dim, output_dim):
        self.actor = ShadowNet(lambda: Actor(input_dim, output_dim, 1e-4), args.tau, "actor")
        self.actor.origin._finish_origin()
        self.critic = ShadowNet(lambda: Critic(input_dim, output_dim, 1e-3), args.tau, "critic")
        self.critic.origin._finish_origin()
        self.summary_writer = tf.train.SummaryWriter(args.log_dir)

        self.saver = tf.train.Saver()
        if args.mode == "train" and args.init:
            logging.info("Initialize variables...")
            tf.get_default_session().run(tf.initialize_all_variables())
            tf.get_default_session().run([self.critic.op_shadow_init])
        else:
            logging.info("Restore variables...")
            self.saver.restore(tf.get_default_session(), args.model_dir)

        self.replay_buffer = ReplayBuffer(1000000)
        self.noise_decay = 1
        self.noise_count = 0

    def reset(self, save=False):
        self.noise = OUProcess()
        self.noise_count += 1
        if self.noise_count % 100 == 0:
            self.noise_decay *= 0.8
        if save:
            self.saver.save(tf.get_default_session(), args.model_dir)

    def feedback(self, state, action, reward, done, new_state):
        reward = np.array([reward])
        done = np.array([int(done)])

        experience = state, action, reward, done, new_state
        self.replay_buffer.add(experience)

        if len(self.replay_buffer) >= 10000:
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

            tf.get_default_session().run([self.actor.op_shadow_train, self.critic.op_shadow_train])

    def action(self, state, show=False):
        # if len(self.replay_buffer.queue) >= 10:
        #     states, _, _, _, _ = self.replay_buffer.sample(31)
        #     state = np.vstack([state, states])
        action = self.actor.origin.infer(state)
        if show:
            logging.info("action = %s, state = %s" % (action, state))
        return action + self.noise.next() * self.noise_decay
