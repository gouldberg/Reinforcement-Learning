# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from collections import defaultdict
import math

import gym

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ----------
# REFERENCE
# https://github.com/icoxfog417/baby-steps-of-rl-ja


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# EL.frozen_lake_util.py
# -----------------------------------------------------------------------------------------------------------

def show_q_value(Q):
    """
    Show Q-values for FrozenLake-v0.
    To show each action's evaluation,
    a state is shown as 3 x 3 matrix like following.

    +---+---+---+
    |   | u |   |  u: up value
    | l | m | r |  l: left value, r: right value, m: mean value
    |   | d |   |  d: down value
    +---+---+---+
    """
    env = gym.make("FrozenLake-v0")
    # env = gym.make("FrozenLake-v1")
    nrow = env.unwrapped.nrow
    ncol = env.unwrapped.ncol
    state_size = 3
    q_nrow = nrow * state_size
    q_ncol = ncol * state_size
    reward_map = np.zeros((q_nrow, q_ncol))

    for r in range(nrow):
        for c in range(ncol):
            s = r * ncol + c
            state_exist = False
            if isinstance(Q, dict) and s in Q:
                state_exist = True
            elif isinstance(Q, (np.ndarray, np.generic)) and s < Q.shape[0]:
                state_exist = True

            if state_exist:
                # At the display map, the vertical index is reversed.
                _r = 1 + (nrow - 1 - r) * state_size
                _c = 1 + c * state_size
                reward_map[_r][_c - 1] = Q[s][0]  # LEFT = 0
                reward_map[_r - 1][_c] = Q[s][1]  # DOWN = 1
                reward_map[_r][_c + 1] = Q[s][2]  # RIGHT = 2
                reward_map[_r + 1][_c] = Q[s][3]  # UP = 3
                reward_map[_r][_c] = np.mean(Q[s])  # Center

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(reward_map, cmap=cm.RdYlGn, interpolation="bilinear",
               vmax=abs(reward_map).max(), vmin=-abs(reward_map).max())
    ax.set_xlim(-0.5, q_ncol - 0.5)
    ax.set_ylim(-0.5, q_nrow - 0.5)
    ax.set_xticks(np.arange(-0.5, q_ncol, state_size))
    ax.set_yticks(np.arange(-0.5, q_nrow, state_size))
    ax.set_xticklabels(range(ncol + 1))
    ax.set_yticklabels(range(nrow + 1))
    ax.grid(which="both")
    plt.show()

    return reward_map


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Base Agent with epsilon greedy policy:  ELAgent
# -----------------------------------------------------------------------------------------------------------

class ELAgent():

    def __init__(self, epsilon):
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []

    # ----------
    # policy is epsilon greedy
    def policy(self, s, actions):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(actions))
        else:
            if s in self.Q and sum(self.Q[s]) != 0:
                return np.argmax(self.Q[s])
            else:
                return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(
                   episode, mean, std))
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds,
                             alpha=0.1, color="g")
            plt.plot(indices, means, "o-", color="g",
                     label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show()


#############################################################################################################
# Implementation of Agent

# -----------------------------------------------------------------------------------------------------------
# Monte Carlo Agent
#   - 1. Experience:                actual  (Update Q[s][a] after each complete episode)
#   - 2. Objective for updating:    value (Q[s][a])
#   - 3. Action Base:               Off-policy (when estimating value, take action for maximizing value)
#   - Large dependency on sampled episode.  (requires many episodes)
# -----------------------------------------------------------------------------------------------------------

class MonteCarloAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              render=False, report_interval=50):

        self.init_log()
        actions = list(range(env.action_space.n))

        self.Q = defaultdict(lambda: [0] * len(actions))
        N = defaultdict(lambda: [0] * len(actions))

        for e in range(episode_count):
            s = env.reset()
            # ----------
            # s = s[0]
            # ----------
            done = False

            # ----------
            # Play until the end of episode and store experience
            experience = []
            while not done:
                if render:
                    env.render()
                a = self.policy(s, actions)
                # ----------
                # n_state, reward, done, info, _ = env.step(a)
                n_state, reward, done, info = env.step(a)
                # ----------
                experience.append({"state": s, "action": a, "reward": reward})
                s = n_state
            else:
                self.log(reward)

            # ----------
            # Evaluate each state, action.
            for i, x in enumerate(experience):
                s, a = x["state"], x["action"]

                ####################
                # Calculate discounted future reward of s.
                G, t = 0, 0
 
                # Every-Visit:  starting from i
                for j in range(i, len(experience)):
 
                # First-Visit:  starting from first(s, a)
                # for j in range(first(s, a), len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1

                N[s][a] += 1  # count of s, a pair
                alpha = 1 / N[s][a]

                # update Q[s][a]
                self.Q[s][a] += alpha * (G - self.Q[s][a])
                ####################

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


# -----------------------------------------------------------------------------------------------------------
# Q-Learning Agent (Temporal Difference)
# Value base (Off-policy)
#   - 1. Experience:                estimation  (Update Q[s][a] after 1 action:  TD(0))
#   - 2. Objective for updating:    value (Q[s][a])
#   - 3. Action Base:               Off-policy (when estimating value, take action for maximizing value)
#   - Large dependency on sampled episode.  (requires many episodes)
#   - Not large dependency on episode, and learning is basically efficient
#   - but result is in uncertainty due to updating by estimation
# -----------------------------------------------------------------------------------------------------------

class QLearningAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):

        self.init_log()
        actions = list(range(env.action_space.n))

        self.Q = defaultdict(lambda: [0] * len(actions))

        for e in range(episode_count):
            s = env.reset()
            # ----------
            # s = s[0]
            # ----------
            done = False
            while not done:
                if render:
                    env.render()

                a = self.policy(s, actions)
                # ----------
                # n_state, reward, done, info, _ = env.step(a)
                n_state, reward, done, info = env.step(a)
                # ----------
                ####################
                # THIS IS Update by Temporal Difference method

                # value-based:
                # 1. take action to maximize value
                #    note that Q table is updated later by learning rate * temporal difference error.
                gain = reward + gamma * max(self.Q[n_state])
                # ----------
                estimated = self.Q[s][a]

                # 2. update Q baesd on expericence (= temporal difference error = gain - estimated)
                #    'Learning from Delayed Rewards'
                self.Q[s][a] += learning_rate * (gain - estimated)
                ####################

                s = n_state

            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


# -----------------------------------------------------------------------------------------------------------
# SARSA (State-Action-Reward-State-Action)
# Policy-base (On-policy)
#   - 1. Experience:                estimation
#   - 2. Objective for updating:    both (policy --> value)
#   - 3. Action Base:               On-policy (action is probability distribution)
# -----------------------------------------------------------------------------------------------------------

class SARSAAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):

        self.init_log()
        actions = list(range(env.action_space.n))

        self.Q = defaultdict(lambda: [0] * len(actions))

        for e in range(episode_count):
            s = env.reset()
            # ----------
            # s = s[0]
            # ----------
            done = False

            a = self.policy(s, actions)

            while not done:
                if render:
                    env.render()
                # ----------
                # n_state, reward, done, info, _ = env.step(a)
                n_state, reward, done, info = env.step(a)
                # ----------
                ####################
                # 1. take action based on policy (this requires updated Q table)
                n_action = self.policy(n_state, actions)  # On-policy
                gain = reward + gamma * self.Q[n_state][n_action]
                # ----------
                estimated = self.Q[s][a]
                # 2. update Q[s][a] table
                self.Q[s][a] += learning_rate * (gain - estimated)
                ####################

                s = n_state
                a = n_action
            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


# -----------------------------------------------------------------------------------------------------------
# Actor Critic (value and policy base)
#   - 1. Experience:                estimation
#   - 2. Objective for updating:    both (action value AND state value)
#   - 3. Action Base:               On-policy (action is probability distribution)
#                                   if Off-Policy, this is Deterministic Policy Gradient Algorithm (DPG)
# -----------------------------------------------------------------------------------------------------------

# policy base
class Actor(ELAgent):

    def __init__(self, env):
        super().__init__(epsilon=-1)
        nrow = env.observation_space.n
        ncol = env.action_space.n
        self.actions = list(range(env.action_space.n))

        # initialize by random uniform (same probability for each action)
        self.Q = np.random.uniform(0, 1, nrow * ncol).reshape((nrow, ncol))

    # convert self.Q[s] to probability by each action
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    # policy base: take action based on action probability at given state
    def policy(self, s):
        a = np.random.choice(self.actions, 1,
                             p=self.softmax(self.Q[s]))
        return a[0]


class Critic():

    def __init__(self, env):
        states = env.observation_space.n
        self.V = np.zeros(states)


class ActorCritic():

    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class

    def train(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
 
        actor = self.actor_class(env)
        critic = self.critic_class(env)

        actor.init_log()
 
        for e in range(episode_count):
            s = env.reset()
            # ----------
            # s = s[0]
            # ----------
            done = False
            while not done:
                if render:
                    env.render()

                # actor action on policy
                a = actor.policy(s)

                # ----------
                # n_state, reward, done, info, _ = env.step(a)
                n_state, reward, done, info = env.step(a)
                # ----------
                ####################
                # gain is calculated by critic.V
                gain = reward + gamma * critic.V[n_state]
                estimated = critic.V[s]
                td = gain - estimated

                # ----------
                # temporal difference error (experience) is used to update BOTH for Actor and Critic
                # for actor: update Q (action value)
                actor.Q[s][a] += learning_rate * td
                # for critic: update V (state value)
                critic.V[s] += learning_rate * td
                ####################

                s = n_state

            else:
                actor.log(reward)

            if e != 0 and e % report_interval == 0:
                actor.show_reward_log(episode=e)

        return actor, critic


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Register new environment
# -----------------------------------------------------------------------------------------------------------

from gym.envs.registration import register

# register as is_slippery = False
register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"is_slippery": False})


env = gym.make("FrozenLakeEasy-v0")


# Discrete(4)
print(env.action_space)
print(env.action_space.n)


# Discrete(16)
print(env.observation_space)
print(env.observation_space.n)


# -----------------------------------------------------------------------------------------------------------
# Training:  Frozen Lake
# -----------------------------------------------------------------------------------------------------------

# select agent

# Monte Carlo Agent
# agent = MonteCarloAgent(epsilon=0.1)

# Q-Learning Agent (value-base (off-policy), Temporal Difference)
# agent = QLearningAgent(epsilon=0.1)

# SSRSA (State-Action-Reward-State-Action) (policy-base (on-policy))
agent = SARSAAgent(epsilon=0.1)


# ----------
agent.learn(env, episode_count=500)

for i in range(len(agent.Q)):
    print(agent.Q[i])

print(len(agent.Q))


# ----------
reward_map = show_q_value(agent.Q)

print(reward_map.shape)
print(f'min: {reward_map.min()}   max: {reward_map.max()}')


# ----------
agent.show_reward_log()


# ---------
# env.close()


# ----------
nrow = env.unwrapped.nrow
ncol = env.unwrapped.ncol
state_size = 3
q_nrow = nrow * state_size
q_ncol = ncol * state_size

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# plt.imshow(reward_map, cmap=cm.RdYlGn, interpolation="bilinear",
#            vmax=abs(reward_map).max(), vmin=-abs(reward_map).max())
plt.imshow(reward_map, cmap=cm.RdYlGn,
           vmax=abs(reward_map).max(), vmin=-abs(reward_map).max())
ax.set_xlim(-0.5, q_ncol - 0.5)
ax.set_ylim(-0.5, q_nrow - 0.5)
ax.set_xticks(np.arange(-0.5, q_ncol, state_size))
ax.set_yticks(np.arange(-0.5, q_nrow, state_size))
ax.set_xticklabels(range(ncol + 1))
ax.set_yticklabels(range(nrow + 1))
ax.grid(which="both")
plt.show()


# -----------------------------------------------------------------------------------------------------------
# Actor Critic
#  - Requires many episodes but at the end reaches to stable mean rewards. (learning curve is good)
# -----------------------------------------------------------------------------------------------------------

env = gym.make("FrozenLakeEasy-v0")

trainer = ActorCritic(Actor, Critic)

actor, critic = trainer.train(env, episode_count=3000)

reward_map = show_q_value(actor.Q)

actor.show_reward_log()

# env.close()


# -----------------------------------------------------------------------------------------------------------
# Comparison:  Value base (Q-Learning) and Policy base (SARSA) with high epsilon
# -----------------------------------------------------------------------------------------------------------

epsilon_val = 0.3

agent_q = QLearningAgent(epsilon=epsilon_val)
agent_s = SARSAAgent(epsilon=epsilon_val)


# ----------
agent_q.learn(env, episode_count=5000)
agent_s.learn(env, episode_count=5000)


# ----------
# Q-Learning:  higher valuees compared to SARSA
# Q-Learning:  always taking best action and opportunistic
# SARSA:  taking action by policy, leading to realistic, taking into account the possiblity fall in holes

show_q_value(agent_q.Q)
show_q_value(agent_s.Q)

# agent_q.show_reward_log()
# agent_s.show_reward_log()



# -----------------------------------------------------------------------------------------------------------
# Compare Agent
# -----------------------------------------------------------------------------------------------------------

# from multiprocessing import Pool

class CompareAgent(ELAgent):

    def __init__(self, q_learning=True, epsilon=0.33):
        self.q_learning = q_learning
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        self.init_log()
        self.Q = defaultdict(lambda: [0] * len(actions))
        actions = list(range(env.action_space.n))
        for e in range(episode_count):
            s = env.reset()
            # ----------
            # s = s[0]
            # ----------
            done = False
            a = self.policy(s, actions)
            while not done:
                if render:
                    env.render()
                # ----------
                # n_state, reward, done, info, _ = env.step(a)
                n_state, reward, done, info = env.step(a)
                # ----------
                if done and reward == 0:
                    reward = -0.5  # Reward as penalty

                n_action = self.policy(n_state, actions)

                ############################
                if self.q_learning:
                    gain = reward + gamma * max(self.Q[n_state])
                else:
                    gain = reward + gamma * self.Q[n_state][n_action]
                ############################

                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state

                ############################
                if self.q_learning:
                    a = self.policy(s, actions)
                else:
                    a = n_action
                ############################
            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


# ----------
env = gym.make("FrozenLakeEasy-v0")
# env = gym.make("FrozenLakeEasy-v1")

agent_t = CompareAgent(q_learning=True)
agent_t.learn(env, episode_count=3000)
results_t = dict(agent_t.Q)
print(len(results_t))


# ----------
env = gym.make("FrozenLakeEasy-v0")
# env = gym.make("FrozenLakeEasy-v1")

agent_f = CompareAgent(q_learning=False)
agent_f.learn(env, episode_count=3000)
results_f = dict(agent_f.Q)
print(len(results_f))

show_q_value(results_t[15])

for r in results_t:
    show_q_value(r)


# def train(q_learning):
#     env = gym.make("FrozenLakeEasy-v0")
#     agent = CompareAgent(q_learning=q_learning)
#     agent.learn(env, episode_count=3000)
#     return dict(agent.Q)
#
#
# with Pool() as pool:
#     results = pool.map(train, ([True, False]))
#     for r in results:
#         show_q_value(r)
