# -*- coding: utf-8 -*-

import random
import numpy as np

import pandas as pd
import matplotlib

# ----------
# check path of 'matplotlibrc' file
# and add 'backend: tkgg'
# matplotlib.matplotlib_fname()

import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")

# ----------
# REFERENCE
# https://github.com/icoxfog417/baby-steps-of-rl-ja


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Environment (CoinToss task)
#   - reset
#   - step:  coin toss
# -----------------------------------------------------------------------------------------------------------

class CoinToss():

    def __init__(self, head_probs, max_episode_steps=30):
        self.head_probs = head_probs
        self.max_episode_steps = max_episode_steps
        self.toss_count = 0

    def __len__(self):
        # number of coins
        return len(self.head_probs)

    def reset(self):
        self.toss_count = 0

    def step(self, action):

        # ----------
        # done condition
        final = self.max_episode_steps - 1
        if self.toss_count < final:
            done = False
        elif self.toss_count == final:
            done = True
        else:
            raise Exception("The step count exceeded maximum. \
                            Please reset env.")

        # ----------
        if action >= len(self.head_probs):
            raise Exception("The No.{} coin doesn't exist.".format(action))
        else:
            head_prob = self.head_probs[action]
            if random.random() < head_prob:
                reward = 1.0
            else:
                reward = 0.0
            self.toss_count += 1
            return reward, done


# -----------------------------------------------------------------------------------------------------------
# Agent:  Epsilon Greedy Agent
#   - policy
#   - play
# -----------------------------------------------------------------------------------------------------------

class EpsilonGreedyAgent():

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.V = []

    # ----------
    # epsilon greedy
    def policy(self):
        coins = range(len(self.V))
        if random.random() < self.epsilon:
            return random.choice(coins)
        else:
            # select coin index of max expectation
            return np.argmax(self.V)
    # ----------

    def play(self, env):
        # Initialize estimation
        # N:  number of tosses by each coin
        # V:  expectation by each coin
        N = [0] * len(env)
        self.V = [0] * len(env)

        env.reset()
        done = False
        rewards = []

        idx = 0
        while not done:
            idx += 1
            selected_coin = self.policy()
            # ----------
            # this is step !!
            reward, done = env.step(selected_coin)
            # ----------
            rewards.append(reward)

            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward) / (n + 1)
            N[selected_coin] += 1
            self.V[selected_coin] = new_average

            # ========================
            print(f'{idx} -- seleted coin: {selected_coin}   reward: {reward}   coin average before: {coin_average: .3f}  after: {new_average: .3f}')
            print(f'            updated V: {self.V}')
            # ========================

        return rewards


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# play coin tosses
# -----------------------------------------------------------------------------------------------------------

# environment
# 5 coins, each probability of head --> head_probs
env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
print(len(env))


# ----------
# agent
epsilon = 0.1
agent = EpsilonGreedyAgent(epsilon=epsilon)


# ----------
# 30 tosses
# if set larger tosses (such as 500), you can check conversion of V to ground truth probability
env.max_episode_steps = 30
# env.max_episode_steps = 500


# ----------
# play
rewards = agent.play(env)
print(rewards)

# rewards averaged
print(np.mean(rewards))


# ----------
# final sample estimation of had probability of each coin
print(agent.V)


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# simulation by
#  - epsilon (exploration and exploitation balance)
#  - game steps (max episode, number of tosses)
# -----------------------------------------------------------------------------------------------------------

# 5 coins, each probability of head --> head_probs
env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])

epsilons = [0.0, 0.1, 0.2, 0.5, 0.8]

# number of tosses
game_steps = list(range(10, 310, 10))

result = {}
for e in epsilons:
    agent = EpsilonGreedyAgent(epsilon=e)
    means = []
    for s in game_steps:
        env.max_episode_steps = s
        rewards = agent.play(env)

        # mean of rewards
        means.append(np.mean(rewards))
    result["epsilon={}".format(e)] = means

result["coin toss count"] = game_steps
result = pd.DataFrame(result)
print(result)


# ----------
# epsilon = 0.1, 0.2 (small exploration and large exploitation)
# is good:  the larger coin tossses, the larger mean rewards

result.set_index("coin toss count", drop=True, inplace=True)
result.plot.line(figsize=(10, 5))
plt.show()

