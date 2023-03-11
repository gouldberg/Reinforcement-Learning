# -*- coding: utf-8 -*-

import os, sys
import numpy as np

import matplotlib.pyplot as plt


# ----------
# REFERENCE
# https://github.com/oreilly-japan/deep-learning-from-scratch-4


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------------------------------------

# Stationary Bandit (Slot Machine)
# Rate is random
class Bandit:
    def __init__(self, arms=10):
        # stationary bandit
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


# ----------
# Non-Stationary Bandit (Slot Machine)
class NonStatBandit:
    def __init__(self, arms=10, noise=0.1):
        # stationary bandit
        self.arms = arms
        self.noise = noise
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        # ----------
        # noise added
        self.rates += self.noise * np.random.randn(self.arms)
        # ----------
        if rate > np.random.rand():
            return 1
        else:
            return 0


# -----------------------------------------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------------------------------------

class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        # calculate sample average by incremental way
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    # select max Qs slot machine
    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


class AlphaAgent:
    def __init__(self, epsilon, alpha, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.alpha = alpha

    def update(self, action, reward):
        # Not sample average, but constant alpha is applied
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    # select max Qs slot machine
    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Play
# -----------------------------------------------------------------------------------------------------------

# Environment: 10 Slot Machines
bandit = Bandit(arms=10)

# rates to get coins by each machines
print(bandit.rates)

# 10 plays for slot machin 0
for i in range(10):
    print(bandit.play(0))


# ----------
epsilon = 0.1
agent = Agent(epsilon=epsilon, action_size=10)


# ----------
total_reward = 0
total_rewards = []
rates = []

steps = 1000

for step in range(steps):
    action = agent.get_action()
    reward = bandit.play(action)
    agent.update(action, reward)
    total_reward += reward

    total_rewards.append(total_reward)
    rates.append(total_reward / (step + 1))


print(total_reward)


# ----------
# total reward
plt.ylabel('Total reward')
plt.xlabel('Steps')
plt.plot(total_rewards)
plt.show()


# rates
plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(rates)
plt.show()


# -----------------------------------------------------------------------------------------------------------
# Stationary Bandit + Sample Average Agent
# 200 experiments and take average at each step, by each epsilon
# -----------------------------------------------------------------------------------------------------------

runs = 200
steps = 1000
epsilon_list = [0.1, 0.3, 0.01]

all_rates = np.zeros((len(epsilon_list), runs, steps))

for i in range(len(epsilon_list)):
    for run in range(runs):
        bandit = Bandit(arms=10)
        agent = Agent(epsilon=epsilon_list[i], action_size=10)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[i, run] = rates


# ----------
print(all_rates)

avg_rates = np.average(all_rates, axis=1)

# rates:  epsilon = 0.1 is good
plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates[0])
plt.plot(avg_rates[1])
plt.plot(avg_rates[2])
plt.show()


# -----------------------------------------------------------------------------------------------------------
# Non-Stationary Bandit + Sample Average Agent
# 200 experiments and take average at each step, by each epsilon
# -----------------------------------------------------------------------------------------------------------

runs = 200
steps = 1000
epsilon_list = [0.1, 0.3, 0.01]

# noise to environment (bandit)
noise = 0.1

all_rates = np.zeros((len(epsilon_list), runs, steps))

for i in range(len(epsilon_list)):
    for run in range(runs):
        nonstat_bandit = NonStatBandit(arms=10, noise=noise)
        agent = Agent(epsilon=epsilon_list[i], action_size=10)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = nonstat_bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[i, run] = rates


# ----------
print(all_rates)

avg_rates = np.average(all_rates, axis=1)

# rates:  epsilon = 0.1 is good
plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates[0])
plt.plot(avg_rates[1])
plt.plot(avg_rates[2])
plt.show()



# -----------------------------------------------------------------------------------------------------------
# Comparison in Non-Stationary Bandit (environment)
# Sample average Agent   vs.   Alpha (constant) agent
# 200 experiments and take average at each step, by each epsilon
# -----------------------------------------------------------------------------------------------------------

runs = 200
steps = 1000
epsilon = 0.1
alpha = 0.8

# noise to environment (bandit)
noise = 0.1

all_rates = np.zeros((runs, steps))
all_rates2 = np.zeros((runs, steps))

for run in range(runs):
    nonstat_bandit = NonStatBandit(arms=10, noise=noise)
    # ----------
    agent = Agent(epsilon=epsilon, action_size=10)
    # ----------
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = nonstat_bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        rates.append(total_reward / (step + 1))

    all_rates[run] = rates

for run in range(runs):
    nonstat_bandit = NonStatBandit(arms=10, noise=noise)
    # ----------
    agent = AlphaAgent(epsilon=epsilon, alpha=alpha, action_size=10)
    # ----------
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = nonstat_bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        rates.append(total_reward / (step + 1))

    all_rates2[run] = rates


# ----------
avg_rates = np.average(all_rates, axis=0)
avg_rates2 = np.average(all_rates2, axis=0)


# ----------
# alpha constant update is better
plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.plot(avg_rates2)
plt.show()

