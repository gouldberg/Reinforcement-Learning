# -*- coding: utf-8 -*-

import os, sys
import gym


# ----------
# REFERENCE
# https://github.com/icoxfog417/baby-steps-of-rl-ja


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# environment FrozenLake
# -----------------------------------------------------------------------------------------------------------

# REFERENCE: gym environment
# https://github.com/openai/gym/blob/8e5a7ca3e6b4c88100a9550910dfb1a6ed8c5277/gym/envs/__init__.py#L50

# env = gym.make("FrozenLake-v0")
env = gym.make("FrozenLake-v1")

print(f'action space: {env.action_space}')
print(f'observation space: {env.observation_space}')
print(f'reward range: {env.reward_range}')

# episode length is greater than 100
print(f'max episode step: {env.spec.max_episode_steps}')

# considered SOLVED when average reward is greater than or equal to 0.7 ?
print(f'reward threshold: {env.spec.reward_threshold}')


# -----------------------------------------------------------------------------------------------------------
# basics
# https://github.com/openai/gym/wiki/FrozenLake-v0
# The goal of this game is to go from the starting state (S) to the goal state (G)
# by walking only on frozen tiles (F) and avoid holes (H).
# However, the ice is slippery, so you won't always move in the direction you intend (stochastic environment)
# -----------------------------------------------------------------------------------------------------------

# env = gym.make('FrozenLake-v0')
env = gym.make('FrozenLake-v1')


# ----------
# observation space Discrete(16)
# For 4x4 square, counting each position from left to right, top to bottom
# reference says for Starting State
# Starting state is at the top left corner (= 0)

observation = env.reset()
print(observation)


# ----------
# action space Discrete(4)
# 0: Move Left  1: Move Down  2: Move Right  3: Move Up
# random actions
for i in range(5):
    action = env.action_space.sample()
    print(action)


# -----------------------------------------------------------------------------------------------------------
# trial and rendering
# -----------------------------------------------------------------------------------------------------------

# render_mode requires 'human'
env = gym.make('FrozenLake-v1', render_mode='human')

# Reward is 0 for every step taken, 0 for falling in the hole, 1 for reaching the final goal

for i_episode in range(5):
    observation = env.reset()
    print(f'episode: {i_episode}  -  starting state: {observation}')
    # ----------
    for t in range(100):
        env.render()  # render game screen
        # ----------
        action = env.action_space.sample()  # this is random action. replace here to your algorithm!
        print(f'episode: {i_episode}  -  trial: {t+1}')
        print(f'     action: {action}')
        # ----------
        observation, reward, done, _, info = env.step(action)  # get reward and next scene
        print(f'     observation: {observation}')
        print(f'     reward:      {reward}')
        # ----------
        if done:
            print(f'episode: {i_episode}  -  finished after {t+1} trial')
            break

env.close()


