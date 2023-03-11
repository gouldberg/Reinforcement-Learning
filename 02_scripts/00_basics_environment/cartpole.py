# -*- coding: utf-8 -*-

import os, sys
import gym


# ----------
# REFERENCE
# https://github.com/icoxfog417/baby-steps-of-rl-ja


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# environment CartPole
# -----------------------------------------------------------------------------------------------------------

# REFERENCE: gym environment
# https://github.com/openai/gym/blob/8e5a7ca3e6b4c88100a9550910dfb1a6ed8c5277/gym/envs/__init__.py#L50

env = gym.make('CartPole-v0')

print(f'action space: {env.action_space}')
print(f'observation space: {env.observation_space}')
print(f'reward range: {env.reward_range}')

# episode length is greater than 200
print(f'max episode step: {env.spec.max_episode_steps}')

# considered SOLVED when average reward is greater than or equal to 195.0 over 100 consective trials
print(f'reward threshold: {env.spec.reward_threshold}')


# ----------
env = gym.make('CartPole-v1')

print(f'action space: {env.action_space}')
print(f'observation space: {env.observation_space}')
print(f'reward range: {env.reward_range}')
print(f'max episode step: {env.spec.max_episode_steps}')
print(f'reward threshold: {env.spec.reward_threshold}')


# -----------------------------------------------------------------------------------------------------------
# basics
# https://github.com/openai/gym/wiki/CartPole-v0
# -----------------------------------------------------------------------------------------------------------

env = gym.make('CartPole-v0')


# ----------
# observation space Box(4,)
# [Cart Position, Cart Velocity, Pole Angle, Pole Velocity at Tip]
# reference says for Starting State
# All observations are assigned a uniform random value between ±0.05.

for i in range(5):
    observation = env.reset()
    print(observation)


# ----------
# action space Discrete(2)
# 0: Push cart to the left  1: Push cart to the right
# random actions
for i in range(5):
    action = env.action_space.sample()
    print(action)


# -----------------------------------------------------------------------------------------------------------
# trial and rendering
# -----------------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/reinforcement_learning'


from gym.wrappers import RecordVideo

env = RecordVideo(gym.make("CartPole-v0", render_mode="rgb_array"), os.path.join(base_path, '04_output/cartpole/videos'))

o = env.reset()

for _ in range(100):
    # step() の中で、自動的にビデオ録画されるため、 render() を明示的に呼ぶ必要はない。
    o, r, d, t, i = env.step(env.action_space.sample())
    if d:
        o = env.reset()
