# -*- coding: utf-8 -*-

import os, sys
import gym



#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Pendulum - v0
# -----------------------------------------------------------------------------------------------------------

env = gym.make('Pendulum-v0')

env.reset()

# state: pendulum angle and angle velocity
# action:  force to pendulum [-2, 2]

for i in range(100):
    action = env.action_space.sample()
    # state, reward, done, info, _ = env.step(action)
    state, reward, done, info = env.step(action)
    print(f'action: {action},  state: {state},  reward: {reward}')

