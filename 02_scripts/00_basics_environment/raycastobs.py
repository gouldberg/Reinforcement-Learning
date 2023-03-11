# -*- coding: utf-8 -*-

import os, sys

from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment


base_path = '/home/kswada/kw/reinforcement_learninig'


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# environment RaycastObservation
# -----------------------------------------------------------------------------------------------------------

# 1. THAT THIS WILL GENERATE NEW WHOLE SCREEN WINDOW (Unity)
# --> 2. Ctrl + Alt + D to come back to original 
# --> 3. Re-Click Visual Studio Code (2 windws there) 

# unity_env = UnityEnvironment('./unity_app/RaycastObservation_app')
unity_env = UnityEnvironment('./unity_app/RaycastObservation_app', no_graphics=True)


# ----------
# wrap to gym
# env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
env = UnityToGymWrapper(unity_env)

print(dir(env))


# ----------
print(f'name: {env.name}')
print(f'observation space: {env.observation_space}')
print(f'action space: {env.action_space}')
print(f'reward_range: {env.reward_range}')


# ----------
state = env.reset()

print(state)


# ----------
while True:
    env.render()

    action = env.action_space.sample()

    # ----------
    state, reward, done, info = env.step(action)
    print('reward:', reward)

    if done:
        print('done')
        state = env.reset()


