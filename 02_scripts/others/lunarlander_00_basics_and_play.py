# -*- coding: utf-8 -*-

import os, sys
import gym


# ----------
# REFERENCE
# https://www.gymlibrary.dev/environments/box2d/
# https://www.gymlibrary.dev/environments/box2d/lunar_lander/


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# environment LunarLander
#   - This environment is a classic rocket trajectory optimization problem
#     According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or
#     turn it off. This is the reason why this environment has discrete actions: engine on or off.
# -----------------------------------------------------------------------------------------------------------

env = gym.make('LunarLanderContinuous-v2')

print(f'action space: {env.action_space}')
print(f'observation space: {env.observation_space}')
print(f'reward range: {env.reward_range}')

# episode length is greater than 1000
print(f'max episode step: {env.spec.max_episode_steps}')

# considered SOLVED when average reward is greater than or equal to 200 ?
print(f'reward threshold: {env.spec.reward_threshold}')


# -----------------------------------------------------------------------------------------------------------
# basics
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

env = gym.make('LunarLanderContinuous-v2')

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
        observation, reward, done, info = env.step(action)  # get reward and next scene
        print(f'     observation: {observation}')
        print(f'     reward:      {reward}')
        # ----------
        if done:
            print(f'episode: {i_episode}  -  finished after {t+1} trial')
            break

env.close()


