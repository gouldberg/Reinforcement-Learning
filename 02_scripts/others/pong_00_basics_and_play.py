# -*- coding: utf-8 -*-

import os, sys
import gym


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# environment Pong  (Pong-v0)
# -----------------------------------------------------------------------------------------------------------

env = gym.make('Pong-v0')

print(f'action space: {env.action_space}')
print(f'observation space: {env.observation_space}')
print(f'reward range: {env.reward_range}')


# Rewards:
# You get score points for getting the ball to pass the opponent's paddle.
# You lose points if the ball passes your paddle.


# -----------------------------------------------------------------------------------------------------------
# basics
# https://www.gymlibrary.dev/environments/atari/pong/
# -----------------------------------------------------------------------------------------------------------

# v0: deterministic
env = gym.make('Pong-v0')


# ----------
# observation space Box(0, 255, (210, 160, 3), uint8)

observation = env.reset()

print(len(observation))
print(observation[0])
print(observation[0].shape)
print(observation[1])


# ----------
# action space Discrete(6)
# 0: NOOP  1: FIRE  2: RIGHT  3: LEFT  4: RIGHTFIRE  5: LEFTFIRE
# random actions
for i in range(10):
    action = env.action_space.sample()
    print(action)


# ----------
obs = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())


# ----------
# PongNoFrameskip-v4
env = gym.make('PongNoFrameskip-v4')
observation = env.reset()

print(len(observation))
print(observation[0])
print(observation[0].shape)
print(observation[1])

obs, reward, terminated, truncated, info = env.step(env.action_space.sample())


# -----------------------------------------------------------------------------------------------------------
# trial and rendering
# (RecordVideo does not work for Pong)
# -----------------------------------------------------------------------------------------------------------

# render_mode requires 'human'
env = gym.make('Pong-v0', render_mode='human')

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
        observation, reward, done, info, _ = env.step(action)  # get reward and next scene
        print(f'     observation: {observation}')
        print(f'     reward:      {reward}')
        # ----------
        if done:
            print(f'episode: {i_episode}  -  finished after {t+1} trial')
            break

env.close()


# -----------------------------------------------------------------------------------------------------------
# trial and rendering:  PongNoFrameskip-v4
# -----------------------------------------------------------------------------------------------------------

env = gym.make('PongNoFrameskip-v4', render_mode='human')

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
        observation, reward, done, info, _ = env.step(action)  # get reward and next scene
        print(f'     observation: {observation}')
        print(f'     reward:      {reward}')
        # ----------
        if done:
            print(f'episode: {i_episode}  -  finished after {t+1} trial')
            break

env.close()


