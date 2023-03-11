#!/usr/bin/env python3
import argparse
import gym
import pybullet_envs

import numpy as np
import torch
import torch.nn as nn


##############################################################################################
# --------------------------------------------------------------------------------------------
# ModelActor
# --------------------------------------------------------------------------------------------

HID_SIZE = 64

class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, act_size),
            # tanh nonlinearity.
            nn.Tanh(),
        )

        # The variance is modeled as a separate network parameter
        # and interpreted as a logarithm of the standard deviation.
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        return self.mu(x)


##############################################################################################
# --------------------------------------------------------------------------------------------
# play
# --------------------------------------------------------------------------------------------

ENV_ID = "HalfCheetahBulletEnv-v0"

# device = torch.device("cuda")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    spec = gym.envs.registry.spec(args.env)
    spec._kwargs['render'] = False
    env = gym.make(args.env)

    # ----------
    env._max_episode_steps = 1000*2
    # ----------

    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    # ----------
    # this should be CPU (not GPU)
    # net = ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    net = ModelActor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(args.model))

    obs = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor(obs)
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        if np.isscalar(action): 
            action = [action]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))

