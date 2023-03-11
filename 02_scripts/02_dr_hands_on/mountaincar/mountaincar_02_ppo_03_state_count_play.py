#!/usr/bin/env python3
import argparse
import gym

import numpy as np
import torch
import torch.nn as nn

import ptan


##############################################################################################
# --------------------------------------------------------------------------------------------
# Net
# --------------------------------------------------------------------------------------------

class MountainCarBasePPO(nn.Module):
    def __init__(self, obs_size, n_actions, hid_size: int = 64):
        super(MountainCarBasePPO, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


##############################################################################################
# --------------------------------------------------------------------------------------------
# play
# --------------------------------------------------------------------------------------------

ENV_ID = "MountainCar-v0"

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    # spec = gym.envs.registry.spec(args.env)
    # spec._kwargs['render'] = False
    env = gym.make(args.env)

    # ----------
    # env._max_episode_steps = 1000
    # ----------

    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    # ----------
    net = MountainCarBasePPO(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load(args.model))

    agent = ptan.agent.PolicyAgent(net.actor, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)

    obs = env.reset()
    total_reward = 0.0
    total_steps = 0

    while True:
        acts, _ = agent([obs])
        obs, reward, done, _ = env.step(acts[0])
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
