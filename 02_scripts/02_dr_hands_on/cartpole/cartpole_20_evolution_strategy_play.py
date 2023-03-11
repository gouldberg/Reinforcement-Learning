#!/usr/bin/env python3
import argparse
import gym

import numpy as np
import torch
import torch.nn as nn


##############################################################################################
# --------------------------------------------------------------------------------------------
# Net
# --------------------------------------------------------------------------------------------

# simpe one-hidden-layer NN,
# which gives us the action to take from the observation.

HID_SIZE = 32

class Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


##############################################################################################
# --------------------------------------------------------------------------------------------
# play
# --------------------------------------------------------------------------------------------

ENV_ID = "CartPole-v0"

# device = torch.device("cuda")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    spec = gym.envs.registry.spec(args.env)
    # spec._kwargs['render'] = False
    env = gym.make(args.env)

    # ----------
    # env._max_episode_steps = 1000*2
    # ----------

    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    # ----------
    net = Net(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load(args.model))

    obs = env.reset()
    reward = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        act_prob = net(obs_v)
        acts = act_prob.max(dim=1)[1]
        obs, r, done, _ = env.step(acts.data.numpy()[0])
        reward += r
        steps += 1
        if done:
            break
    print("In %d steps we got %.3f reward" % (steps, reward))
