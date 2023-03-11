import os, sys
import time
import numpy as np
# import collections
import math

import gym

# pybullet_envs is required
import pybullet_envs

import ptan

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.utils as nn_utils
import torch.optim as optim


##############################################################################################
# --------------------------------------------------------------------------------------------
# ModelA2C, ModelCritic
# --------------------------------------------------------------------------------------------

# Both the actor and critic are placed in the separate networks without sharing weights.
# Critic estimate the mean and the variance for the actions,
# but now the variance is not a separate head of the base network,
# it is just a single parameter of the model.
# This parameter will be adjusted during the training by SGD, but it does not depend on the observation.

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


class ModelCritic(nn.Module):
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x)


# --------------------------------------------------------------------------------------------
# AgentA2C
# --------------------------------------------------------------------------------------------

class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        rnd = np.random.normal(size=logstd.shape)

        # apply noise with variance
        actions = mu + np.exp(logstd) * rnd
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


##############################################################################################
# --------------------------------------------------------------------------------------------
# test_net
# calc_logpob
# --------------------------------------------------------------------------------------------

def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            if np.isscalar(action): 
                action = [action]
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


# --------------------------------------------------------------------------------------------
# calc_adv_ref
# --------------------------------------------------------------------------------------------

GAMMA = 0.99

# lambda factor in the estimator, the value of 0.95 was used in the PPO paper.
GAE_LAMBDA = 0.95


def calc_adv_ref(trajectory, net_crt, states_v, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]),
                                     reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv)))
    ref_v = torch.FloatTensor(list(reversed(result_ref)))
    return adv_v.to(device), ref_v.to(device)


##############################################################################################
# --------------------------------------------------------------------------------------------
# HalfCheetah Agent Learning:  Proximal Policy Optimization (PPO)
# --------------------------------------------------------------------------------------------

# ----------
# device
device = torch.device("cuda")


# ----------
# environment
ENV_ID = "HalfCheetahBulletEnv-v0"

env = gym.make(ENV_ID)
test_env = gym.make(ENV_ID)


# ----------
base_path = '/home/kswada/kw/reinforcement_learning'

args_name = 'trial'
writer = SummaryWriter(comment="-halfcheetah_ppo_" + args_name)

save_path = os.path.join(base_path, f'04_output/02_deep_reinforcement_learning_hands_on/halfcheetah/ppo_{args_name}/model')
os.makedirs(save_path, exist_ok=True)


# ----------
# net:  ModelA2C and ModelCritic
net_act = ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
net_crt = ModelCritic(env.observation_space.shape[0]).to(device)
print(net_act)
print(net_crt)


# ----------
# agent
agent = AgentA2C(net_act, device=device)


# ----------
# experience source
exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)


# ----------
# optimizer
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4

opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR)
opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)


# ----------
trajectory = []
best_reward = None


PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64

# ----------
TRAJECTORY_SIZE = 2049

TEST_ITERS = 10000

with ptan.common.utils.RewardTracker(writer) as tracker:
    for step_idx, exp in enumerate(exp_source):
        rewards_steps = exp_source.pop_rewards_steps()
        if rewards_steps:
            rewards, steps = zip(*rewards_steps)
            writer.add_scalar("episode_steps", np.mean(steps), step_idx)
            tracker.reward(np.mean(rewards), step_idx)

        if step_idx % TEST_ITERS == 0:
            ts = time.time()
            rewards, steps = test_net(net_act, test_env, device=device)
            print("Test done in %.2f sec, reward %.3f, steps %d" % (
                time.time() - ts, rewards, steps))
            writer.add_scalar("test_reward", rewards, step_idx)
            writer.add_scalar("test_steps", steps, step_idx)
            if best_reward is None or best_reward < rewards:
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                    name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                    fname = os.path.join(save_path, name)
                    torch.save(net_act.state_dict(), fname)
                best_reward = rewards

        trajectory.append(exp)
        if len(trajectory) < TRAJECTORY_SIZE:
            continue

        traj_states = [t[0].state for t in trajectory]
        traj_actions = [t[0].action for t in trajectory]
        traj_states_v = torch.FloatTensor(traj_states)
        traj_states_v = traj_states_v.to(device)
        traj_actions_v = torch.FloatTensor(traj_actions)
        traj_actions_v = traj_actions_v.to(device)
        traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, device=device)
        mu_v = net_act(traj_states_v)
        old_logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)

        # normalize advantages
        traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
        traj_adv_v /= torch.std(traj_adv_v)

        # drop last entry from the trajectory, an our adv and ref value calculated without it
        trajectory = trajectory[:-1]
        old_logprob_v = old_logprob_v[:-1].detach()

        sum_loss_value = 0.0
        sum_loss_policy = 0.0
        count_steps = 0

        for epoch in range(PPO_EPOCHES):
            for batch_ofs in range(0, len(trajectory), PPO_BATCH_SIZE):
                batch_l = batch_ofs + PPO_BATCH_SIZE
                states_v = traj_states_v[batch_ofs:batch_l]
                actions_v = traj_actions_v[batch_ofs:batch_l]
                batch_adv_v = traj_adv_v[batch_ofs:batch_l]
                batch_adv_v = batch_adv_v.unsqueeze(-1)
                batch_ref_v = traj_ref_v[batch_ofs:batch_l]
                batch_old_logprob_v = old_logprob_v[batch_ofs:batch_l]

                # critic training
                opt_crt.zero_grad()
                value_v = net_crt(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_ref_v)
                loss_value_v.backward()
                opt_crt.step()

                # actor training
                opt_act.zero_grad()
                mu_v = net_act(states_v)
                logprob_pi_v = calc_logprob(mu_v, net_act.logstd, actions_v)
                ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                surr_obj_v = batch_adv_v * ratio_v
                c_ratio_v = torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                clipped_surr_v = batch_adv_v * c_ratio_v
                loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()
                loss_policy_v.backward()
                opt_act.step()

                sum_loss_value += loss_value_v.item()
                sum_loss_policy += loss_policy_v.item()
                count_steps += 1

        trajectory.clear()
        writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
        writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
        writer.add_scalar("loss_policy", sum_loss_policy / count_steps, step_idx)
        writer.add_scalar("loss_value", sum_loss_value / count_steps, step_idx)


# ----------
name = "final_%d.dat" % (step_idx)
fname = os.path.join(save_path, name)
torch.save(net_act.state_dict(), fname)


##############################################################################################
# --------------------------------------------------------------------------------------------
# play by command line (xvfv-run)
# https://manpages.ubuntu.com/manpages/trusty/man1/xvfb-run.1.html
# --------------------------------------------------------------------------------------------

# xvfb-run:  run specified X client or command in a virtual X server environment
# -s:  --server-args, default is '-screen 0 640x480x8'
# +extension GLX:  enable OpenGL Extension to the X Window System

xvfb-run -s "-screen 0 640x480x24 +extension GLX" \
    ./halfcheetah_pybullet_10_PPO_play.py \
        -m /home/kswada/kw/reinforcement_learning/04_output/02_deep_reinforcement_learning_hands_on/halfcheetah/ppo_trial_202301032258/model/final_10556447.dat \
            -r /home/kswada/kw/reinforcement_learning/04_output/02_deep_reinforcement_learning_hands_on/halfcheetah/ppo_trial_202301032258/video/final_10556447

xvfb-run -s "-screen 0 640x480x24 +extension GLX" \
    ./halfcheetah_pybullet_10_PPO_play.py \
        -m /home/kswada/kw/reinforcement_learning/04_output/02_deep_reinforcement_learning_hands_on/halfcheetah/ppo_trial_202301032258/model/best_+2222.032_9460000.dat \
            -r /home/kswada/kw/reinforcement_learning/04_output/02_deep_reinforcement_learning_hands_on/halfcheetah/ppo_trial_202301032258/video/best_+2222.032_9460000

