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


# --------------------------------------------------------------------------------------------
# unpack_batch_a2c
# --------------------------------------------------------------------------------------------

def unpack_batch_a2c(batch, net, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v


##############################################################################################
# --------------------------------------------------------------------------------------------
# test_net and calc_logpob
# --------------------------------------------------------------------------------------------

def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs])
            obs_v = obs_v.to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


# Caluculate the logarithm of the taken actions given the policy.
# torch.clamn() to prevent the division on zero when the returned variance is too small.
def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


##############################################################################################
# --------------------------------------------------------------------------------------------
# HalfCheetah Agent Learning:  Advantage Actor-Critic (A2C)
# --------------------------------------------------------------------------------------------

# ----------
# device
device = torch.device("cuda")


# ----------
# environment
ENV_ID = "HalfCheetahBulletEnv-v0"

# We have 16 parallel environments used to gather experience during the training.
ENVS_COUNT = 16

envs = [gym.make(ENV_ID) for _ in range(ENVS_COUNT)]
test_env = gym.make(ENV_ID)


# ----------
base_path = '/home/kswada/kw/reinforcement_learning'

args_name = 'baseline'
writer = SummaryWriter(comment="-halfcheetah_a2c_" + args_name)

save_path = os.path.join(base_path, f'04_output/02_deep_reinforcement_learning_hands_on/halfcheetah/a2c_{args_name}/model')
os.makedirs(save_path, exist_ok=True)


# ----------
# net:  ModelA2C and ModelCritic
net_act = ModelActor(envs[0].observation_space.shape[0], envs[0].action_space.shape[0]).to(device)
net_crt = ModelCritic(envs[0].observation_space.shape[0]).to(device)
print(net_act)
print(net_crt)


# ----------
# agent
agent = AgentA2C(net_act, device=device)


# ----------
# experience source
REWARD_STEPS = 5

GAMMA = 0.99
exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count=REWARD_STEPS)


# ----------
# optimizer
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-3

opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR)
opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)


# ----------
batch = []
best_reward = None


BATCH_SIZE = 32
ENTROPY_BETA = 1e-3


# ----------
TEST_ITERS = 100000
batch_size_tracker = 100

with ptan.common.utils.RewardTracker(writer) as tracker:
    with ptan.common.utils.TBMeanTracker(writer, batch_size=batch_size_tracker) as tb_tracker:
        for step_idx, exp in enumerate(exp_source):
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                tb_tracker.track("episode_steps", np.mean(steps), step_idx)
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

            batch.append(exp)
            if len(batch) < BATCH_SIZE:
                continue

            states_v, actions_v, vals_ref_v = unpack_batch_a2c(batch, net_crt, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
            batch.clear()

            opt_crt.zero_grad()
            value_v = net_crt(states_v)
            loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
            loss_value_v.backward()
            opt_crt.step()

            opt_act.zero_grad()
            mu_v = net_act(states_v)
            adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
            log_prob_v = adv_v * calc_logprob(mu_v, net_act.logstd, actions_v)
            loss_policy_v = -log_prob_v.mean()
            entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*torch.exp(net_act.logstd)) + 1)/2).mean()
            loss_v = loss_policy_v + entropy_loss_v
            loss_v.backward()
            opt_act.step()

            tb_tracker.track("advantage", adv_v, step_idx)
            tb_tracker.track("values", value_v, step_idx)
            tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
            tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
            tb_tracker.track("loss_policy", loss_policy_v, step_idx)
            tb_tracker.track("loss_value", loss_value_v, step_idx)
            tb_tracker.track("loss_total", loss_v, step_idx)

