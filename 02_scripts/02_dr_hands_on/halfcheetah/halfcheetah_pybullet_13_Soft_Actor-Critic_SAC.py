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
# ModelA2C, ModelCritic + ModelSACTwinQ
# --------------------------------------------------------------------------------------------

# Both the actor and critic are placed in the separate networks without sharing weights.
# Critic estimate the mean and the variance for the actions,
# but now the variance is not a separate head of the base network,
# it is just a single parameter of the model.
# This parameter will be adjusted during the training by SGD, but it does not depend on the observation.

### In original paper: HiD_SIZE = 256
# HID_SIZE = 256
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


class ModelSACTwinQ(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelSACTwinQ, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.q1(x), self.q2(x)


# --------------------------------------------------------------------------------------------
# AgentDDPG
# --------------------------------------------------------------------------------------------

class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, net, device="cpu", ou_enabled=True,
                 ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2,
                 ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(
                        shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        actions = np.clip(actions, -1, 1)
        return actions, new_a_states


##############################################################################################
# --------------------------------------------------------------------------------------------
# test_net
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


##############################################################################################
# --------------------------------------------------------------------------------------------
# unpack_batch_a2c + unpack_batch_sac
# --------------------------------------------------------------------------------------------

import torch.distributions as distr

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


@torch.no_grad()
def unpack_batch_sac(batch, val_net, twinq_net, policy_net,
                     gamma: float, ent_alpha: float,
                     device="cpu"):
    """
    Unpack Soft Actor-Critic batch
    """
    states_v, actions_v, ref_q_v = unpack_batch_a2c(batch, val_net, gamma, device)

    # references for the critic network
    mu_v = policy_net(states_v)
    act_dist = distr.Normal(mu_v, torch.exp(policy_net.logstd))
    acts_v = act_dist.sample()
    q1_v, q2_v = twinq_net(states_v, acts_v)

    # element-wise minimum
    # We give the agent a bonus for getting into situations
    # where the entropy is at maximum, which is very similar to the 
    # advanced exploration methods.
    ref_vals_v = torch.min(q1_v, q2_v).squeeze() - ent_alpha * act_dist.log_prob(acts_v).sum(dim=1)
    return states_v, actions_v, ref_vals_v, ref_q_v



##############################################################################################
# --------------------------------------------------------------------------------------------
# HalfCheetah Agent Learning:  Soft Actor-Critic (SAC)
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
writer = SummaryWriter(comment="-halfcheetah_sac_" + args_name)

save_path = os.path.join(base_path, f'04_output/02_deep_reinforcement_learning_hands_on/halfcheetah/sac_{args_name}/model')
os.makedirs(save_path, exist_ok=True)


# ----------
# net:  ModelA2C and ModelCritic
act_net = ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
crt_net = ModelCritic(env.observation_space.shape[0]).to(device)
twinq_net = ModelSACTwinQ(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

print(act_net)
print(crt_net)
print(twinq_net)


tgt_crt_net = ptan.agent.TargetNet(crt_net)


# ----------
# agent
agent = AgentDDPG(act_net, device=device)


# ----------
# experience source and replay buffer
GAMMA = 0.99

# in original paper: 10**6
# REPLAY_SIZE = 100000*10
REPLAY_SIZE = 100000

exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)

buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)


# ----------
# optimizer
# In original paper:  3 * 1e-4
# LR_ACTS = 3e-4
# LR_VALS = 3e-4
LR_ACTS = 1e-4
LR_VALS = 1e-4

act_opt = optim.Adam(act_net.parameters(), lr=LR_ACTS)
crt_opt = optim.Adam(crt_net.parameters(), lr=LR_VALS)
twinq_opt = optim.Adam(twinq_net.parameters(), lr=LR_VALS)


# ----------
frame_idx = 0
best_reward = None

# temperature parameter to determine the relative importance
# of the entropy term against the reward, and thus
# controls the stochasticity of the optimal policy
SAC_ENTROPY_ALPHA = 0.1


# ----------
### In original paper: 256
# BATCH_SIZE = 256
BATCH_SIZE = 64

REPLAY_INITIAL = 10000
TEST_ITERS = 10000
batch_size_tracker = 10

with ptan.common.utils.RewardTracker(writer) as tracker:
    with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                tb_tracker.track("episode_steps", steps[0], frame_idx)
                tracker.reward(rewards[0], frame_idx)

            if len(buffer) < REPLAY_INITIAL:
                continue

            batch = buffer.sample(BATCH_SIZE)
            states_v, actions_v, ref_vals_v, ref_q_v = unpack_batch_sac(
                    batch, tgt_crt_net.target_model,
                    twinq_net, act_net, GAMMA,
                    SAC_ENTROPY_ALPHA, device)

            tb_tracker.track("ref_v", ref_vals_v.mean(), frame_idx)
            tb_tracker.track("ref_q", ref_q_v.mean(), frame_idx)

            # train TwinQ
            twinq_opt.zero_grad()
            q1_v, q2_v = twinq_net(states_v, actions_v)
            q1_loss_v = F.mse_loss(q1_v.squeeze(), ref_q_v.detach())
            q2_loss_v = F.mse_loss(q2_v.squeeze(), ref_q_v.detach())
            q_loss_v = q1_loss_v + q2_loss_v
            q_loss_v.backward()
            twinq_opt.step()
            tb_tracker.track("loss_q1", q1_loss_v, frame_idx)
            tb_tracker.track("loss_q2", q2_loss_v, frame_idx)

            # Critic
            crt_opt.zero_grad()
            val_v = crt_net(states_v)
            v_loss_v = F.mse_loss(val_v.squeeze(), ref_vals_v.detach())
            v_loss_v.backward()
            crt_opt.step()
            tb_tracker.track("loss_v", v_loss_v, frame_idx)

            # Actor
            act_opt.zero_grad()
            acts_v = act_net(states_v)
            q_out_v, _ = twinq_net(states_v, acts_v)
            act_loss = -q_out_v.mean()
            act_loss.backward()
            act_opt.step()
            tb_tracker.track("loss_act", act_loss, frame_idx)

            ### In original paper alpha = 1 - 5e-3
            tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

            if frame_idx % TEST_ITERS == 0:
                ts = time.time()
                rewards, steps = test_net(act_net, test_env, device=device)
                print("Test done in %.2f sec, reward %.3f, steps %d" % (
                    time.time() - ts, rewards, steps))
                writer.add_scalar("test_reward", rewards, frame_idx)
                writer.add_scalar("test_steps", steps, frame_idx)
                if best_reward is None or best_reward < rewards:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                        name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                        fname = os.path.join(save_path, name)
                        torch.save(act_net.state_dict(), fname)
                    best_reward = rewards


# ----------
name = "final_%d.dat" % (frame_idx)
fname = os.path.join(save_path, name)
torch.save(act_net.state_dict(), fname)

