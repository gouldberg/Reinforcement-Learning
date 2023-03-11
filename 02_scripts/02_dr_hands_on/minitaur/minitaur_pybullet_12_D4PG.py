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
# DDPG Actor and D4PG Critic
# --------------------------------------------------------------------------------------------

class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# ----------
# Instead of returning the single Q-value for the given state and the action,
# it now returns N_ATOMS (=51) values, corresponding to the probabilities of values from the predefined range.
# The distribution range is Vmin (=-10) and Vmax (=10).
# So the critic return 51 numbers, representing the probabilities of the discounted reward falling into bins
# with bounds in [-10, -9.6, -9.2, ..., 9.6, 10]  (51 numbers)

class D4PGCritic(nn.Module):
    def __init__(self, obs_size, act_size,
                 n_atoms, v_min, v_max):
        super(D4PGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, n_atoms)
        )

        # ----------
        # helper PyTorch buffer with reward supports,
        # which will be used to get from the probability distribution to the single mean Q-value.
        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(
            v_min, v_max + delta, delta))

    def forward(self, x, a):
        # transform the observations with small network
        obs = self.obs_net(x)
        # concatenate the output and given actors to transform them into one single value of Q
        return self.out_net(torch.cat([obs, a], dim=1))

    # distr_to_q() to convert from the probability distribution to the single mean Q-value using support atoms
    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)


# --------------------------------------------------------------------------------------------
# AgentDDPG
#   - Exploration:
#      Our policy is deterministic, so we have to explore the environment somehow.
#      We can do this by adding noise to the actions returned by the actor before we pass them to the environment.
#      We apply stochastic processes model (Ornstein-Uhlenbeck (OU) processs).
#      In a discrete-time case, the OU process could be written as x(t+1) = x(t) + theta * (mu - x(t)) + sigma * noise,
#      which generate temporally correlated exploration for exploration efficiency in physical control problems with inertia.
#      OU process models the velocity of a Brownian particle with friction.
# --------------------------------------------------------------------------------------------

class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    # The constructor accepts a lot of parameters, most of which are the default values of OU
    # taken from the paper 'Continuous Control with Deep Reinforcement Learning'
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

    # ----------
    # This method is derived from the BaseAgent class and has to return the initial state of the agent
    # when a new episode is started.
    # As our initial state has to have the same dimension as the actions
    # (we want to have individual exploration trajectories for every action of the einvironment),
    # we postpone the initialization of the state until the __call__ method.
    def initial_state(self):
        return None

    # ----------
    # This convert the observed state and internal agent state into the action.
    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        # ask the actor network to convert into deterministic actions
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)

                # ----------
                # OU process in a discrete-time case

                a_state += self.ou_teta * (self.ou_mu - a_state)
                # add normal noise to OU process
                a_state += self.ou_sigma * np.random.normal(size=action.shape)
                # ----------

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        # finally clip the actions to enforce them to fall into the -1 ...1 range,
        # otherwise PyBullet will throw an exception.
        actions = np.clip(actions, -1, 1)

        return actions, new_a_states


# --------------------------------------------------------------------------------------------
# AgentD4PG
# --------------------------------------------------------------------------------------------

class AgentD4PG(ptan.agent.BaseAgent):
    """
    Agent implementing noisy agent
    """
    def __init__(self, net, device="cpu", epsilon=0.3):
        self.net = net
        self.device = device
        self.epsilon = epsilon

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        # add gaussian noise to the actions, scaled by the epsilon value.
        actions += self.epsilon * np.random.normal(size=actions.shape)

        # finally clip the actions to enforce them to fall into the -1 ...1 range,
        # otherwise PyBullet will throw an exception.
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


##############################################################################################
# --------------------------------------------------------------------------------------------
# distr_projection
#   - Calculate the result of the Bellman operator and 
#     project the resulting probability distribution to the same support atoms as the original distribution
# --------------------------------------------------------------------------------------------

Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

def distr_projection(next_distr_v, rewards_v, dones_mask_t,
                     gamma, device="cpu"):

    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    # dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool_)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    for atom in range(N_ATOMS):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

    return torch.FloatTensor(proj_distr).to(device)


##############################################################################################
# --------------------------------------------------------------------------------------------
# unpack_batch
# --------------------------------------------------------------------------------------------

def unpack_batch_ddqn(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = ptan.agent.float32_preprocessor(actions).to(device)
    rewards_v = ptan.agent.float32_preprocessor(rewards).to(device)
    last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v


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
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


##############################################################################################
# --------------------------------------------------------------------------------------------
# Minitaur Agent Learning:  Distributed Distributional Deep Deterministic Policy Gradients (D4PG)
# --------------------------------------------------------------------------------------------

# ----------
# device
device = torch.device("cuda")


# ----------
# environment
ENV_ID = "MinitaurBulletEnv-v0"

env = gym.make(ENV_ID)
test_env = gym.make(ENV_ID)


# ----------
base_path = '/home/kswada/kw/reinforcement_learning'

args_name = 'trial'
writer = SummaryWriter(comment="-minitaur_d4pg_" + args_name)

save_path = os.path.join(base_path, '04_output/minitaur/d4pg_trial/model')
os.makedirs(save_path, exist_ok=True)


# ----------
# net:  DDPG Actor and D4PG Critic
act_net = DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
crt_net = D4PGCritic(env.observation_space.shape[0], env.action_space.shape[0], N_ATOMS, Vmin, Vmax).to(device)

print(act_net)
print(crt_net)

tgt_act_net = ptan.agent.TargetNet(act_net)

tgt_crt_net = ptan.agent.TargetNet(crt_net)


# ----------
# agent is DDPG
agent = AgentDDPG(act_net, device=device)


# ----------
# experience source and replay buffer

GAMMA = 0.99
REWARD_STEPS = 5
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

# In the D4PG paper, the authors used 1M transitions in the buffer,
# but a smaller replay buffer works here.
REPLAY_SIZE = 100000
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
 

# ----------
# optimizer
# we use 2 different optimiers to simplify the way that we handle gradients for the actor and critic training steps.
LEARNING_RATE = 1e-4
act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)


# ----------
frame_idx = 0
best_reward = None


# ----------
BATCH_SIZE = 64
TEST_ITERS = 1000

# The buffer is prepopulated with 10000 samples from the environment
# and then the training starts.
REPLAY_INITIAL = 10000

batch_size_tracker = 10

with ptan.common.utils.RewardTracker(writer) as tracker:

    # This wrapper is responsible for writing into TensorBoard the mean of the measured parameters for the last 10 steps.
    # This is helpful, as training can take millions of steps and we do not want to write millions of points into TensorBoard,
    # but rather write smoothed values every 10 steps.
    with ptan.common.utils.TBMeanTracker(writer, batch_size=batch_size_tracker) as tb_tracker:
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
            states_v, actions_v, rewards_v, dones_mask, last_states_v = unpack_batch_ddqn(batch, device)

            # train critic
            crt_opt.zero_grad()
            crt_distr_v = crt_net(states_v, actions_v)
            last_act_v = tgt_act_net.target_model(last_states_v)
            last_distr_v = F.softmax(tgt_crt_net.target_model(last_states_v, last_act_v), dim=1)
            #################
            # Bellman projection of the distribution:
            #  - calculate the transformation of the last_states probability distribution,
            #    which is shifted according to the immediate reward and scaled to respect the discount factor
            proj_distr_v = distr_projection(last_distr_v, rewards_v, dones_mask, gamma=GAMMA**REWARD_STEPS, device=device)
            # calculate cross-entropy loss
            prob_dist_v = -F.log_softmax(crt_distr_v, dim=1) * proj_distr_v
            #################
            critic_loss_v = prob_dist_v.sum(dim=1).mean()
            critic_loss_v.backward()
            crt_opt.step()
            tb_tracker.track("loss_critic", critic_loss_v, frame_idx)

            # train actor
            act_opt.zero_grad()
            cur_actions_v = act_net(states_v)
            crt_distr_v = crt_net(states_v, cur_actions_v)
            #################
            # distr_to_q() to convert from the probability distribution to the single mean Q-value using support atoms
            actor_loss_v = -crt_net.distr_to_q(crt_distr_v)
            #################
            actor_loss_v = actor_loss_v.mean()
            actor_loss_v.backward()
            act_opt.step()
            tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

            tgt_act_net.alpha_sync(alpha=1 - 1e-3)
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


##############################################################################################
# --------------------------------------------------------------------------------------------
# play
# https://github.com/openai/gym/blob/master/gym/envs/registration.py
# --------------------------------------------------------------------------------------------

ENV_ID = "MinitaurBulletEnv-v0"

spec = gym.envs.registry.spec(ENV_ID)   
dir(spec)

spec._kwargs['render'] = False

env = gym.make(ENV_ID)


# ----------
# Monitor wrapper
model_path = os.path.join(base_path, '04_output/minitaur/d4pg_trial/model/best_+8.805_979000.dat')
record_dir = os.path.join(base_path, '04_output/minitaur/d4pg_trial/video')

env = gym.wrappers.Monitor(env, record_dir)


# ----------
# load model
net = DDPGActor(env.observation_space.shape[0], env.action_space.shape[0])

net.load_state_dict(torch.load(model_path))


# ----------
obs = env.reset()

total_reward = 0.0
total_steps = 0

while True:
    obs_v = torch.FloatTensor([obs])
    mu_v = net(obs_v)
    action = mu_v.squeeze(dim=0).data.numpy()
    action = np.clip(action, -1, 1)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    total_steps += 1
    if done:
        break

print("In %d steps we got %.3f reward" % (total_steps, total_reward))


##############################################################################################
# --------------------------------------------------------------------------------------------
# play by command line (xvfv-run)
# https://manpages.ubuntu.com/manpages/trusty/man1/xvfb-run.1.html
# --------------------------------------------------------------------------------------------

# xvfb-run:  run specified X client or command in a virtual X server environment
# -s:  --server-args, default is '-screen 0 640x480x8'
# +extension GLX:  enable OpenGL Extension to the X Window System

xvfb-run -s "-screen 0 640x480x24 +extension GLX" \
    ./minitaur_pybullet_12_D4PG_play.py \
        -m /home/kswada/kw/reinforcement_learning/04_output/minitaur/d4pg_trial/model/best_+8.805_979000.dat \
            -r /home/kswada/kw/reinforcement_learning/04_output/minitaur/d4pg_trial/video
