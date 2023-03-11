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
# TRPO step
# --------------------------------------------------------------------------------------------

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10, device="cpu"):
    x = torch.zeros(b.size()).to(device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(model,
               f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f().data
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + fullstep * stepfrac
        set_flat_params_to(model, xnew)
        newfval = f().data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            return True, xnew
    return False, x


def trpo_step(model, get_loss, get_kl, max_kl, damping, device="cpu"):
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        v_v = v.clone().detach().to(device)
        kl_v = (flat_grad_kl * v_v).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * damping

    stepdir = conjugate_gradients(Fvp, -loss_grad, 10, device=device)

    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)

    prev_params = get_flat_params_from(model)
    success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])
    set_flat_params_to(model, new_params)

    return loss


##############################################################################################
# --------------------------------------------------------------------------------------------
# HalfCheetah Agent Learning:  Trust Region Policy Optimization (TRPO)
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
writer = SummaryWriter(comment="-halfcheetah_trpo_" + args_name)

save_path = os.path.join(base_path, f'04_output/02_deep_reinforcement_learning_hands_on/halfcheetah/trpo_{args_name}/model')
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
LEARNING_RATE_CRITIC = 1e-3

opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)


# ----------
trajectory = []
best_reward = None


TRPO_MAX_KL = 0.01
TRPO_DAMPING = 0.1

# ----------
TRAJECTORY_SIZE = 2049

# TEST_ITERS = 100000
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
        traj_states_v = torch.FloatTensor(traj_states).to(device)
        traj_actions_v = torch.FloatTensor(traj_actions).to(device)
        traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, device=device)
        mu_v = net_act(traj_states_v)
        old_logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)

        # normalize advantages
        traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)

        # drop last entry from the trajectory, an our adv and ref value calculated without it
        trajectory = trajectory[:-1]
        old_logprob_v = old_logprob_v[:-1].detach()
        traj_states_v = traj_states_v[:-1]
        traj_actions_v = traj_actions_v[:-1]
        sum_loss_value = 0.0
        sum_loss_policy = 0.0
        count_steps = 0

        # critic step
        opt_crt.zero_grad()
        value_v = net_crt(traj_states_v)
        loss_value_v = F.mse_loss(value_v.squeeze(-1), traj_ref_v)
        loss_value_v.backward()
        opt_crt.step()

        # actor step
        def get_loss():
            mu_v = net_act(traj_states_v)
            logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)
            dp_v = torch.exp(logprob_v - old_logprob_v)
            action_loss_v = -traj_adv_v.unsqueeze(dim=-1)*dp_v
            return action_loss_v.mean()

        def get_kl():
            mu_v = net_act(traj_states_v)
            logstd_v = net_act.logstd
            mu0_v = mu_v.detach()
            logstd0_v = logstd_v.detach()
            std_v = torch.exp(logstd_v)
            std0_v = std_v.detach()
            v = (std0_v ** 2 + (mu0_v - mu_v) ** 2) / (2.0 * std_v ** 2)
            kl = logstd_v - logstd0_v + v - 0.5
            return kl.sum(1, keepdim=True)

        trpo_step(net_act, get_loss, get_kl, TRPO_MAX_KL, TRPO_DAMPING, device=device)

        trajectory.clear()
        writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
        writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
        writer.add_scalar("loss_value", loss_value_v.item(), step_idx)

