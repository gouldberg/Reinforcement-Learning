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
# KFACOptimizer (K-FAC: Kronecker-factored approximate curvature)
# --------------------------------------------------------------------------------------------

# Code taken here https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/kfac.py (with minor modifications)
import math

# TODO: In order to make this code faster:
# 1) Implement _extract_patches as a single cuda kernel
# 2) Compute QR decomposition in a separate process
# 3) Actually make a general KFAC optimizer so it fits PyTorch


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def _extract_patches(x, kernel_size, stride, padding):
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2), x.size(3) * x.size(4) * x.size(5))
    return x


def compute_cov_a(a, classname, layer_info, fast_cnn):
    batch_size = a.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            a = _extract_patches(a, *layer_info)
            a = a.view(a.size(0), -1, a.size(-1))
            a = a.mean(1)
        else:
            a = _extract_patches(a, *layer_info)
            a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
    elif classname == 'AddBias':
        is_cuda = a.is_cuda
        a = torch.ones(a.size(0), 1)
        if is_cuda:
            a = a.cuda()

    return a.t() @ (a / batch_size)


def compute_cov_g(g, classname, layer_info, fast_cnn):
    batch_size = g.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)
        else:
            g = g.transpose(1, 2).transpose(2, 3).contiguous()
            g = g.view(-1, g.size(-1)).mul_(g.size(1)).mul_(g.size(2))
    elif classname == 'AddBias':
        g = g.view(g.size(0), g.size(1), -1)
        g = g.sum(-1)

    g_ = g * batch_size
    return g_.t() @ (g_ / g.size(0))


def update_running_stat(aa, m_aa, momentum):
    # Do the trick to keep aa unchanged and not create any additional tensors
    m_aa *= momentum / (1 - momentum)
    m_aa += aa
    m_aa *= (1 - momentum)


class SplitBias(nn.Module):
    def __init__(self, module):
        super(SplitBias, self).__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None

    def forward(self, input):
        x = self.module(input)
        x = self.add_bias(x)
        return x


class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.25,
                 momentum=0.9,
                 stat_decay=0.99,
                 kl_clip=0.001,
                 damping=1e-2,
                 weight_decay=0,
                 fast_cnn=False,
                 Ts=1,
                 Tf=10):
        defaults = dict()

        def split_bias(module):
            for mname, child in module.named_children():
                if hasattr(child, 'bias'):
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)

        split_bias(model)

        super(KFACOptimizer, self).__init__(model.parameters(), defaults)

        self.known_modules = {'Linear', 'Conv2d', 'AddBias'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}

        self.momentum = momentum
        self.stat_decay = stat_decay

        self.lr = lr
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay

        self.fast_cnn = fast_cnn

        self.Ts = Ts
        self.Tf = Tf

        self.optim = optim.SGD(
            model.parameters(),
            lr=self.lr * (1 - self.momentum),
            momentum=self.momentum)

    def _save_input(self, module, input):
        if self.steps % self.Ts == 0:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            aa = compute_cov_a(input[0].data, classname, layer_info,
                               self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = aa.clone()

            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.acc_stats:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            gg = compute_cov_g(grad_output[0].data, classname,
                               layer_info, self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = gg.clone()

            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                assert not ((classname in ['Linear', 'Conv2d']) and module.bias is not None), \
                                    "You must have a bias as a separate layer"

                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def step(self):
        # Add weight decay
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(self.weight_decay, p.data)

        updates = {}
        for i, m in enumerate(self.modules):
            assert len(list(m.parameters())
                       ) == 1, "Can handle only one parameter at the moment"
            classname = m.__class__.__name__
            p = next(m.parameters())

            la = self.damping + self.weight_decay

            if self.steps % self.Tf == 0:
                # My asynchronous implementation exists, I will add it later.
                # Experimenting with different ways to this in PyTorch.
                self.d_a[m], self.Q_a[m] = torch.symeig(
                    self.m_aa[m], eigenvectors=True)
                self.d_g[m], self.Q_g[m] = torch.symeig(
                    self.m_gg[m], eigenvectors=True)

                self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                self.d_g[m].mul_((self.d_g[m] > 1e-6).float())

            if classname == 'Conv2d':
                p_grad_mat = p.grad.data.view(p.grad.data.size(0), -1)
            else:
                p_grad_mat = p.grad.data

            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            v2 = v1 / (
                self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()

            v = v.view(p.grad.data.size())
            updates[p] = v

        vg_sum = 0
        for p in self.model.parameters():
            if p not in updates:
#                print("Not found in updates: %s" % p)
                continue
            v = updates[p]
            vg_sum += (v * p.grad.data * self.lr * self.lr).sum()

        nu = min(1, math.sqrt(self.kl_clip / vg_sum))

        for p in self.model.parameters():
            if p not in updates:
#                print("Not found in updates: %s" % p)
                continue
            v = updates[p]
            p.grad.data.copy_(v)
            p.grad.data.mul_(nu)

        self.optim.step()
        self.steps += 1


##############################################################################################
# --------------------------------------------------------------------------------------------
# HalfCheetah Agent Learning:  ACKTR
# --------------------------------------------------------------------------------------------

# ----------
# device
device = torch.device("cuda")


# ----------
# environment
ENV_ID = "HalfCheetahBulletEnv-v0"


ENVS_COUNT = 16
envs = [gym.make(ENV_ID) for _ in range(ENVS_COUNT)]
test_env = gym.make(ENV_ID)


# ----------
base_path = '/home/kswada/kw/reinforcement_learning'

args_name = 'trial'
writer = SummaryWriter(comment="-halfcheetah_acktr_" + args_name)

save_path = os.path.join(base_path, f'04_output/02_deep_reinforcement_learning_hands_on/halfcheetah/acktr_{args_name}/model')
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
GAMMA = 0.99
REWARD_STEPS = 5

exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count=REWARD_STEPS)


# ----------
# optimizer
LEARNING_RATE_ACTOR = 1e-3
LEARNING_RATE_CRITIC = 1e-3

opt_act = KFACOptimizer(net_act, lr=LEARNING_RATE_ACTOR)
opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)


# ----------
batch = []
best_reward = None

ENTROPY_BETA = 1e-3


# ----------
BATCH_SIZE = 32

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

            mu_v = net_act(states_v)
            log_prob_v = calc_logprob(mu_v, net_act.logstd, actions_v)
            if opt_act.steps % opt_act.Ts == 0:
                opt_act.zero_grad()
                pg_fisher_loss = -log_prob_v.mean()
                opt_act.acc_stats = True
                pg_fisher_loss.backward(retain_graph=True)
                opt_act.acc_stats = False

            opt_act.zero_grad()
            adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
            loss_policy_v = -(adv_v * log_prob_v).mean()
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
