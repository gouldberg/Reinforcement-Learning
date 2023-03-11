import os, sys
import time
import gym
import ptan
import numpy as np
import collections

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim


##############################################################################################
# --------------------------------------------------------------------------------------------
# RewardTracker 
# --------------------------------------------------------------------------------------------

class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


# --------------------------------------------------------------------------------------------
# AtariPGN
# --------------------------------------------------------------------------------------------

class AtariPGN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariPGN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


##############################################################################################
# --------------------------------------------------------------------------------------------
# others
# --------------------------------------------------------------------------------------------

def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


# ----------
# the baseline is estimated with a moving average for past transitions,
# instead of all examples
# To make moving average calculations faster, a deque-backed buffer is created.

class MeanBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.deque = collections.deque(maxlen=capacity)
        self.sum = 0.0

    def add(self, val):
        if len(self.deque) == self.capacity:
            self.sum -= self.deque[0]
        self.deque.append(val)
        self.sum += val

    def mean(self):
        if not self.deque:
            return 0.0
        return self.sum / len(self.deque)



##############################################################################################
# --------------------------------------------------------------------------------------------
# Pong Agent Learning:  Policy Gradient
#
#  1. The baseline is estimated with a moving average fo 1M past transitions, instead of all examples
#  2. Severla concurrent environments are used.
#  3. Gradients are clipped to improve training stability.
#
#  Overall, hyperparameter tuning is required ...  
# --------------------------------------------------------------------------------------------

# ----------
# device
device = torch.device("cuda")


# ----------
# environment
# For working with multiple environments, we have to pass the array of Env objects
# to the ExperienceSource class. All the rest is done automatically.
# In the case of several environments, the experience source asks them for transitions
# in round-robin, providing us with less-correlated training samples.
ENV_COUNT = 32
envs = [make_env() for _ in range(ENV_COUNT)]


# ----------
# Policy Gradient Net
net = AtariPGN(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
print(net)


# ----------
# agent
agent = ptan.agent.PolicyAgent(net, apply_softmax=True, device=device)


# ----------
# experience source
# how many steps ahead the Bellman equation is unrolled
# to estimate the discounted total reward of every transition
# For CartPole, short episodes is fine
REWARD_STEPS = 10

GAMMA = 0.99
exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)


# ----------
# optimizer
LEARNING_RATE = 0.0001
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)


# ----------
total_rewards = []
step_idx = 0
done_episodes = 0
train_step_idx = 0

# the baseline is estimated with a moving average for 1M past transitions,
# instead of all examples
BASELINE_STEPS = 1000000
baseline_buf = MeanBuffer(BASELINE_STEPS)

ENTROPY_BETA = 0.01
BATCH_SIZE = 128
GRAD_L2_CLIP = 0.1

batch_states, batch_actions, batch_scales = [], [], []
m_baseline, m_batch_scales, m_loss_entropy, m_loss_policy, m_loss_total = [], [], [], [], []

m_grad_max, m_grad_mean = [], []

sum_reward = 0.0


# ----------
args_name = 'trial'
writer = SummaryWriter(comment="-pong-pg-byptan" + args_name)

with RewardTracker(writer, stop_reward=18) as tracker:
    for step_idx, exp in enumerate(exp_source):

        # 'baseline' for the policy scale
        baseline_buf.add(exp.reward)
        baseline = baseline_buf.mean()
        batch_states.append(np.array(exp.state, copy=False))
        batch_actions.append(int(exp.action))
        # 'batch_scales' is used for scaling
        batch_scales.append(exp.reward - baseline)

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            if tracker.reward(new_rewards[0], step_idx):
                break

        if len(batch_states) < BATCH_SIZE:
            continue

        train_step_idx += 1
        states_v = torch.FloatTensor(np.array(batch_states, copy=False)).to(device)
        batch_actions_t = torch.LongTensor(batch_actions).to(device)

        scale_std = np.std(batch_scales)
        batch_scale_v = torch.FloatTensor(batch_scales).to(device)

        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()

        # add entropy bonus to the loss
        # As entropy has a maximum for uniform probability distribution and 
        # we want to push the training toward this maximum, we need to subtract from the loss
        # (to minimize all loss)
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        loss_v = loss_policy_v + entropy_loss_v
        loss_v.backward()

        # gradients are clipped to improve training stability
        nn_utils.clip_grad_norm_(net.parameters(), GRAD_L2_CLIP)
        optimizer.step()


        ####################################
        # for monitoring
        # ----------------------------------
        # calc KL-div between the new policy and the old policy.
        # High spikes in KL are usually a bad sign, showing that our policy was pushed too far
        # from the previous policy.
        new_logits_v = net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        writer.add_scalar("kl", kl_div_v.item(), step_idx)

        # calculate the statistics about the gradients on this training step.
        # it is usually good practice to show the graph of the maximum and L2 norm of gradients
        # to get an idea about the training dynamics.
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in net.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1

        writer.add_scalar("baseline", baseline, step_idx)
        writer.add_scalar("entropy", entropy_v.item(), step_idx)
        writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
        writer.add_scalar("batch_scales_std", scale_std, step_idx)
        writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
        writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
        writer.add_scalar("loss_total", loss_v.item(), step_idx)
        writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
        writer.add_scalar("grad_max", grad_max, step_idx)
        ####################################

        # ----------
        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

writer.close()
