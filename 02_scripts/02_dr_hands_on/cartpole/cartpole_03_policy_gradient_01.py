import os
import gym
import ptan
import numpy as np
from typing import Optional

from tensorboardX import SummaryWriter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


##############################################################################################
# --------------------------------------------------------------------------------------------
# PolicyGradientNet
# --------------------------------------------------------------------------------------------

class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------------------------
# smooth:  used for monitor statistics about the gradients on training step.
# --------------------------------------------------------------------------------------------

def smooth(old: Optional[float], val: float, alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha) * val


##############################################################################################
# --------------------------------------------------------------------------------------------
# CartPole training:  Policy Gradient
#
#   - Full episodes are required:
#     Both REINFORCE and the cross-entropy method behave better with more episodes used for training.
#     This situation is fine for short episodes in the CartPole, but completely different for Pong.
#     For Pong, we need to communicate with the environment a lot just to perform a single training step
#     in order to get as accurate a Q-estimation as possible.
#     In order to get as accurate a Q-estimation as possible, we can do the Bellman equation,
#     unrolling N steps ahead, which will effectively exploit the fact that the value contribution decrease
#     when gamma is less than 1.
#
#   - High gradient variance:
#     For policy gradient method, one lucky episode will dominate in the final gradient.
#     The policy gradient has high variance, and usual approach to handling this is subtracting
#     a value called the 'baseline' from the Q.
#     Possible choices for the baseline are as follows:
#        - Some constant value, which is normally the mean of the discounted rewards.
#        - The moving average of the discounted rewards
#        - The value of the state, V(s)
#
#   - Exploration:
#     Even with the policy represented as the probability distribution,
#     there is a high chance that the agent will converge to some locally optimal policy
#     and stop exploring the environment.
#     We use entropy bonus.
#
#  - Correlation between samples:
#    parallel environemnts are normally used:
#    instead of communicating with one environment,
#    we use several and use their transitions as training data.
# --------------------------------------------------------------------------------------------

# ----------
# environment
env = gym.make("CartPole-v0")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n


# ----------
# Policy Gradient Net

net = PGN(env.observation_space.shape[0], env.action_space.n)
print(net)


# ----------
# agent
agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                apply_softmax=True)


# ----------
# experience source
# how many steps ahead the Bellman equation is unrolled
# to estimate the discounted total reward of every transition
# For CartPole, short episodes is fine
REWARD_STEPS = 10

GAMMA = 0.9
exp_source = ptan.experience.ExperienceSourceFirstLast(
    env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)


# ----------
# optimizer
LEARNING_RATE = 0.001
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


# ----------
total_rewards = []
step_rewards = []
step_idx = 0
done_episodes = 0
reward_sum = 0.0
bs_smoothed = entropy = l_entropy = l_policy = l_total = None

# scale of the entropy bonus
ENTROPY_BETA = 0.01
BATCH_SIZE = 8

batch_states, batch_actions, batch_scales = [], [], []


# ----------
# args_baseline = True or False to compare gradient variance

args_baseline = True
# args_baseline = False
writer = SummaryWriter(comment="-cartpole-pg" + "-baseline=%s" % args_baseline)

for step_idx, exp in enumerate(exp_source):
    reward_sum += exp.reward
    # 'baseline' for the policy scale
    baseline = reward_sum / (step_idx + 1)
    writer.add_scalar("baseline", baseline, step_idx)
    batch_states.append(exp.state)
    batch_actions.append(int(exp.action))

    # ----------
    # 'batch_scales' is used for scaling
    if args_baseline:
        batch_scales.append(exp.reward - baseline)
    else:
        batch_scales.append(exp.reward)


    # handle new rewards
    new_rewards = exp_source.pop_total_rewards()
    if new_rewards:
        done_episodes += 1
        reward = new_rewards[0]
        total_rewards.append(reward)
        mean_rewards = float(np.mean(total_rewards[-100:]))
        print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
            step_idx, reward, mean_rewards, done_episodes))
        writer.add_scalar("reward", reward, step_idx)
        writer.add_scalar("reward_100", mean_rewards, step_idx)
        writer.add_scalar("episodes", done_episodes, step_idx)
        if mean_rewards > 195:
            print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
            break

    if len(batch_states) < BATCH_SIZE:
        continue

    states_v = torch.FloatTensor(batch_states)
    batch_actions_t = torch.LongTensor(batch_actions)
    batch_scale_v = torch.FloatTensor(batch_scales)

    optimizer.zero_grad()
    logits_v = net(states_v)
    log_prob_v = F.log_softmax(logits_v, dim=1)
    log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
    loss_policy_v = -log_prob_actions_v.mean()

    # ----------
    # On every training loop, we gather the gradients from the policy loss and 
    # use this data to calculate the variance
    # retain_graph=True instructs PyTorch to keep the graph structure of the variables.
    # Normally, graph is destroyed by the backward() call, but in our case,
    # this is not what we want. In general, retaining the graph could be useful when we need
    # to backpropagate the loss multiple times before the call to the optimizer, although this is
    # not a very common situation.
    loss_policy_v.backward(retain_graph=True)
    grads = np.concatenate([p.grad.data.numpy().flatten()
                            for p in net.parameters()
                            if p.grad is not None])
    # ----------

    prob_v = F.softmax(logits_v, dim=1)
    entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()

    # add entropy bonus to the loss
    # As entropy has a maximum for uniform probability distribution and 
    # we want to push the training toward this maximum, we need to subtract from the loss
    # (to minimize all loss)
    entropy_loss_v = -ENTROPY_BETA * entropy_v
    loss_v = loss_policy_v + entropy_loss_v

    # backward again
    loss_v.backward()
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

    bs_smoothed = smooth(bs_smoothed, np.mean(batch_scales))
    entropy = smooth(entropy, entropy_v.item())
    l_entropy = smooth(l_entropy, entropy_loss_v.item())
    l_policy = smooth(l_policy, loss_policy_v.item())
    l_total = smooth(l_total, loss_v.item())

    writer.add_scalar("baseline", baseline, step_idx)
    writer.add_scalar("entropy", entropy, step_idx)
    writer.add_scalar("loss_entropy", l_entropy, step_idx)
    writer.add_scalar("loss_policy", l_policy, step_idx)
    writer.add_scalar("loss_total", l_total, step_idx)
    writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
    writer.add_scalar("grad_max", grad_max, step_idx)
    writer.add_scalar("grad_var", np.var(grads), step_idx)
    writer.add_scalar("batch_scales", bs_smoothed, step_idx)
    ####################################

    # ----------
    batch_states.clear()
    batch_actions.clear()
    batch_scales.clear()

writer.close()



# -->
# baseline:
#  - We expect the baseline to converge to 1 + 0.99 + 0.99^2 + ..+ 0.99^9,
#    which is approximately 9.56
#
# batch scale:
#  - Scales of policy gradients should oscillate around zero.
#
# entropy:
#  - The entropy is decreasing over time from 0.92 to 0.52.
#    The starting value corresponds to the maximum entropy with tow actions, which is approximately 0.69.
#    - (0.5 * log(0.5) + 0.5 * log(0.5)) = 0.69
#  - The fact that the entropy is decreasing during the training is to show
#    that our policy is moving from uniform distribution to more deterministic actions.
#
# gradient's L2 values, maximum, and KL:
#  - Show healthiness during the whole training.
#  - They are not too large and not too small, and there are no huge spikes.
#  - KL charts also look normal, as there are some spikes, but they don't exceed 1e-3.
#
# grads variance
#  - the version with the baseline is two-to-three orders of magnitude lower than
#    the version without one, which helps the system to converge faster.

