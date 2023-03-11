import os, sys
import time
import numpy as np
import collections

import gym
import ptan

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

        # ----------
        conv_out_size = self._get_conv_out(input_shape)

        # ----------
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


# --------------------------------------------------------------------------------------------
# AtariP2C
# --------------------------------------------------------------------------------------------

class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # ----------
        conv_out_size = self._get_conv_out(input_shape)

        # ----------
        # Actor (Policy net)
        # return the policy with the probability distribution over our actions
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        # ----------
        # Critic (Value net)
        # return one single number, which will approximate the state's value.
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )


    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    # ---------
    # In ptactice, the policy and value networks partially overlap,
    # mostly due to efficiency and convergence considerations.
    # The policy and value are implemented as different heads of the network,
    # taking the output from the common body and transforming it into
    # the probability distribution and a sngle number representing
    # the value of the state.
    # This helps both networks to share low-leve features.
    def forward(self, x):
        fx = x.float() / 256
        # shared body
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        # return two heads (policy net and value net)
        return self.policy(conv_out), self.value(conv_out)


# --------------------------------------------------------------------------------------------
# unpack_batch
# --------------------------------------------------------------------------------------------

def unpack_batch(batch, net, device='cpu'):
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
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))

        # reward value already contains the discounted reward for REWARD_STEPS, as we use the ptan.ExperienceSourceFirstLast class.
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    # extra call to np.array() might look redundant, but without it, the performance of tensor creation
    # degrades 5-10x
    states_v = torch.FloatTensor(
        np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)

        # get V(state) appximation.
        last_vals_v = net(last_states_v)[1]

        # discounted and added to immediate rewards
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        last_vals_np *= GAMMA ** REWARD_STEPS
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)

    return states_v, actions_t, ref_vals_v


##############################################################################################
# --------------------------------------------------------------------------------------------
# Pong Agent Learning:  Advantage Actor-Critic (A2C)
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
NUM_ENVS = 50
make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
envs = [make_env() for _ in range(NUM_ENVS)]


# ----------
# net:  Atari A2C
net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
print(net)


# ----------
# agent
agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)


# ----------
# experience source
# How many steps ahead we will tke to approximate the total discounted reward for every action.
# In the policy gradient methods, we used about 10 steps, but in A2C, we will use our value approximation
# to get a state value for further steps, so it will be fine to decrease the number of steps.
REWARD_STEPS = 4

GAMMA = 0.99
exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)


# ----------
# optimizer
# if 0.004 or larger --> does not converge ...
LEARNING_RATE = 0.001

# Normally eps, which is added to denominator to prevent zero division situation,
# is set to some small number such as 1e-8 or 1e-10,
# but in our case, these values turned out to be too small and the method does not converge at all.
# so we set large eps such as 1e-3
EPS_VAL = 1e-3
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=EPS_VAL)

# ----------
batch = []


BATCH_SIZE = 128
ENTROPY_BETA = 0.01

# gradient clipping for L" norm of the gradient, 
# which basically prevents our gradients from becoming too large at the optimization stage
# and pushing our policy too far.
CLIP_GRAD = 0.1

# ----------
args_name = 'trial'
writer = SummaryWriter(comment="-pong-a2c_" + args_name)

stop_reward = 18
batch_size_tracker = 10

# RewardTracker computes the mean reward for the last 100 episodes and
# tell us when this mean reward exceeds the desired threshold.
with RewardTracker(writer, stop_reward=stop_reward) as tracker:

    # This wrapper is responsible for writing into TensorBoard the mean of the measured parameters for the last 10 steps.
    # This is helpful, as training can take millions of steps and we do not want to write millions of points into TensorBoard,
    # but rather write smoothed values eveyr 10 steps.
    with ptan.common.utils.TBMeanTracker(writer, batch_size=batch_size_tracker) as tb_tracker:
        for step_idx, exp in enumerate(exp_source):
            batch.append(exp)

            # handle new rewards
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if tracker.reward(new_rewards[0], step_idx):
                    break

            if len(batch) < BATCH_SIZE:
                continue

            # ----------
            states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
            batch.clear()
            optimizer.zero_grad()

            # policy with the probability distribution over actions and single number to appximate the state value
            logits_v, value_v = net(states_v)
            # ----------

            # calculate MSE loss to improve the value approximation in the same way as DQN
            loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

            log_prob_v = F.log_softmax(logits_v, dim=1)

            # detach():  we do not want to propagate the policy gradient into our value approximation head.
            adv_v = vals_ref_v - value_v.detach()

            log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
            loss_policy_v = -log_prob_actions_v.mean()

            prob_v = F.softmax(logits_v, dim=1)
            entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

            # ----------
            # calculate policy gradients only:
            # retain_graph=True instructs PyTorch to keep the graph structure of the variables.
            # Normally, graph is destroyed by the backward() call, but in our case,
            # this is not what we want. In general, retaining the graph could be useful when we need
            # to backpropagate the loss multiple times before the call to the optimizer, although this is
            # not a very common situation.
            loss_policy_v.backward(retain_graph=True)
            grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                    for p in net.parameters()
                                    if p.grad is not None])

            # apply entropy and value gradients
            loss_v = entropy_loss_v + loss_value_v
            loss_v.backward()

            # gradient clipping
            nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            optimizer.step()
            # get full loss
            loss_v += loss_policy_v

            tb_tracker.track("advantage",       adv_v, step_idx)
            tb_tracker.track("values",          value_v, step_idx)
            tb_tracker.track("batch_rewards",   vals_ref_v, step_idx)
            tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
            tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
            tb_tracker.track("loss_value",      loss_value_v, step_idx)
            tb_tracker.track("loss_total",      loss_v, step_idx)
            tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
            tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
            tb_tracker.track("grad_var",        np.var(grads), step_idx)



# -->
# With the original hyperparameters, it requires more than 8 million frames to solve,
# which is approximately 3 hours on GPU.
