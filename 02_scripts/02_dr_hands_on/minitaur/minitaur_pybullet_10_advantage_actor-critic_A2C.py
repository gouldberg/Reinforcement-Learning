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
# ModelA2C, AgentA2C
# --------------------------------------------------------------------------------------------

HID_SIZE = 128

class ModelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelA2C, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )

        # mean value:
        # return by activation function of a hyperbolic tanget,
        # which is the squashed output to the range of -1 ...1.
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )

        # variance of the actions:
        # variance is transformed with the softplus activation function,
        # which is log(1 + e**x) and has the shape of a smoothed rectified linear unit (ReLU) function.
        # this activation helps to make our variance positive.
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )

        # value of state (critic)
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), \
               self.value(base_out)

# Now we are dealing with continuous problem, we need to write our own agent class.
# override the __call__ method, which needs to convert observations into actions.
class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)

        # To prevent the actions from going outside of the environemnt's -1 ... 1 bounds,
        # we use np.clip, which replaces all values less than -1 with -1,
        # and values more than 1 with 1.
        actions = np.clip(actions, -1, 1)

        # The agent_states is not used, but it needs to be returned with the chosen actions,
        # as our BaseAgent supports keeping the state of the agent.
        return actions, agent_states


# --------------------------------------------------------------------------------------------
# unpack_batch
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
        last_vals_v = net(last_states_v)[2]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v


##############################################################################################
# --------------------------------------------------------------------------------------------
# test_net and calc_logpob
# --------------------------------------------------------------------------------------------

# Perform periodical tests of our model on the separate testing environment.
# During the testing, we don't need to do any exploration.
# We will just use the mean value returned by the model directly,
# without any random sampling.
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
# Minitaur Agent Learning:  Advantage Actor-Critic (A2C)
#  - This is continuous action domain. Representation of the policy is changed.
#    The alternative representation of an action is something stochasitic, network returning parameters of the Gaussian distribution.
#    For N actions, this will be two vectors (mean and variance) of size N.
#    The policy will be represented as a random N-dimensional vectorl of uncorrelated, normally distrbuted random variables,
#    and our network can make a selection about the mean and the variance of every variable.
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
writer = SummaryWriter(comment="-minitaur_a2c_" + args_name)

save_path = os.path.join(os.path.join(base_path, '04_output/minitaur/a2c_trial/model')
os.makedirs(save_path, exist_ok=True)


# ----------
# net:  ModelA2C
net = ModelA2C(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
print(net)


# ----------
# agent
agent = AgentA2C(net, device=device)


# ----------
# experience source
# How many steps ahead we will tke to approximate the total discounted reward for every action.
# In the policy gradient methods, we used about 10 steps, but in A2C, we will use our value approximation
# to get a state value for further steps, so it will be fine to decrease the number of steps.
REWARD_STEPS = 2

GAMMA = 0.99
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)


# ----------
# optimizer
LEARNING_RATE = 5e-5

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


# ----------
batch = []
best_reward = None


BATCH_SIZE = 32
ENTROPY_BETA = 1e-4


# ----------
TEST_ITERS = 1000
batch_size_tracker = 10

with ptan.common.utils.RewardTracker(writer) as tracker:

    # This wrapper is responsible for writing into TensorBoard the mean of the measured parameters for the last 10 steps.
    # This is helpful, as training can take millions of steps and we do not want to write millions of points into TensorBoard,
    # but rather write smoothed values eveyr 10 steps.
    with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
        for step_idx, exp in enumerate(exp_source):
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                tb_tracker.track("episode_steps", steps[0], step_idx)
                tracker.reward(rewards[0], step_idx)

            # ----------
            # Every TEST_ITERS frames, the model is tested, and in the case of the bset reward obtained,
            # the model weights are saved.
            if step_idx % TEST_ITERS == 0:
                ts = time.time()
                rewards, steps = test_net(net, test_env, device=device)
                print("Test done is %.2f sec, reward %.3f, steps %d" % (time.time() - ts, rewards, steps))
                writer.add_scalar("test_reward", rewards, step_idx)
                writer.add_scalar("test_steps", steps, step_idx)
                if best_reward is None or best_reward < rewards:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                        name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                        fname = os.path.join(save_path, name)
                        torch.save(net.state_dict(), fname)
                    best_reward = rewards

            batch.append(exp)
            if len(batch) < BATCH_SIZE:
                continue

            states_v, actions_v, vals_ref_v = unpack_batch_a2c(batch, net, device=device, last_val_gamma=GAMMA ** REWARD_STEPS)
            batch.clear()

            optimizer.zero_grad()
            mu_v, var_v, value_v = net(states_v)

            loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

            adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
            log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
            loss_policy_v = -log_prob_v.mean()

            # entropy of Gaussian distribution
            ent_v = -(torch.log(2*math.pi*var_v) + 1)/2
            entropy_loss_v = ENTROPY_BETA * ent_v.mean()

            loss_v = loss_policy_v + entropy_loss_v + loss_value_v
            loss_v.backward()
            optimizer.step()

            tb_tracker.track("advantage", adv_v, step_idx)
            tb_tracker.track("values", value_v, step_idx)
            tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
            tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
            tb_tracker.track("loss_policy", loss_policy_v, step_idx)
            tb_tracker.track("loss_value", loss_value_v, step_idx)
            tb_tracker.track("loss_total", loss_v, step_idx)


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
model_path = os.path.join(base_path, '04_output/minitaur/a2c_trial/model/best_+0.404_266000.dat')
record_dir = os.path.join(base_path, '04_output/minitaur/a2c_trial/video')

env = gym.wrappers.Monitor(env, record_dir)


# ----------
# load model
net = ModelA2C(env.observation_space.shape[0], env.action_space.shape[0])

net.load_state_dict(torch.load(model_path))


# ----------
obs = env.reset()

total_reward = 0.0
total_steps = 0

while True:
    obs_v = torch.FloatTensor([obs])
    mu_v, var_v, val_v = net(obs_v)
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
    ./minitaur_pybullet_10_advantage_actor-critic_A2C_play.py \
        -m /home/kswada/kw/reinforcement_learning/04_output/minitaur/a2c_trial/model/best_+0.404_266000.dat \
            -r /home/kswada/kw/reinforcement_learning/04_output/minitaur/a2c_trial/video
