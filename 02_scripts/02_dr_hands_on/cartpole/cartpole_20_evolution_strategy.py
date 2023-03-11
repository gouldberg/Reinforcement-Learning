import os
import gym
import time
import numpy as np

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter


##############################################################################################
# --------------------------------------------------------------------------------------------
# Net
# --------------------------------------------------------------------------------------------

# simpe one-hidden-layer NN,
# which gives us the action to take from the observation.

HID_SIZE = 32

class Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------------------------

def evaluate(env, net):
    obs = env.reset()
    reward = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        act_prob = net(obs_v)
        acts = act_prob.max(dim=1)[1]
        obs, r, done, _ = env.step(acts.data.numpy()[0])
        reward += r
        steps += 1
        if done:
            break
    return reward, steps


# --------------------------------------------------------------------------------------------
# sample_noise
# eval_with_noise
# --------------------------------------------------------------------------------------------

NOISE_STD = 0.01

# mirrored sampling to improve the stability of the convergence.
# without the negative noise, the convergence becomes very unstable.
def sample_noise(net):
    pos = []
    neg = []
    for p in net.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise)
        pos.append(noise_t)
        neg.append(-noise_t)
    return pos, neg


# evaluate the network with noise added.
def eval_with_noise(env, net, noise):
    old_params = net.state_dict()

    # add the noise to the network's parameters and 
    # call the evaluate function to obtain the reward and number of steps taken.
    for p, p_n in zip(net.parameters(), noise):
        p.data += NOISE_STD * p_n
    r, s = evaluate(env, net)
    
    # restore the network weighs to their orignal state.
    net.load_state_dict(old_params)
    return r, s


# --------------------------------------------------------------------------------------------
# train_step
# --------------------------------------------------------------------------------------------

# The coefficient used to adjust the weights on the training step
LEARNING_RATE = 0.001

def train_step(net, batch_noise, batch_reward, writer, step_idx):
    weighted_noise = None

    # ----------
    # normalize rewards to have zero mean and unit variance,
    # which improve the stability
    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    s = np.std(norm_reward)
    if abs(s) > 1e-6:
        norm_reward /= s

    # iterate every pair (noise, reward) in batch and
    # multiply the noise values with the normalized reward,
    # summing together the respective noise for every parameter
    # in our policy.
    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                # ks comment:  w_n --> weighted_noise ??
                w_n += reward * p_n

    # ----------
    # update weights (p.data)
    # Technically what we do here is a gradient ascent, although the gradient
    # was not obtained from backpropagation but from the Monte Carlo sampling method.
    m_updates = []
    for p, p_update in zip(net.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        p.data += LEARNING_RATE * update
        m_updates.append(torch.norm(update))

    writer.add_scalar("update_l2", np.mean(m_updates), step_idx)


##############################################################################################
# --------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------

# ----------
# environment
ENV_ID = "CartPole-v0"

env = gym.make(ENV_ID)

writer = SummaryWriter(comment="-cartpole-es")

# ----------
base_path = '/home/kswada/kw/reinforcement_learning'

args_name = 'trial'
writer = SummaryWriter(comment="-cartpole-es")

save_path = os.path.join(base_path, f'04_output/02_dr_hands_on/cartpole/es_{args_name}/model')
os.makedirs(save_path, exist_ok=True)


# ----------
# Net
net = Net(env.observation_space.shape[0], env.action_space.n)
print(net)


# ----------
MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 10000

step_idx = 0

while True:
    t_start = time.time()
    batch_noise = []
    batch_reward = []
    batch_steps = 0

    # ----------
    # we gather data
    # until we reach the limit of episodes in the batch
    # or the limit of the total steps.
    for _ in range(MAX_BATCH_EPISODES):
        noise, neg_noise = sample_noise(net)
        batch_noise.append(noise)
        batch_noise.append(neg_noise)
        reward, steps = eval_with_noise(env, net, noise)
        batch_reward.append(reward)
        batch_steps += steps
        reward, steps = eval_with_noise(env, net, neg_noise)
        batch_reward.append(reward)
        batch_steps += steps
        if batch_steps > MAX_BATCH_STEPS:
            break

    step_idx += 1

    # if batch reward > threshold, we solved
    m_reward = np.mean(batch_reward)
    if m_reward > 199:
        print("Solved in %d steps" % step_idx)
        break

    # ----------
    # takes the batch with noise and respective rewards and calculates the update
    # to the network parameters
    train_step(net, batch_noise, batch_reward, writer, step_idx)

    # ----------
    writer.add_scalar("reward_mean", m_reward, step_idx)
    writer.add_scalar("reward_std", np.std(batch_reward), step_idx)
    writer.add_scalar("reward_max", np.max(batch_reward), step_idx)
    writer.add_scalar("batch_episodes", len(batch_reward), step_idx)
    writer.add_scalar("batch_steps", batch_steps, step_idx)
    speed = batch_steps / (time.time() - t_start)
    writer.add_scalar("speed", speed, step_idx)
    print("%d: reward=%.2f, speed=%.2f f/s" % (
        step_idx, m_reward, speed))


# ----------
name = "final_%d.dat" % (step_idx)
fname = os.path.join(save_path, name)
torch.save(net.state_dict(), fname)


##############################################################################################
# --------------------------------------------------------------------------------------------
# play by command line (xvfv-run)
# https://manpages.ubuntu.com/manpages/trusty/man1/xvfb-run.1.html
# --------------------------------------------------------------------------------------------

# xvfb-run:  run specified X client or command in a virtual X server environment
# -s:  --server-args, default is '-screen 0 640x480x8'
# +extension GLX:  enable OpenGL Extension to the X Window System

xvfb-run -s "-screen 0 640x480x24 +extension GLX" \
    ./cartpole_20_evolution_strategy_play.py \
        -m /home/kswada/kw/reinforcement_learning/04_output/02_dr_hands_on/cartpole/es_trial_202301050839/model/final_89.dat \
            -r /home/kswada/kw/reinforcement_learning/04_output/02_dr_hands_on/cartpole/es_trial_202301050839/video/final_89




##############################################################################################
##############################################################################################
# --------------------------------------------------------------------------------------------
# check step by step
# --------------------------------------------------------------------------------------------

ENV_ID = "CartPole-v0"

env = gym.make(ENV_ID)

net = Net(env.observation_space.shape[0], env.action_space.n)


# ----------
# 4 set of net parameters
# Linear, ReLU, Linear, Softmax
print(net)

for i, p in enumerate(net.parameters()):
    print(f'{i}: {p.shape}')
    print(f'data:  {p.data}')

# 0. Linear:  HID_SIZE(32:output) * observation_space(4:input)
print(list(net.parameters())[0])

# 1. ReLU: 32
print(list(net.parameters())[1])

# 2. Linear:  in 32  out 2
print(list(net.parameters())[2])

# 3. Softmaxr: 2
print(list(net.parameters())[3])


# ----------
MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 10000

batch_noise = []
batch_reward = []
batch_steps = 0

# ----------
# we gather data
for _ in range(MAX_BATCH_EPISODES):
    noise, neg_noise = sample_noise(net)
    batch_noise.append(noise)
    batch_noise.append(neg_noise)
    reward, steps = eval_with_noise(env, net, noise)
    batch_reward.append(reward)
    batch_steps += steps
    reward, steps = eval_with_noise(env, net, neg_noise)
    batch_reward.append(reward)
    batch_steps += steps
    if batch_steps > MAX_BATCH_STEPS:
        break

# ----------
# MAX_BATCH_EPISODES * 2 = 200 (positive and negative)  
print(len(batch_noise))
print(len(batch_noise[0]))

print(batch_noise[0][0].shape)
print(batch_noise[0][0])

print(batch_noise[0][1].shape)
print(batch_noise[0][1])

print(batch_noise[0][2].shape)
print(batch_noise[0][3])

print(batch_noise[0][3].shape)
print(batch_noise[0][3])


print(len(batch_reward))
print(batch_reward)


# ----------
LEARNING_RATE = 0.001

norm_reward = np.array(batch_reward)
norm_reward -= np.mean(norm_reward)
s = np.std(norm_reward)

print(f'norm reward: {norm_reward}')
print(f'std of norm reward: {s}')


weighted_noise = None

for noise, reward in zip(batch_noise, norm_reward):
    if weighted_noise is None:
        weighted_noise = [reward * p_n for p_n in noise]
    else:
        for w_n, p_n in zip(weighted_noise, noise):
            w_n += reward * p_n

print(weighted_noise[0])
print(w_n)


# ----------
m_updates = []
for p, p_update in zip(net.parameters(), weighted_noise):
    update = p_update / (len(batch_reward) * NOISE_STD)
    p.data += LEARNING_RATE * update
    m_updates.append(torch.norm(update))
