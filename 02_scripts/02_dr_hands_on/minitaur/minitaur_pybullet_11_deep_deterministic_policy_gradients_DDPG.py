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


# ----------
# Reference:
# paper 'Continuous Control with Deep Reinforcement Learning'


##############################################################################################
# --------------------------------------------------------------------------------------------
# DDPG Actor-Critic
# --------------------------------------------------------------------------------------------

# Note:
# When learning from pixels, use 3 layers (no pooling) with 32 filters at each layer,
# followed by 2 fully connected layers with 200 units (around 430,000 parameters)

class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        # 2 hidden layers with 400 and 300 units respectively (around 130,000 parameters)
        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            # actions were not included until 2nd hidden layer
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    # ----------
    def forward(self, x, a):
        # transform the observations with small network
        obs = self.obs_net(x)
        # concatenate the output and given actors to transform them into one single value of Q
        return self.out_net(torch.cat([obs, a], dim=1))


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
# Minitaur Agent Learning:  Deep Deterministic Policy Gradients (DDPG)
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
writer = SummaryWriter(comment="Minitaur_ddpg_" + args_name)

save_path = os.path.join(base_path, '04_output/minitaur/ddpg_trial/model')
os.makedirs(save_path, exist_ok=True)


# ----------
# net:  DDPG Actor and Critic
act_net = DDPGActor(
    env.observation_space.shape[0],
    env.action_space.shape[0]).to(device)

crt_net = DDPGCritic(
    env.observation_space.shape[0],
    env.action_space.shape[0]).to(device)

print(act_net)
print(crt_net)

tgt_act_net = ptan.agent.TargetNet(act_net)

tgt_crt_net = ptan.agent.TargetNet(crt_net)


# ----------
# agent
agent = AgentDDPG(act_net, device=device)


# ----------
# experience source and replay buffer

GAMMA = 0.99
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)

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
REPLAY_INITIAL = 10000

batch_size_tracker = 10

with ptan.common.utils.RewardTracker(writer) as tracker:

    # This wrapper is responsible for writing into TensorBoard the mean of the measured parameters for the last 10 steps.
    # This is helpful, as training can take millions of steps and we do not want to write millions of points into TensorBoard,
    # but rather write smoothed values eveyr 10 steps.
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

            # ----------
            # train critic:
            # To train the critic, we need to calculate the target Q-value using the 1-step Bellman equation,
            # with the target critic network as the approximation of the next state.
            crt_opt.zero_grad()
            q_v = crt_net(states_v, actions_v)
            last_act_v = tgt_act_net.target_model(last_states_v)
            q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
            q_last_v[dones_mask] = 0.0
            q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA

            critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
            critic_loss_v.backward()
            crt_opt.step()
            tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
            tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

            # ----------
            # train actor:
            # We need to update the actor's weights in a direction that will increase the critic's output.
            # As both the actor and critic are represented as differentiable functions,
            # we pass the actor's output to the critic and then minimize the negated value returned by the critic.
            act_opt.zero_grad()
            cur_actions_v = act_net(states_v)
            actor_loss_v = -crt_net(states_v, cur_actions_v)
            actor_loss_v = actor_loss_v.mean()

            # We don't want to touch the critic's weights,
            # so it is important to ask only the actor's optimizer to do the optimization step.
            # The weights of the critic will still keep the gradients from this call,
            # but they will be discarded on the next optimization step.
            actor_loss_v.backward()
            act_opt.step()
            tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

            # ----------
            # Not in continuous action problems, we synced the weights from the optimized network into the target
            # every n steps.
            # In continuous action problems, such syncing works worse than so-called 'soft sync'.
            # Soft sync is carried out on every step, but only a small ratio of the optimized network's weights
            # are added to the target network.
            # This makes a smooth and slow transition from the old weight to the new ones.
            tgt_act_net.alpha_sync(alpha=1 - 1e-3)
            tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

            # ----------
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
