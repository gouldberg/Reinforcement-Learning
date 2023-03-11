
import os
import gym
# import argparse
import random
import collections
import numpy as np
import warnings

import math

from datetime import timedelta, datetime
from types import SimpleNamespace
from typing import Iterable, Tuple, List
from typing import Union, Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import ptan
import ptan.ignite as ptan_ignite
from ignite.engine import Engine
from ignite.metrics import RunningAverage

from ignite.contrib.handlers import tensorboard_logger as tb_logger


##############################################################################################
# --------------------------------------------------------------------------------------------
# Net
# --------------------------------------------------------------------------------------------

class MountainCarBasePPO(nn.Module):
    def __init__(self, obs_size, n_actions, hid_size: int = 64):
        super(MountainCarBasePPO, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features,
                 sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        if not self.training:
            return super(NoisyLinear, self).forward(input)
        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * \
                   self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + \
            self.weight
        return F.linear(input, v, bias)

    def sample_noise(self):
        self.epsilon_weight.normal_()
        if self.bias is not None:
            self.epsilon_bias.normal_()


class MountainCarNoisyNetsPPO(nn.Module):
    def __init__(self, obs_size, n_actions, hid_size: int = 128):
        super(MountainCarNoisyNetsPPO, self).__init__()

        self.noisy_layers = [
            NoisyLinear(hid_size, n_actions)
        ]

        self.actor = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            self.noisy_layers[0],
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def sample_noise(self):
        for l in self.noisy_layers:
            l.sample_noise()


class MountainCarNetDistillery(nn.Module):
    def __init__(self, obs_size: int, hid_size: int = 128):
        super(MountainCarNetDistillery, self).__init__()

        self.ref_net = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )
        self.ref_net.train(False)

        self.trn_net = nn.Sequential(
            nn.Linear(obs_size, 1),
        )

    def forward(self, x):
        return self.ref_net(x), self.trn_net(x)

    def extra_reward(self, obs):
        r1, r2 = self.forward(torch.FloatTensor([obs]))
        return (r1 - r2).abs().detach().numpy()[0][0]

    def loss(self, obs_t):
        r1_t, r2_t = self.forward(obs_t)
        return F.mse_loss(r2_t, r1_t).mean()


##############################################################################################
# --------------------------------------------------------------------------------------------
# Batch:
#   - process_batch
#   - batch_generator
#   - new_ppo_batch
# --------------------------------------------------------------------------------------------

def process_batch(engine, batch):
    states_t, actions_t, adv_t, ref_t, old_logprob_t = batch

    opt_critic.zero_grad()
    value_t = net.critic(states_t)
    loss_value_t = F.mse_loss(value_t.squeeze(-1), ref_t)
    loss_value_t.backward()
    opt_critic.step()

    opt_actor.zero_grad()
    policy_t = net.actor(states_t)
    logpolicy_t = F.log_softmax(policy_t, dim=1)

    prob_t = F.softmax(policy_t, dim=1)
    loss_entropy_t = (prob_t * logpolicy_t).sum(dim=1).mean()

    logprob_t = logpolicy_t.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
    ratio_t = torch.exp(logprob_t - old_logprob_t)
    surr_obj_t = adv_t * ratio_t
    clipped_surr_t = adv_t * torch.clamp(ratio_t, 1.0 - params.ppo_eps, 1.0 + params.ppo_eps)
    loss_policy_t = -torch.min(surr_obj_t, clipped_surr_t).mean()
    loss_polent_t = params.entropy_beta * loss_entropy_t + loss_policy_t
    loss_polent_t.backward()
    opt_actor.step()

    res = {
        "loss": loss_value_t.item() + loss_polent_t.item(),
        "loss_value": loss_value_t.item(),
        "loss_policy": loss_policy_t.item(),
        "adv": adv_t.mean().item(),
        "ref": ref_t.mean().item(),
        "loss_entropy": loss_entropy_t.item(),
    }

    if net_distill is not None:
        opt_distill.zero_grad()
        loss_distill_t = net_distill.loss(states_t)
        loss_distill_t.backward()
        opt_distill.step()
        res['loss_distill'] = loss_distill_t.item()

    return res


def batch_generator(exp_source: ptan.experience.ExperienceSource,
                    net: nn.Module,
                    trajectory_size: int, ppo_epoches: int,
                    batch_size: int, gamma: float, gae_lambda: float,
                    device: Union[torch.device, str] = "cpu", trim_trajectory: bool = True,
                    new_batch_callable: Optional[Callable] = None):
    trj_states = []
    trj_actions = []
    trj_rewards = []
    trj_dones = []
    last_done_index = None
    for (exp,) in exp_source:
        trj_states.append(exp.state)
        trj_actions.append(exp.action)
        trj_rewards.append(exp.reward)
        trj_dones.append(exp.done)
        if exp.done:
            last_done_index = len(trj_states)-1
        if len(trj_states) < trajectory_size:
            continue
        # ensure that we have at least one full episode in the trajectory
        if last_done_index is None or last_done_index == len(trj_states)-1:
            continue

        if new_batch_callable is not None:
            new_batch_callable()

        # trim the trajectory till the last done plus one step (which will be discarded).
        # This increases convergence speed and stability
        if trim_trajectory:
            trj_states = trj_states[:last_done_index+2]
            trj_actions = trj_actions[:last_done_index + 2]
            trj_rewards = trj_rewards[:last_done_index + 2]
            trj_dones = trj_dones[:last_done_index + 2]

        trj_states_t = torch.FloatTensor(trj_states).to(device)
        trj_actions_t = torch.tensor(trj_actions).to(device)
        policy_t, trj_values_t = net(trj_states_t)
        trj_values_t = trj_values_t.squeeze()

        adv_t, ref_t = calc_adv_ref(trj_values_t.data.cpu().numpy(),
                                    trj_dones, trj_rewards, gamma, gae_lambda)
        adv_t = adv_t.to(device)
        ref_t = ref_t.to(device)

        logpolicy_t = F.log_softmax(policy_t, dim=1)
        old_logprob_t = logpolicy_t.gather(1, trj_actions_t.unsqueeze(-1)).squeeze(-1)
        adv_t = (adv_t - torch.mean(adv_t)) / torch.std(adv_t)
        old_logprob_t = old_logprob_t.detach()

        # make our trajectory splittable on even batch chunks
        trj_len = len(trj_states) - 1
        trj_len -= trj_len % batch_size
        trj_len += 1
        indices = np.arange(0, trj_len-1)

        # generate needed amount of batches
        for _ in range(ppo_epoches):
            np.random.shuffle(indices)
            for batch_indices in np.split(indices, trj_len // batch_size):
                yield (
                    trj_states_t[batch_indices],
                    trj_actions_t[batch_indices],
                    adv_t[batch_indices],
                    ref_t[batch_indices],
                    old_logprob_t[batch_indices],
                )

        trj_states.clear()
        trj_actions.clear()
        trj_rewards.clear()
        trj_dones.clear()


def new_ppo_batch():
    # In noisy networks we need to reset the noise
    if args_params == 'noisynet':
        net.sample_noise()


##############################################################################################
# --------------------------------------------------------------------------------------------
# Calulate
#   - calc_adv_ref
# --------------------------------------------------------------------------------------------

def calc_adv_ref(values, dones, rewards, gamma, gae_lambda):
    last_gae = 0.0
    adv, ref = [], []

    for val, next_val, done, reward in zip(reversed(values[:-1]), reversed(values[1:]),
                                           reversed(dones[:-1]), reversed(rewards[:-1])):
        if done:
            delta = reward - val
            last_gae = delta
        else:
            delta = reward + gamma * next_val - val
            last_gae = delta + gamma * gae_lambda * last_gae
        adv.append(last_gae)
        ref.append(last_gae + val)
    adv = list(reversed(adv))
    ref = list(reversed(ref))
    return torch.FloatTensor(adv), torch.FloatTensor(ref)


##############################################################################################
# --------------------------------------------------------------------------------------------
# setup_ignite
# --------------------------------------------------------------------------------------------

def setup_ignite(engine: Engine, params: SimpleNamespace,
                 exp_source, run_name: str,
                 extra_metrics: Iterable[str] = ()):
    warnings.simplefilter("ignore", category=UserWarning)
    handler = ptan_ignite.EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.stop_reward)
    handler.attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Episode %d: reward=%.2f, steps=%s, "
              "speed=%.1f f/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps,
            trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=int(passed))))

    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Game solved in %s, after %d episodes "
              "and %d iterations!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    now = datetime.now().isoformat(timespec='minutes')
    logdir = f"runs/{now}-{params.run_name}-{run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    metrics = ['reward', 'steps', 'avg_reward']
    handler = tb_logger.OutputHandler(
        tag="episodes", metric_names=metrics)
    event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # write to tensorboard every 100 iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(
        tag="train", metric_names=metrics,
        output_transform=lambda a: a)
    event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)


##############################################################################################
# --------------------------------------------------------------------------------------------
#  Environment Wrapper
#    - PseudoCountRewardWrapper
#    - counts_hash 
#    - NetworkDistillationRewardWrapper
# --------------------------------------------------------------------------------------------

class PseudoCountRewardWrapper(gym.Wrapper):
    def __init__(self, env, hash_function = lambda o: o,
                 reward_scale: float = 1.0):
        super(PseudoCountRewardWrapper, self).__init__(env)
        self.hash_function = hash_function
        self.reward_scale = reward_scale
        # ----------
        # map hashed state into the count of times we have seen it.
        self.counts = collections.Counter()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        extra_reward = self._count_observation(obs)
        return obs, reward + self.reward_scale * extra_reward, done, info

    def _count_observation(self, obs) -> float:
        """
        Increments observation counter and returns pseudo-count reward
        :param obs: observation
        :return: extra reward
        """
        h = self.hash_function(obs)
        # ----------
        # map hashed state into the count of times we have seen it.
        self.counts[h] += 1
        return np.sqrt(1/self.counts[h])


def counts_hash(obs):
    r = obs.tolist()
    return tuple(map(lambda v: round(v, 3), r))


class NetworkDistillationRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_callable, reward_scale: float = 1.0, sum_rewards: bool = True):
        super(NetworkDistillationRewardWrapper, self).__init__(env)
        self.reward_scale = reward_scale
        self.reward_callable = reward_callable
        self.sum_rewards = sum_rewards

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        extra_reward = self.reward_callable(obs)
        if self.sum_rewards:
            res_rewards = reward + self.reward_scale * extra_reward
        else:
            res_rewards = np.array([reward, extra_reward * self.reward_scale])
        return obs, res_rewards, done, info


##############################################################################################
# --------------------------------------------------------------------------------------------
# HYPERPARAMS
# --------------------------------------------------------------------------------------------

HYPERPARAMS = {
    'debug': SimpleNamespace(**{
        'env_name':         "CartPole-v0",
        'stop_reward':      None,
        'stop_test_reward': 190.0,
        'run_name':         'debug',
        'actor_lr':         1e-4,
        'critic_lr':        1e-4,
        'gamma':            0.9,
        'ppo_trajectory':   2049,
        'ppo_epoches':      10,
        'ppo_eps':          0.2,
        'batch_size':       32,
        'gae_lambda':       0.95,
        'entropy_beta':     0.1,
    }),
    'ppo': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        'run_name':         'ppo',
        'actor_lr':         1e-4,
        'critic_lr':        1e-4,
        'gamma':            0.99,
        'ppo_trajectory':   2049,
        'ppo_epoches':      10,
        'ppo_eps':          0.2,
        'batch_size':       32,
        'gae_lambda':       0.95,
        'entropy_beta':     0.1,
    }),
    'noisynet': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        'run_name':         'noisynet',
        'actor_lr':         1e-4,
        'critic_lr':        1e-4,
        'gamma':            0.99,
        'ppo_trajectory':   2049,
        'ppo_epoches':      10,
        'ppo_eps':          0.2,
        'batch_size':       32,
        'gae_lambda':       0.95,
        'entropy_beta':     0.1,
    }),
    'counts': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        # 'stop_test_reward': -100,
        'run_name':         'counts',
        'actor_lr':         1e-4,
        'critic_lr':        1e-4,
        'gamma':            0.99,
        'ppo_trajectory':   2049,
        'ppo_epoches':      10,
        'ppo_eps':          0.2,
        'batch_size':       32,
        'gae_lambda':       0.95,
        'entropy_beta':     0.1,
        'counts_reward_scale': 0.5,
    }),

    'distill': SimpleNamespace(**{
        'env_name': "MountainCar-v0",
        'stop_reward': None,
        'stop_test_reward': -130.0,
        'run_name': 'distill',
        'actor_lr': 1e-4,
        'critic_lr': 1e-4,
        'gamma': 0.99,
        'ppo_trajectory': 2049,
        'ppo_epoches': 10,
        'ppo_eps': 0.2,
        'batch_size': 32,
        'gae_lambda': 0.95,
        'entropy_beta': 0.1,
        'reward_scale': 100.0,
        'distill_lr': 1e-5,
    }),
}


##############################################################################################
# --------------------------------------------------------------------------------------------
# Train Agent:  PPO with variety of exploration
# --------------------------------------------------------------------------------------------

SEED = 123

random.seed(SEED)
torch.manual_seed(SEED)


# ----------
# args_params = 'debug'
# args_params = 'ppo'
# args_params = 'noisynet'
args_params = 'counts'
# args_params = 'distill'


params = HYPERPARAMS[args_params]


# ----------
# environment and net
env = gym.make(params.env_name)

test_env = gym.make(params.env_name)

if args_params == 'counts':
    env = PseudoCountRewardWrapper(env, reward_scale=params.counts_reward_scale,
                                            hash_function=counts_hash)
net_distill = None
if args_params == 'distill':
    net_distill = MountainCarNetDistillery(env.observation_space.shape[0])
    env = NetworkDistillationRewardWrapper(env, net_distill.extra_reward, reward_scale=params.reward_scale)

env.seed(SEED)

# --
# change max_episode_steps to 1000
# env = env.env
# env._max_episode_steps = 1000

# test_env = test_env.env
# test_env._max_episode_steps = 1000
# --

if args_params == 'noisynet':
    net = MountainCarNoisyNetsPPO(env.observation_space.shape[0], env.action_space.n)
else:
    net = MountainCarBasePPO(env.observation_space.shape[0], env.action_space.n)

print(net)


# ----------
# Agent
agent = ptan.agent.PolicyAgent(net.actor, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)


# ----------
# ExperienceSource
exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)


# ----------
# optimizer
opt_actor = optim.Adam(net.actor.parameters(), lr=params.actor_lr)
opt_critic = optim.Adam(net.critic.parameters(), lr=params.critic_lr)

if net_distill is not None:
    opt_distill = optim.Adam(net_distill.trn_net.parameters(), lr=params.distill_lr)


# ----------
# engine
engine = Engine(process_batch)


# ----------
# setup ignite
args_name = 'trial'

setup_ignite(engine, params, exp_source, args_name, extra_metrics=(
    'test_reward', 'avg_test_reward', 'test_steps'))


# ----------
@engine.on(ptan_ignite.PeriodEvents.ITERS_1000_COMPLETED)
def test_network(engine):
    net.actor.train(False)
    obs = test_env.reset()
    reward = 0.0
    steps = 0

    while True:
        acts, _ = agent([obs])
        obs, r, is_done, _ = test_env.step(acts[0])
        reward += r
        steps += 1
        if is_done:
            break

    test_reward_avg = getattr(engine.state, "test_reward_avg", None)

    if test_reward_avg is None:
        test_reward_avg = reward
    else:
        test_reward_avg = test_reward_avg * 0.95 + 0.05 * reward

    engine.state.test_reward_avg = test_reward_avg
    print("Test done: got %.3f reward after %d steps, avg reward %.3f" % (
        reward, steps, test_reward_avg
    ))
    engine.state.metrics['test_reward'] = reward
    engine.state.metrics['avg_test_reward'] = test_reward_avg
    engine.state.metrics['test_steps'] = steps

    if test_reward_avg > params.stop_test_reward:
        print("Reward boundary has crossed, stopping training. Contgrats!")
        engine.should_terminate = True

    net.actor.train(True)


engine.run(batch_generator(exp_source, net, params.ppo_trajectory,
                                params.ppo_epoches, params.batch_size,
                                params.gamma, params.gae_lambda, new_batch_callable=new_ppo_batch))



# ----------
base_path = '/home/kswada/kw/reinforcement_learning'
save_path = os.path.join(base_path, '04_output/02_dr_hands_on/mountaincar/ppo_state_count_202301061824/model')
os.makedirs(save_path, exist_ok=True)

fname = os.path.join(save_path, 'final_79869.dat')
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
    ./mountaincar_02_ppo_03_state_count_play.py \
        -m /home/kswada/kw/reinforcement_learning/04_output/02_dr_hands_on/mountaincar/ppo_state_count_202301061824/model/final_79869.dat \
            -r /home/kswada/kw/reinforcement_learning/04_output/02_dr_hands_on/mountaincar/ppo_state_count_202301061824/video/final_79869
