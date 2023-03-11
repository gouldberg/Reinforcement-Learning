
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import ptan
import ptan.ignite as ptan_ignite
from ignite.engine import Engine
from ignite.metrics import RunningAverage

from ignite.contrib.handlers import tensorboard_logger as tb_logger


##############################################################################################
# --------------------------------------------------------------------------------------------
# Net:  MountainCarNoisyNetDQN
# --------------------------------------------------------------------------------------------

# This version has an explicit method, sample_noise(), 
# to update the noise tensors, so we need to call this method on every training iterations.

# This is needed for future experiments with policy-based methods,
# which require the noise to be constant during the relatively long period of trajectories.

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


class MountainCarNoisyNetDQN(nn.Module):
    def __init__(self, obs_size, n_actions, hid_size: int = 128):
        super(MountainCarNoisyNetDQN, self).__init__()

        self.noisy_layers = [
            NoisyLinear(hid_size, n_actions),
        ]

        self.net = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            self.noisy_layers[0]
        )

    def forward(self, x):
        return self.net(x)

    def sample_noise(self):
        for l in self.noisy_layers:
            l.sample_noise()


##############################################################################################
# --------------------------------------------------------------------------------------------
# Batch
#   - unpack_batch
#   - process_bath
#   - batch_generator
# --------------------------------------------------------------------------------------------

def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state  # the result will be masked anyway
        else:
            lstate = np.array(exp.last_state, copy=False)
        last_states.append(lstate)
    return np.array(states, copy=False, dtype=np.float32), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False, dtype=np.float32)

def process_batch(engine, batch):
    if not training_enabled:
        return {
            "loss": 0.0,
            "epsilon": selector.epsilon
        }

    optimizer.zero_grad()
    loss_v = calc_loss_double_dqn(batch, net, tgt_net.target_model,
                                            gamma=params.gamma**N_STEPS)
    loss_v.backward()
    optimizer.step()
    res = {
        "loss": loss_v.item(),
        "epsilon": 0.0,
    }
    if engine.state.iteration % params.target_net_sync == 0:
        tgt_net.sync()

    if args_params.startswith("egreedy"):
        epsilon_tracker.frame(engine.state.iteration - epsilon_tracker_frame)
        res['epsilon'] = selector.epsilon
    # reset noise every training step, this is fine in off-policy method
    if args_params == 'noisynet':
        net.sample_noise()
    return res


def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer,
                    initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


##############################################################################################
# --------------------------------------------------------------------------------------------
# Calulate Loss
#   - calc_loss_double_dqn
# --------------------------------------------------------------------------------------------

def calc_loss_double_dqn(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        next_state_acts = net(next_states_v).max(1)[1]
        next_state_acts = next_state_acts.unsqueeze(-1)
        next_state_vals = tgt_net(next_states_v).gather(
                1, next_state_acts).squeeze(-1)
        next_state_vals[done_mask] = 0.0
        exp_sa_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, exp_sa_vals)


##############################################################################################
# --------------------------------------------------------------------------------------------
# Tracker
#   - EpsilonTracker
# --------------------------------------------------------------------------------------------

class EpsilonTracker:
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector,
                 params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)


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
# HYPERPARAMS
# --------------------------------------------------------------------------------------------

HYPERPARAMS = {
    'egreedy': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        'run_name':         'egreedy',
        'replay_size':      100000,
        'replay_initial':   100,
        'target_net_sync':  100,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'eps_decay_trigger': False,
    }),
    'egreedy-long': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        'run_name':         'egreedy-long',
        'replay_size':      100000,
        'replay_initial':   1000,
        'target_net_sync':  100,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'eps_decay_trigger': True,
    }),
    'noisynet': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        'run_name':         'noisynet',
        'replay_size':      100000,
        'replay_initial':   1000,
        'target_net_sync':  1000,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'eps_decay_trigger': False,
    }),
}


##############################################################################################
# --------------------------------------------------------------------------------------------
# Train Agent:  DQN method with noisy networks
# --------------------------------------------------------------------------------------------

SEED = 123

random.seed(SEED)
torch.manual_seed(SEED)


# ----------
args_params = 'noisynet'

params = HYPERPARAMS[args_params]


# ----------
# environment
env = gym.make(params.env_name)
test_env = gym.make(params.env_name)

env.seed(SEED)


# ----------
# net: MountainCarNoisyNetDQN
net = MountainCarNoisyNetDQN(env.observation_space.shape[0], env.action_space.n)

tgt_net = ptan.agent.TargetNet(net)

print(net)


# ----------
# selector, tracker:  THIS IS CHANGED FROM DQN with epsilon-greedy case
selector = ptan.actions.ArgmaxActionSelector()
training_enabled = True


# ----------
# Agent
agent = ptan.agent.DQNAgent(net, selector, preprocessor=ptan.agent.float32_preprocessor)


# ----------
# ExperienceSource and Buffer
N_STEPS = 4

exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params.gamma, steps_count=N_STEPS)

buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params.replay_size)


# ----------
# optimizer
optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)


# ----------
# engine
engine = Engine(process_batch)


# ----------
# setup ignite

args_name = 'trial'

setup_ignite(engine, params, exp_source, args_name, extra_metrics=(
    'test_reward', 'avg_test_reward', 'test_steps'))


# ----------
@engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
def check_reward_trigger(trainer: Engine):
    global training_enabled, epsilon_tracker_frame
    if training_enabled:
        return
    # check trigger condition to enable epsilon decay
    if trainer.state.episode_reward > -200:
        training_enabled = True
        epsilon_tracker_frame = trainer.state.iteration
        print("Epsilon decay triggered!")


@engine.on(ptan_ignite.PeriodEvents.ITERS_1000_COMPLETED)
def test_network(engine):
    net.train(False)
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
    net.train(True)


# ----------
engine.run(batch_generator(buffer, params.replay_initial, params.batch_size))

