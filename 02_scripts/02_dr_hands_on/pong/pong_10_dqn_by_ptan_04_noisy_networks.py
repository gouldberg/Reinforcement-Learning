
import os
import gym
import ptan
import ptan.ignite as ptan_ignite
# import argparse
import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import warnings
from datetime import timedelta, datetime
from types import SimpleNamespace
from typing import Iterable, Tuple, List

from ignite.engine import Engine
from ignite.metrics import RunningAverage

from ignite.contrib.handlers import tensorboard_logger as tb_logger


# -----------
# From paper
# "Noisy Networks for Exploration"


########################################################################################################
# ------------------------------------------------------------------------------------------------------
# common
# hyperparameters
# ------------------------------------------------------------------------------------------------------

HYPERPARAMS = {
    'pong': SimpleNamespace(**{
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    })}


# ------------------------------------------------------------------------------------------------------
# common
# ------------------------------------------------------------------------------------------------------

def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state  # the result will be masked anyway
        else:
            lstate = np.array(exp.last_state)
        last_states.append(lstate)
    return np.array(states, copy=False), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = \
        unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)

    # ----------
    with torch.no_grad():
        next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[done_mask] = 0.0

    bellman_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, bellman_vals)


class EpsilonTracker:
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector,
                 params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - \
              frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)


def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer,
                    initial: int, batch_size: int):
    # in the beginning, the function ensures that the buffer contains the required amount of samples
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


@torch.no_grad()
def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


def setup_ignite(engine: Engine, params: SimpleNamespace,
                 exp_source, run_name: str,
                 extra_metrics: Iterable[str] = ()):
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    # Emits the Ignite event every time a game episode ends.
    # It can also fire an event when the averaged reward for episodes crosses some boundary.
    # We use this to detect when the game is finally solved. 
    handler = ptan_ignite.EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.stop_reward)

    handler.attach(engine)

    # Tracks the time the episode has taken and the amount of interactions that we have had with the environment.
    # From this we calculate frames per second (FPS).
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    # called at the end of an episode.
    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Episode %d: reward=%.0f, steps=%s, "
              "speed=%.1f f/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps,
            trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=int(passed))))

    # called when the average reward grows above the boundary defined in the hyperparameters (18.0 in the case of Pong)
    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Game solved in %s, after %d episodes "
              "and %d iterations!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    # ----------
    now = datetime.now().isoformat(timespec='minutes').replace(':', '')
    logdir = f"runs/{now}-{params.run_name}-{run_name}"

    # TensorboardLogger, provided by Ignite to write into TensorBoard.
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    # OutputHandler write into TensorBoard information about the episode every time it is completed.
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


########################################################################################################
# ------------------------------------------------------------------------------------------------------
# Noisy Networks
# ------------------------------------------------------------------------------------------------------

# for independent Gaussian noise:
# for every weight in a fully connected layer, we have a random value
# that we draw from the normal distribution.
# Parameters of the noise, mu and sigam, are stored inside the layer and get trained
# using backpropagation in the same way that we train weights of the standard linear layer.
# The noise applied to each weight and bias is independent.

class NoisyLinear(nn.Linear):
    # The initial value for sigmas (0.017) was taken from the Noisy Networks article.
    # From author's article:
    #  "This particular initialiaation was chosen because similar values worked well
    #   for the supervised learning tasks described in Fortunato et al. (2017),
    #   where the initialisation of the variances of the posteriors and the variances of
    #   the prior are related. We have not tuned for this parameter, but we believe
    #   different values on the same scale provide similar results."
    def __init__(self, in_features, out_features,
                 sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=bias)

        # a matrix for sigmas
        w = torch.full((out_features, in_features), sigma_init)
        # to make sigmas trainable, we need to wrap the tensor in an nn.Parameter
        self.sigma_weight = nn.Parameter(w)

        # ----------
        # The register_buffer method creates a tensor in the network
        # that won't be updated during backpropagation,
        # but will be handled by the nn.Module machinery
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
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * \
                   self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + \
            self.weight
        return F.linear(input, v, bias)


# ----------
# for factorized noise variant:
# to minimize the number of random values to be sampled to reduce computing time of
# random number generation 
# the authors proposed keeping only two random vectors:
#   - one with ths size of the input and nother with the size of the output of the layer
#   - then a random matrix for the layer is created by calculating the outer product of the vectors

class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    """
    # Based on author's article, sigma_zero=0.5, but here 0.4.
    def __init__(self, in_features, out_features,
                 sigma_zero=0.5, bias=True):
        super(NoisyFactorizedLinear, self).__init__(
            in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z1 = torch.zeros(1, in_features)
        self.register_buffer("epsilon_input", z1)
        z2 = torch.zeros(out_features, 1)
        self.register_buffer("epsilon_output", z2)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        # this is the function proposed in author's article
        func = lambda x: torch.sign(x) * \
                         torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        v = self.weight + self.sigma_weight * noise_v
        return F.linear(input, v, bias)


class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        # !! THIS IS ADDED on baseline normal DQN
        # add noise to the weights of fully connected layers of the network
        # and adjust the parameters of this noise during training using backpropagation.
        self.noisy_layers = [
            NoisyLinear(conv_out_size, 512),
            NoisyLinear(512, n_actions)
        ]

        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

    # !! THIS IS ADDED on baseline normal DQN
    # To check the internal noise level during training,
    # we can monitor the signal-to-noise ratio (SNR) of our noisy layers.
    # SNR = RMS(mu) / RMS(sigma).
    # SNR shows how many times the stationary component of the noisy layer is larger than
    # ths injected noise.

    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]


########################################################################################################
# ------------------------------------------------------------------------------------------------------
# Training by Noisy Net - DQN
#   1.  Epsilon-greedy is no longer used, 
#       but instead the policy greedily optimises the (randomized) action value function
#   2.  The fully connected layers of the value network are parametrised as a noisy network,
#       where the parameters are drawn from the noisy network parameter distribution after
#       every replay step.
# ------------------------------------------------------------------------------------------------------

# ----------
# parameters
SEED = 123

random.seed(SEED)

torch.manual_seed(SEED)

params = HYPERPARAMS['pong']

device = torch.device("cuda")


# ----------
# !!!
NOISY_SNR_EVERY_ITERS = 100


# ----------
# environment
env = gym.make(params.env_name)

env = ptan.common.wrappers.wrap_dqn(env)

env.seed(SEED)


# ----------
# main and target net

# Net is NoisyDQN !!
net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)

tgt_net = ptan.agent.TargetNet(net)


# ----------
# selector and agent
# remove all the code related to the epsilon-greedy strategy

# selector is ArgmaxActionSelector (not including epsilon greedy)
selector = ptan.actions.ArgmaxActionSelector()

# epsilon_tracker = EpsilonTracker(selector, params)

agent = ptan.agent.DQNAgent(net, selector, device=device)


# ----------
# experience source and replay buffer
exp_source = ptan.experience.ExperienceSourceFirstLast(
    env, agent, gamma=params.gamma)

buffer = ptan.experience.ExperienceReplayBuffer(
    exp_source, buffer_size=params.replay_size)


# ----------
# optimizer
optimizer = optim.Adam(net.parameters(),
                        lr=params.learning_rate)


# ----------
# training

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100

def process_batch(engine, batch):
    optimizer.zero_grad()
    loss_v = calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params.gamma, device=device)
    loss_v.backward()
    optimizer.step()
    # epsilon_tracker.frame(engine.state.iteration)

    # ----------
    # sync
    if engine.state.iteration % params.target_net_sync == 0:
        tgt_net.sync()
    if engine.state.iteration % NOISY_SNR_EVERY_ITERS == 0:
        for layer_idx, sigma_l2 in enumerate(net.noisy_layers_sigma_snr()):
            engine.state.metrics[f'snr_{layer_idx+1}'] = sigma_l2
    return {
        "loss": loss_v.item(),
    }

engine = Engine(process_batch)

NAME = "04_noisy"
setup_ignite(engine, params, exp_source, NAME, extra_metrics=('snr_1', 'snr_2'))

engine.run(batch_generator(buffer, params.replay_initial, params.batch_size))


# ----------
# save model
base_path = '/home/kswada/kw/reinforcement_learning'
model_path = os.path.join(base_path, '04_output/pong/deep_q_learning_ptan/model/PongNoFrameskip-v4-04_noisy.pth')

with open(model_path, 'wb') as fd:
    torch.save(net.state_dict(), fd)

