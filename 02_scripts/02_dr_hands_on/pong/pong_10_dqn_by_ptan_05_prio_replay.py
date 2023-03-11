
import os
import gym
import ptan
import ptan.ignite as ptan_ignite
# import argparse
import random
import numpy as np

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
# "Prioritized Experience Replay"


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


# def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
#     states, actions, rewards, dones, next_states = unpack_batch(batch)

#     states_v = torch.tensor(states).to(device)
#     next_states_v = torch.tensor(next_states).to(device)
#     actions_v = torch.tensor(actions).to(device)
#     rewards_v = torch.tensor(rewards).to(device)
#     done_mask = torch.BoolTensor(dones).to(device)

#     actions_v = actions_v.unsqueeze(-1)
#     state_action_vals = net(states_v).gather(1, actions_v)
#     state_action_vals = state_action_vals.squeeze(-1)

#     # ----------
#     with torch.no_grad():
#         next_state_vals = tgt_net(next_states_v).max(1)[0]
#         next_state_vals[done_mask] = 0.0

#     bellman_vals = next_state_vals.detach() * gamma + rewards_v
#     return nn.MSELoss()(state_action_vals, bellman_vals)


# !!! Here Added !!!
def calc_loss(batch, batch_weights, net, tgt_net,
              gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)

    # ----------
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        next_s_vals = tgt_net(next_states_v).max(1)[0]
        next_s_vals[done_mask] = 0.0
        exp_sa_vals = next_s_vals.detach() * gamma + rewards_v

    l = (state_action_vals - exp_sa_vals) ** 2
    losses_v = batch_weights_v * l
    return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()


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
# PrioReplyBuffer
# ------------------------------------------------------------------------------------------------------

# replay buffer params
BETA_START = 0.4
BETA_FRAMES = 100000

class PrioReplayBuffer:
    def __init__(self, exp_source, buf_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros(
            (buf_size, ), dtype=np.float32)
        self.beta = BETA_START

    def update_beta(self, idx):
        v = BETA_START + idx * (1.0 - BETA_START) / BETA_FRAMES
        self.beta = min(1.0, v)
        return self.beta

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        max_prio = self.priorities.max() if \
            self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer),
                                   batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, \
               np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices,
                          batch_priorities):
        for idx, prio in zip(batch_indices,
                             batch_priorities):
            self.priorities[idx] = prio


########################################################################################################
# ------------------------------------------------------------------------------------------------------
# Deep Q-Net 
# ------------------------------------------------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

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


########################################################################################################
# ------------------------------------------------------------------------------------------------------
# Training by 
# ------------------------------------------------------------------------------------------------------

# ----------
# parameters
SEED = 123

random.seed(SEED)

torch.manual_seed(SEED)

params = HYPERPARAMS['pong']

device = torch.device("cuda")


# ----------
# environment
env = gym.make(params.env_name)

env = ptan.common.wrappers.wrap_dqn(env)

env.seed(SEED)


# ----------
# main and target net

net = DQN(env.observation_space.shape, env.action_space.n).to(device)

tgt_net = ptan.agent.TargetNet(net)


# ----------
# selector and agent

selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)

epsilon_tracker = EpsilonTracker(selector, params)

agent = ptan.agent.DQNAgent(net, selector, device=device)


# ----------
# experience source and replay buffer
exp_source = ptan.experience.ExperienceSourceFirstLast(
    env, agent, gamma=params.gamma)


PRIO_REPLAY_ALPHA = 0.6
buffer = PrioReplayBuffer(
    exp_source, params.replay_size, PRIO_REPLAY_ALPHA)


# ----------
# optimizer
optimizer = optim.Adam(net.parameters(),
                        lr=params.learning_rate)


# ----------
# training

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100

def process_batch(engine, batch_data):

    batch, batch_indices, batch_weights = batch_data
    optimizer.zero_grad()

    # here calc_loss is applied
    loss_v, sample_prios = calc_loss(
        batch, batch_weights, net, tgt_net.target_model,
        gamma=params.gamma, device=device)

    loss_v.backward()
    optimizer.step()

    buffer.update_priorities(batch_indices, sample_prios)

    epsilon_tracker.frame(engine.state.iteration)

    if engine.state.iteration % params.target_net_sync == 0:
        tgt_net.sync()
    return {
        "loss": loss_v.item(),
        "epsilon": selector.epsilon,
        "beta": buffer.update_beta(engine.state.iteration),
    }


engine = Engine(process_batch)

NAME = "05_prio_replay"
setup_ignite(engine, params, exp_source, NAME)

engine.run(batch_generator(buffer, params.replay_initial, params.batch_size))


# ----------
# save model
base_path = '/home/kswada/kw/reinforcement_learning'
model_path = os.path.join(base_path, '04_output/pong/deep_q_learning_ptan/model/PongNoFrameskip-v4-05_prio_replay.pth')

with open(model_path, 'wb') as fd:
    torch.save(net.state_dict(), fd)

