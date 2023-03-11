
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

import warnings
from datetime import timedelta, datetime
from types import SimpleNamespace
from typing import Iterable, Tuple, List

from ignite.engine import Engine
from ignite.metrics import RunningAverage

from ignite.contrib.handlers import tensorboard_logger as tb_logger


# -----------
# From paper
# "Learning to Predict by the Methods of Temporal Differences"  (Richard Sutton)


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
# evaluate_states:  THIS IS ADDED
# ------------------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_states(states, net, device, engine):
    s_v = torch.tensor(states).to(device)
    adv, val = net.adv_val(s_v)
    engine.state.metrics['adv'] = adv.mean().item()
    engine.state.metrics['val'] = val.mean().item()


########################################################################################################
# ------------------------------------------------------------------------------------------------------
# Dueling DQN
# ------------------------------------------------------------------------------------------------------

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        adv, val = self.adv_val(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc_adv(conv_out), self.fc_val(conv_out)


########################################################################################################
# ------------------------------------------------------------------------------------------------------
# Training by Dueling DQN
# ------------------------------------------------------------------------------------------------------

# ----------
# parameters
SEED = 123

random.seed(SEED)

torch.manual_seed(SEED)

params = HYPERPARAMS['pong']

device = torch.device("cuda")


# ----------
STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100


# ----------
# environment
env = gym.make(params.env_name)

env = ptan.common.wrappers.wrap_dqn(env)

env.seed(SEED)


# ----------
# main and target net

# net = DQN(env.observation_space.shape, env.action_space.n).to(device)
net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)

tgt_net = ptan.agent.TargetNet(net)


# ----------
# selector and agent
selector = ptan.actions.EpsilonGreedyActionSelector(
    epsilon=params.epsilon_start)

epsilon_tracker = EpsilonTracker(selector, params)

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

def process_batch(engine, batch):
    optimizer.zero_grad()
    loss_v = calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params.gamma, device=device)
    loss_v.backward()
    optimizer.step()

    epsilon_tracker.frame(engine.state.iteration)

    # ----------
    # sync
    if engine.state.iteration % params.target_net_sync == 0:
        tgt_net.sync()
    if engine.state.iteration % EVAL_EVERY_FRAME == 0:
        eval_states = getattr(engine.state, "eval_states", None)
        if eval_states is None:
            eval_states = buffer.sample(STATES_TO_EVALUATE)
            eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
            eval_states = np.array(eval_states, copy=False)
            engine.state.eval_states = eval_states
        evaluate_states(eval_states, net, device, engine)
    return {
        "loss": loss_v.item(),
        "epsilon": selector.epsilon,
    }


engine = Engine(process_batch)

NAME = "06_dueling"
setup_ignite(engine, params, exp_source, NAME, extra_metrics=('adv', 'val'))

engine.run(batch_generator(buffer, params.replay_initial, params.batch_size))


# ----------
# save model
base_path = '/home/kswada/kw/reinforcement_learning'
model_path = os.path.join(base_path, '04_output/pong/deep_q_learning_ptan/model/PongNoFrameskip-v4-06_dueling.pth')

with open(model_path, 'wb') as fd:
    torch.save(net.state_dict(), fd)

