
import gym
# import argparse
import random
import collections
import numpy as np
import warnings

from datetime import timedelta, datetime
from types import SimpleNamespace
from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim

import ptan
import ptan.ignite as ptan_ignite
from ignite.engine import Engine
from ignite.metrics import RunningAverage

from ignite.contrib.handlers import tensorboard_logger as tb_logger


##############################################################################################
# --------------------------------------------------------------------------------------------
# Net:  MountainCarBaseDQN
# --------------------------------------------------------------------------------------------

# class BaselineDQN(nn.Module):
#     """
#     Dueling net
#     """
#     def __init__(self, input_shape, n_actions):
#         super(BaselineDQN, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32,
#                       kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )

#         conv_out_size = self._get_conv_out(input_shape)
#         self.fc_adv = nn.Sequential(
#             nn.Linear(conv_out_size, 256),
#             nn.ReLU(),
#             nn.Linear(256, n_actions)
#         )
#         self.fc_val = nn.Sequential(
#             nn.Linear(conv_out_size, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def _get_conv_out(self, shape):
#         o = self.conv(torch.zeros(1, *shape))
#         return int(np.prod(o.size()))

#     def forward(self, x):
#         adv, val = self.adv_val(x)
#         return val + (adv - adv.mean(dim=1, keepdim=True))

#     def adv_val(self, x):
#         fx = x.float() / 256
#         conv_out = self.conv(fx).view(fx.size()[0], -1)
#         return self.fc_adv(conv_out), self.fc_val(conv_out)


class MountainCarBaseDQN(nn.Module):
    def __init__(self, obs_size, n_actions, hid_size: int = 128):
        super(MountainCarBaseDQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


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
}


##############################################################################################
# --------------------------------------------------------------------------------------------
# Train Agent:  DQN method with epsilon-greedy
# --------------------------------------------------------------------------------------------

SEED = 123

random.seed(SEED)
torch.manual_seed(SEED)


# ----------
# egreedy: epsilon_frames = 10 ** 5 (training steps)
# --> 500 episodes
# args_params = 'egreedy'

# egreedy-long: epsilon_frames = 10 ** 6 (training steps)
# --> 5000 episodes
args_params = 'egreedy-long'

params = HYPERPARAMS[args_params]


# ----------
# environment
env = gym.make(params.env_name)
test_env = gym.make(params.env_name)

env.seed(SEED)


# ----------
# net
net = MountainCarBaseDQN(env.observation_space.shape[0], env.action_space.n)

tgt_net = ptan.agent.TargetNet(net)

print(net)


# ----------
# selector, tracker
selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
epsilon_tracker = EpsilonTracker(selector, params)
training_enabled = not params.eps_decay_trigger
epsilon_tracker_frame = 0


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


# -->
# Unfortunately, even after many episodes,
# it still have not faced even a single example of the goal .....
# Rewared stays at -200 ...
