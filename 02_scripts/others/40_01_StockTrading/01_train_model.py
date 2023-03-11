
import os
import csv
import glob
import collections
import pathlib
# import argparse
import numpy as np
import enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
from typing import Iterable
from datetime import datetime, timedelta

import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import gym.wrappers

import ptan
import ptan.ignite as ptan_ignite

from ignite.engine import Engine
from ignite.metrics import RunningAverage

from ignite.contrib.handlers import tensorboard_logger as tb_logger


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# lib.environ.py 
# ----------------------------------------------------------------------------------------------------------------

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    def __init__(self, bars_count, commission_perc,
                 reset_on_close, reward_on_close=True,
                 volumes=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    def reset(self, prices, offset):
        assert isinstance(prices, Prices)
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        # [h, l, c] * bars + position_flag + rel_profit
        if self.volumes:
            return 4 * self.bars_count + 1 + 1,
        else:
            return 3 * self.bars_count + 1 + 1,

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            ofs = self._offset + bar_idx
            res[shift] = self._prices.high[ofs]
            shift += 1
            res[shift] = self._prices.low[ofs]
            shift += 1
            res[shift] = self._prices.close[ofs]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[ofs]
                shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = self._cur_close() / self.open_price - 1.0
        return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close / self.open_price - 1.0)
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1

        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close / prev_close - 1.0)

        return reward, done


class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        start = self._offset-(self.bars_count-1)
        stop = self._offset+1
        res[0] = self._prices.high[start:stop]
        res[1] = self._prices.low[start:stop]
        res[2] = self._prices.close[start:stop]
        if self.volumes:
            res[3] = self._prices.volume[start:stop]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = self._cur_close() / self.open_price - 1.0
        return res


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("StocksEnv-v0")

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC,
                 reset_on_close=True, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=False,
                 volumes=False):
        assert isinstance(prices, dict)
        self._prices = prices
        if state_1d:
            self._state = State1D(
                bars_count, commission, reset_on_close,
                reward_on_close=reward_on_close, volumes=volumes)
        else:
            self._state = State(
                bars_count, commission, reset_on_close,
                reward_on_close=reward_on_close, volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        self._instrument = self.np_random.choice(
            list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(
                prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {
            "instrument": self._instrument,
            "offset": self._state._offset
        }
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {
            file: load_relative(file)
            for file in price_files(data_dir)
        }
        return StocksEnv(prices, **kwargs)


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# lib.data.py 
# ----------------------------------------------------------------------------------------------------------------

Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])


def read_csv(file_name, sep=',', filter_data=True, fix_open_price=False):
    print("Reading", file_name)
    with open(file_name, 'rt', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter=sep)
        h = next(reader)
        if '<OPEN>' not in h and sep == ',':
            return read_csv(file_name, ';')
        indices = [h.index(s) for s in ('<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>')]
        o, h, l, c, v = [], [], [], [], []
        count_out = 0
        count_filter = 0
        count_fixed = 0
        prev_vals = None
        for row in reader:
            vals = list(map(float, [row[idx] for idx in indices]))
            if filter_data and all(map(lambda v: abs(v-vals[0]) < 1e-8, vals[:-1])):
                count_filter += 1
                continue

            po, ph, pl, pc, pv = vals

            # fix open price for current bar to match close price for the previous bar
            if fix_open_price and prev_vals is not None:
                ppo, pph, ppl, ppc, ppv = prev_vals
                if abs(po - ppc) > 1e-8:
                    count_fixed += 1
                    po = ppc
                    pl = min(pl, po)
                    ph = max(ph, po)
            count_out += 1
            o.append(po)
            c.append(pc)
            h.append(ph)
            l.append(pl)
            v.append(pv)
            prev_vals = vals
    print("Read done, got %d rows, %d filtered, %d open prices adjusted" % (
        count_filter + count_out, count_filter, count_fixed))
    return Prices(open=np.array(o, dtype=np.float32),
                  high=np.array(h, dtype=np.float32),
                  low=np.array(l, dtype=np.float32),
                  close=np.array(c, dtype=np.float32),
                  volume=np.array(v, dtype=np.float32))


def prices_to_relative(prices):
    """
    Convert prices to relative in respect to open price
    :param ochl: tuple with open, close, high, low
    :return: tuple with open, rel_close, rel_high, rel_low
    """
    assert isinstance(prices, Prices)
    rh = (prices.high - prices.open) / prices.open
    rl = (prices.low - prices.open) / prices.open
    rc = (prices.close - prices.open) / prices.open
    return Prices(open=prices.open, high=rh, low=rl, close=rc, volume=prices.volume)


def load_relative(csv_file):
    return prices_to_relative(read_csv(csv_file))


def price_files(dir_name):
    result = []
    for path in glob.glob(os.path.join(dir_name, "*.csv")):
        result.append(path)
    return result


def load_year_data(year, basedir='data'):
    y = str(year)[-2:]
    result = {}
    for path in glob.glob(os.path.join(basedir, "*_%s*.csv" % y)):
        result[path] = load_relative(path)
    return result


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# lib.common.py
# ----------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_actions = net(next_states_v).max(1)[1]
    next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer,
                    initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


def setup_ignite(engine: Engine, exp_source, run_name: str,
                 extra_metrics: Iterable[str] = ()):
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    handler = ptan_ignite.EndOfEpisodeHandler(exp_source, subsample_end_of_episode=100)
    handler.attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Episode %d: reward=%.0f, steps=%s, "
              "speed=%.1f f/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps,
            trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=int(passed))))

    now = datetime.now().isoformat(timespec='minutes')
    logdir = f"runs/{now}-{run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    metrics = ['reward', 'steps', 'avg_reward']
    handler = tb_logger.OutputHandler(
        tag="episodes", metric_names=metrics)
    event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    ptan_ignite.PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(
        tag="train", metric_names=metrics,
        output_transform=lambda a: a)
    event = ptan_ignite.PeriodEvents.ITERS_1000_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)
    return tb


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# lib.model.py 
# ----------------------------------------------------------------------------------------------------------------

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
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
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class SimpleFFDQN(nn.Module):
    def __init__(self, obs_len, actions_n):
        super(SimpleFFDQN, self).__init__()

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class DQNConv1D(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class DQNConv1DLarge(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1DLarge, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# lib.validation.py 
# ----------------------------------------------------------------------------------------------------------------

METRICS = (
    'episode_reward',
    'episode_steps',
    'order_profits',
    'order_steps',
)


def validation_run(env, net, episodes=100, device="cpu", epsilon=0.02, comission=0.1):
    stats = { metric: [] for metric in METRICS }

    for episode in range(episodes):
        obs = env.reset()

        total_reward = 0.0
        position = None
        position_steps = None
        episode_steps = 0

        while True:
            obs_v = torch.tensor([obs]).to(device)
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = Actions(action_idx)

            close_price = env._state._cur_close()

            if action == Actions.Buy and position is None:
                position = close_price
                position_steps = 0
            elif action == Actions.Close and position is not None:
                profit = close_price - position - (close_price + position) * comission / 100
                profit = 100.0 * profit / position
                stats['order_profits'].append(profit)
                stats['order_steps'].append(position_steps)
                position = None
                position_steps = None

            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward
            episode_steps += 1
            if position_steps is not None:
                position_steps += 1
            if done:
                if position is not None:
                    profit = close_price - position - (close_price + position) * comission / 100
                    profit = 100.0 * profit / position
                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)
                break

        stats['episode_reward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    return { key: np.mean(vals) for key, vals in stats.items() }


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# agent learning 
# ----------------------------------------------------------------------------------------------------------------

device = torch.device("cuda")


# ----------
# save path
SAVES_DIR = pathlib.Path('/home/kswada/kw/reinforcement_learning/04_output/stocktrading')
args_run = "first"

saves_path = SAVES_DIR / f"simple-{args_run}"

saves_path.mkdir(parents=True, exist_ok=True)


# ----------
# data
data_base_path = '/home/kswada/kw/reinforcement_learning/02_scripts/40_01_StockTrading/data'
STOCKS = os.path.join(data_base_path, "YNDX_160101_161231.csv")
VAL_STOCKS = os.path.join(data_base_path, "YNDX_150101_151231.csv")

data_path = pathlib.Path(STOCKS)
val_path = pathlib.Path(VAL_STOCKS)

BARS_COUNT = 10
args_year = None

if args_year is not None or data_path.is_file():
    if args_year is not None:
        stock_data = load_year_data(args_year, basedir=data_base_path)
    else:
        stock_data = {"YNDX": load_relative(data_path)}
    env = StocksEnv(stock_data, bars_count=BARS_COUNT)
    env_tst = StocksEnv(stock_data, bars_count=BARS_COUNT)
elif data_path.is_dir():
    env = StocksEnv.from_dir(data_path, bars_count=BARS_COUNT)
    env_tst = StocksEnv.from_dir(data_path, bars_count=BARS_COUNT)
else:
    raise RuntimeError("No data to train on")


val_data = {"YNDX": load_relative(val_path)}
env_val = StocksEnv(val_data, bars_count=BARS_COUNT)

print(stock_data)
print(val_data)
print(env)
print(env_tst)


# ----------
# environment
env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)


# ----------
# model
net = SimpleFFDQN(env.observation_space.shape[0], env.action_space.n).to(device)

tgt_net = ptan.agent.TargetNet(net)


# ----------
# selector, tracker, agent
EPS_START = 1.0
selector = ptan.actions.EpsilonGreedyActionSelector(EPS_START)


EPS_FINAL = 0.1
EPS_STEPS = 1000000
eps_tracker = ptan.actions.EpsilonTracker(
    selector, EPS_START, EPS_FINAL, EPS_STEPS)

agent = ptan.agent.DQNAgent(net, selector, device=device)


# ----------
# experience and replay buffer
REWARD_STEPS = 2
REPLAY_SIZE = 100000
GAMMA = 0.99
exp_source = ptan.experience.ExperienceSourceFirstLast(
    env, agent, GAMMA, steps_count=REWARD_STEPS)

buffer = ptan.experience.ExperienceReplayBuffer(
    exp_source, REPLAY_SIZE)


# ----------
# optimizer
LEARNING_RATE = 0.0001
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


# ----------
# engine

STATES_TO_EVALUATE = 1000

def process_batch(engine, batch):
    optimizer.zero_grad()
    loss_v = calc_loss(
        batch, net, tgt_net.target_model,
        gamma=GAMMA ** REWARD_STEPS, device=device)
    loss_v.backward()
    optimizer.step()
    eps_tracker.frame(engine.state.iteration)

    if getattr(engine.state, "eval_states", None) is None:
        eval_states = buffer.sample(STATES_TO_EVALUATE)
        eval_states = [np.array(transition.state, copy=False)
                        for transition in eval_states]
        engine.state.eval_states = np.array(eval_states, copy=False)

    return {
        "loss": loss_v.item(),
        "epsilon": selector.epsilon,
    }


engine = Engine(process_batch)

tb = setup_ignite(engine, exp_source, f"simple-{args_run}",
                            extra_metrics=('values_mean',))


@engine.on(ptan.ignite.PeriodEvents.ITERS_1000_COMPLETED)
def sync_eval(engine: Engine):
    tgt_net.sync()

    mean_val = calc_values_of_states(
        engine.state.eval_states, net, device=device)
    engine.state.metrics["values_mean"] = mean_val
    if getattr(engine.state, "best_mean_val", None) is None:
        engine.state.best_mean_val = mean_val
    if engine.state.best_mean_val < mean_val:
        print("%d: Best mean value updated %.3f -> %.3f" % (
            engine.state.iteration, engine.state.best_mean_val,
            mean_val))
        path = saves_path / ("mean_value-%.3f.data" % mean_val)
        torch.save(net.state_dict(), path)
        engine.state.best_mean_val = mean_val


@engine.on(ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED)
def validate(engine: Engine):
    res = validation_run(env_tst, net, device=device)
    print("%d: tst: %s" % (engine.state.iteration, res))
    for key, val in res.items():
        engine.state.metrics[key + "_tst"] = val
    res = validation_run(env_val, net, device=device)
    print("%d: val: %s" % (engine.state.iteration, res))
    for key, val in res.items():
        engine.state.metrics[key + "_val"] = val
    val_reward = res['episode_reward']
    if getattr(engine.state, "best_val_reward", None) is None:
        engine.state.best_val_reward = val_reward
    if engine.state.best_val_reward < val_reward:
        print("Best validation reward updated: %.3f -> %.3f, model saved" % (
            engine.state.best_val_reward, val_reward
        ))
        engine.state.best_val_reward = val_reward
        path = saves_path / ("val_reward-%.3f.data" % val_reward)
        torch.save(net.state_dict(), path)


event = ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED

tst_metrics = [m + "_tst" for m in METRICS]

tst_handler = tb_logger.OutputHandler(
    tag="test", metric_names=tst_metrics)

tb.attach(engine, log_handler=tst_handler, event_name=event)

val_metrics = [m + "_val" for m in METRICS]

val_handler = tb_logger.OutputHandler(
    tag="validation", metric_names=val_metrics)

tb.attach(engine, log_handler=val_handler, event_name=event)


REPLAY_INITIAL = 10000
BATCH_SIZE = 32
engine.run(batch_generator(buffer, REPLAY_INITIAL, BATCH_SIZE))

