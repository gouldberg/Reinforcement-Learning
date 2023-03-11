
# import argparse
import os
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

import cv2
import gym
import gym.spaces


from tensorboardX import SummaryWriter


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# wrappers
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
# https://alexandervandekleut.github.io/gym-wrappers/
# -----------------------------------------------------------------------------------------------------------

# ----------
# Presses the FIRE button in environements that require that for the game to start.
# In addition to pressing FIRE, this wrapper checks for several corner cases that are
# present in some games.

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


# ----------
# Combines the repetition of actions during K frames and pixels from two consecutive frames.

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


# ----------
# Convert input observations from the emulator, which normally has a resolution of 210 * 160 pixels
# with RGB color channels, to a grayscale 84 * 84 iamge.
# It does this using a colorimetric grayscale conversion (which is closer to human color perception)
# than a simple averaging of color channels), resizing the image, and cropping the top and bottom
# parts of the result.

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + \
              img[:, :, 2] * 0.114
        resized_screen = cv2.resize(
            img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


# ----------
# Changes the shape of the observation from HWC (height, width, channel)
# to the CHW (channel, height, width) format required by PyTorch.

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


# ----------
# Converts observation data from bytes to floats, and scales every pixel's value to the range [0, 1.0].

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


# ----------
# Creats a stack of subsequent frames along the 1st dimension and returns them as an observation.
# The purpose is to give the network an idea about the dynamics of the objects,
# such as the speed and direction of the ball in Pong or how enemies are moving.
# This is very important information, which it is not possible to obtain from a single image.

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(
            self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Experience Buffer
# -----------------------------------------------------------------------------------------------------------

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


# -----------------------------------------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------------------------------------

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # 1 in gather(1, ..) corresponds to 1st dimension (= actions)
    # Keep in mind that the result of gather() applied to tensor is a differentiable operation
    # that will keep all gradients with respect to the final loss value.
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        # we apply the target network to our next state observations and 
        # calculate the maximum Q-value along the same action dimension, 1.
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        # detach() is used to prevent gradients from flowing into the target network's graph
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v

    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Deep Q-Learning model
# -----------------------------------------------------------------------------------------------------------

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
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Deep Q-Learning
# -----------------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/reinforcement_learning'

device = torch.device("cuda")


# ----------
# environment
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
env_name = DEFAULT_ENV_NAME

env = make_env(env_name)


# ----------
# Deep Q-Learning model
# net calculate gradients
net = DQN(env.observation_space.shape, env.action_space.n).to(device)

# calcualte values for the next states (this should not affect gradients)
tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)

print(net)


# ----------
# maximum capacity of the buffer
REPLAY_SIZE = 10000

# counts of frames we wait for before starting training to populate the replay buffer
REPLAY_START_SIZE = 10000

buffer = ExperienceBuffer(REPLAY_SIZE)

agent = Agent(env, buffer)


# ----------
# optimizer
LEARNING_RATE = 1e-4
# LEARNING_RATE = 1e-3
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


# ----------
# We start with epsilon = 1.0 at the early stages of training,
# which causes all actions to be selected randomly.
# Then, during the first 150,000 frames epsilon is linearly decayed to 0.01,
# which corresponds to the random action taken in 1% of steps.
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
epsilon = EPSILON_START
EPSILON_DECAY_LAST_FRAME = 150000


MEAN_REWARD_BOUND = 19

# gamma value for Bellman approzimation
GAMMA = 0.99

# batch size sampled from the replay buffer
BATCH_SIZE = 32

# How frequently we sync model weights from the training model
# to the target model, which is used to get the value of the next state
# in the Bellman approximation
SYNC_TARGET_FRAMES = 1000


# ----------
# training
total_rewards = []
frame_idx = 0
ts_frame = 0
ts = time.time()
best_m_reward = None

writer = SummaryWriter(comment="-" + env_name)

while True:
    frame_idx += 1
    epsilon = max(EPSILON_FINAL, EPSILON_START -
                    frame_idx / EPSILON_DECAY_LAST_FRAME)

    reward = agent.play_step(net, epsilon, device=device)

    if reward is not None:
        total_rewards.append(reward)
        speed = (frame_idx - ts_frame) / (time.time() - ts)
        ts_frame = frame_idx
        ts = time.time()
        # ----------
        # mean reward for the last 100 episodes
        m_reward = np.mean(total_rewards[-100:])
        # ----------
        print("%d: done %d games, reward %.3f, "
                "eps %.2f, speed %.2f f/s" % (
            frame_idx, len(total_rewards), m_reward, epsilon,
            speed
        ))
        # ----------
        writer.add_scalar("epsilon", epsilon, frame_idx)
        writer.add_scalar("speed", speed, frame_idx)
        writer.add_scalar("reward_100", m_reward, frame_idx)
        writer.add_scalar("reward", reward, frame_idx)
        # ----------
        if best_m_reward is None or best_m_reward < m_reward:
            torch.save(net.state_dict(),  os.path.join(base_path, "04_output/pong/deep_q_learning", env_name + "-best_%.0f.dat" % m_reward))
            if best_m_reward is not None:
                print("Best reward updated %.3f -> %.3f" % (best_m_reward, m_reward))
            best_m_reward = m_reward
        # ----------
        if m_reward > MEAN_REWARD_BOUND:
            print("Solved in %d frames!" % frame_idx)
            break

    if len(buffer) < REPLAY_START_SIZE:
        continue

    if frame_idx % SYNC_TARGET_FRAMES == 0:
        tgt_net.load_state_dict(net.state_dict())

    optimizer.zero_grad()
    batch = buffer.sample(BATCH_SIZE)
    loss_t = calc_loss(batch, net, tgt_net, device=device)
    loss_t.backward()
    optimizer.step()


writer.close()



#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Visualize learned play
# -----------------------------------------------------------------------------------------------------------

from gym.wrappers.monitoring import video_recorder

model_path = os.path.join(base_path, '04_output/pong/deep_q_learning/model/PongNoFrameskip-v4-best_19.dat')

# record_path = os.path.join(base_path, '04_output/pong/deep_q_learning/video/PongNoFrameskip-v4-best_19')
record_path = os.path.join(base_path, '04_output/pong/deep_q_learning/video/PongNoFrameskip-v4-best_19_video.mp4')


# ----------
# environment
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
env_name = DEFAULT_ENV_NAME

env = make_env(env_name)
# env = gym.wrappers.Monitor(env, record_path)

vid = video_recorder.VideoRecorder(env, path=record_path)


# ----------
# model

net = DQN(env.observation_space.shape, env.action_space.n)

state = torch.load(model_path, map_location=lambda stg, _: stg)

net.load_state_dict(state)


# ----------
state = env.reset()


# ----------
total_reward = 0.0

c = collections.Counter()

FPS = 25

while True:
    start_ts = time.time()

    # ----------
    env.render()
    vid.capture_frame()
    # ----------
    state_v = torch.tensor(np.array([state], copy=False))
    q_vals = net(state_v).data.numpy()[0]
    action = np.argmax(q_vals)
    c[action] += 1
    state, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        break
    delta = 1/FPS - (time.time() - start_ts)
    if delta > 0:
        time.sleep(delta)

print("Total reward: %.2f" % total_reward)
print("Action counts:", c)

env.env.close()

