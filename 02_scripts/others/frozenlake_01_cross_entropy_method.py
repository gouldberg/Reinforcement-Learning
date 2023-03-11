
import gym, gym.spaces
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# NN 
# ----------------------------------------------------------------------------------------------------------------

# We do not apply softmax to increase the numerical stability of the training process.
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# ----------
# Rather than calculating softmax (which uses exponentiation) and then calculating cross-entropy loss (which uses a logarithm of probabilities),
# we can use the PyTorch class nn.CrossEntropyLoss which combines both softmax and cross-entropy in a single, more numerically stable expression.
# CrossEntropyLoss requires raw, unnormalized values from the NN (also called logits).

objective = nn.CrossEntropyLoss()


# ----------------------------------------------------------------------------------------------------------------
# DiscreteOneHotWrapper
# ----------------------------------------------------------------------------------------------------------------

class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space,
                          gym.spaces.Discrete)
        shape = (env.observation_space.n, )
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape, dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


# ----------------------------------------------------------------------------------------------------------------
# yield batch 
# ----------------------------------------------------------------------------------------------------------------

Episode = namedtuple('Episode', field_names=['reward', 'steps'])

# Used to represent one single step that our agent made in the episode, and it stores the observation from the environment
# and what action the agent completed.
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    # ----------
    # obs = env.reset()
    obs = env.reset()[0]
    # ----------
    sm = nn.Softmax(dim=1)

    # ----------
    while True:
        obs_v = torch.FloatTensor([obs])
        # softmax
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)

        # ----------
        # next_obs, reward, is_done, _ = env.step(action)
        next_obs, reward, is_done, _, _ = env.step(action)
        # ----------

        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            # ----------
            # next_obs = env.reset()
            next_obs = env.reset()[0]
            # ----------
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


# ----------------------------------------------------------------------------------------------------------------
# episode filtering 
# ----------------------------------------------------------------------------------------------------------------

# This function is at the core of the cross-entropy method.
# From the given batch of episodes and percentile value, it calculates a boundary reward, which is used to filter "elite" episodes to train on.

def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))

    # boundary reward
    reward_bound = np.percentile(rewards, percentile)
    # mean reward for monitoring purpose
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# FozenLake environment
# ----------------------------------------------------------------------------------------------------------------

env = DiscreteOneHotWrapper(gym.make("FrozenLake-v1"))
# env = gym.wrappers.Monitor(env, directory="mon", force=True)

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

print(f'observation space: {obs_size}')
print(f'n_actions: {n_actions}')


# ----------
obs = env.reset()

# 4 values
print(obs)
print(obs[0])
print(obs[1])


# ----------------------------------------------------------------------------------------------------------------
# Cross-Entropy method on FrozenLake
# ----------------------------------------------------------------------------------------------------------------
#   - model-free, policy-based, and on-policy
#     - It does not build any model of the environment, it just says to the agen what to do at every step
#     - It approximates the policy of the agent:  policy as probability distribution over actions
#     - It requires fresh data obtained from the environment
# ----------------------------------------------------------------------------------------------------------------
#   - the steps of the method
#     - 1. Play N number of episodes using our current model and environement
#     - 2. Calculate the total reward for every episode and decide on a reward boundary.
#          Usually, we use some percentile of all rewards, such as 50th or 70th.
#     - 3. Throw away all episodes with a reward below the boundary.
#     - 4. Train on the remaining "elite" episodes using observations as the input and issued actions as the desired output.
#     - 5. Repeat from step 1 until we become satisfied with the result.
# ----------------------------------------------------------------------------------------------------------------
# Naive method fails:
#   - We get the reward of 1.0 only when we reach the goal, and this reward says nothing about how good each episode was.
#     The distribution of rewards for our episodes are also problematic.
#     There are only two kinds of episodes possible, with zero reward (failed) and one reward (successful),
#     and failed episodes will obviously dominate in the beginning of the training.
#     So our percentile selection of "elite" episodes is totally wrong.
# ----------------------------------------------------------------------------------------------------------------

HIDDEN_SIZE = 128
net = Net(obs_size, HIDDEN_SIZE, n_actions)

print(net)


# ----------
optimizer = optim.Adam(params=net.parameters(), lr=0.01)

writer = SummaryWriter(comment="-frozenlake-naive")


# ----------
BATCH_SIZE = 16

# The percentile of episodes' total rewards that we use for "elite" episode filtering.
# We take the 70th percentile, which means that we will leave the top 30% of episodes sorted by reward.
PERCENTILE = 70

for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):

    # ----------
    # filter "elite" episodes
    obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)

    # ----------
    optimizer.zero_grad()
    action_scores_v = net(obs_v)
    loss_v = objective(action_scores_v, acts_v)
    loss_v.backward()
    optimizer.step()

    print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
        iter_no, loss_v.item(), reward_m, reward_b))

    # ----------
    writer.add_scalar("loss", loss_v.item(), iter_no)
    writer.add_scalar("reward_bound", reward_b, iter_no)
    writer.add_scalar("reward_mean", reward_m, iter_no)
    if reward_m > 0.8:
        print("Solved!")
        break

# ----------
writer.close()



# ----------------------------------------------------------------------------------------------------------------
# Tweaked:
#   - 1. Larger batches of played episodes
#   - 2. Discount factor applied to the reward
#   - 3. Keeping "elite" episodes for a longer time
#   - 4. Decreasing learning rate
#   - 5. Much longer training time
# -->
# The training of the model stopped improving at around 55% of solved episodes.
# ----------------------------------------------------------------------------------------------------------------

# ----------
# slippery version
env = DiscreteOneHotWrapper(gym.make("FrozenLake-v1"))
# env = gym.wrappers.Monitor(env, directory="mon", force=True)

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

print(f'observation space: {obs_size}')
print(f'n_actions: {n_actions}')


# ----------
# non-slippery version:  this can be solved in 120140 batch iteration ...
# env = gym.envs.toy_text.frozen_lake.FrozenLakeEnv(is_slippery=False)
# env.spec = gym.spec("FrozenLake-v1")
# env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
# env = DiscreteOneHotWrapper(env)


GAMMA = 0.9

def filter_batch_rev(batch, percentile):
    # ----------
    # Discount factor applied to the reward
    filter_fun = lambda s: s.reward * (GAMMA ** len(s.steps))
    disc_rewards = list(map(filter_fun, batch))

    # ----------
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation,
                                 example.steps))
            train_act.extend(map(lambda step: step.action,
                                 example.steps))
            elite_batch.append(example)

    return elite_batch, train_obs, train_act, reward_bound


# ----------
HIDDEN_SIZE = 128
net = Net(obs_size, HIDDEN_SIZE, n_actions)

print(net)


# ----------
# decreasing learning rate 0.01 to 0.001
lr = 0.001
optimizer = optim.Adam(params=net.parameters(), lr=lr)

writer = SummaryWriter(comment="-frozenlake-tweaked")


# ----------
# larger batches of played episodes.
BATCH_SIZE = 100

PERCENTILE = 30

full_batch = []

for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):

    # keeping "elite" episodes for a longer time:
    # store previous "elite" episodes to pass them to the preceding function
    # on the next training iteration.
    reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
    full_batch, obs, acts, reward_bound = filter_batch_rev(full_batch + batch, PERCENTILE)
    if not full_batch:
        continue
    obs_v = torch.FloatTensor(obs)
    acts_v = torch.LongTensor(acts)
    full_batch = full_batch[-500:]

    # ----------
    optimizer.zero_grad()
    action_scores_v = net(obs_v)
    loss_v = objective(action_scores_v, acts_v)
    loss_v.backward()
    optimizer.step()

    print("%d: loss=%.3f, rw_mean=%.3f, ""rw_bound=%.3f, batch=%d" % (
        iter_no, loss_v.item(), reward_mean, reward_bound, len(full_batch)))

    # ----------
    writer.add_scalar("loss", loss_v.item(), iter_no)
    writer.add_scalar("reward_mean", reward_mean, iter_no)
    writer.add_scalar("reward_bound", reward_bound, iter_no)
    if reward_mean > 0.8:
        print("Solved!")
        break

# ----------
writer.close()
