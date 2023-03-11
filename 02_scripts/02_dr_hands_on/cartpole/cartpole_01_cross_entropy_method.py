
import gym
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
    obs = env.reset()
    # obs = env.reset()[0]
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
        next_obs, reward, is_done, _ = env.step(action)
        # next_obs, reward, is_done, _, _ = env.step(action)
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
            next_obs = env.reset()
            # next_obs = env.reset()[0]
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
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# CartPole environment
# ----------------------------------------------------------------------------------------------------------------

env = gym.make("CartPole-v0")
# env = gym.make("CartPole-v1")
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
# Cross-Entropy method on CartPole 
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
#   - Every step of the environment gives us the reward 1.0, until the moment that the pole falls.
#     So, the longer our agen balanced the pole, the more reward it obtained.
#     Due to randomness in our agent's behavior, different episodes were of different lengths,
#     which gave us a pretty normal distribution of the epsodes' rewards.
# ----------------------------------------------------------------------------------------------------------------

HIDDEN_SIZE = 128
net = Net(obs_size, HIDDEN_SIZE, n_actions)

print(net)


# ----------
optimizer = optim.Adam(params=net.parameters(), lr=0.01)

writer = SummaryWriter(comment="-cartpole")


# ----------
BATCH_SIZE = 16

# The percentile of episodes' total rewards that we use for "elite" episode filtering.
# We take the 70th percentile, which means that we will leave the top 30% of episodes sorted by reward.
PERCENTILE = 70

for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
    # print(f'{iter_no}  :  {batch}')

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
    if reward_m > 199:
        print("Solved!")
        break

# ----------
writer.close()
