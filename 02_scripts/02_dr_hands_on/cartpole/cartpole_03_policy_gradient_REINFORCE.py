import os
import gym
import ptan
import numpy as np
from typing import Optional

from tensorboardX import SummaryWriter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


##############################################################################################
# --------------------------------------------------------------------------------------------
# PolicyGradientNet
# --------------------------------------------------------------------------------------------

# Note that despite the fact our network returns probabilities, 
# we are not applying softmax nonlinearity to the output. (raw scores)
# We will later use PyTorch log_softmax function to calculate the logarithm of the softmax output at once.
# This method of calculation is much more numerically stable.

class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)



# ----------
# for comparison:  this is for DQN --> almost same...
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        # return as float
        return self.net(x.float())


# --------------------------------------------------------------------------------------------
# calc_qvals
# --------------------------------------------------------------------------------------------

# sum_r contains the total reward for the previous steps,
# so to get the total reward for the previous step,
# we need to mltiply sum_r by gamma and sum the local reward.

def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


##############################################################################################
# --------------------------------------------------------------------------------------------
# CartPole Agent Learning (REINFORCE method)
#   REINFORCE:  REward Increment = Nonnegative Factor * Offset Reinforcement * Characteristic Eligibility
#   - for REINFORCE method, use Q(state, action) for training instead of just 0 and 1,
#     (use Q to scale gradient)
#   - More fine-grained separation of episode
#       - Transitions of the episode with the total reward of 10 should contribute to the gradient
#         more than transitions from the episode with the reward of 1.
#   - Increase probabilities of good actions in the beginning of the episode and
#     decrease the actions closer to the end of the episode
#     (as Q incorporates the discount factor, uncertainty for longer sequences of actions
#      is automatically taken into account)
#
#   - No explicit exploration is needed. Now, with probabilities returned by the network,
#     the exploration is performed automatically.
#   - No replay buffer is used. We cannot train on data obtained from the old policy.
#     Faster convergence, but require much more inteaction with the environment than off-policy method such as DQN.
#     (less sample efficient, but no need replay buffer)
#   - No target network is needed.
#     In DQN, target network is used to break the correlation in Q-values approximation,
#     but we are not approximating anymore.
#
#   - The REINFORCE method should be able to solve CartPole in 300 - 500 episodes
#
# --------------------------------------------------------------------------------------------

# ----------
# environment
env = gym.make("CartPole-v0")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n


# ----------
# Policy Gradient Net

net = PGN(env.observation_space.shape[0], env.action_space.n)
print(net)


# ----------
# agent
# As our network returns the policy as the probabilities of the actions,
# in order to select the action to take, we need to obtain the probabilities from the network
# and then perform random sampling from this probability distribution.

# PolicyAgent internally calls the NumPy random.choice function with probabilities from the network.
# apply_softmax instructs it to convert the network output to probabilities by calling softmax first.
# preprocessor is a way to get around the fact that the CartPole environment in Gym returns
# the observation as float64 instead of the float32 required by PyTorch.
agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                apply_softmax=True)


# ----------
# experience source
GAMMA = 0.9
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)


# ----------
# training 
LEARNING_RATE = 0.01
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


# ----------
total_rewards = []
step_idx = 0
done_episodes = 0

# how many complete episodes we will use for training
EPISODES_TO_TRAIN = 4

batch_episodes = 0

# batch_qvals is used for scaling policy gradient.
# for REINFORCE method, use Q(state, action) for training instead of just 0 and 1.
batch_states, batch_actions, batch_qvals = [], [], []

# list of local rewards for the episode being currently played.
cur_rewards = []

writer = SummaryWriter(comment="-cartpole-reinforce")

for step_idx, exp in enumerate(exp_source):

    # we just save the state, action, and local reward in our lists
    batch_states.append(exp.state)
    batch_actions.append(int(exp.action))
    cur_rewards.append(exp.reward)

    # At the episode reaches the end
    if exp.last_state is None:
        # calculate the discounted total rewards from local rewards
        # using calc_qvals function and append them to the batch_qvals list.
        batch_qvals.extend(calc_qvals(cur_rewards))
        cur_rewards.clear()
        batch_episodes += 1

    # ----------
    # This part is performed at the end of the episode and
    # is responsible for reporting the current progress and writing metrics to TensorBoard.
    new_rewards = exp_source.pop_total_rewards()
    if new_rewards:
        done_episodes += 1
        reward = new_rewards[0]
        total_rewards.append(reward)
        mean_rewards = float(np.mean(total_rewards[-100:]))
        print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
            step_idx, reward, mean_rewards, done_episodes))
        writer.add_scalar("reward", reward, step_idx)
        writer.add_scalar("reward_100", mean_rewards, step_idx)
        writer.add_scalar("episodes", done_episodes, step_idx)

        if mean_rewards > 195:
            print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
            break

    if batch_episodes < EPISODES_TO_TRAIN:
        continue

    # ----------
    # When enough episodes have passed since the last training step,
    # we perform optimization on the gathered examples.
    optimizer.zero_grad()
    states_v = torch.FloatTensor(batch_states)
    batch_actions_t = torch.LongTensor(batch_actions)
    batch_qvals_v = torch.FloatTensor(batch_qvals)

    # -----------
    logits_v = net(states_v)
    log_prob_v = F.log_softmax(logits_v, dim=1)

    # ----------
    # Policy gradient itself is equal to the gradient of log probability of the action taken.
    # This means that we are trying to increase the probability of actions
    # that have given us good total reward and decrease the probability of actions with bad final outcomes.

    # we select log probabilities from the actions taken and scale them with Q-values
    log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]

    # we want to maximize expectation of our policy gradient = log_prob_actions_v.mean()
    # we average those scaled values and do negation to obtain the loss to minimize
    loss_v = -log_prob_actions_v.mean()

    loss_v.backward()
    optimizer.step()

    batch_episodes = 0
    batch_states.clear()
    batch_actions.clear()
    batch_qvals.clear()

writer.close()


##############################################################################################
# --------------------------------------------------------------------------------------------
# Check
# --------------------------------------------------------------------------------------------

GAMMA = 0.9
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

count = 0
for step_idx, exp in enumerate(exp_source):
    count += 1
    if count <= 1000 or exp_source.pop_total_rewards():
        print(f'count:  {count}')
        print(f'state:  {exp.state}')
        print(f'action:  {exp.action}')
        print(f'reward:  {exp.reward}')
        print(f'last state:  {exp.last_state}')
    else:
        break


