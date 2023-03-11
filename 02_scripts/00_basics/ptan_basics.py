import ptan
import numpy as np


#############################################################################################
# -------------------------------------------------------------------------------------------
# Action Selector
#  - Object that helps with going from network output to concrete action values
#  - Action selector is used by the Agent
#  - The most common cases:
#     - Argmax: commonly used by Q-value methods when the network predicts Q-values for
#               a set of actions and the desired action is the action with the largest Q(s, a)
#     - Policy-based: the network outputs the probability distribution (in the form of logits
#               or normalized distribution), and an action needs to be sampled from this
#               distribution.
# -------------------------------------------------------------------------------------------

q_vals = np.array([[1, 2, 3], [1, -1, 0]])

print(q_vals)


# ----------
# ArgmaxActionSelector:
#  - applies argmax on the second axis of a passed tensor.
#    It assumes a matrix with batch dimension along the 1st axis.

selector = ptan.actions.ArgmaxActionSelector()

# The selector returns indices of actions with the largest values
print(f'argmax: {selector(q_vals)}')


# ----------
# EpsilonGreedyActionSelector:
#  - has the parameter epsilon, which specifies the probability of a random action to be taken.

# epsilon = 0.0 means no random actions are taken.
# If we change epsilon to 1, actions will be random.

selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.0)

print(f'epsilon=0.0: {selector(q_vals)}')

selector.epsilon = 1.0
print(f'epsilon=1.0: {selector(q_vals)}')

selector.epsilon = 0.5
for _ in range(10):
    print(f'epsilon=0.5: {selector(q_vals)}')

selector.epsilon = 0.3
for _ in range(20):
    print(f'epsilon=0.3: {selector(q_vals)}')


# ----------
# ProbabilityActionSelector:
#  - samples from the probability distribution of a discrete set of actions

selector = ptan.actions.ProbabilityActionSelector()

print("Actions sampled from three prob distributions:")

for _ in range(10):
    acts = selector(np.array([
        [0.1, 0.8, 0.1],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0]
    ]))
    print(acts)


#############################################################################################
# -------------------------------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

class DQNNet(nn.Module):
    def __init__(self, actions: int):
        super(DQNNet, self).__init__()
        self.actions = actions

    def forward(self, x):
        # we always produce diagonal tensor of shape (batch_size, actions)
        return torch.eye(x.size()[0], self.actions)


class PolicyNet(nn.Module):
    def __init__(self, actions: int):
        super(PolicyNet, self).__init__()
        self.actions = actions

    def forward(self, x):
        # Now we produce the tensor with first two actions
        # having the same logit scores
        shape = (x.size()[0], self.actions)
        res = torch.zeros(shape, dtype=torch.float32)
        res[:, 0] = 1
        res[:, 1] = 1
        return res


# ----------
# DQN Agent:
#   - This class is applicable in Q-learning when the action space is not very large,
#     which covers Atari games and lots of classical problems.
#   - DQN Agent takes a batch of observations on input, applies the network on them to get Q-values,
#     and then uses the provided ActionSelector to convert Q-values to indices of actions.

net = DQNNet(actions=3)

net_out = net(torch.zeros(2, 10))

print(net_out)


# DQN Agent with ArgmaxActionSelector
selector = ptan.actions.ArgmaxActionSelector()
agent = ptan.agent.DQNAgent(dqn_model=net, action_selector=selector)
ag_out = agent(torch.zeros(2, 5))
print("Argmax:", ag_out)


# DQN Agent with EpsilonGreedyActionSelector
selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
agent = ptan.agent.DQNAgent(dqn_model=net, action_selector=selector)

ag_out = agent(torch.zeros(10, 5))[0]
print("eps=1.0:", ag_out)

selector.epsilon = 0.5
ag_out = agent(torch.zeros(10, 5))[0]
print("eps=0.5:", ag_out)

selector.epsilon = 0.1
ag_out = agent(torch.zeros(10, 5))[0]
print("eps=0.1:", ag_out)


# ----------
# Policy Agent:
#   - Policy Agent expects the network to produce policy distribution over a discrete set of actions.
#     Policy distribution could be either logits (unnormalized) or a normalized distribution.
#     In practice, you should always use logits to improve the numeric stability of the training process.

net = PolicyNet(actions=5)

net_out = net(torch.zeros(6, 10))

print("policy_net:")
print(net_out)


# We can use PolicyAgent in combination with ProbabilityActionSelector.
# As the latter expects normalized probabilities, we need to ask PolicyAgent
# to apply softmax to the network's output.

selector = ptan.actions.ProbabilityActionSelector()
agent = ptan.agent.PolicyAgent(model=net, action_selector=selector, apply_softmax=True)

ag_out = agent(torch.zeros(6, 5))[0]
print(ag_out)


# note that the softmax operation produces non-zero probabilities for zero logits,
# so our agent can still select actions > 1.

torch.nn.functional.softmax(net(torch.zeros(1, 10)), dim=1)


#############################################################################################
# -------------------------------------------------------------------------------------------
# Experience Source
# -------------------------------------------------------------------------------------------

import gym
from typing import List, Optional, Tuple, Any


class ToyEnv(gym.Env):
    """
    Environment with observation 0..4 and actions 0..2
    Observations are rotated sequentialy mod 5, reward is equal to given action.
    Episodes are having fixed length of 10
    """

    def __init__(self):
        super(ToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=5)
        self.action_space = gym.spaces.Discrete(n=3)
        self.step_index = 0

    def reset(self):
        self.step_index = 0
        return self.step_index

    def step(self, action):
        is_done = self.step_index == 10
        if is_done:
            return self.step_index % self.observation_space.n, \
                   0.0, is_done, {}
        self.step_index += 1
        return self.step_index % self.observation_space.n, \
               float(action), self.step_index == 10, {}


class DullAgent(ptan.agent.BaseAgent):
    """
    Agent always returns the fixed action
    """
    def __init__(self, action: int):
        self.action = action

    def __call__(self, observations: List[Any],
                 state: Optional[List] = None) \
            -> Tuple[List[int], Optional[List]]:
        return [self.action for _ in observations], state


# ----------
# check toy environment (gym env): reset and step
env = ToyEnv()

# reset
s = env.reset()
print("env.reset() -> %s" % s)

# step
s = env.step(1)
print("env.step(1) -> %s" % str(s))

s = env.step(2)
print("env.step(2) -> %s" % str(s))

for _ in range(10):
    r = env.step(0)
    print(r)


# ----------
# check agent (ptan.agent.BaseAgent)
# This agent always generates fixed actions regardless of observations

action = 1
agent = DullAgent(action=action)

observations = 1
state = 2
print("agent:", agent([observations, state]))


###################################
# ---------------------------------
# ptan.experience.ExperienceSource
#  - generates chunks of agent full subtrajectories of the given length.
#  - return namedtuple with (state, action, reward, done)
#  - the implementation automatically handles the end of episode situation
#    (when the step() method in the environment returns is_done = True) and resets the environment.
#  - The class instance provides the standard Python iterator interface, so you can just iterate
#    over this to get subtrajectories.

env = ToyEnv()

agent = DullAgent(action=1)

# the length of subtrajectories to be generated.
steps_count = 2
exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=steps_count)

# On every iteration, ExperienceSource returns
# a piece of the agent's trajectory in environment communication. 
# We asked for 2-step subtrajectories, so tuples will be of length 2 or 1 (at the end of episodes)
# If the episode reaches the end, the subtrajectory will be shorter and the underlying
# environment will be reset automatically.
for idx, exp in enumerate(exp_source):
    if idx > 3:
        break
    print(f'idx: {idx}  len(exp):  {len(exp)}')
    print(exp)


steps_count = 4
exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=steps_count)
print(next(iter(exp_source)))


# ----------
# We can pass it several instances of gym.Env.
steps_count = 2
exp_source = ptan.experience.ExperienceSource(env=[ToyEnv(), ToyEnv()], agent=agent, steps_count=steps_count)
for idx, exp in enumerate(exp_source):
    if idx > 3:
        break
    # len(exp) is 4 = 2 steps * 2 instances
    print(f'idx: {idx}  len(exp):  {len(exp)}')
    print(exp)


###################################
# ---------------------------------
# ptan.experience.ExperienceSourceFirstLast
#  - The class ExperienceSource provides us with full subtrajectories of the given length
#    as the list of (state, action, reward) objects. The next state is returned in the next tuple,
#    which is not always convenient.
#    For example, in DQN training, we want to have tuples (state, action, reward, next state) at once
#    to do one-step Bellman approximation during the training.
#    In addition, some extension of DQN, like n-step DQN, might want to collapse longer sequences
#    of observations into (1st state, action, total-reward-for-n-steps, state-after-step-n)
#
#  - return namedtuple with (state, action, reward, last_state)
#  - reward here is the partial accumulated reward for steps_count,
#    so the case of steps_count = 1 is equal to the immediate reward.
#  - last_state:  the state we got after executing the action. If our episode ends, we have None here.
#
#  - This data is much more convenient for DQN training, as we can apply Bellman approximation
#    directly to it.

gamma = 1.0
steps_count = 2
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma, steps_count=steps_count)
for idx, exp in enumerate(exp_source):
    print(f'idx: {idx}')
    print(exp)
    if idx > 10:
        break


#############################################################################################
# -------------------------------------------------------------------------------------------
# Experience Replay Buffers
#  - ExperienceReplayBuffer:  a simple replay buffer of predefined size with uniform sampling
#  - PrioReplayBufferNaive:  a simple, but not very efficient, prioritized replay buffer implementation.
#                            The complexity of sampling is O(n), which might become an issue with
#                            large buffers.
#  - PrioritizedReplayBuffer:  uses segment trees for sampling, the complexity is O(log(n)).
# -------------------------------------------------------------------------------------------

env = ToyEnv()

agent = DullAgent(action=1)

exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=1)


# ----------
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=100)
print(f'len(buffer): {len(buffer)}')

# populate(N) to get samples from the experience source and put them into the buffer.
# buffer.populate(1) to get a fresh sample from the environemnt
buffer.populate(1)
print(f'len(buffer): {len(buffer)}')

# sample(N) to get the batch of N experience objects.
batch = buffer.sample(1)
print(batch)


# -->
# All the rest happens automatically: resetting the environment,
# handling subtrajectories, buffer size maintenance, and so on.

# ----------
for step in range(6):

    buffer.populate(1)
    # if buffer is small enough, do nothing
    if len(buffer) < 5:
        continue

    # ----------
    batch = buffer.sample(4)
    print("Train time, %d batch samples:" % len(batch))
    for s in batch:
        print(s)


#############################################################################################
# -------------------------------------------------------------------------------------------
# TargetNet class
#  - TargetNet is a small but usefule class that allows us to synchronize two NNs of the same architecture,
#    useful to improve training stability.
#  - TargetNet supports two modes of synchronization:
#     1. sync(): weights from the source network are copied into the target network.
#                This is standard way for discrete action space problems
#     2. alpha_sync(): the source network's weights are blended into the target network
#                      with some alpha weight (between 0 and 1)
#                      This is used in continuous control problems, where the transition
#                      between two networks' parameters should be smooth.
# -------------------------------------------------------------------------------------------

class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()
        self.ff = nn.Linear(5, 3)

    def forward(self, x):
        return self.ff(x)


net = DQNNet()
print(net)


# main net and target net
tgt_net = ptan.agent.TargetNet(net)
print("Main net:", net.ff.weight)
print("Target net:", tgt_net.target_model.ff.weight)


# They are independent of each other, however, just having the same architecture.
net.ff.weight.data += 1.0
print("After update")
print("Main net:", net.ff.weight)
print("Target net:", tgt_net.target_model.ff.weight)


# To synchoronize them again, the sync() method can be used.
tgt_net.sync()
print("After sync")
print("Main net:", net.ff.weight)
print("Target net:", tgt_net.target_model.ff.weight)
