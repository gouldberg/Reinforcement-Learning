import gym
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


##############################################################################################
# --------------------------------------------------------------------------------------------
# Net
# --------------------------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())


# --------------------------------------------------------------------------------------------
# unpack_batch
# --------------------------------------------------------------------------------------------

@torch.no_grad()
def unpack_batch(batch, net, gamma):
    states = []
    actions = []
    rewards = []
    done_masks = []
    last_states = []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        done_masks.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)

    states_v = torch.tensor(states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    last_states_v = torch.tensor(last_states)

    # ----------
    # use last state
    last_state_q_v = net(last_states_v)
    best_last_q_v = torch.max(last_state_q_v, dim=1)[0]
    best_last_q_v[done_masks] = 0.0
    # ----------

    return states_v, actions_v, best_last_q_v * gamma + rewards_v


##############################################################################################
# --------------------------------------------------------------------------------------------
# CartPole Agent Learning:  DQN
#   - During training we need to access our NN to process 2 batches of states:
#      one for the current state and another for the next state in the Belman update.
# --------------------------------------------------------------------------------------------

# ----------
# environment
env = gym.make("CartPole-v0")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n


# ----------
# main net and target net
HIDDEN_SIZE = 128

net = Net(obs_size, HIDDEN_SIZE, n_actions)

tgt_net = ptan.agent.TargetNet(net)


# ----------
# action selector and agent
selector = ptan.actions.ArgmaxActionSelector()

epsilon_start = 1
selector = ptan.actions.EpsilonGreedyActionSelector(
    epsilon=epsilon_start, selector=selector)

agent = ptan.agent.DQNAgent(net, selector)


# ----------
# experience source and replay buffer
GAMMA = 0.9
exp_source = ptan.experience.ExperienceSourceFirstLast(
    env, agent, gamma=GAMMA)

# Training samples in a simple episode are usually heavily correlated, which is bad for SGD training.
# In the case of DQN, we solve this issue by having a large replay buffer with 100k-1M observations
# what we sampled our training batch from.
REPLAY_SIZE = 1000
buffer = ptan.experience.ExperienceReplayBuffer(
    exp_source, buffer_size=REPLAY_SIZE)


# ----------
# training 
LR = 1e-3
optimizer = optim.Adam(net.parameters(), LR)

step = 0
episode = 0
solved = False

BATCH_SIZE = 16

# ask the target network to sync every TGT_NET_SYNC iterations
TGT_NET_SYNC = 10

# decay epsilon to zero at training step 500
EPS_DECAY=0.99

epsilon_cur = epsilon_start
for i in range(500):
    epsilon_cur *= EPS_DECAY
    print(f'{i} : {epsilon_cur}')


while True:
    # ----------
    step += 1

    # In the beginning of every training loop iteration,
    # we ask the buffer to fetch one sample from the experience source and
    # then check for the finished episode.
    buffer.populate(1)

    # pop_rewards_steps() returns the list of tuples with information about episodes
    # completed since the last call to the method.
    for reward, steps in exp_source.pop_rewards_steps():
        episode += 1
        print("%d: episode %d done, reward=%.3f, epsilon=%.2f" % (
            step, episode, reward, selector.epsilon))
        solved = reward > 195
    if solved:
        print("Congrats!")
        break

    if len(buffer) < 2 * BATCH_SIZE:
        continue

    # ----------
    # sample training batches
    batch = buffer.sample(BATCH_SIZE)

    # 1. unpack_batch access to our NN for next state
    states_v, actions_v, tgt_q_v = unpack_batch(
        batch, tgt_net.target_model, GAMMA)

    # ----------
    optimizer.zero_grad()

    # 2. here access to our NN for current state
    q_v = net(states_v)
    q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    loss_v = F.mse_loss(q_v, tgt_q_v)
    loss_v.backward()
    optimizer.step()
    selector.epsilon *= EPS_DECAY

    # ask the target network to sync every TGT_NET_SYNC iterations
    if step % TGT_NET_SYNC == 0:
        tgt_net.sync()


##############################################################################################
# --------------------------------------------------------------------------------------------
# Check
# --------------------------------------------------------------------------------------------

# experience source:  ptan.experience.ExperienceSourceFirstLast:

GAMMA = 0.9
exp_source = ptan.experience.ExperienceSourceFirstLast(
    env, agent, gamma=GAMMA)

count = 0
for reward, steps in exp_source.pop_rewards_steps():
    count += 1
    if count <= 1000:
        print(f'count:  {count}')
        print(f'reward:  {reward}')
        print(f'steps:  {steps}')
    else:
        break


