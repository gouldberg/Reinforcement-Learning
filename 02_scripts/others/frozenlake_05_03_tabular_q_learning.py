
import gym
import collections

from tensorboardX import SummaryWriter


#################################################################################################
# -----------------------------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------------------------

GAMMA = 0.9
ALPHA = 0.2

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        # self.state = self.env.reset()
        self.state = self.env.reset()[0]
        # ----------
        # We do not need to track the history of rewards and transition counters,
        # just our value table.
        # This will make memory footprint smaller.
        self.values = collections.defaultdict(float)

    # sample only one step from the environment
    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        # new_state, reward, is_done, _ = self.env.step(action)
        new_state, reward, is_done, _, _ = self.env.step(action)
        # self.state = self.env.reset() if is_done else new_state
        self.state = self.env.reset()[0] if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    # Here we update value table using one step from the environment
    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v
        old_v = self.values[(s, a)]
        # ----------
        # update Q with approzimations using blending (alpha) technique.
        # this technique allows values of Q to converge smoothly, even if environment is noisy.
        self.values[(s, a)] = old_v * (1-ALPHA) + new_v * ALPHA

    # We don't tourch Q-values during the test,
    # which causes more iterations before the environemnt gets solved.
    def play_episode(self, env):
        total_reward = 0.0
        # state = env.reset()
        state = env.reset()[0]
        while True:
            _, action = self.best_value_and_action(state)
            # new_state, reward, is_done, _ = env.step(action)
            new_state, reward, is_done, _, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


#################################################################################################
# -----------------------------------------------------------------------------------------------
# tabular Q-learning
#  - Need more iterations to solve the problem compared to the value iteration method,
#    since we are no longer using the experience obtained during training.
#    Total number of samples required from the environment is almost the same.
# -----------------------------------------------------------------------------------------------

# ENV_NAME = "FrozenLake-v0"
ENV_NAME = "FrozenLake-v1"

# ENV_NAME = "FrozenLake8x8-v0"      # uncomment for larger version
# ENV_NAME = "FrozenLake8x8-v1"      # uncomment for larger version


# ----------
# test to play TEST_EPISODES
TEST_EPISODES = 20


# ----------
test_env = gym.make(ENV_NAME)
agent = Agent()
writer = SummaryWriter(comment="-q-learning")

iter_no = 0
best_reward = 0.0

while True:
    iter_no += 1

    s, a, r, next_s = agent.sample_env()
    agent.value_update(s, a, r, next_s)
    # print(f'----------------------------------')
    # print(f'iter_no: {iter_no}')
    # print(f'{agent.values}')

    # ----------
    # test: play several full episodes to check our improvements using the updated value table.
    reward = 0.0
    for _ in range(TEST_EPISODES):
        reward += agent.play_episode(test_env)
    reward /= TEST_EPISODES

    writer.add_scalar("reward", reward, iter_no)

    if reward > best_reward:
        print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
        best_reward = reward

    if reward > 0.80:
        print("Solved in %d iterations!" % iter_no)
        break

# ----------
writer.close()

