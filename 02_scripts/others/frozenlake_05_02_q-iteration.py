
import gym
import collections

from tensorboardX import SummaryWriter


#################################################################################################
# -----------------------------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------------------------

GAMMA = 0.9

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        # self.state = self.env.reset()
        self.state = self.env.reset()[0]

        # --------------------
        # Tables
        # --------------------
        # 1. Reward table:
        #   A dictionary with the composite key "source state" + "action" + "target state".
        #   The value is obtained from the immediate reward.
        self.rewards = collections.defaultdict(float)
        # ----------
        # 2. Transition table:
        #   A dictionary keeping counters of the experienced transitions.
        #   The key is the composite "state" + "action", and the value is another dictionary
        #   that maps the target state into a count of times that we have seen it.
        #   For example, if in state 0 we execute action-1 10 times
        #     - after 3 times it will lead us to state 4
        #     - after 7 times it will lead us to state 5
        #     --> entry key (0,1) and dict value {4: 3, 5: 7}
        #   We can use this table to estimate the probabilities of our transitions   
        self.transits = collections.defaultdict(collections.Counter)
        # ----------
        # 3. Value table:
        #   A dictionary that maps a state into the calculated value of this state.
        self.values = collections.defaultdict(float)

    # ----------
    # gather random experience from the environment and update the reward and transition tables
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            # new_state, reward, is_done, _ = self.env.step(action)
            new_state, reward, is_done, _, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            # self.state = self.env.reset() \
            #     if is_done else new_state
            self.state = self.env.reset()[0] \
                if is_done else new_state

    # ----------
    # THIS PART IS INCORPORATED IN value_iteration

    # def calc_action_value(self, state, action):
    #     target_counts = self.transits[(state, action)]
    #     total = sum(target_counts.values())
    #     action_value = 0.0
    #     for tgt_state, count in target_counts.items():
    #         reward = self.rewards[(state, action, tgt_state)]
    #         val = reward + GAMMA * self.values[tgt_state]
    #         action_value += (count / total) * val
    #     return action_value

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            # ----------
            # action_value = self.calc_action_value(state, action)

            # now action value is from values table
            action_value = self.values[(state, action)]
            # ----------
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    # test by updated value table
    def play_episode(self, env):
        total_reward = 0.0
        # state = env.reset()
        state = env.reset()[0]
        while True:
            action = self.select_action(state)
            # new_state, reward, is_done, _ = env.step(action)
            new_state, reward, is_done, _, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    # ----------
    # def value_iteration(self):
    #     # value iteration loop over all states, updating our value table.
    #     for state in range(self.env.observation_space.n):
    #         state_values = [
    #             self.calc_action_value(state, action)
    #             for action in range(self.env.action_space.n)
    #         ]
    #         self.values[state] = max(state_values)

    def value_iteration(self):
        # q-iteration: loop over (state, action)  -->  values are stored each (state, action)
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    key = (state, action, tgt_state)
                    reward = self.rewards[key]
                    best_action = self.select_action(tgt_state)
                    val = reward + GAMMA * self.values[(tgt_state, best_action)]
                    action_value += (count / total) * val
                self.values[(state, action)] = action_value


#################################################################################################
# -----------------------------------------------------------------------------------------------
# play and value learning
# -----------------------------------------------------------------------------------------------

# ENV_NAME = "FrozenLake-v0"
ENV_NAME = "FrozenLake-v1"

# ENV_NAME = "FrozenLake8x8-v0"      # uncomment for larger version
# ENV_NAME = "FrozenLake8x8-v1"      # uncomment for larger version


# ----------
# test to play TEST_EPISODES to check improvements using the updated value table.
TEST_EPISODES = 20


# ----------
test_env = gym.make(ENV_NAME)
agent = Agent()
writer = SummaryWriter(comment="-q-iteration")

iter_no = 0
best_reward = 0.0

while True:
    iter_no += 1

    # we play 100 random steps from the environment, populating the reward and transition tables.
    agent.play_n_random_steps(100)

    # value iteration loop over all states, updating our value table
    agent.value_iteration()
    print(f'----------------------------------')
    print(f'iter_no: {iter_no}')
    print(f'{agent.values}')

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


# -----------------------------------------------------------------------------------------------
# step by step
# -----------------------------------------------------------------------------------------------

agent = Agent()


# ----------
# play 100 random steps
agent.play_n_random_steps(100)

# (state, action, new_state): reward
print(agent.rewards)

# (state, action)[new_state]:  counter
print(agent.transits)


# ----------
# value iteration loop over all states, updating our value tables
agent.value_iteration()

print(agent.values)

