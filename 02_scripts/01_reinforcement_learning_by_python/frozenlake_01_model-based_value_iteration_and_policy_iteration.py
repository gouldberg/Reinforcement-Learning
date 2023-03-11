
import os
import random
from enum import Enum
import numpy as np


#############################################################################################
# -------------------------------------------------------------------------------------------
# 4 elements in MDP (Markov Decision Process):
#   -  State
#   -  Action
#   -  Environment:  Transition Function and Reward Function
# -------------------------------------------------------------------------------------------

# State is position (cell) in grid:  row and column
class State():

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


# ----------
class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


# ----------
class Environment():

    def __init__(self, grid, move_prob=0.8):
        # grid is 2d-array. Its values are treated as an attribute.
        # Kinds of attribute is following.
        #  0: ordinary cell
        #  -1: damage cell (game end)
        #  1: reward cell (game end)
        #  9: block cell (can't locate agent)
        self.grid = grid
        self.agent_state = State()

        # Default reward is minus. Just like a poison swamp.
        # It means the agent has to reach the goal fast!
        self.default_reward = -0.04

        # Agent can move to a selected direction in move_prob.
        # It means the agent will move different direction
        # in (1 - move_prob).
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN,
                Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # Block cells are not included to the state.
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    ############################################
    # Transition Probability
    #   - selected action:  prob = move_prob
    #   - opposite direction of selected action:  prob = 0
    #   - other direction:  (1 - move_prob) / 2
    ############################################
    def transit_func(self, state, action):
        transition_probs = {}
        if not self.can_action_at(state):
            # Already on the terminal cell.
            return transition_probs

        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # Execute an action (move).
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # Check whether a state is out of the grid.
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # Check whether the agent bumped a block cell.
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    ############################################
    # reward and done
    ############################################
    def reward_func(self, state):
        reward = self.default_reward
        done = False

        # Check an attribute of next state.
        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            # Get reward! and the game ends.
            reward = 1
            done = True
        elif attribute == -1:
            # Get damage! and the game ends.
            reward = -1
            done = True

        return reward, done

    ############################################
    # reset, step and transit
    ############################################
    def reset(self):
        # Locate the agent at lower left corner.
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    def transit(self, state, action):
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        # ----------
        # next state is selected based on probs
        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done



#############################################################################################
# -------------------------------------------------------------------------------------------
# Planner (Base)
# -------------------------------------------------------------------------------------------

class Planner():

    def __init__(self, env):
        self.env = env
        self.log = []

    def initialize(self):
        self.env.reset()
        self.log = []

    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("Planner have to implements plan method.")

    ############################################
    # Transition Function: T(s'|s, a)
    # returns
    #   - next state
    #   - prob of move to next state
    #   - reward at next state
    ############################################
    def transitions_at(self, state, action):
        transition_probs = self.env.transit_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]

            # reward and done
            reward, _ = self.env.reward_func(next_state)
            yield prob, next_state, reward

    def dict_to_grid(self, state_reward_dict):
        grid = []
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            grid.append(row)
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]

        return grid


# -------------------------------------------------------------------------------------------
# Value Iteration Planner
# -------------------------------------------------------------------------------------------

class ValueIterationPlanner(Planner):

    def __init__(self, env, fpath_dump):
        super().__init__(env)
        self.fpath_dump = fpath_dump

    def plan(self, gamma=0.9, threshold=0.0001):

        # ----------
        # initialize
        self.initialize()
        actions = self.env.actions
        V = {}
        for s in self.env.states:
            # Initialize each state's expected reward.
            V[s] = 0
        # ===================================
        f = open(self.fpath_dump, 'w')
        f.write(f'-----------------------------------------------------\n')
        f.write(f'Initialization \n')
        f.write(f'  actions: \n')
        f.write(f'        {actions} \n')
        f.write(f'  value by state: \n')
        f.write(f'        {V} \n')
        f.write(f'-----------------------------------------------------\n')
        # ===================================

        # ----------
        idx = 0

        while True:
            delta = 0
            self.log.append(self.dict_to_grid(V))

            # ----------
            # loop over state
            for s in V:
                if not self.env.can_action_at(s):
                    continue
                expected_rewards = []
                # ===================================
                f.write(f'-----------------------------------------------------\n')
                f.write(f'  state: {s}\n')
                # ===================================

                # ----------
                # loop over action
                for a in actions:
                    idx += 1
                    # ===================================
                    f.write(f'-----------------------------------------------------\n')
                    f.write(f'    {idx} -- at {s}, try to take action {a}, but selected action have only prob = {self.env.move_prob: .3f} \n')
                    # ===================================

                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r_tmp = prob * (reward + gamma * V[next_state]) 
                        r += r_tmp
                        # ===================================
                        f.write(f'       --  cumsum reward = {r: .3f}   next_state: {next_state}:  {r_tmp: .3f} = prob({prob: .3f}) * ( reward({reward: .3f})  + gamma({gamma}) * V[{next_state}]({V[next_state]: .3f}) )\n')
                        # ===================================
                    expected_rewards.append(r)
                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                # ===================================
                f.write(f'   -----------------------\n')
                f.write(f'   V[{s}] = max reward({max_reward: .3f})    max(delta, abs(max_reward - V[s])) = {delta: .3f}\n')
                # ===================================
                V[s] = max_reward

            if delta < threshold:
                break

        V_grid = self.dict_to_grid(V)
        f.close()
        return V_grid


# -------------------------------------------------------------------------------------------
# Policy Iteration Planner
# 1. update V (expected rewards under current policy)  --> 2. update policy[s][a]
# -------------------------------------------------------------------------------------------

class PolicyIterationPlanner(Planner):

    def __init__(self, env, fpath_dump):
        super().__init__(env)
        self.policy = {}
        self.fpath_dump = fpath_dump

    def initialize(self):
        super().initialize()
        self.policy = {}
        actions = self.env.actions
        states = self.env.states
        for s in states:
            self.policy[s] = {}
            for a in actions:
                # Initialize policy.
                # At first, each action is taken uniformly.
                self.policy[s][a] = 1 / len(actions)

    # V[s] is updated
    def estimate_by_policy(self, gamma, threshold):
        V = {}
        for s in self.env.states:
            # Initialize each state's expected reward.
            V[s] = 0

        while True:
            delta = 0
            for s in V:
                expected_rewards = []
                for a in self.policy[s]:
                    action_prob = self.policy[s][a]
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        # action probability is taken into account.
                        r += action_prob * prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                value = sum(expected_rewards)
                delta = max(delta, abs(value - V[s]))
                V[s] = value
            if delta < threshold:
                break

        return V

    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        states = self.env.states
        actions = self.env.actions
        # ===================================
        f = open(self.fpath_dump, 'w')
        f.write(f'-----------------------------------------------------\n')
        f.write(f'Initialization \n')
        f.write(f'  actions: \n')
        f.write(f'        {actions} \n')
        f.write(f'-----------------------------------------------------\n')
        # ===================================

        def take_max_action(action_value_dict):
            return max(action_value_dict, key=action_value_dict.get)

        # ----------
        idx = 0

        while True:
            update_stable = True
            # Estimate expected rewards under current policy.
            V = self.estimate_by_policy(gamma, threshold)
            # ===================================
            f.write(f'=====================================================\n')
            f.write(f'  1. updated Expected Rewards Under Current Policy (V): \n')
            f.write(f'        {V} \n')
            f.write(f'=====================================================\n')
            # ===================================
            self.log.append(self.dict_to_grid(V))

            # ----------
            # loop over state
            for s in states:
                # Get an action following to the current policy.
                policy_action = take_max_action(self.policy[s])
                # ===================================
                f.write(f'-----------------------------------------------------\n')
                f.write(f'  at {s}, policy[{s}]: {self.policy[s]}\n')
                f.write(f'  at {s}, try to take policy action {policy_action}, but selected action have only prob = {self.env.move_prob: .3f} \n')
                f.write(f'-----------------------------------------------------\n')
                # ===================================

                # Compare with other actions.
                action_rewards = {}

                # ----------
                # loop over action
                for a in actions:

                    idx += 1
                    # ===================================
                    f.write(f'-----------------------------------------------------\n')
                    f.write(f'    {idx} -- action {a}\n')
                    # ===================================

                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r_tmp = prob * (reward + gamma * V[next_state]) 
                        r += r_tmp
                        # ===================================
                        f.write(f'       --  cumsum reward = {r: .3f}   next_state: {next_state}:  {r_tmp: .3f} = prob({prob: .3f}) * ( reward({reward: .3f})  + gamma({gamma}) * V[{next_state}]({V[next_state]: .3f}) )\n')
                        # ===================================
                    action_rewards[a] = r
                    # ===================================
                    f.write(f'   action_rewards[{a}] = {r: .3f}\n')
                    # ===================================

                best_action = take_max_action(action_rewards)
                if policy_action != best_action:
                    update_stable = False
                # ===================================
                f.write(f'   -----------------------\n')
                f.write(f'   best_action = {best_action}\n')
                # ===================================

                # Update policy (set best_action prob=1, otherwise=0 (greedy))
                for a in self.policy[s]:
                    prob = 1 if a == best_action else 0
                    self.policy[s][a] = prob
               # ===================================
                f.write(f'   =======================\n')
                f.write(f'   2. updated policy[{s}]: {self.policy[s]}\n')
                # ===================================

            if update_stable:
                # If policy isn't updated, stop iteration
                break

        # Turn dictionary to grid
        V_grid = self.dict_to_grid(V)
        f.close()
        return V_grid


#############################################################################################
# -------------------------------------------------------------------------------------------
# play:  Value Iteration
# -------------------------------------------------------------------------------------------

# environment
grid = [
    [0, 0, 0, 1],
    [0, 9, 0, -1],
    [0, 0, 0, 0]
]

# probability to move to the direction following selected action
move_prob = 0.8
env = Environment(grid, move_prob=move_prob)

print(env.grid)

# At start, agent is posisioned at [2,0] (bottom left)
print(env.agent_state)


# ----------
# plan
base_path = '/home/kswada/kw/reinforcement_learning'
fpath_dump = os.path.join(base_path, '04_output/01_reinforcement_learning_by_python/valute_iteration.txt')

planner = ValueIterationPlanner(env, fpath_dump=fpath_dump)

result = planner.plan()

planner.log.append(result)

# history of value updates
for log_ in planner.log:
    print(log_)


# -----------------------------
# check
move_prob = 0.8
env = Environment(grid, move_prob=move_prob)

# environment
env.reset()

# set action list:  4 action
print(Action.Up)

actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
print(actions)


# state:  11 states (excluding wall(=9))
print(env.states)
print(len(env.states))


# value: state_reward_dict
V = {}
for s in env.states:
    V[s] = 0

print(V)
print(len(V))


#############################################################################################
# -------------------------------------------------------------------------------------------
# play:  Policy Iteration
# -------------------------------------------------------------------------------------------

# environment
grid = [
    [0, 0, 0, 1],
    [0, 9, 0, -1],
    [0, 0, 0, 0]
]

# probability to move to the direction following selected action
move_prob = 0.8
env = Environment(grid, move_prob=move_prob)

print(env.grid)

# At start, agent is posisioned at [2,0] (bottom left)
print(env.agent_state)


# ----------
# plan
base_path = '/home/kswada/kw/reinforcement_learning'
fpath_dump = os.path.join(base_path, '04_output/01_reinforcement_learning_by_python/policy_iteration.txt')

planner = PolicyIterationPlanner(env, fpath_dump=fpath_dump)

result = planner.plan()

planner.log.append(result)
