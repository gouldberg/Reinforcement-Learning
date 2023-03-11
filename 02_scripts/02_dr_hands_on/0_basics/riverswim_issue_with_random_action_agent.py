#!/usr/bin/env python3

import random
# import argparse
import collections


# ----------
# Reference Paper:
# 'An analysis of model-based Interval Estimation for Markov Decision Process'


##############################################################################################
# --------------------------------------------------------------------------------------------
# get_action
# do_action
# --------------------------------------------------------------------------------------------

# only 2 actions
def get_action(state: int, total_states: int) -> int:
    """
    Return action from the given state. Actions are selected randomly
    :param state: state we're currently in
    :return: 0 means left, 1 is right
    """
    if state == 1:
        return 1
    if state == total_states:
        return 0
    return random.choice([0, 1])


def do_action(state: int, action: int) -> int:
    """
    Simulate the action from the given state
    """
    # left action always succeeds and brings us to the left
    if action == 0:
        return state-1

    if state == 1:
        return random.choices([1, 2], weights=[0.4, 0.6])[0]

    # the rest of states are the same
    delta = random.choices([-1, 0, 1], weights=[0.05, 0.6, 0.35])[0]
    return state + delta


##############################################################################################
# --------------------------------------------------------------------------------------------
# 'River Swim" example:
#   - Illustrate the issue with random actions in exploration.
#     By acting randomly, our agent does not try to actively explore the environment.
# --------------------------------------------------------------------------------------------

# Amount of steps to simulate
N_STEPS = 100
N_STEPS = 1000
N_STEPS = 10000

# Limit of one episode
EPISODE_LENGTH = 10

# Amount of states in the eivironment
ENV_LEN = 6


# ----------
SEED = 2
random.seed(SEED)

states_count = collections.Counter()
state = 1
episode_step = 0

for _ in range(N_STEPS):
    action = get_action(state, ENV_LEN)
    state = do_action(state, action)
    states_count[state] += 1
    episode_step += 1
    if episode_step == EPISODE_LENGTH:
        state = 1
        episode_step = 0

for state in range(1, ENV_LEN + 1):
    print("%d:\t%d" % (state, states_count[state]))


# -->
# ENV_LEN = 6 and N_STEPS = 100
# Agent never reached state 6 and was only in state 5 once.

# N_STEPS = 1000
# with 10 times more episodes simultated, we still did not visit state 6,
# so the agent had no idea about the large rewared there.

# N_STEPS = 10000
# Only with 100 episodes simulated were we able to get to state 6, but only 5 times, which is 0.05%
# of all steps.