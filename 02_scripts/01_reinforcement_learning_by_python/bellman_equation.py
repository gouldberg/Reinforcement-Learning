# -*- coding: utf-8 -*-

# ----------
# REFERENCE
# https://github.com/icoxfog417/baby-steps-of-rl-ja


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Components in Bellman Equation
# -----------------------------------------------------------------------------------------------------------

# ----------
# Value Function:
# here Value is determined only be state (=s)
def V(s, gamma=0.99):
    V = R(s) + gamma * max_V_on_next_state(s)
    return V


# ----------
# Reward Function
def R(s):
    if s == "happy_end":
        return 1
    elif s == "bad_end":
        return -1
    else:
        return 0


# ----------
# select action for values to be maximized
def max_V_on_next_state(s):
    # If game end, expected value is 0.
    if s in ["happy_end", "bad_end"]:
        return 0

    actions = ["up", "down"]
    values = []
    for a in actions:
        transition_probs = transit_func(s, a)
        v = 0
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            v += prob * V(next_state)
        values.append(v)
    return max(values)


# ----------
# trainsition function
def transit_func(s, a):
    """
    Make next state by adding action str to state.
    ex: (s = 'state', a = 'up') => 'state_up'
        (s = 'state_up', a = 'down') => 'state_up_down'
    """

    actions = s.split("_")[1:]
    LIMIT_GAME_COUNT = 5
    HAPPY_END_BORDER = 4
    MOVE_PROB = 0.9

    def next_state(state, action):
        return "_".join([state, action])

    if len(actions) == LIMIT_GAME_COUNT:
        up_count = sum([1 if a == "up" else 0 for a in actions])
        state = "happy_end" if up_count >= HAPPY_END_BORDER else "bad_end"
        prob = 1.0
        return {state: prob}
    else:
        opposite = "up" if a == "down" else "down"
        return {
            next_state(s, a): MOVE_PROB,
            next_state(s, opposite): 1 - MOVE_PROB
        }


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# play
# -----------------------------------------------------------------------------------------------------------

print(V("state"))

print(V("state_up_up"))

print(V("state_down_down"))
