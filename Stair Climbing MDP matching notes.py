import numpy as np
import random as rm

STATE_STRING = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]

STATE_SPACE = {"S1": 0,
               "S2": 1,
               "S3": 2,
               "S4": 3,
               "S5": 4,
               "S6": 5,
               "S7": 6}

ACTION_SPACE = {"Down": 0,
                "Up": 1}

TERMINAL_STATES = ["S1", "S7"]  # Absorbing states

# Question: is the transition probability matrix different for each action? Do I need a 3D array?

PROB_TRANS_L = np.array([[1, 1, 0, 0, 0, 0, 0],  # Always down
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1]])

PROB_TRANS_R = np.array([[1, 0, 0, 0, 0, 0, 0],   # Always up
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1]])

REWARD_L = np.array([[0, -10, 0, 0, 0, 0, 0],  # Always down
                     [0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0]])

REWARD_R = np.array([[0, 0, 0, 0, 0, 0, 0],    # Always up
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, -1, 0, 0, 0, 0, 0],
                     [0, 0, -1, 0, 0, 0, 0],
                     [0, 0, 0, -1, 0, 0, 0],
                     [0, 0, 0, 0, -1, 0, 0],
                     [0, 0, 0, 0, 0, 10, 0]])

DISCOUNT = 0.9
THRES = 0.05

# The action will always be the same regardless of the state - deterministic
# POLICY = np.array([0, 1, 1, 1, 1, 1, 0])

# Stochastic:
POLICY = np.array([[0, 0.5, 0.5, 0.5, 0.5, 0.5, 0],   # Down
                   [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0]])  # Up


def value_function(state_name, current_values_of_states):

    state_pos = STATE_SPACE[state_name]
    policy_vector = POLICY[:, state_pos]
    result = []

    for idx, act in enumerate(policy_vector):

        if idx == 0:
            transition_prob_matrix = PROB_TRANS_L
            reward_matrix = REWARD_L
        else:
            transition_prob_matrix = PROB_TRANS_R
            reward_matrix = REWARD_R

        # Based on action pluck out columns

        state_trans_prob = transition_prob_matrix[:, state_pos]   # vector
        reward = reward_matrix[:, state_pos]                      # vector
        policy = policy_vector[idx]

        result.append(policy * np.sum(state_trans_prob * (reward + DISCOUNT * np.asarray(current_values_of_states))))

    return np.sum(result)


def iterative_value_function():

    state_values = np.zeros(7)

    delta = THRES + 1

    while delta > THRES:
        new_state_values = []
        for idx, st in enumerate(STATE_STRING):

            state_value = value_function(st, state_values)
            new_state_values.append(state_value)

        delta = np.max(np.asarray(state_values)-np.asarray(new_state_values))
        state_values = new_state_values

    return new_state_values


final_value_function = iterative_value_function()
print(final_value_function)
