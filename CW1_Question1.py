import numpy as np


STATE_SPACE = {"S1": 0,
               "S2": 1,
               "S3": 2}


# STATE_SEQUENCE = [0, 1, 1, 1, 1, 0, 1]
STATE_SEQUENCE = [0, 0, 1, 1, 2, 0, 1]

# REWARD_SEQUENCE = [1, 0, 1, 0, 1, 1, 0]
REWARD_SEQUENCE = [1, 1, 0, 0, 0, 1, 1]

gamma = 1
alpha = 0.25


def temporal_difference_estimation(state_sequence, reward_sequence):
    state_values = np.zeros(3)
    valid_states = len(state_sequence) - 1

    for idx, st in enumerate(state_sequence[0:valid_states]):
        next_state = state_sequence[idx + 1]
        delta = reward_sequence[idx] + gamma * state_values[next_state] - state_values[st]
        state_values[st] = state_values[st] + alpha * delta

    return state_values


print(temporal_difference_estimation(STATE_SEQUENCE, REWARD_SEQUENCE))