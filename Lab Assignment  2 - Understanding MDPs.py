import numpy as np
import random as rm
import matplotlib.pyplot as plt

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

REWARD_L = np.array([[0, -100, 0, 0, 0, 0, 0],  # Always down
                     [0, 0, 10, 0, 0, 0, 0],
                     [0, 0, 0, 10, 0, 0, 0],
                     [0, 0, 0, 0, 10, 0, 0],
                     [0, 0, 0, 0, 0, 10, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0]])

REWARD_R = np.array([[0, 0, 0, 0, 0, 0, 0],    # Always up
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, -10, 0, 0, 0, 0, 0],
                     [0, 0, -10, 0, 0, 0, 0],
                     [0, 0, 0, -10, 0, 0, 0],
                     [0, 0, 0, 0, -10, 0, 0],
                     [0, 0, 0, 0, 0, 100, 0]])

DISCOUNT = 0.5

THRES = 0.05

# The action will always be the same regardless of the state - deterministic
POLICY = np.array([0, 1, 1, 1, 1, 1, 0])

# Stochastic:
# POLICY = np.array([[0, 0.5, 0.5, 0.5, 0.5, 0.5, 0],   # Down
#                     [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0]]) # Up


def value_function(policy, state_name, current_values_of_states):

    if policy == "Always Down":
        transition_prob_matrix = PROB_TRANS_L
        reward_matrix = REWARD_L
    else:
        transition_prob_matrix = PROB_TRANS_R
        reward_matrix = REWARD_R

    # Based on action pluck out columns
    state_pos = STATE_SPACE[state_name]
    state_trans_prob = transition_prob_matrix[:, state_pos]   # vector
    reward = reward_matrix[:, state_pos]                      # vector
    policy = POLICY[state_pos]                                # scalar

    result = policy * np.sum(state_trans_prob * (reward + DISCOUNT * np.asarray(current_values_of_states)))

    return result


def iterative_value_function(policy):

    state_values = np.zeros(7)

    delta = THRES + 1

    while delta > THRES:
        new_state_values = []

        for idx, st in enumerate(STATE_STRING):
            state_value = value_function(policy, st, state_values)
            new_state_values.append(state_value)

        delta = np.max(np.asarray(state_values) - np.asarray(new_state_values))
        state_values = new_state_values

    return new_state_values


final_value_function = iterative_value_function("Always Down")
print(final_value_function)

# GTA:
# Question 5: Run the above code for multple values of gamma above, and mentally compute the path from S4 that results in the most return.
# Plot this value against gamma.

# My take:
# Look at S4 for an always down and an always up policy for different values of gamma which should result in an always up policy being more rewarding.
# GTA said it was okay!

gamma_values = np.asarray([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
always_down_values_s4 = np.asarray([10, 10.25, 10, 9.25, 8, 6.25, 4, 1.25, -2, -5.75, -10, -14.75, -20, -25.75, -32, -38.75, -46, -53.75, -62, -70.76, -80])
always_up_values_s4 = always_down_values_s4 * -1

plt.plot(gamma_values, always_down_values_s4)
plot = plt.plot(gamma_values, always_up_values_s4)
plt.show()