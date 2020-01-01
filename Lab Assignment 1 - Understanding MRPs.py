import numpy as np
import random as rm

# Constants

STATE_STRING = ["Class 1", "Class 2", "Class 3", "Pass", "Facebook", "Pub", "Sleep"]

STATE_SPACE = {"Class 1": 0,
               "Class 2": 1,
               "Class 3": 2,
               "Pass": 3,
               "Facebook": 4,
               "Pub": 5,
               "Sleep": 6}

PROB_MATRIX = np.array([[0, 0.5, 0, 0, 0.5, 0, 0],
                        [0, 0, 0.8, 0, 0, 0, 0.2],
                        [0, 0, 0, 0.6, 0, 0.4, 0],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0.1, 0, 0, 0, 0.9, 0, 0],
                        [0.2, 0.4, 0.4, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1]])

REWARD = {"Class 1": -2,   # TODO consider merging this with a list in STATE_SPACE key
          "Class 2": -2,
          "Class 3": -2,
          "Pass": 10,
          "Facebook": -1,
          "Pub": 1,
          "Sleep": 0}

INT_STATE = "Class 1"
TERM_STATE = "Sleep"
DISCOUNT = 0.5
ITER_NUM = range(1000)

# Functions


def state_transition_probability(state_name):
    """
    parameters: particular state in state space
    :return: transition probability vector for that state, Pss'
    """
    state_trans_pos = STATE_SPACE[state_name]

    return PROB_MATRIX[state_trans_pos, :]


def immediate_reward(state_name):
    """
    :param state_name:
    :return: immediate reward recieved upon departing the state
    """
    return REWARD[state_name]


def trace_sample(state_space):
    """
    parameters: the previous state
    :return: the next state visited
    """

    # For probabilities generate random numbers between 0 and 1 and then split with inequalities

    rando_num = rm.random()

    transition_probs = [x for x in state_space if x > 0]
    transition_loc = [val for val, x in enumerate(state_space) if x > 0]

    ranges = []

    # Divide [0,1] based on probabilities
    init_prob = 0

    for prob in transition_probs:
        rng = [init_prob, init_prob + prob]
        init_prob += prob
        ranges.append(rng)

    # How to associate the ranges with state name
    for val, x in enumerate(ranges):
        if x[0] < rando_num < x[1]:
            next_state_loc = transition_loc[val]
            next_state = STATE_STRING[next_state_loc]

    return next_state


def compute_trace_return(state_trace, immediate_reward_sequence):
    """
    :param state_trace: Takes a sequence of states from the MRP process
    :return: The return for this sequence including the discount factor
    """

    return_sequence = []

    for val, st in enumerate(state_trace):
        intermediate_return = pow(DISCOUNT, val) * immediate_reward_sequence[val]
        return_sequence.append(intermediate_return)

    rtrn = np.sum(return_sequence)

    return rtrn


def compute_state_value(starting_state):
    """
    :param starting_state:
    :return: the sequence of states visited in specific run of MRP process, sequence of immediate rewards received
    """

    state = starting_state
    trace_of_states = [starting_state]
    trace_of_rewards = [STATE_SPACE[starting_state]]

    while state != TERM_STATE:
        following_state = trace_sample(state_transition_probability(state))
        # print(following_state)
        trace_of_states.append(following_state)
        # print(immediate_reward(following_state))
        trace_of_rewards.append(immediate_reward(following_state))
        state = following_state

    return trace_of_states, trace_of_rewards

# To compute for each value of the state , specify that state as the initial state and average for many traces


for name_of_state in STATE_STRING:
    print(name_of_state)
    intermediate_state_vals = []
    for i in ITER_NUM:
        # print(i)
        trace, rewards = compute_state_value(name_of_state)
        total_return = compute_trace_return(trace, rewards)
        intermediate_state_vals.append(total_return)

    state_value = np.mean(intermediate_state_vals)
    print(state_value)


# Note that values do not match the lecture notes, do I need to solve the simultaneous equations?



