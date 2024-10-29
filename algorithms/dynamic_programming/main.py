from dp import DP

"""
5x5 gridworld example.
"""

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
GRID_SIZE = 5
GOAL_STATE = 24  # bottom-right corner (state 24)
OBSTACLE_STATES = [12]  # Example obstacle in the middle
REWARD_GOAL = 100
REWARD_OBSTACLE = -10
REWARD_STEP = -1


def state_to_position(state):
    return (state // GRID_SIZE, state % GRID_SIZE)


def position_to_state(i, j):
    return i * GRID_SIZE + j


def next_state_from_action(s, a):
    i, j = state_to_position(s)
    intended_i, intended_j = i, j
    intended_i, intended_j = i, j
    if a == 'UP' and i > 0:
        intended_i = i - 1
    elif a == 'DOWN' and i < GRID_SIZE - 1:
        intended_i = i + 1
    elif a == 'LEFT' and j > 0:
        intended_j = j - 1
    elif a == 'RIGHT' and j < GRID_SIZE - 1:
        intended_j = j + 1

    intended_state = position_to_state(intended_i, intended_j)

    return intended_state


def transition_probs(s, a, s_prime):
    if s == GOAL_STATE:  # Terminal state
        return 1.0 if s == s_prime else 0.0

    if s in OBSTACLE_STATES:  # Can't move from obstacle
        return 0.0

    intended_state = next_state_from_action(s, a)

    # If intended next state is obstacle, stay in current position
    if intended_state in OBSTACLE_STATES:
        intended_state = s

    return 1.0 if s_prime == intended_state else 0.0


def rewards(s, a):
    """
    Returns R(s,a) - reward for taking action a in state s
    """
    if s == GOAL_STATE:
        return 0.0

    next_state = next_state_from_action(s, a)
    if next_state == GOAL_STATE:
        return REWARD_GOAL
    elif next_state in OBSTACLE_STATES:
        return REWARD_OBSTACLE
    return REWARD_STEP


if __name__ == "__main__":
    # States are numbered from 0 to 24 (5x5 grid)
    states = list(range(GRID_SIZE * GRID_SIZE))

    dp = DP(states, ACTIONS, transition_probs, rewards, gamma=0.9)
    dp.policy_iteration()

    optimal_policy = dp.get_optimal_policy()
    optimal_value_function = dp.get_optimal_value_function()

    # Print the optimal policy in a grid format
    print("\nOptimal Policy:")
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            state = position_to_state(i, j)
            if state == GOAL_STATE:
                print("GOAL".center(8), end=" ")
            elif state in OBSTACLE_STATES:
                print("OBST".center(8), end=" ")
            else:
                print(f"{optimal_policy[state]}".center(8), end=" ")
        print()

    # Print the value function in a grid format
    print("\nOptimal Value Function:")
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            state = position_to_state(i, j)
            print(f"{optimal_value_function[state]:.2f}".center(8), end=" ")
        print()