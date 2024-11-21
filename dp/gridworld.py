from dp import DP


class GridWorld:
    def __init__(self, grid_size=5, goal_state=24, obstacle_states=[12]):
        self.grid_size = grid_size
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.goal_state = goal_state
        self.obstacle_states = obstacle_states
        self.states = list(range(grid_size * grid_size))

        # Rewards
        self.reward_goal = 100
        self.reward_obstacle = -10
        self.reward_step = -1

    def state_to_position(self, state):
        return (state // self.grid_size, state % self.grid_size)

    def position_to_state(self, i, j):
        return i * self.grid_size + j

    def next_state_from_action(self, s, a):
        i, j = self.state_to_position(s)
        intended_i, intended_j = i, j

        if a == 'UP' and i > 0:
            intended_i = i - 1
        elif a == 'DOWN' and i < self.grid_size - 1:
            intended_i = i + 1
        elif a == 'LEFT' and j > 0:
            intended_j = j - 1
        elif a == 'RIGHT' and j < self.grid_size - 1:
            intended_j = j + 1

        return self.position_to_state(intended_i, intended_j)

    def transition_probs(self, s, a, s_prime):
        if s == self.goal_state:  # Terminal state
            return 1.0 if s == s_prime else 0.0

        if s in self.obstacle_states:  # Can't move from obstacle
            return 0.0

        intended_state = self.next_state_from_action(s, a)

        # If intended next state is obstacle, stay in current position
        if intended_state in self.obstacle_states:
            intended_state = s

        return 1.0 if s_prime == intended_state else 0.0

    def rewards(self, s, a):
        if s == self.goal_state:
            return 0.0

        next_state = self.next_state_from_action(s, a)
        if next_state == self.goal_state:
            return self.reward_goal
        elif next_state in self.obstacle_states:
            return self.reward_obstacle
        return self.reward_step

    def solve(self, gamma=0.9):
        dp = DP(
            self.states,
            self.actions,
            self.transition_probs,
            self.rewards,
            gamma
        )
        dp.policy_iteration()
        return dp.get_optimal_policy(), dp.get_optimal_value_function()

    def print_policy(self, policy):
        print("\nOptimal Policy:")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = self.position_to_state(i, j)
                if state == self.goal_state:
                    print("GOAL".center(8), end=" ")
                elif state in self.obstacle_states:
                    print("OBST".center(8), end=" ")
                else:
                    print(f"{policy[state]}".center(8), end=" ")
            print()

    def print_values(self, values):
        print("\nOptimal Value Function:")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = self.position_to_state(i, j)
                print(f"{values[state]:.2f}".center(8), end=" ")
            print()
