from gridworld import GridWorld

if __name__ == "__main__":
    env = GridWorld()
    optimal_policy, optimal_values = env.solve()

    env.print_policy(optimal_policy)
    env.print_values(optimal_values)
