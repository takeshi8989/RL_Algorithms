from epsilon_greedy_bandit import EpsilonGreedyBandit

if __name__ == "__main__":
    k = 10
    epsilon = 0.1
    num_iterations = 1000

    bandit = EpsilonGreedyBandit(k, epsilon)
    bandit.run(num_iterations)

    print("Estimated action values (Q):", bandit.Q)
    print("Number of times each action was taken (N):", bandit.N)
