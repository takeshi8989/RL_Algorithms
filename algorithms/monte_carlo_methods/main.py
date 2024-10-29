from blackjack import BlackJack
from on_policy_mc import OnPolicyMC
from off_policy_mc import OffPolicyMC


def print_table(usable_ace, policy, algorithm):
    print(
        f"\nOptimal policy "
        f"({'usable' if usable_ace else 'no usable'} ace) "
        f"for {algorithm}:")
    print("       Dealer's Card")
    print("       A  2  3  4  5  6  7  8  9  10")
    print("      ----------------------")

    for player_sum in range(21, 11, -1):
        print(f"{player_sum:2d}    ", end="")
        for dealer_card in range(1, 11):
            state = (player_sum, dealer_card, usable_ace)
            action = "H" if policy(state) == 1 else "S"
            print(f"{action:2}", end=" ")
        print(f"    |{player_sum}")


def print_policy(policy, algorithm):
    print_table(usable_ace=False, policy=policy, algorithm=algorithm)
    print_table(usable_ace=True, policy=policy, algorithm=algorithm)


if __name__ == "__main__":
    env = BlackJack()
    num_episodes = int(
        input("Enter number of episodes (default 10000): ") or 10000)

    # On-policy MC
    on_policy = OnPolicyMC(env.states, env.actions, gamma=0.9, epsilon=0.1)
    for _ in range(num_episodes):
        episode = env.play_episode(
            lambda state: on_policy.choose_action(state))
        on_policy.update(episode)

    # Off-policy MC
    off_policy = OffPolicyMC(env.states, env.actions, gamma=0.9)
    for _ in range(num_episodes):
        episode = env.play_episode(
            lambda state: off_policy.behavior_policy_action(state))
        off_policy.update(episode)

    # Print learned policies
    print_policy(
        lambda state: on_policy.choose_action(state),
        "On-policy"
    )
    print_policy(
        lambda state: off_policy.target_policy_action(state),
        "Off-policy"
    )
