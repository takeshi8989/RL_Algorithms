import numpy as np


class BlackJack:
    def __init__(self):
        # Actions: 0 = stick, 1 = hit
        self.actions = [0, 1]
        # States: (player_sum, dealer_showing_card, usable_ace)
        self.states = [(p, d, a)
                       for p in range(12, 22)  # player sum (12-21)
                       for d in range(1, 11)   # dealer showing (1-10)
                       for a in [False, True]]  # usable ace

    def get_card(self):
        # Assume we have an infinite deck of cards.
        card = min(10, np.random.randint(1, 14))
        return card

    def sum_cards(self, cards):
        total = sum(cards)
        n_aces = cards.count(1)

        # Convert aces from 1 to 11 while possible
        while total <= 11 and n_aces > 0:
            total += 10
            n_aces -= 1

        return total, (total <= 21 and n_aces > 0)

    def dealer_policy(self, cards):
        # Dealer hits on sum < 17
        total, _ = self.sum_cards(cards)
        return total < 17

    def play_dealer(self, showing_card):
        dealer_cards = [showing_card, self.get_card()]

        while self.dealer_policy(dealer_cards):
            dealer_cards.append(self.get_card())
            if self.sum_cards(dealer_cards)[0] > 21:
                return dealer_cards, True  # Bust

        return dealer_cards, False

    def get_reward(self, player_sum, dealer_sum):
        if player_sum > 21:
            return -1
        elif dealer_sum > 21 or player_sum > dealer_sum:
            return 1
        elif player_sum < dealer_sum:
            return -1
        else:
            return 0

    def play_episode(self, policy):
        episode = []

        player_cards = [self.get_card(), self.get_card()]
        dealer_card = self.get_card()

        # Player's turn
        while True:
            player_sum, usable_ace = self.sum_cards(player_cards)

            if player_sum > 21:
                return [(s, a, -1) for (s, a) in episode]
            if player_sum == 21 and len(player_cards) == 2:
                return [(s, a, 1) for (s, a) in episode]

            if player_sum >= 12:
                state = (player_sum, dealer_card, usable_ace)
                action = policy(state)
                episode.append((state, action))

                if action == 0:  # stick
                    break
                else:
                    player_cards.append(self.get_card())
            else:
                player_cards.append(self.get_card())

        # Dealer's turn
        dealer_cards, dealer_bust = self.play_dealer(dealer_card)
        dealer_sum = self.sum_cards(dealer_cards)[0]

        # Calculate final reward
        final_reward = self.get_reward(player_sum, dealer_sum)

        # Update rewards in episode
        return [(s, a, final_reward) for (s, a) in episode]
