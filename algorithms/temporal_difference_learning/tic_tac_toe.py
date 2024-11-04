import numpy as np


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Player 1 starts (could be 1 or -1)
        self.done = False
        self.winner = None
        return self.board

    def get_valid_actions(self):
        return [i * 3 + j for i in range(3) for j in range(3)
                if self.board[i, j] == 0]

    def action_to_index(self, action):
        return action // 3, action % 3

    def step(self, action, user_play=False):
        if self.done:
            raise ValueError(
                "Game has already ended. Please reset the environment.")

        row, col = self.action_to_index(action)
        if self.board[row, col] != 0:
            raise ValueError("Invalid action. Cell already occupied.")

        self.board[row, col] = self.current_player

        if self.check_winner(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1
        elif len(self.get_valid_actions()) == 0:
            self.done = True
            self.winner = 0  # Draw
            reward = 0
        else:
            reward = 0

        self.current_player *= -1

        if self.done or user_play:
            return self.board, reward, self.done, {"winner": self.winner}

        # Play random action for opponent
        opponent_action = np.random.choice(self.get_valid_actions())
        row, col = self.action_to_index(opponent_action)
        self.board[row, col] = self.current_player
        self.current_player *= -1

        if self.check_winner(-self.current_player):
            self.done = True
            self.winner = -self.current_player
            reward = -1
        elif len(self.get_valid_actions()) == 0:
            self.done = True
            self.winner = 0
            reward = 0

        return self.board, reward, self.done, {"winner": self.winner}

    def check_winner(self, player):
        for i in range(3):
            if all(self.board[i, :] == player):  # Check row
                return True
            if all(self.board[:, i] == player):  # Check column
                return True
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == player:
            return True
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == player:
            return True
        return False

    def render(self):
        symbols = {1: "X", -1: "O", 0: " "}
        print("\n".join([" | ".join([symbols[self.board[i, j]]
              for j in range(3)]) for i in range(3)]))
        print()
