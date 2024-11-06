import numpy as np


"""
For the later use, I keep the ConnectFour class here.
"""


class ConnectFour:
    def __init__(self):
        self.n_rows = 6
        self.n_cols = 7
        self.board = np.zeros((self.n_rows, self.n_cols), dtype=int)
        self.current_player = 1  # 1 for player 1, -1 for player 2

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        return self.get_state_index()

    def get_state_index(self):
        # A better way to get a unique state index could be required
        return hash(self.board.tostring()) % (10 ** 8)

    def get_valid_actions(self):
        return [col for col in range(self.n_cols) if self.board[0, col] == 0]

    def step(self, action):
        if action not in self.get_valid_actions():
            raise ValueError(f"Column {action} is full.")

        row = self.drop_piece(action)
        done, reward = self.check_game_over(row, action)
        self.current_player *= -1  # Switch player

        return self.get_state_index(), reward, done, None

    def drop_piece(self, col):
        if self.board[0, col] != 0:
            return -1
        for row in range(self.n_rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                return row

    def check_game_over(self, row, col):
        if self.is_winner(row, col):
            return True, 1 if self.current_player == 1 else -1
        elif np.all(self.board != 0):
            return True, 0
        return False, 0

    def is_winner(self, row, col):
        return (self.count_consecutive(row, col, 1, 0) >= 4 or  # Vertical
                self.count_consecutive(row, col, 0, 1) >= 4 or  # Horizontal
                self.count_consecutive(row, col, 1, 1) >= 4 or  # Diagonal /
                self.count_consecutive(row, col, 1, -1) >= 4)   # Diagonal \

    def count_consecutive(self, row, col, delta_row, delta_col):
        count = 1
        for direction in (1, -1):
            r, c = row + direction * delta_row, col + direction * delta_col
            while (0 <= r < self.n_rows and
                   0 <= c < self.n_cols and
                   self.board[r, c] == self.current_player):
                count += 1
                r += direction * delta_row
                c += direction * delta_col
        return count
