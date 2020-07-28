import numpy as np
import mnk_game


class MnkGravityGame(mnk_game.MnkGame):

    def __init__(self, m, n, k):
        super().__init__(m, n, k)
        self.positions = np.full(m, n-1, dtype=np.int8)

    def restart(self):
        super().restart()
        self.positions = np.full(self.m, self.n-1, dtype=np.int8)

    def put(self, position):
        if type(position) == tuple:
            position = position[1]
        done = self.positions[position] < self.n and super().put((self.positions[position], position))
        if done:
            self.positions[position] -= 1
        return done

    def undo(self):
        last_move = self.moves_stack[-1][1]
        done = super().undo()
        if done:
            self.positions[last_move] += 1
        return done

    def _get_moves(self):
        for j in range(self.m):  # Optimize?
            if self.positions[j] >= 0:
                yield self.positions[j], j
        """print(np.where(self.positions > 0))
        return np.where(self.positions > 0)[0]"""

    def _get_equivalents(self):
        return {"id": {"func": lambda i, j: (i, j), "board": lambda: self.board,
                        "arr": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])},
                "lr_sym": {"func": lambda i, j: (i, self.m - 1 - j), "board": lambda: np.fliplr(self.board),
                           "arr": np.array([[1, 0, 0], [0, -1, self.m - 1], [0, 0, 1]])}}
