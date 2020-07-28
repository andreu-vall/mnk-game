import sortedcontainers as sc
import numpy as np
import time


class MnkGame:

    def __init__(self, m, n, k):
        self.m, self.n, self.k = m, n, k  # m columns, n rows, k to win
        self.board = np.zeros((n, m), dtype=np.int8)
        self.turn, self.moves_stack = 1, []
        self.hashes_stack, self.identities_stack = [hash(self.board.tobytes())], [hash(self.board.tobytes())]
        self.finished, self.won, self.heuristic = False, 0, 0
        self.final, self.depth, self.best_value = False, 0, 0
        self.ident_moves, self.sorted_ident_moves = {}, None
        self.trans, self.ident_trans = self._get_equivalents(), {}
        self.cached_data, self.hashes, self.identities = {}, {}, {}
        self.identifying_time, self.status_time = 0, 0
        self._save_cache()
        self._identify_trans()
        self._save_identities()

    def _save_cache(self):
        self.cached_data[self.identities_stack[-1]] = self.finished, self.won, self.heuristic, self.final, self.depth, \
                                                      self.best_value, self.ident_moves, self.sorted_ident_moves

    def _identify_trans(self):
        arr_trans = {dic["arr"].tobytes(): string for string, dic in self.trans.items()}
        for f, dic1 in self.trans.items():
            self.ident_trans[f+"-1"] = arr_trans[np.linalg.inv(dic1["arr"]).astype(np.int32).tobytes()]
            for g, dic2 in self.trans.items():
                self.ident_trans[g+"º"+f] = arr_trans[np.dot(dic1["arr"], dic2["arr"]).tobytes()]

    def _save_identities(self):
        start = time.perf_counter()
        for f, dic in self.trans.items():
            hashed = hash(dic["board"]().tobytes())
            f_inv = self.ident_trans[f+"-1"]
            self.hashes[self.identities_stack[-1], f_inv] = hashed
            if hashed not in self.identities:
                self.identities[hashed] = self.identities_stack[-1], f_inv
        self.identifying_time += time.perf_counter()-start

    def restart(self):
        self.board.fill(0)
        self.turn, self.moves_stack = 1, []
        del self.hashes_stack[1:], self.identities_stack[1:]
        self._load_cache()

    def _load_cache(self):
        self.finished, self.won, self.heuristic, self.final, self.depth, self.best_value, self.ident_moves, \
            self.sorted_ident_moves = self.cached_data[self.identities_stack[-1]]

    def put(self, position):
        if self.finished or self.board[position] != 0:
            return False
        self.board[position], self.turn = self.turn, -self.turn
        self.moves_stack.append(position)
        f = self.identities[self.hashes_stack[-1]][1]
        id_position = self.trans[f]["func"](*position)
        if id_position not in self.ident_moves:
            self.hashes_stack.append(hash(self.board.tobytes()))
            if self.hashes_stack[-1] not in self.identities:
                self.identities_stack.append(self.hashes_stack[-1])
                self.ident_moves[id_position] = self.identities_stack[-1], self.ident_trans[f+"-1"]
                self._compute_status()
                self._save_cache()
                self._save_identities()
                return True

            identity, h = self.identities[self.hashes_stack[-1]]
            self.identities_stack.append(identity)
            g = self.ident_trans[h+"º"+self.ident_trans[f+"-1"]]
            self.ident_moves[id_position] = identity, g
        else:
            move_identifier, g = self.ident_moves[id_position]
            gof = self.ident_trans[g+"º"+f]
            self.hashes_stack.append(self.hashes[move_identifier, gof])
            self.identities_stack.append(self.identities[self.hashes_stack[-1]][0])

        self._load_cache()
        return True

    def undo(self):
        if not self.moves_stack:
            return False
        self.board[self.moves_stack.pop()], self.turn = 0, -self.turn
        del self.hashes_stack[-1], self.identities_stack[-1]
        self._load_cache()
        return True

    def __str__(self):
        return str(self.board)

    def _compute_status(self):
        start = time.perf_counter()
        player = -self.turn
        self.board[self.moves_stack[-1]] = 0
        for line, pos in self._get_lines():
            for part in (line[x:x + self.k] for x in range(max(0, pos - self.k + 1),
                                                           min(line.size - self.k + 1, pos + 1))):
                player1 = np.sum(part == 1)
                zeros = np.sum(part == 0)
                player2 = self.k - (player1 + zeros)
                if player == 1 and player1 == self.k - 1 or player == -1 and player2 == self.k - 1:
                    self.finished = True
                    break
                if player1 == 0 and player2 > 0:
                    self.heuristic += 10 ** (player2 - 1)
                elif player2 == 0 and player1 > 0:
                    self.heuristic -= 10 ** (player1 - 1)
                if player == 1 and player2 == 0:
                    self.heuristic += 10 ** player1
                elif player == -1 and player1 == 0:
                    self.heuristic -= 10 ** player2
            if self.finished:
                break
        if self.finished:
            self.heuristic = player*float("inf")
            self.won = player
        else:
            self.finished = len(self.moves_stack) == self.n * self.m

        self.board[self.moves_stack[-1]] = player
        self.final, self.depth, self.best_value = self.finished, 0, self.heuristic
        self.ident_moves, self.sorted_ident_moves = {}, None
        self.status_time += time.perf_counter()-start

    def _get_lines(self):
        yield self.board[self.moves_stack[-1][0], :], self.moves_stack[-1][1]
        yield self.board[:, self.moves_stack[-1][1]], self.moves_stack[-1][0]
        diagonal1 = self.moves_stack[-1][1] - self.moves_stack[-1][0]
        yield self.board.diagonal(diagonal1), self.moves_stack[-1][1] - max(0, diagonal1)
        diagonal2 = self.m - 1 - sum(self.moves_stack[-1])
        yield np.fliplr(self.board).diagonal(diagonal2), self.moves_stack[-1][0] - max(0, -diagonal2)

    def iterative_deepening_search(self, max_time=1):
        self.identifying_time, self.status_time = 0, 0
        end = time.perf_counter() + max_time
        while not self.final and time.perf_counter() < end:
            start = time.perf_counter()
            self._recursive_alpha_beta(self.depth+1, -float("inf"), float("inf"), end)
            if time.perf_counter() < end:
                print("Finished depth", self.depth, "took", time.perf_counter()-start, "seconds")
            else:
                print("Interrupted depth", self.depth+1)
        print("Final sorted identified moves:", self.sorted_ident_moves)
        print("Identifying time:", self.identifying_time)
        print("Status time:", self.status_time)
        print("Cached:", len(self.cached_data))
        func = self.trans[self.ident_trans[self.identities[self.hashes_stack[-1]][1]+"-1"]]["func"]
        return func(*self.sorted_ident_moves[0][0])

    def _recursive_alpha_beta(self, depth, alpha, beta, end):
        if depth == 0 or self.final or depth <= self.depth:
            return self.best_value, self.final
        f = self.identities[self.hashes_stack[-1]][1]
        if not self.sorted_ident_moves:
            moves_dic = {}
            for move in self._get_moves():
                self.put(move)
                if self.identities_stack[-1] not in moves_dic:
                    moves_dic[self.identities_stack[-1]] = self.trans[f]["func"](*self.moves_stack[-1]), self.best_value
                self.undo()
            self.sorted_ident_moves = sc.SortedKeyList(moves_dic.values(), key=lambda x: -self.turn*x[1])
            self._save_cache()
        best_value, final = -self.turn*float("inf"), True
        new_moves = []
        for id_move, old_value in self.sorted_ident_moves:
            move = self.trans[self.ident_trans[f+"-1"]]["func"](*id_move)
            self.put(move)
            value, final_son = self._recursive_alpha_beta(depth-1, alpha, beta, end)
            self.undo()
            if not final_son:
                final = False
            new_moves.append((id_move, value))
            if value*self.turn > best_value*self.turn:
                best_value = value
                if self.turn == 1:
                    alpha = max(best_value, alpha)
                else:
                    beta = min(best_value, beta)
                if alpha >= beta:
                    break
            if time.perf_counter() >= end:
                break

        if time.perf_counter() < end:
            self.final, self.depth, self.best_value = final, depth, best_value
            if alpha < beta:
                self.sorted_ident_moves = sc.SortedKeyList(new_moves, key=lambda x: -self.turn*x[1])
            else:
                for old, new in zip(self.sorted_ident_moves, new_moves):
                    self.sorted_ident_moves.remove(old)
                    self.sorted_ident_moves.add(new)

            self._save_cache()

        return self.best_value, self.final

    def _get_moves(self):
        return map(tuple, np.transpose(np.where(self.board == 0)))

    def _get_equivalents(self):
        trans = {"id": {"func": lambda i, j: (i, j), "board": lambda: self.board,
                        "arr": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])}}
        if self.n == self.m:
            trans["rot90"] = {"func": lambda i, j: (self.m-1-j, i), "board": lambda: np.rot90(self.board),
                              "arr": np.array([[0, 1, 0], [-1, 0, self.n-1], [0, 0, 1]])}
        trans["rot180"] = {"func": lambda i, j: (self.n-1-i, self.m-1-j), "board": lambda: np.flip(self.board),
                           "arr": np.array([[-1, 0, self.n-1], [0, -1, self.m-1], [0, 0, 1]])}
        if self.n == self.m:
            trans["rot270"] = {"func": lambda i, j: (j, self.n-1-i), "board": lambda: np.rot90(self.board, 3),
                               "arr": np.array([[0, -1, self.n-1], [1, 0, 0], [0, 0, 1]])}
        trans["lr_sym"] = {"func": lambda i, j: (i, self.m-1-j), "board": lambda: np.fliplr(self.board),
                           "arr": np.array([[1, 0, 0], [0, -1, self.m-1], [0, 0, 1]])}
        if self.n == self.m:
            trans["d1_sym"] = {"func": lambda i, j: (j, i), "board": lambda: np.rot90(np.fliplr(self.board)),
                               "arr": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])}
        trans["ud_sym"] = {"func": lambda i, j: (self.n-1-i, j), "board": lambda: np.flipud(self.board),
                           "arr": np.array([[-1, 0, self.n-1], [0, 1, 0], [0, 0, 1]])}
        if self.n == self.m:
            trans["d2_sym"] = {"func": lambda i, j: (self.m-1-j, self.n-1-i),
                               "board": lambda: np.rot90(np.flipud(self.board)),
                               "arr": np.array([[0, -1, self.m-1], [-1, 0, self.n-1], [0, 0, 1]])}
        return trans
