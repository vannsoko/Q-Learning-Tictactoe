import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class TicTacToeENV(gym.Env):
    def __init__(self, render_mode="console", epsilon=0.2):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=np.int8)
        self.board = np.zeros(9, dtype=np.int8)
        self.epsilon = epsilon

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        if random.randint(0, 1) == 1:
            if random.uniform(0, 1) < 0.2:
                self.board[random.randint(0, 8)] = 2
            else:
                _, move = self.minimax(self.board, False)
                self.board[move] = 2

        info = {"message": "Nouvelle partie"}
        return self.board.copy(), info

    def step(self, action):
        reward = 0
        finished = False
        info = {}
        if self.board[action] != 0:
            return self.board, -100, True, False, {"message": "illegal action"}

        self.board[action] = 1
        match self.winner(self.board):
            case 0:
                info = {"message": "match en cour"}
                if self.empty_cases(self.board) == []:
                    reward = 0
                    finished = True
                    info = {"message": "match Null"}
            case 1:
                reward = 1
                finished = True
                info = {"message": "match gagné"}
            case 2:
                reward = -1
                finished = True
                info = {"message": "match perdu"}
        if not finished:
            _, move = self.minimax(self.board, False)
            self.board[move] = 2
            match self.winner(self.board):
                case 0:
                    info = {"message": "match en cour"}
                    if self.empty_cases(self.board) == []:
                        finished = True
                        info = {"message": "match Null"}
                case 1:
                    reward = 1
                    finished = True
                    info = {"message": "match gagné"}
                case 2:
                    reward = -1
                    finished = True
                    info = {"message": "match perdu"}

        return self.board.copy(), reward, finished, False, info

    def render(self):
        board_symbols = self.board.copy()
        if self.render_mode == "console":
            print("\n")
            print(f" {board_symbols[0]} | {board_symbols[1]} | {board_symbols[2]} ")
            print("---+---+---")
            print(f" {board_symbols[3]} | {board_symbols[4]} | {board_symbols[5]} ")
            print("---+---+---")
            print(f" {board_symbols[6]} | {board_symbols[7]} | {board_symbols[8]} ")
            print("\n")

    def winner(self, board) -> int:
        """
        :return 2 for computer winning, 0 for nothing, 1 for  player winning
        """
        lists = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),  # ligne
            (0, 4, 8),
            (2, 4, 6),  # diagonale
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),  # colonne
        ]
        for a, b, c in lists:
            if board[a] == board[b] == board[c] and board[a] != 0:
                return board[a]
        return 0

    @staticmethod
    def empty_cases(board) -> list:
        return [i for i in range(9) if board[i] == 0]

    def minimax(self, board, maximising):
        winner = self.winner(board)
        if winner != 0 or not self.empty_cases(board):
            # Traduction des scores pour le minimax :
            if winner == 1:
                return 1, None   # Le joueur (maximizer) gagne
            elif winner == 2:
                return -1, None  # L'ordi (minimizer) gagne
            else:
                return 0, None   # Match nul

        best_score = float("-inf") if maximising else float("inf")
        best_move = None

        for i in self.empty_cases(board):
            l_board = board.copy()
            l_board[i] = 1 if maximising else 2

            l_score, _ = self.minimax(l_board, not maximising)
            if maximising:
                if l_score > best_score:
                    best_score = l_score
                    best_move = i
            else:
                if l_score < best_score:
                    best_score = l_score
                    best_move = i

        return best_score, best_move

    def epsilon_minimax(self, board, maximising):
        if random.uniform(0, 1) < self.epsilon:
            return None, random.choice(self.empty_cases(board))

        winner = self.winner(board)
        if winner != 0 or not self.empty_cases(board):
            if winner == 1:
                return 1, None
            elif winner == 2:
                return -1, None
            else:
                return 0, None

        best_score = float("-inf") if maximising else float("inf")
        best_move = None

        for i in self.empty_cases(board):
            l_board = board.copy()
            l_board[i] = 1 if maximising else 2
            
            if maximising:
                l_score, _ = self.minimax(l_board, False)
                if l_score > best_score:
                    best_score = l_score
                    best_move = i
            else:
                l_score, _ = self.minimax(l_board, True)
                if l_score < best_score:
                    best_score = l_score
                    best_move = i

        return best_score, best_move
