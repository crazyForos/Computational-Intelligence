import random
from game import Game, Move, Player
from copy import deepcopy, copy
import math
from tqdm import tqdm
import numpy as np


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos, move=random.choice(game.possible_moves(game.get_current_player()))

class QLPlayer(Player):
    def __init__(self, id: int, alfa=0.5, gamma=0.8, epsilon=1):
        super().__init__()
        self.player_id = id
        self.alfa = alfa
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = dict()

    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    
    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]
    

    def choice_action(self, state, actions):
        if np.random.uniform() < self.epsilon:
            return actions[np.random.choice(range(len(actions)))]
        else:
            q_values = np.array([self.get_q_value(state, action) for action in actions])
            maximum = np.max(q_values)
            return actions[np.random.choice(np.where(q_values == maximum)[0])]
        
    
    def update(self, state, action, reward, next_state, next_actions):
        q_value = self.get_q_value(state, action)
        next_q_values = np.array([self.get_q_value(next_state, next_action) for next_action in next_actions])
        maximum = np.max(next_q_values) if len(next_q_values) > 0 else 0
        self.q_table[(state, action)] = q_value + self.alfa * (reward + self.gamma * maximum - q_value)


    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        state = game.get_board()
        actions = game.possible_moves(self.player_id)
        from_pos, slide = self.choice_action(str(state), actions) # >> controllare
        return from_pos, slide
    

    def train_as0(self, episodes=50000):
        epsilon = np.linspace(1, 0.1, num=episodes, endpoint=True)
        players = [self, RandomPlayer()]

        for i in tqdm(range(episodes)):
            self.set_epsilon(epsilon[i])
            g = Game()
            winner = -1
            while winner < 0:
                g.current_player_idx += 1
                g.current_player_idx %= 2
                ok = False
                while not ok:
                    from_pos, slide = players[g.current_player_idx].make_move(
                        g)
                    state = g.get_board()
                    action = (from_pos, slide)
                    ok = g.move(from_pos, slide, g.current_player_idx)
                winner = g.check_winner()
                if winner != -1:
                    if g.current_player_idx == self.player_id:
                        next_state = g.get_board()
                        next_actions = g.possible_moves(self.player_id) 
                        reward = 1 if winner == self.player_id else -1
                        self.update(str(state), action, reward, str(next_state), next_actions)
                else:
                    g.current_player_idx += 1
                    g.current_player_idx %= 2
                    ok = False
                    reward = 0
                    while not ok:
                        from_pos, slide = players[g.current_player_idx].make_move(
                            g)
                        ok = g.move(from_pos, slide, g.current_player_idx)
                    winner = g.check_winner()
                    if winner != -1:
                        reward = 1 if winner == self.player_id else -1
                    next_state = g.get_board()
                    next_actions = g.possible_moves(self.player_id)
                    self.update(str(state), action, reward, str(next_state), next_actions)

Q=QLPlayer(0)
Q.train_as0()





                



    