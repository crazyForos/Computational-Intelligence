import numpy as np
from itertools import permutations
from tqdm import tqdm
from game import Game, Move, Player
from copy import deepcopy
import random
import sys
import struct


class Qlearning:
    def __init__(self, alpha, gamma, epsilon, player):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.player= player

    def compact_string(self, matrix):
        matrix = matrix.flatten()
        compressed_data = struct.pack(f">{len(matrix)}b", *matrix)
              
        return compressed_data
    
    def compact_move(self,t):
        string=str(t[0][0])+str(t[0][1])+str(t[1].value)
        string = string.encode('utf-8')
        return string
    
    def decode_move(self,t):
        t = t.decode('utf-8')
        return (int(t[0]), int(t[1])), Move(int(t[2]))
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_q_table(self):
        return self.q_table
    
    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]
    
    def choice_action(self, state, actions):
        if np.random.uniform() < self.epsilon:
            return actions[np.random.choice(range(len(actions)))]
        else:
            state = self.compact_string(state)
            actions = [self.compact_move(action) for action in actions]
            q_values = np.array([self.get_q_value(state, action) for action in actions])
            maximum = np.max(q_values)
            return self.decode_move(actions[np.random.choice(np.where(q_values == maximum)[0])])
    
    def reward(self, win):
        if win == self.player:
            return 1
        elif win == 1 - self.player:
            return -1
        else:
            return 0
        
    def update(self, trajectory, reward):
        for state, action in trajectory:
            state = self.compact_string(state)
            action = self.compact_move(action)
            self.q_table[(state, action)] = self.get_q_value(state, action) + self.alpha * (reward - self.get_q_value(state, action))
            reward = reward * self.gamma

        


if __name__ == '__main__':
    Q = Qlearning(0.5, 0.9, 1, 1)
    games = 250000
    epsilon = np.linspace(1, 0, num=games, endpoint=True)


    for i in tqdm(range(games)):
        Q.set_epsilon(epsilon[i])
        game = Game()
        player = 0
        trajectory = []
        
        while game.check_winner()==-1:
            state = game.get_board()
            actions = game.possible_moves(player)
            action = random.choice(actions)
            game.move(action[0], action[1], player)
            
            
            if game.check_winner()!=-1:
                break
            player = 1 - player

            state = game.get_board()
            actions = game.possible_moves(player)
            action = Q.choice_action(state, actions)
            game.move(action[0], action[1], player)
            trajectory.append((state, action))

            player = 1 - player
        
        Q.update(trajectory, Q.reward(game.check_winner()))
        trajectory = []
    

    
    with open('Quixo\\Q_table_1.txt', 'w') as f:
        sys.stdout = f 
        for key, value in Q.get_q_table().items():
            print(key[0], key[1], value)