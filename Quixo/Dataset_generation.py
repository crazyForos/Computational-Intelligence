import random
import sys
from game import Game, Move, Player
from copy import deepcopy
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    games = 200001
    history = {}
    for i in (pbar:=tqdm(range(games),bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}" +
     " [{elapsed}<{remaining}, {rate_noinv_fmt}]" )):
        game = Game()
        player=0

        while game.check_winner()==-1 and random.random() < 0.993:
            actions = game.possible_moves(player)
            action = random.choice(actions)
            game.move(action[0], action[1], player)
            player = 1 - player
            
        if game.check_winner()==-1:
            board = game.get_board()
            hashable_board= tuple(map(tuple, board))
            player_checkpoint = player

            for times in range(200):
                pbar.set_description(f"Iteration {times+1}/200")
                game = Game(deepcopy(board))
                player = player_checkpoint
                
                while game.check_winner()==-1:
                    actions = game.possible_moves(player)
                    action = random.choice(actions)
                    game.move(action[0], action[1], player)
                    player = 1 - player
                
                if game.check_winner()==player_checkpoint:
                    reward= (1,0,0)
                elif game.check_winner()==1-player_checkpoint:
                    reward= (0,1,0)
                else:
                    reward= (0,0,1)

                if (hashable_board, player_checkpoint) in history:
                    history[(hashable_board, player_checkpoint)] = tuple(map(sum, zip(history[(hashable_board, player_checkpoint)], reward)))
                else:
                    history[(hashable_board, player_checkpoint)] = (1,0,0)
            
        if i%1000==0 and i!=0:
            with open(f"history{i//1000}.txt", "w") as f:
                sys.stdout = f
                for key, v in history.items():
                    print(key, v)
                history = {}
            
            
                
    
       
            
        







    
    

