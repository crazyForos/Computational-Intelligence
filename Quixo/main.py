import random
from game import Game, Move, Player
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import struct

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos, move=random.choice(game.possible_moves(game.get_current_player()))
        return from_pos, move
    
class QlearningPlayer(Player):
    def __init__(self, player) -> None:
        super().__init__()
        self.q_table= {}
        self.player=player
        file = "Quixo\\Q_table_0.txt" if self.player==0 else "Quixo\\Q_table_1.txt"
        print("Charging Q_table...")
        with open(file, 'r') as f:
            for line in f:
                line = line.split(" ")
                self.q_table[(line[0], line[1])] = float(line[2])

    def get_q_table(self):
        return self.q_table

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
    
    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]
    
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        player = game.get_current_player()
        actions = game.possible_moves(player)
        state = game.get_board()
        state = self.compact_string(state)
        actions = [self.compact_move(action) for action in actions]
        q_values = np.array([self.get_q_value(state, action) for action in actions])
        maximum = np.max(q_values)
        return self.decode_move(actions[np.random.choice(np.where(q_values == maximum)[0])])
            
    
        


class MinMaxPlayer(Player):
    def __init__(self, player: int) -> None:
        super().__init__()
        self.player = player
        self.DEPTH = 4

    def evaluate(self, game: 'Game') -> int:
        evaluation = 0
        if game.check_winner() == 0:
            evaluation= 1000
        elif game.check_winner() == 1:
            evaluation = -1000
        else:
            board = game.get_board()
            number_of_player_0 = 0
            number_of_player_1 = 0
            for i in range(5):
                for j in range(5):
                    if board[i][j] == 0:
                        number_of_player_0 += 1
                    elif board[i][j] == 1:
                        number_of_player_1 += 1
            evaluation = number_of_player_0 - number_of_player_1

            number_of_4_player_0= 0
            number_of_4_player_1= 0
            row_player_0 = [0, 0, 0, 0, 0]
            row_player_1 = [0, 0, 0, 0, 0]
            column_player_0 = [0, 0, 0, 0, 0]
            column_player_1 = [0, 0, 0, 0, 0]
            diag_player_1=0
            diag_player_0=0
            anti_diag_player_0=0
            anti_diag_player_1=0
            for i in range(5):
                for j in range(5):
                    if board[i][j] == 0:
                        row_player_0[i] += 1
                        column_player_0[j] += 1
                    elif board[i][j] == 1:
                        row_player_1[i] += 1
                        column_player_1[j] += 1

                    if i==j:
                        if board[i][j] == 0:
                            diag_player_0 += 1
                        elif board[i][j] == 1:
                            diag_player_1 += 1
                    
                    if i+j==4:
                        if board[i][j] == 0:
                            anti_diag_player_0 += 1
                        elif board[i][j] == 1:
                            anti_diag_player_1 += 1

            for i in range(5):
                if row_player_0[i] == 4:
                    number_of_4_player_0 += 1
                elif row_player_1[i] == 4:
                    number_of_4_player_1 += 1
                if column_player_0[i] == 4:
                    number_of_4_player_0 += 1
                elif column_player_1[i] == 4:
                    number_of_4_player_1 += 1
            
            if diag_player_0 == 4:
                number_of_4_player_0 += 1
            elif diag_player_1 == 4:
                number_of_4_player_1 += 1
            
            if anti_diag_player_0 == 4:
                number_of_4_player_0 += 1
            elif anti_diag_player_1 == 4:
                number_of_4_player_1 += 1

            evaluation += 3* (number_of_4_player_0 - number_of_4_player_1)
        return evaluation if self.player == 0 else -evaluation
        
    def max(self, game: 'Game', depth: int, alpha: int, beta: int):
        if depth == 0 or game.check_winner() != -1:
            return self.evaluate(game)
        max_value = -10000

        for move in game.possible_moves(game.get_current_player()):
            game_copy = deepcopy(game)
            game_copy.move(move[0], move[1], game_copy.get_current_player())
            minimum = self.min(game_copy, depth - 1, alpha, beta)
            if max_value < minimum:
                best_move= (move[0],move[1])
                max_value = minimum
            
            if max_value >= beta:
                break

            if max_value > alpha:
                alpha = max_value
        
        if depth == self.DEPTH:
            return best_move

        return max_value
    
    def min(self, game: 'Game', depth: int, alpha: int, beta: int) -> int:
        if depth == 0 or game.check_winner() != -1:
            return self.evaluate(game)
        min_value = 10000

        for move in game.possible_moves(game.get_current_player()):
            game_copy = deepcopy(game)
            game_copy.move(move[0], move[1],  game_copy.get_current_player())
            min_value = min(min_value, self.max(game_copy, depth - 1, alpha, beta))

            if min_value <= alpha:
                break   

            if min_value < beta:
                beta = min_value

        return min_value

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos, move = self.max(game, self.DEPTH, -10000, 10000)
        return from_pos, move

class MinMaxPlayerGA(Player):
    def __init__(self, player: int) -> None:
        super().__init__()
        self.player = player
        self.DEPTH= 4

    def evaluate(self, game: 'Game') -> int:
        evaluation = 0
        if game.check_winner() == 0:
            evaluation= 1000
        elif game.check_winner() == 1:
            evaluation = -1000
        else:
            board = game.get_board()
            number_of_player_0 = 0
            number_of_player_1 = 0
            for i in range(5):
                for j in range(5):
                    if board[i][j] == 0:
                        number_of_player_0 += 1
                    elif board[i][j] == 1:
                        number_of_player_1 += 1
            evaluation = 13.29*(number_of_player_0 - number_of_player_1)

            number_of_4_player_0= 0
            number_of_4_player_1= 0
            row_player_0 = [0, 0, 0, 0, 0]
            row_player_1 = [0, 0, 0, 0, 0]
            column_player_0 = [0, 0, 0, 0, 0]
            column_player_1 = [0, 0, 0, 0, 0]
            diag_player_1=0
            diag_player_0=0
            anti_diag_player_0=0
            anti_diag_player_1=0
            for i in range(5):
                for j in range(5):
                    if board[i][j] == 0:
                        row_player_0[i] += 1
                        column_player_0[j] += 1
                    elif board[i][j] == 1:
                        row_player_1[i] += 1
                        column_player_1[j] += 1

                    if i==j:
                        if board[i][j] == 0:
                            diag_player_0 += 1
                        elif board[i][j] == 1:
                            diag_player_1 += 1
                    
                    if i+j==4:
                        if board[i][j] == 0:
                            anti_diag_player_0 += 1
                        elif board[i][j] == 1:
                            anti_diag_player_1 += 1

            number_of_3_player_0=0
            number_of_3_player_1=0

            for i in range(5):
                if row_player_0[i] == 4:
                    number_of_4_player_0 += 1
                elif row_player_1[i] == 4:
                    number_of_4_player_1 += 1
                if column_player_0[i] == 4:
                    number_of_4_player_0 += 1
                elif column_player_1[i] == 4:
                    number_of_4_player_1 += 1
            
            if diag_player_0 == 4:
                number_of_4_player_0 += 1
            elif diag_player_1 == 4:
                number_of_4_player_1 += 1
            
            if anti_diag_player_0 == 4:
                number_of_4_player_0 += 1
            elif anti_diag_player_1 == 4:
                number_of_4_player_1 += 1

            for i in range(5):
                if row_player_0[i] == 3:
                    number_of_3_player_0 += 1
                elif row_player_1[i] == 3:
                    number_of_3_player_1 += 1
                if column_player_0[i] == 3:
                    number_of_3_player_0 += 1
                elif column_player_1[i] == 3:
                    number_of_3_player_1 += 1
            
            if diag_player_0 == 3:
                number_of_3_player_0 += 1
            elif diag_player_1 == 3:
                number_of_3_player_1 += 1
            
            if anti_diag_player_0 == 3:
                number_of_3_player_0 += 1
            elif anti_diag_player_1 == 3:
                number_of_3_player_1 += 1

            evaluation += 3.98* (number_of_4_player_0 - number_of_4_player_1) + 0.58*(number_of_3_player_0 - number_of_3_player_1)
        return evaluation if self.player == 0 else -evaluation
        
    def max(self, game: 'Game', depth: int, alpha: int, beta: int):
        if depth == 0 or game.check_winner() != -1:
            return self.evaluate(game)
        max_value = -10000

        for move in game.possible_moves(game.get_current_player()):
            game_copy = deepcopy(game)
            game_copy.move(move[0], move[1], game_copy.get_current_player())
            minimum = self.min(game_copy, depth - 1, alpha, beta)
            if max_value < minimum:
                best_move= (move[0],move[1])
                max_value = minimum
            
            if max_value >= beta:
                break

            if max_value > alpha:
                alpha = max_value
        
        if depth == self.DEPTH:
            return best_move

        return max_value
    
    def min(self, game: 'Game', depth: int, alpha: int, beta: int) -> int:
        if depth == 0 or game.check_winner() != -1:
            return self.evaluate(game)
        min_value = 10000

        for move in game.possible_moves(game.get_current_player()):
            game_copy = deepcopy(game)
            game_copy.move(move[0], move[1],  game_copy.get_current_player())
            min_value = min(min_value, self.max(game_copy, depth - 1, alpha, beta))

            if min_value <= alpha:
                break   

            if min_value < beta:
                beta = min_value

        return min_value

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos, move = self.max(game, self.DEPTH, -10000, 10000)
        return from_pos, move


class MinMaxPlayerGACut(Player):
    def __init__(self, player: int) -> None:
        super().__init__()
        self.player = player
        self.DEPTH=5

    def evaluate(self, game: 'Game') -> int:
        evaluation = 0
        if game.check_winner() == 0:
            evaluation= 1000
        elif game.check_winner() == 1:
            evaluation = -1000
        else:
            board = game.get_board()
            number_of_player_0 = 0
            number_of_player_1 = 0
            for i in range(5):
                for j in range(5):
                    if board[i][j] == 0:
                        number_of_player_0 += 1
                    elif board[i][j] == 1:
                        number_of_player_1 += 1
            evaluation = 13.29*(number_of_player_0 - number_of_player_1)

            number_of_4_player_0= 0
            number_of_4_player_1= 0
            row_player_0 = [0, 0, 0, 0, 0]
            row_player_1 = [0, 0, 0, 0, 0]
            column_player_0 = [0, 0, 0, 0, 0]
            column_player_1 = [0, 0, 0, 0, 0]
            diag_player_1=0
            diag_player_0=0
            anti_diag_player_0=0
            anti_diag_player_1=0
            for i in range(5):
                for j in range(5):
                    if board[i][j] == 0:
                        row_player_0[i] += 1
                        column_player_0[j] += 1
                    elif board[i][j] == 1:
                        row_player_1[i] += 1
                        column_player_1[j] += 1

                    if i==j:
                        if board[i][j] == 0:
                            diag_player_0 += 1
                        elif board[i][j] == 1:
                            diag_player_1 += 1
                    
                    if i+j==4:
                        if board[i][j] == 0:
                            anti_diag_player_0 += 1
                        elif board[i][j] == 1:
                            anti_diag_player_1 += 1

            number_of_3_player_0=0
            number_of_3_player_1=0

            for i in range(5):
                if row_player_0[i] == 4:
                    number_of_4_player_0 += 1
                elif row_player_1[i] == 4:
                    number_of_4_player_1 += 1
                if column_player_0[i] == 4:
                    number_of_4_player_0 += 1
                elif column_player_1[i] == 4:
                    number_of_4_player_1 += 1
            
            if diag_player_0 == 4:
                number_of_4_player_0 += 1
            elif diag_player_1 == 4:
                number_of_4_player_1 += 1
            
            if anti_diag_player_0 == 4:
                number_of_4_player_0 += 1
            elif anti_diag_player_1 == 4:
                number_of_4_player_1 += 1

            for i in range(5):
                if row_player_0[i] == 3:
                    number_of_3_player_0 += 1
                elif row_player_1[i] == 3:
                    number_of_3_player_1 += 1
                if column_player_0[i] == 3:
                    number_of_3_player_0 += 1
                elif column_player_1[i] == 3:
                    number_of_3_player_1 += 1
            
            if diag_player_0 == 3:
                number_of_3_player_0 += 1
            elif diag_player_1 == 3:
                number_of_3_player_1 += 1
            
            if anti_diag_player_0 == 3:
                number_of_3_player_0 += 1
            elif anti_diag_player_1 == 3:
                number_of_3_player_1 += 1

            evaluation += 3.98* (number_of_4_player_0 - number_of_4_player_1) + 0.58*(number_of_3_player_0 - number_of_3_player_1)
        return evaluation if self.player == 0 else -evaluation
        
    def max(self, game: 'Game', depth: int, alpha: int, beta: int):
        evaluation = self.evaluate(game)
        if depth == 0 or game.check_winner() != -1 or (evaluation < -5 and depth != self.DEPTH):
            return evaluation
        max_value = -10000

        for move in game.possible_moves(game.get_current_player()):
            game_copy = deepcopy(game)
            game_copy.move(move[0], move[1], game_copy.get_current_player())


            minimum = self.min(game_copy, depth - 1, alpha, beta)
            if max_value < minimum:
                best_move= (move[0],move[1])
                max_value = minimum
            
            if max_value >= beta:
                break

            if max_value > alpha:
                alpha = max_value
        
        if depth == self.DEPTH:
            return best_move

        return max_value
    
    def min(self, game: 'Game', depth: int, alpha: int, beta: int) -> int:
        evaluation = self.evaluate(game)
        if depth == 0 or game.check_winner() != -1 or evaluation > 5:
            return evaluation
        min_value = 10000

        for move in game.possible_moves(game.get_current_player()):
            game_copy = deepcopy(game)
            game_copy.move(move[0], move[1],  game_copy.get_current_player())
            min_value = min(min_value, self.max(game_copy, depth - 1, alpha, beta))

            if min_value <= alpha:
                break   

            if min_value < beta:
                beta = min_value

        return min_value

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos, move = self.max(game, self.DEPTH, -10000, 10000)
        return from_pos, move
    
class MinMaxPlayerDataset(Player):
    def __init__(self, player: int) -> None:
        super().__init__()
        self.player = player
        self.DEPTH=4
        self.alpha = 0.4
        print("Charging dataset...")
        self.dataset_0, self.dataset_1 = self.charge_dataset(["Quixo\\dataset.txt"])
        print("Creating dictionaries...")
        self.indicies = []
        for _ in range(5):
            l= random.sample(range(25), 25)
            tmp=[]
            for i in range(3):
                tmp.append(l[i*8:(i+1)*8])
            self.indicies.append(tmp)

        self.dizionaries = self.diz(self.dataset_0, self.indicies) if self.player==0 else self.diz(self.dataset_1, self.indicies)
        self.dataset_0 = {}
        self.dataset_1 = {}
        print("Done!")


    def charge_dataset(self, names):
        dataset_0 = {}
        dataset_1 = {}
        i=0
        for name in names:
            with open(name, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        line = line.split("))")
                        line[0] = line[0][2:]
                        key1= line[0].split(")")
                        key1 = [key1[i].replace(", (", "(")[1:] for i in range(len(key1))]
                        key1 = tuple(tuple(map(int, key1[i].split(", "))) for i in range(len(key1)))
                        key2 = int(line[1][2])
                        value = line[1][6:-1].split(", ")
                        value = tuple(map(int, value))
                        simmetries_0, simmetries_1 = self.simmetry(key1, key2, value)
                        for (key1, value) in simmetries_0:
                            if key1 in dataset_0:
                                dataset_0[key1] = tuple(map(sum, zip(dataset_0[key1], value)))
                            else:
                                dataset_0[key1] = value

                        for (key1, value) in simmetries_1:
                            if key1 in dataset_1:
                                dataset_1[key1] = tuple(map(sum, zip(dataset_1[key1], value)))
                            else:
                                dataset_1[key1] = value 
                    
        return dataset_0, dataset_1
    
    def simmetry(self, key1, key2, value):
        simmetries_0 = []
        simmetries_1=[]
        key1= np.array(key1)
        if key2==0:
            simmetries_0.append((tuple(map(tuple, key1)), value))
            simmetries_0.append((tuple(map(tuple, np.rot90(key1))), value))
            simmetries_0.append((tuple(map(tuple, np.rot90(key1, 2))), value))
            simmetries_0.append((tuple(map(tuple, np.rot90(key1, 3))), value))
            simmetries_0.append((tuple(map(tuple, np.fliplr(key1))), value))
            simmetries_0.append((tuple(map(tuple, np.rot90(np.fliplr(key1)))), value))
            simmetries_0.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 2))), value))
            simmetries_0.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 3))), value))
            
            map1= key1==0
            map2= key1==1
            key1[map1] = 1
            key1[map2] = 0
            value = (value[1], value[0], value[2])
            
            simmetries_1.append((tuple(map(tuple, key1)), value))
            simmetries_1.append((tuple(map(tuple, np.rot90(key1))), value))
            simmetries_1.append((tuple(map(tuple, np.rot90(key1, 2))), value))
            simmetries_1.append((tuple(map(tuple, np.rot90(key1, 3))), value))
            simmetries_1.append((tuple(map(tuple, np.fliplr(key1))), value))
            simmetries_1.append((tuple(map(tuple, np.rot90(np.fliplr(key1)))), value))
            simmetries_1.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 2))), value))
            simmetries_1.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 3))), value))
        else:
            simmetries_1.append((tuple(map(tuple, key1)), value))
            simmetries_1.append((tuple(map(tuple, np.rot90(key1))), value))
            simmetries_1.append((tuple(map(tuple, np.rot90(key1, 2))), value))
            simmetries_1.append((tuple(map(tuple, np.rot90(key1, 3))), value))
            simmetries_1.append((tuple(map(tuple, np.fliplr(key1))), value))
            simmetries_1.append((tuple(map(tuple, np.rot90(np.fliplr(key1)))), value))
            simmetries_1.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 2))), value))
            simmetries_1.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 3))), value))

            map1= key1==0
            map2= key1==1
            key1[map1] = 1
            key1[map2] = 0
            value = (value[1], value[0], value[2])

            simmetries_0.append((tuple(map(tuple, key1)), value))
            simmetries_0.append((tuple(map(tuple, np.rot90(key1))), value))
            simmetries_0.append((tuple(map(tuple, np.rot90(key1, 2))), value))
            simmetries_0.append((tuple(map(tuple, np.rot90(key1, 3))), value))
            simmetries_0.append((tuple(map(tuple, np.fliplr(key1))), value))
            simmetries_0.append((tuple(map(tuple, np.rot90(np.fliplr(key1)))), value))
            simmetries_0.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 2))), value))
            simmetries_0.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 3))), value))


        return simmetries_0, simmetries_1 
    
    def diz(self, dataset, indicies):
        dizionaries=[]
        for index in indicies:
            diz = {}
            for key, value in dataset.items():
                hash = self.hash_sum(key, index) 
                if hash in diz:
                    diz[hash][key]= value
                else:
                    diz[hash] = {key: value}
            dizionaries.append(diz)
        
        return dizionaries
    
    def hash_sum(self,key, vec):
        key = np.array(key).reshape(25)
        hash = []
        for v in vec:
            values=key[v]
            s=0
            s += np.sum(values)

            hash.append(s)
        return tuple(hash)
    
    def distance(self, base, state):
        state = np.array(state)
        state_k= np.array(base)
        diff = np.subtract(state, state_k)
        num_diff_values = np.count_nonzero(diff)
        return num_diff_values

    def value(self, state):
        best_dist = 25
        best_value = (0,0,0)
        best_state = None
        for diz_0, index in zip(self.dizionaries, self.indicies):
            hash = self.hash_sum(state, index) 
            try:
                diz_0[hash]
            except KeyError:
                continue

            for key, value in diz_0[hash].items():
                dist = self.distance(key, state)
                if dist < best_dist:
                    best_dist = dist
                    best_value = value
                    best_state = key
        
        return best_state, best_value
    
    def evaluate(self, game: 'Game') -> int:
        evaluation = 0
        if game.check_winner() == 0:
            evaluation= 1000
        elif game.check_winner() == 1:
            evaluation = -1000
        else:
            board = game.get_board()
            number_of_player_0 = 0
            number_of_player_1 = 0
            for i in range(5):
                for j in range(5):
                    if board[i][j] == 0:
                        number_of_player_0 += 1
                    elif board[i][j] == 1:
                        number_of_player_1 += 1
            evaluation = 13.29*(number_of_player_0 - number_of_player_1)

            number_of_4_player_0= 0
            number_of_4_player_1= 0
            row_player_0 = [0, 0, 0, 0, 0]
            row_player_1 = [0, 0, 0, 0, 0]
            column_player_0 = [0, 0, 0, 0, 0]
            column_player_1 = [0, 0, 0, 0, 0]
            diag_player_1=0
            diag_player_0=0
            anti_diag_player_0=0
            anti_diag_player_1=0
            for i in range(5):
                for j in range(5):
                    if board[i][j] == 0:
                        row_player_0[i] += 1
                        column_player_0[j] += 1
                    elif board[i][j] == 1:
                        row_player_1[i] += 1
                        column_player_1[j] += 1

                    if i==j:
                        if board[i][j] == 0:
                            diag_player_0 += 1
                        elif board[i][j] == 1:
                            diag_player_1 += 1
                    
                    if i+j==4:
                        if board[i][j] == 0:
                            anti_diag_player_0 += 1
                        elif board[i][j] == 1:
                            anti_diag_player_1 += 1

            number_of_3_player_0=0
            number_of_3_player_1=0

            for i in range(5):
                if row_player_0[i] == 4:
                    number_of_4_player_0 += 1
                elif row_player_1[i] == 4:
                    number_of_4_player_1 += 1
                if column_player_0[i] == 4:
                    number_of_4_player_0 += 1
                elif column_player_1[i] == 4:
                    number_of_4_player_1 += 1
            
            if diag_player_0 == 4:
                number_of_4_player_0 += 1
            elif diag_player_1 == 4:
                number_of_4_player_1 += 1
            
            if anti_diag_player_0 == 4:
                number_of_4_player_0 += 1
            elif anti_diag_player_1 == 4:
                number_of_4_player_1 += 1

            for i in range(5):
                if row_player_0[i] == 3:
                    number_of_3_player_0 += 1
                elif row_player_1[i] == 3:
                    number_of_3_player_1 += 1
                if column_player_0[i] == 3:
                    number_of_3_player_0 += 1
                elif column_player_1[i] == 3:
                    number_of_3_player_1 += 1
            
            if diag_player_0 == 3:
                number_of_3_player_0 += 1
            elif diag_player_1 == 3:
                number_of_3_player_1 += 1
            
            if anti_diag_player_0 == 3:
                number_of_3_player_0 += 1
            elif anti_diag_player_1 == 3:
                number_of_3_player_1 += 1

            evaluation += 3.98* (number_of_4_player_0 - number_of_4_player_1) + 0.58*(number_of_3_player_0 - number_of_3_player_1)
        return evaluation if self.player == 0 else -evaluation
    
    def complete_evaluate(self, game: 'Game') -> int:
        if game.check_winner() == 0:
            return 1000
        elif game.check_winner() == 1:
            return -1000
        
        ev_now= self.evaluate(game)
        state, val = self.value(game.get_board()) 
        ev_near = self.evaluate(Game(np.array(state)))
        return self.alpha * (ev_now/200 - ev_near/200 ) + (1-self.alpha)*(((val[0]-val[1])/(val[0]+ val[1] + val[2])))
        
    def max(self, game: 'Game', depth: int, alpha: int, beta: int):
        evaluation = self.evaluate(game) if depth != 0 else self.complete_evaluate(game)
        if depth == 0 or game.check_winner() != -1 or (evaluation < -5 and depth != self.DEPTH):
            return evaluation
        max_value = -10000

        for move in game.possible_moves(game.get_current_player()):
            game_copy = deepcopy(game)
            game_copy.move(move[0], move[1], game_copy.get_current_player())


            minimum = self.min(game_copy, depth - 1, alpha, beta)
            if max_value < minimum:
                best_move= (move[0],move[1])
                max_value = minimum
            
            if max_value >= beta:
                break

            if max_value > alpha:
                alpha = max_value
        
        if depth == self.DEPTH:
            return best_move

        return max_value
    
    def min(self, game: 'Game', depth: int, alpha: int, beta: int) -> int:
        evaluation = self.evaluate(game) if depth != 0 else self.complete_evaluate(game)
        if depth == 0 or game.check_winner() != -1 or evaluation > 5:
            return evaluation
        min_value = 10000

        for move in game.possible_moves(game.get_current_player()):
            game_copy = deepcopy(game)
            game_copy.move(move[0], move[1],  game_copy.get_current_player())
            min_value = min(min_value, self.max(game_copy, depth - 1, alpha, beta))

            if min_value <= alpha:
                break   

            if min_value < beta:
                beta = min_value

        return min_value

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos, move = self.max(game, self.DEPTH, -10000, 10000)
        return from_pos, move

def test():
    players_0 =  [QlearningPlayer(0),MinMaxPlayer(0), MinMaxPlayerGA(0), MinMaxPlayerGACut(0), MinMaxPlayerDataset(0)]
    players_1 =  [QlearningPlayer(1),MinMaxPlayer(1), MinMaxPlayerGA(1), MinMaxPlayerGACut(1), MinMaxPlayerDataset(1)] 

    names= ["Qlearning","MinMax", "MinMaxGA", "MinMaxGACut", "MinMaxDataset"] 

    for player_0 in range(len(players_0)):
        for player_1 in range(len(players_1)):
            if player_0 != player_1:
                win = 0
                loss = 0
                tie = 0
                g = Game()
                g.play(players_0[player_0], players_1[player_1])
                check = g.check_winner()
                if check == 0:
                    win += 1
                elif check == 1:
                    loss += 1
                else:
                    tie += 1
                print("Player 0:", names[player_0], "Vs Player 1:", names[player_1], "Win:", win, "Loss:", loss, "Tie:", tie)  

    for player_0 in range(len(players_0)):
        win = 0
        loss = 0
        tie = 0
        for _ in tqdm(range(1000)):
            
            g = Game()
            g.play(players_0[player_0], RandomPlayer())
            check = g.check_winner()
            if check == 0:
                win += 1
            elif check == 1:
                loss += 1
            else:
                tie += 1
        print("Player 0:", names[player_0], "Vs Player 1:", "Random", "Win:", win, "Loss:", loss, "Tie:", tie)  
    

    for player_1 in range(len(players_1)):
        win = 0
        loss = 0
        tie = 0
        for _ in tqdm(range(1000)):
            
            g = Game()
            g.play(RandomPlayer(), players_1[player_1])
            check = g.check_winner()
            if check == 1:
                win += 1
            elif check == 0:
                loss += 1
            else:
                tie += 1
        print("Player 0:", "Random", "Vs Player 1:", names[player_1], "Win:", win, "Loss:", loss, "Tie:", tie) 
        
if __name__ == '__main__':
    test()

