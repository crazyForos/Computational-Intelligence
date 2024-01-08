import random
from game import Game, Move, Player
from random import random, choice, randint, sample, shuffle, uniform
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import itertools

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos, move=choice(game.possible_moves(game.get_current_player()))
        return from_pos, move
    
class MinMaxPlayer(Player):
    def __init__(self, player: int, genotype: list[float]) -> None:
        super().__init__()
        self.player = player
        self.genotype = genotype
    
    def evaluate(self, game: 'Game') -> int:
        evaluation = 0
        if game.check_winner() == 0:
            evaluation= 1000
        elif game.check_winner() == 1:
            evaluation = -1000
        else:
            if self.genotype[0]*fase(game) < 13:
                evaluation = self.genotype[1]*rule1(game) + self.genotype[2]*rule2(game) + self.genotype[3]*rule3(game)
            else:
                evaluation = self.genotype[4]*rule1(game) + self.genotype[5]*rule2(game) + self.genotype[6]*rule3(game)
        
        return evaluation if self.player == 0 else -evaluation
    
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        best_move = None
        best_score = -np.inf
        for move in game.possible_moves(self.player):
            g = deepcopy(game)
            g.move(move[0], move[1], g.get_current_player())
            score = self.evaluate(g)
            
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

def rule1(game: 'Game'):
    board = game.get_board()
    number_of_player_0 = 0
    number_of_player_1 = 0
    for i in range(5):
        for j in range(5):
            if board[i][j] == 0:
                number_of_player_0 += 1
            elif board[i][j] == 1:
                number_of_player_1 += 1
    return number_of_player_0 - number_of_player_1
    


def rule2(game: 'Game'):
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
    board = game.get_board()
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

    return number_of_4_player_0 - number_of_4_player_1

def rule3(game: 'Game'):
    number_of_3_player_0=0
    number_of_3_player_1=0
    row_player_0 = [0, 0, 0, 0, 0]
    row_player_1 = [0, 0, 0, 0, 0]
    column_player_0 = [0, 0, 0, 0, 0]
    column_player_1 = [0, 0, 0, 0, 0]
    diag_player_1=0
    diag_player_0=0
    anti_diag_player_0=0
    anti_diag_player_1=0
    board = game.get_board()
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
    
    return number_of_3_player_0 - number_of_3_player_1

def fase(game: 'Game'):
    board = game.get_board()
    number_of_sign = 0
    for i in range(5):
        for j in range(5):
            if board[i][j] == 0 or board[i][j] == 1:
                number_of_sign += 1
    return number_of_sign

NUM_GEN=20
TOURNAMENT_SIZE = 2
MUTATION_PROBABILITY = .15
POPULATION_SIZE=15
OFFSPRING_SIZE=18

@dataclass
class Individual:
    fitness: float
    genotype: list[float]

def select_parent(pop):
    pool = [choice(pop) for _ in range(TOURNAMENT_SIZE)]
    champion = max(pool, key=lambda i: i.fitness)
    return champion

def mutate(ind: Individual) -> Individual:
    offspring = deepcopy(ind)
    pos = randint(0, len(ind.genotype)-1)
    offspring.genotype[pos] += np.random.normal(0, 3)
    offspring.fitness = None
    return offspring

def uniform_xover(ind1: Individual, ind2: Individual) -> Individual:
    offspring = Individual(fitness=None,
                           genotype=[(g1+g2)/2
                                     for g1, g2 in zip(ind1.genotype, ind2.genotype)])
    return offspring

def fitness(population):
    for ind in population:
        ind.fitness= 0
    product = itertools.product(population, population)
    for (ind1,ind2) in tqdm(product, total=len(population)**2, desc='Fitness'):
        if ind1==ind2:
            continue
        player1 = MinMaxPlayer(genotype= ind1.genotype, player=0)
        player2 = MinMaxPlayer(genotype= ind2.genotype, player=1)       
        g = Game()
        winner=g.play(player1, player2)
        if winner==0:
            ind1.fitness +=1    
        elif winner==1:
            ind2.fitness +=1
        else:
            ind1.fitness -=0.1
            ind2.fitness +=0.1


        

population = [
    Individual(
        genotype=[uniform(-10.0, 10.0) for _ in range(7)],
        fitness=None,
    )
    for _ in range(POPULATION_SIZE)
]

fitness(population)
for ind in population:
    print(ind.fitness)

for generation in tqdm(range(NUM_GEN), desc='Generation'):
    offspring = list()
    for counter in range(OFFSPRING_SIZE):
        if random() < MUTATION_PROBABILITY:  # self-adapt mutation probability
            # mutation  # add more clever mutations
            p = select_parent(population)
            o = mutate(p)
        else:
            # xover # add more xovers
            p1 = select_parent(population)
            p2 = select_parent(population)
            o = uniform_xover(p1, p2)
        offspring.append(o)

    population.extend(offspring)
    fitness(population)
    population.sort(key=lambda i: i.fitness, reverse=True)
    population = population[:POPULATION_SIZE]
    print(f'Best fitness: {population[0].fitness} at generation {generation}')
print(population[0].genotype)