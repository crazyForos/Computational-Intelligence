### Implementations ###
#### Q_learning ####
A basic Q-learning implementation trained without the Bellman recoursive equation, but with the trajectory in which I use only the states of the Q-learning player.
It requaires a lot of memory, so the state and the action are compressed.

#### MinMax ####    
A standard MinMax algorithm with a maximum depth = 4 to maintain acceptable execution times per move, the evaluation function is made first evaluating if the state is a final state, otherwise counting the number of sign for each player and how many 4 sign in a row they have.

#### MinMax tuned with GA #### 
The same implementation as before but the weights of the different rules of the evaluation function are tuned by a GA, since the execution time will be too large in the training step the deoth is reduced to 1

#### MinMax pruned ####
Adding pruning when a state is too bad (so too small for max and too large for min), this reduce the exploration of the MinMax tree and so it's possible to use a depth = 5

#### MinMax Hashing ####  
The idea is to change the evaluation function exploiting some knowladge, first I created a dataset of random states and evaluate them with a MonteCarlo aproach, then the dataset is increased by the simmetries (and translated for player 0 or 1 invereting the zeros and the ones). Then the evaluation function of the MinMax should be based on the nearest state in particular by the formula $\alpha [f(CurrentState) - f(NearestState)] + (1-\alpha) (\frac{win-loss}{total})$ where f is the same evaluation function as before, win, loss and total are the value of the MonteCarlo simulation associated with the nearest state and $\alpha$ is a parameter, the meaning of the formula is that we in same way know the empirical value of the nearest state (based on the MonteCarlo simulation) so we use that value, but since the two state are not the same, this evaluation is not correct for the Current State, so the first term of the formula tries to correct the differences by the two state.
However, is not possible to search in the complete dataset to find the nearest solution (too expensive) so I created a dictionary in which "similar" state should be in the same entry, in this way we only have to compare with few element.
To create this dictionary I hashed the state in the following method:
* take a subset of the elements of the state
* hash: sum of the elements
* repeat for n times, creating n different hash

So a dictionary will have with a key in the form $$(Hash(sub_1), Hash(sub_2),..., Hash(sub_n))$$ 
This hash is not a classical hash, in fact it's aim is to maintaing in the same spot nearest state.
I created 4 hash, and repeated the process for 5 times creating 5 different dizionaries.

So I compared searching on all the dataset and with this method and I obtained: $$avgDistDataset : 6.07 \ \ \ avgExecutionTime: 2.03s$$ $$avgDistDictionaries : 7.78 \ \ \ avgExecutionTime: 0.003s$$

#### Evaluation ####
I evaluated vs the random player and all the agent win evry games, except for the Q-learning(first and second):

* Win: 32.5% - 28.8%
* Lose: 26.8% - 30.4%
* tie: 40.7% - 40.8%

This probably because the states are too many (also implementing simmetries probably is not sufficient because they reduce only by an 8 factor).

Agent vs Agent winner:
| VS        |Player0  | MinMax    | MinMaxGA  | MinMaxCut | MinMaxHash| Q-learning|
|-----------|  -      |-----------|-----------|-----------|-----------|-----------|
|Player1    |  -      | -         |  -        | -         | -         | -         |
| MinMax    |  -      | -         |  Player0  | Player0   | Player0   | Player1   |
| MinMaxGA  |  -      | Player1   |    -      | Player0   | Player1   | Player1   |
| MinMaxCut |  -      | Player1   | Player1   |     -     | Tie       | Player1   |
| MinMaxHash|  -      | Player1   | Player1   | Tie       |   -       | Player1   |
| Q-learning|  -      | Player0   | Player0   | Player0   | Player0   |    -      |


#### Note ####
* I slighty changed the game to ensure to make it episodic, so after 50 moves it's considered a tie.
* I changed the move method without inverting rows and columns because it confused me
* I cannot charge in the GitHub the Dataset and the Q_table because they are too large 

#### Resorces #### 
* The previous lab
* ChatGpt for compressing the states
* I have simplified and adapted the ideas of the LSH
* For MinMax: https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python/
* Worked with Alessio Cappello (s309450)
