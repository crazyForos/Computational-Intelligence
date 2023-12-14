### TicTacToe ###
I have implemented a TicTacToe Class that implement all the methods needed to use Qlearning:
* The state is the board.
* The action are all the possible moves.
* The win check is based on the magic square.

### Qlearning ###
I implemented two version: 
* The first one is more similar to the standard approach, but since is a two player game the Qlearning pass from $$the \ state \ \ S_t \ \ to \ the \ state \ \  S_{t+2}$$
This because the intermediate state is played by another palyer, and for learning I used a complete MonteCarlo simulation.

* The second is a method that is based on the fact that the first player tries to maximize the profit and the second tries to minimize it (like MinMax algorithm) so I changed the Bellman equation and the best move for the second player.

#### Exploration-Exploitation #### 

For the choice move function I tried to balance the exploration and exploitation, so based on a epsilon variable we will have 
$$With \ probability \ \ P(\epsilon) \ a \ random \ possible \ move \\ With \ probability \ \ P(1 - \epsilon) \ the \ best \ possible \ move \ based \ on \ the \ Qtable.$$

The epsilon decrese during the training in a linear way, passing from exploration to exploitation.

### Results ###
The first method is usable only for the first or the second player, it seems to converge to a solution in which it can only win or tie. (if played first otherwise it has a small percentage of lose).

The second method is usable for both the player and again seems to converge to a very good agent.

### Reference ### 
* Used the code on Github of Professor Squillero.
* Used only resources as https://plainenglish.io/blog/building-a-tic-tac-toe-game-with-reinforcement-learning-in-python 
* Worked with Alessio Cappello (s309450)

