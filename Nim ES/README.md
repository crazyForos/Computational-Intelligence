### Genome ###
I have tried three different strategy:
* Only two gene: the first is a real-valued number that means the "percentuage" index respect of remaining rows and it rappresents the row that we will choose; the second instead means the "percentuage" respect the remaining number in that row that we will take.
* Similar to the previous one, it has the same comportament as before but we split it in function of an another value. It's possible to choose different function.
* A simple Feed Foward Neural Network with only two layers, so two matrix and two vectors.

### Tweak ###

I implemented the standard version and the self-adapt version in which every gene have it's variance.

### Fitness ###

I implemented two version:
* The first simulate n games against a random startegy returns how many the individual wins.
* The second simulate all games beetwen the individual in the population + offspring.

#### Note #### 
1. The second strategy is a generalization of the first, but seems to converge to the same solution, so it's probably useless (but maybe there is a function that make an increase on performance).
1. The first fitness seems more "powerfull", since usually we converge to a "better" solution.
1. Every generation population and offsprings are merge together and then the fitness is evaluated this because:
    * If we use the first fitness we can recalculated fitness of previous generation, since the fitness is slightly random this make a greater probability to calculate it correctly.
    * I we use the second we want to evaluate also the games from the previous generation and the offspring.





