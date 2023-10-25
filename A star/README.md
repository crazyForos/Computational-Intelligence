## Heuristics ##
Explanation of the heuristcs that I used:
* SimpleDistance: if we are at the goal h(state) = 0 otherwise check if one set is enough to cover all the remaining set, if it is so h(state) = 1 else h(state) = 2. Note that is possible to interate this concept and verify if 2 is enough, but the computational complexity of h increase. 
* LenDistance: the same of the professor h3.
* FastDistance: the same as LenDistance but we calculate all the sums at the start and we don't recalculate for the specific state, it's faster but less precise.
* ImbalanceDistance and RandomSamplingDistance: they are similar and work like LenDistance, the difference is that this two heuristics check what's the minimum cost to cover only a subsets of the 'column', so not to solve the problem but only to solve a part of that, this could be usefull because since we sum the lenght using less column we reduce these lenghts. In specific:
    * ImbalanceDistance: takes only the columns with the smaller number of sets that have that colums to True.
    * RandomSamplingDistance: takes columns at random.

    Note: we have to provide a function that returns the number of columns to take. For ImbalanceDistance I have used $$   \lfloor \frac{x}{2} \rfloor \hspace{1cm} and \hspace{1cm}\lfloor \sqrt x \rfloor $$ , for the RandomSamplingDistance a normal distribution  $$ \mathcal{N}(\lfloor \frac{2x}{3} \rfloor , \lfloor \frac{x}{4} \rfloor) $$ but restricted between $ 1 $ and $ x $

#### Results ####
The best in number of node expanded are LenDistance, RandomSamplingDistance and ImbalanceDistance. The first and the second work about the same, however the RandomSamplingDistance is simple to slightly improve increasing the number of tries, but also it became more slowly (it's possible to find a good trade off). ImbalanceDistance performs better if the sets are imbalanced, so there are some columns with less True than others.