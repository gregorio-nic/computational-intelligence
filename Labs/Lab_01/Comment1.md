# LABORATORY 1 - A*
## Some comment related to my A* algorithm

I've worked with Lorenzo Ugoccioni to reach this heuristic function h().
We shifted our attention to the part of the set that still needs to be covered. So, at each step, h() computes the number of tiles that would remain uncovered with the current_set (previous step best_set + one of the not_taken ).

In a first analysis we thought it was a good solution, but we ran into some inconvenience:

The first set taken will influence future choices, even leading the algorithm to using more sets than necessary. In fact, the solution identified (also not correctly) as best, will always be the one on top of the Priority Queue.

Unfortunately, we thought this was a parallel solution to what Professor Squillero showed us in class (h3 function) but it turned out not to be so.