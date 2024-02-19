## EXAM PROJECT - QUIXO
### Comment:
My player exploits the minimax algorithm as its strategy. In particular i implemented an alpha-beta pruned version of it, to limit the tree exploration width if I'm able to decide before that a particular branch is not gonna be worth exploring as I know I'm going anyway to opt for another, better, decision.
In fact, every time my player has to make a move it evaluates the possible future consequences of its possible moves, given the actual state of the game board, in order to decide which move gives it the highest probability to win.
Minimax, in fact is a strategy in which the player that implements it wants to maximize its probability of winning while assuming that the other player is going to make itself the best decision for it.

In order to do so, the minimax function is a recursive function which explore all the possible future states, given an action and and actual state on which it is performed. In order to be 100% accurate and give certain results, it would need to go in depth of the (action -> state) tree until my player or the opponent wins or there is a draw.

The fact is that quixo is a "possible never ending game" and, also, exploring all  the possible outcomes for all the possible moves that my player is going to make in order to give them a certain score is in practice impossible to realize.

So, what I needed was to first define a way in order to limit the width of my tree, and alpha beta pruning came into rescue. Then I needed an evaluation function that, given a certain non-ending state, gives me anyway a score with a certain probability.

I came up with the idea of an evaluation function that checks obviously if one of hte consequences of my action brings me or the opponent to a win and gives it an according score. Another check that i make, in a non-game-ending state, is by checking if my player or the other one took the central position of the board and i also check whether there are four of my blocks or of the other player aligned whether on a line, column or diagonal, giving according scores. Finally i also check the overall number of blocks my player and the other's in order to see which "controls" more board overall.

To reduce the number of actions that my player or the other are trying to make in my minimax function I limit the search to the only margin of the game board and the only moves that can be effectively done i.e. my player cannot choose a block that has the pther player's sign on it or slide from the top when the chosen block comes from the top of the board.

Finally, I cut the depth of the tree to 2 so I'm evaluating the states only two moves ahead of the current move, obiously if in the meantime some branches haven't reached a winning state.

### Results:
Overall, I got a winning rate over the RandomPlayer() of nearly 97% by playing 50% of the times as firstPlayer and 50% as second on 1000 games.