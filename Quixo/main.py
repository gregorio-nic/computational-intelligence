import random
from copy import deepcopy

from game import Game, Move, Player


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        # game.print()
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        # print(f"{game.current_player_idx} trying move {from_pos}, {move}")
        return from_pos, move


class MyPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self.my_player_idx = 0

    def get_my_player_idx(self):
        return self.my_player_idx

    def set_my_player_idx(self, idx):
        self.my_player_idx = idx

    def available_pos(self, game: 'Game'):
        available_pos = []
        # percorro solo i bordi della matrice e seleziono le sole posizioni in cui e' presente -1 o il mio segno
        for i in range(0, 5):
            for j in range(0, 5) if (i == 0 or i == 4) else (0, 4):
                if game.get_board()[j][i] == -1 or game.get_board()[j][i] == game.current_player_idx:
                    available_pos.append((i, j))
        return available_pos

    def available_slides(self, position):
        available_slide = []
        # data la posizione di pick-up scelta mostro solo le slides che posso fare
        if position[0] == 0 and position[1] == 0:
            available_slide = [Move.BOTTOM, Move.RIGHT]
        elif position[0] == 4 and position[1] == 0:
            available_slide = [Move.BOTTOM, Move.LEFT]
        elif position[0] == 0 and position[1] == 4:
            available_slide = [Move.TOP, Move.RIGHT]
        elif position[0] == 4 and position[1] == 4:
            available_slide = [Move.TOP, Move.LEFT]
        elif position[1] == 0:
            available_slide = [Move.BOTTOM, Move.LEFT, Move.RIGHT]
        elif position[1] == 4:
            available_slide = [Move.TOP, Move.LEFT, Move.RIGHT]
        elif position[0] == 0:
            available_slide = [Move.TOP, Move.BOTTOM, Move.RIGHT]
        elif position[0] == 4:
            available_slide = [Move.TOP, Move.BOTTOM, Move.LEFT]
        return available_slide

    def actions(self, game: 'Game'):
        actions = [(x,y) for x in self.available_pos(game) for y in self.available_slides(x)]
        return actions

    # def blocks_player_zero(self, game: 'Game'):
    #     count = 0
    #     for i in range(5):
    #         for j in range(5):
    #             if game.get_board()[i][j] == 0:
    #                 count += 1
    #     return count

    def four_blocks_aligned_my_player(self, game: 'Game'):
        four_blocks_aligned_rows = 0
        for i in range(5):
            for j in range(2):
                if (game.get_board()[i][j] == self.my_player_idx) and (
                        game.get_board()[i][j + 1] == self.my_player_idx) and (
                        game.get_board()[i][j + 2] == self.my_player_idx) and (
                        game.get_board()[i][j + 3] == self.my_player_idx):
                    four_blocks_aligned_rows += 1
                    break

        # print(four_blocks_aligned_rows)

        four_blocks_aligned_cols = 0
        for i in range(5):
            for j in range(2):
                if (game.get_board()[j][i] == self.my_player_idx) and (
                        game.get_board()[j + 1][i] == self.my_player_idx) and (
                        game.get_board()[j + 2][i] == self.my_player_idx) and (
                        game.get_board()[j + 3][i] == self.my_player_idx):
                    four_blocks_aligned_cols += 1
                    break

        # print(four_blocks_aligned_cols)

        four_blocks_aligned_diag_princ = 0
        for i in range(2):
            if (game.get_board()[i][i] == self.my_player_idx) and (
                    game.get_board()[i + 1][i + 1] == self.my_player_idx) and (
                    game.get_board()[i + 2][i + 2] == self.my_player_idx) and (
                    game.get_board()[i + 3][i + 3] == self.my_player_idx):
                four_blocks_aligned_diag_princ += 1
                break

        # print(four_blocks_aligned_diag_princ)

        four_blocks_aligned_diag_sec = 0
        for i in range(2):
            if (game.get_board()[i][4 - i] == self.my_player_idx) and (
                    game.get_board()[i + 1][4 - (i + 1)] == self.my_player_idx) and (
                    game.get_board()[i + 2][4 - (i + 2)] == self.my_player_idx) and (
                    game.get_board()[i + 3][4 - (i + 3)] == self.my_player_idx):
                four_blocks_aligned_diag_sec += 1
                break

        # print(four_blocks_aligned_diag_sec)

        return four_blocks_aligned_rows + four_blocks_aligned_cols + four_blocks_aligned_diag_princ + four_blocks_aligned_diag_sec

    def four_blocks_aligned_other_player(self, game: 'Game'):
        four_blocks_aligned_rows = 0
        for i in range(5):
            for j in range(2):
                if (game.get_board()[i][j] == (self.my_player_idx+1) % 2) and (
                        game.get_board()[i][j + 1] == (self.my_player_idx+1) % 2) and (
                        game.get_board()[i][j + 2] == (self.my_player_idx+1) % 2) and (
                        game.get_board()[i][j + 3] == (self.my_player_idx+1) % 2):
                    four_blocks_aligned_rows += 1
                    break

        # print(four_blocks_aligned_rows)

        four_blocks_aligned_cols = 0
        for i in range(5):
            for j in range(2):
                if (game.get_board()[j][i] == (self.my_player_idx+1) % 2) and (
                        game.get_board()[j + 1][i] == (self.my_player_idx+1) % 2) and (
                        game.get_board()[j + 2][i] == (self.my_player_idx+1) % 2) and (
                        game.get_board()[j + 3][i] == (self.my_player_idx+1) % 2):
                    four_blocks_aligned_cols += 1
                    break

        # print(four_blocks_aligned_cols)

        four_blocks_aligned_diag_princ = 0
        for i in range(2):
            if (game.get_board()[i][i] == (self.my_player_idx+1) % 2) and (
                    game.get_board()[i + 1][i + 1] == (self.my_player_idx+1) % 2) and (
                    game.get_board()[i + 2][i + 2] == (self.my_player_idx+1) % 2) and (
                    game.get_board()[i + 3][i + 3] == (self.my_player_idx+1) % 2):
                four_blocks_aligned_diag_princ += 1
                break

        # print(four_blocks_aligned_diag_princ)

        four_blocks_aligned_diag_sec = 0
        for i in range(2):
            if (game.get_board()[i][4 - i] == (self.my_player_idx+1) % 2) and (
                    game.get_board()[i + 1][4 - (i + 1)] == (self.my_player_idx+1) % 2) and (
                    game.get_board()[i + 2][4 - (i + 2)] == (self.my_player_idx+1) % 2) and (
                    game.get_board()[i + 3][4 - (i + 3)] == (self.my_player_idx+1) % 2):
                four_blocks_aligned_diag_sec += 1
                break

        # print(four_blocks_aligned_diag_sec)

        return four_blocks_aligned_rows + four_blocks_aligned_cols + four_blocks_aligned_diag_princ + four_blocks_aligned_diag_sec



    def blocks_my_player(self, game: 'Game'):
        count = 0
        for i in range(5):
            for j in range(5):
                if game.get_board()[i][j] == self.my_player_idx:
                    count += 1
        return count

    def blocks_other_player(self, game: 'Game'):
        count = 0
        for i in range(5):
            for j in range(5):
                if game.get_board()[i][j] == (self.my_player_idx+1) % 2:
                    count += 1
        return count

    def evaluation_function(self, game: 'Game'):
        score = 0
        board = game.get_board()

        # punteggio positivo in caso di numero maggiore di blocchi in possesso sulla board, viceversa negativo
        if self.blocks_my_player(game) >= self.blocks_other_player(game):
            score += 10
        else:
            score -= 10

        # punteggio positivo in caso il player ottenga la posizione centrale, viceversa negativo
        if board[2][2] == self.my_player_idx:
            score += 10
        if board[2][2] == (self.my_player_idx+1) % 2:
            score -= 10

        # punteggio positivo se player 1 ha allineato 4 caselle
        four_aligned = self.four_blocks_aligned_my_player(game)
        if four_aligned != 0:
             score += 8 * four_aligned

         # punteggio positivo se player 2 ha allineato 4 caselle
        four_aligned = self.four_blocks_aligned_other_player(game)
        if four_aligned != 0:
            score -= 8 * four_aligned

        # game.print()
        # print(score)

        return score

    # TO BE DEFINED
    def minimax(self, game: 'Game', depth, alpha, beta):
        winner = game.check_winner()
        if depth >= 2 or winner != -1:
            if winner == self.my_player_idx:
                # game.print()
                return 100, depth

            elif winner == (self.my_player_idx+1) % 2:
                # game.print()
                return -100, depth

            elif depth >= 2:
                return self.evaluation_function(game), depth

        actions = self.actions(game)

        # maximizing player
        if game.current_player_idx == self.my_player_idx:
            best_score = -10_000
            best_depth = 10
            for a in actions:
                game_copied = deepcopy(game)
                game_copied._Game__move(a[0], a[1], game_copied.current_player_idx)
                game_copied.current_player_idx = (self.my_player_idx+1) % 2
                score, curr_depth = self.minimax(game_copied, depth+1, alpha, beta)
                if score > best_score or (score == best_score and curr_depth < best_depth):
                    best_score = score
                    best_action = a
                    best_depth = curr_depth
                alpha = max(alpha, score)
                if beta <= alpha:
                    break

        # minimizing player
        if game.current_player_idx == (self.my_player_idx+1) % 2:
            best_score = 10_000
            best_depth = 10
            for a in actions:
                game_copied = deepcopy(game)
                game_copied._Game__move(a[0], a[1], game_copied.current_player_idx)
                game_copied.current_player_idx = self.my_player_idx
                score, curr_depth = self.minimax(game_copied, depth+1, alpha, beta)
                if score < best_score or (score == best_score and curr_depth < best_depth):
                    best_score = score
                    best_action = a
                    best_depth = curr_depth
                beta = min(beta, score)
                if beta <= alpha:
                    break

        # print(depth)
        if depth == 0:
            return best_score, best_action
        else:
            return best_score, depth

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        # game.print()
        # actions = self.actions(game)
        # print(actions)
        score, chosen_action = self.minimax(game, 0, -10_000, 10_000)

        # chosen_action = random.choice(actions)
        from_pos, move = chosen_action[0], chosen_action[1]

        print(f"{game.current_player_idx} trying move {from_pos}, {move} with score {score}")
        return from_pos, move


if __name__ == '__main__':

    my_player = MyPlayer()
    other_player = RandomPlayer()

    '''
    g = Game()
    winner = g.play(player1, player2)
    print(f"Player {winner} won!")
    '''

    count = 0.0
    count_starting_first = 0
    count_starting_second = 0
    num_games = 100
    for i in range(num_games):
        g = Game()
        g.print()

        my_player_idx = random.choice([0, 1])
        my_player.set_my_player_idx(my_player_idx)

        if my_player_idx == 0:
            count_starting_first += 1
            winner = g.play(my_player, other_player)
        else:
            count_starting_second += 1
            winner = g.play(other_player, my_player)

        print(f"Player {winner} won!")
        if winner == my_player.get_my_player_idx():
            count += 1.0

    print(f"My player winning rate = {count*100.0/num_games} % ! \nStarting {count_starting_first} times as first and {count_starting_second} times as second!")
