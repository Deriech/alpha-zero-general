import logging

from tqdm import tqdm
from copy import deepcopy
import numpy as np
from astar_router import astar_router
log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.gamep1 = game
        self.gamep2 = None
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        board_with_nets = []
        nets_with_DRCs = set()
        rng = np.random.default_rng()
        n = self.gamep1.n
        star_board = astar_router(n)
        number_nets = 3

        while 1 not in nets_with_DRCs:
            board_with_nets = []
            for i in range(n*n):
                board_with_nets.append([])
            numbers = rng.choice(n*n, size=(number_nets,2), replace=False)
            for i in range(0, number_nets):
                source, dest = numbers[i]
                for node in star_board.astar(source,dest):
                    board_with_nets[node].append(i+1)
            for node in board_with_nets:
                if len(node) > 1:
                    for net in node:
                        nets_with_DRCs.add(net)
            
        source, dest = numbers[0]
        init_board = star_board.convert_to_router_problem(board_with_nets, 1)
        curPlayer = 1
        boardp1 = self.gamep1.getInitBoard(source, dest, init_board)
        self.gamep2 = deepcopy(self.gamep1)
        boardp2 = self.gamep2.getCanonicalForm(boardp1, curPlayer)
        it_p1 = 0
        it_p2 = 0

        while self.gamep1.getGameEnded(boardp1, curPlayer) == 0:
            if verbose:
                self.gamep1.display(self.gamep1.getCanonicalForm(boardp1, 1))
            it_p1 += 1
            action = self.player1(self.gamep1.getCanonicalForm(boardp1, curPlayer))
            valids = self.gamep1.getValidMoves(self.gamep1.getCanonicalForm(boardp1, curPlayer), 1)
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            boardp1, curPlayer = self.gamep1.getNextState(boardp1, curPlayer, action)
        
        curPlayer = 1
        while self.gamep2.getGameEnded(boardp2, curPlayer) == 0:
            it_p2 += 1
            action = self.player2(self.gamep2.getCanonicalForm(boardp2, curPlayer))
            valids = self.gamep2.getValidMoves(self.gamep2.getCanonicalForm(boardp2, curPlayer), 1)
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            boardp2, curPlayer = self.gamep2.getNextState(boardp2, curPlayer, action)

        p1_score = self.gamep1.getGameEnded(boardp1, curPlayer)
        if p1_score < 0:
            it_p1 -= 25
        p2_score = self.gamep2.getGameEnded(boardp2, curPlayer)
        if p2_score < 0:
            it_p2 -= 25

        if it_p1 > it_p2:
            return 1
        elif it_p1 < it_p2:
            return -1
        else:
            return 0

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
