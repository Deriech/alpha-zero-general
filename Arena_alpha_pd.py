import logging

from tqdm import tqdm
from copy import deepcopy
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

        curPlayer = 1
        boardp1 = self.gamep1.getInitBoard()
        self.gamep2 = deepcopy(self.gamep1)
        boardp2 = self.gamep2.getCanonicalForm(boardp1, curPlayer)
        it_p1 = 0
        it_p2 = 0

        while self.gamep1.getGameEnded(boardp1, curPlayer) == 0:
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
        p2_score = self.gamep2.getGameEnded(boardp2, curPlayer)

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
