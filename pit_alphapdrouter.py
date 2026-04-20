import Arena_alpha_pd
from MCTS import MCTS
from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.TicTacToePlayers import *
#from tictactoe.keras.NNet import NNetWrapper as NNet

from alphapdrouter.AlphaPDRouterGame import AlphaPDRouterGame
from alphapdrouter.keras.NNet import NNetWrapper as NNet
import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

g = AlphaPDRouterGame(5)

# all players
rp = RandomPlayer(g).play
hp = HumanTicTacToePlayer(g).play



# nnet players
n1 = NNet(g)

n1.load_checkpoint('temp/','best.weights.h5')

args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

arena = Arena_alpha_pd.Arena(n1p, n1p, g, display=AlphaPDRouterGame.display)

print(arena.playGames(2, verbose=True))
