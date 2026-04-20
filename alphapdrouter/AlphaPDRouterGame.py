from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .AlphaPDRouterLogic import Board
import numpy as np
import json
import pickle


class AlphaPDRouterGame(Game):
    def __init__(self, n=5):
        self.n = n 
        self.source = None
        self.destination = None
        self.init_board = None



    def index_to_coord(self, index):
        assert index <= self.n * self.n
        anchor_y = index%self.n
        anchor_x = int((index - anchor_y) / self.n)
        return (anchor_x, anchor_y)
    
    def coord_to_index(self, coord):
        return self.n*coord[0] + coord[1]
    
    
    def getInitBoard(self, source, destination, init_board=None):
       #figure out A* procedure
       self.source = self.index_to_coord(source)
       self.destination = self.index_to_coord(destination)
       self.init_board = init_board
       b = Board(self.source, self.destination, self.init_board, self.n)
       return b.pieces

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.n, self.n)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 4
    def add_tuples(self, a,b):
        a = np.array(a)
        b = np.array(b)
        total = np.add(a,b)
        return tuple(total.tolist())
    

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        b = Board(self.source, self.destination, self.init_board, self.n)
        b.pieces = dict(pickle.loads(pickle.dumps(board, -1)))
        if action not in b.get_legal_moves(player):
            pass
        b.execute_move(action, player)
        b.pieces["target"], b.pieces["anchor"] = b.pieces["anchor"], b.pieces["target"]
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.source, self.destination, self.init_board, self.n)
        b.pieces = dict(pickle.loads(pickle.dumps(board, -1)))
        legalMoves =  b.get_legal_moves(player)
        for x in legalMoves:
            valids[x]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        b = Board(self.source, self.destination, self.init_board, self.n)
        b.pieces = dict(pickle.loads(pickle.dumps(board, -1)))

        if b.is_win(player):
            return 1
        
        if b.has_legal_moves(player):
            return 0
        # draw has a very little value 
        return -1

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        
                            
        """
        if player == 1:
            board["anchor"] = "trav_source"
            board["target"] = "trav_dest"
        else:
            board["anchor"] = "trav_dest"
            board["target"] = "trav_source"
            temp = pickle.loads(pickle.dumps(board["board"][1], -1))
            board["board"][1] = pickle.loads(pickle.dumps(board["board"][2], -1))
            board["board"][2] = temp

        return board

    def shift(self, key, array):
        return np.concatenate((array[-key:],array[:-key]))
    
    def coord_rot90(self, key, coord):
        coord = list(coord)
        for _ in range(key):
            coord[0], coord[1] = (self.n - 1 - coord[1]), coord[0]
        return(coord[0], coord[1])
    def coord_flip(self, coord):
        return (self.n - coord[0]-1, coord[1])
    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """

        assert(len(pi) == 4)
        
        l = []
        

        for i in range(1, 5):
            for j in [True, False]:
                board_to_flip = pickle.loads(pickle.dumps(board["board"], -1))
                newB = np.rot90(board_to_flip, i, (1,2))
                source = self.index_to_coord(board_to_flip[1][0][0])
                source_rot90 = self.coord_rot90(i, source)
                newB[1] = np.full(np.shape(board_to_flip[1]), self.coord_to_index(source_rot90))
                dest = self.index_to_coord(board_to_flip[2][0][0])
                dest_rot90 = self.coord_rot90(i, dest)
                newB[2] = np.full(np.shape(board_to_flip[2]), self.coord_to_index(dest_rot90))
                
                newPi = self.shift(-i, pi)
                if j:
                    newB = np.fliplr(newB)
                    newPi[0], newPi[2] = newPi[2], newPi[0]
                    source_rot90 = self.coord_flip(source_rot90)
                    dest_rot90 = self.coord_flip(dest_rot90)
                
                newB[1] = np.full(np.shape(board_to_flip[1]), self.coord_to_index(source_rot90))
                newB[2] = np.full(np.shape(board_to_flip[2]), self.coord_to_index(dest_rot90))
                l += [(newB, newPi)]
        return l

        #return[[(board, pi)], [(board, pi)]]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return json.dumps(board['board'].tolist()) + json.dumps(str(board["trav_source"])) + json.dumps(str(board["trav_dest"]))+ json.dumps(str(board["anchor"]))

    @staticmethod
    def display(board):
        n = board['board'].shape[1]
        print("   ", end="")
        for y in range(n):
            print (y,"", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|",end="")    # print the row #
            for x in range(n):
                piece = board['board'][0][y][x]    # get the piece to print
                cur_coord = (y,x)
                index = (y * n) + x
                if index == board['board'][1][0][0]: 
                    print("S ",end="")
                elif index == board['board'][2][0][0]: print("D ",end="") 
                elif cur_coord == board['trav_source']: print("+ ",end="")
                elif cur_coord == board['trav_dest']: print("+ ",end="")
                elif piece == 1: print("O ",end="")
                else:
                    if x==n:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")