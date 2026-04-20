from astar import AStar
import numpy as np
import math


class astar_router(AStar):
    __directions = [(1,0),(0,-1),(-1,0),(0,1)]
    def __init__(self, n):
        self.n = n
        self.board = np.arange(0,n*n,1)
        self.board = np.reshape(self.board, (n,n))
        self.rng = np.random.default_rng()
    
    def index_to_coord(self, index):
        assert index <= self.n * self.n
        anchor_y = index%self.n
        anchor_x = int((index - anchor_y) / self.n)
        return (anchor_x, anchor_y)
      
    def _is_legal(self, move):
        is_in_boundary = all([x < self.n and x >= 0 for x in move])
        return is_in_boundary       
    
    def neighbors(self, node):
        moves = set()  # stores the legal moves.

        # Get all the empty squares (color==0)
        anchor_index = self.index_to_coord(node)
        for direction in self.__directions:
            possible_move = self.add_tuples(direction, anchor_index)
            if(self._is_legal(possible_move)):
                x,y = possible_move
                moves.add(self.board[x][y])
        return list(moves)
    
    def add_tuples(self, a,b):
        total = np.add(np.array(a),np.array(b))
        return tuple(total.tolist())
    
    def subtract_tuples(self, a,b):
        total = np.subtract(np.array(a),np.array(b))
        return tuple(total.tolist())
    
    
    def distance_between(self, n1, n2):
        index_1 = self.index_to_coord(n1)
        index_2 = self.index_to_coord(n2)
        
        x,y = self.subtract_tuples(index_1, index_2)

        return math.sqrt(x*x + y*y)

    def heuristic_cost_estimate(self, current, goal):
        return 1

    def is_goal_reached(self, current, goal):
        return current == goal
    
    def get_board_with_connected_nets(self, num_nets):
        board_with_nets = []
        for i in range(self.n*self.n):
            board_with_nets.append([])
        
        numbers = self.rng.choice(self.n*self.n, size=(num_nets,2), replace=False)
        for i in range(0, num_nets):
            source, dest = numbers[i]
            for node in self.astar(source,dest):
                board_with_nets[node].append(i+1)
        return (board_with_nets, numbers)
    
    def get_nets_with_drc(self, board):
        nets_with_DRCs = set()
        for node in board:
                if len(node) > 1:
                    for net in node:
                        nets_with_DRCs.add(net)
        return nets_with_DRCs
    
    def convert_to_router_problem(self, board, net):
        n = self.n
        init_board = []
        for node in board:
            if len(node) > 1:
                init_board.append(1)
            elif net not in node:
                init_board.append(len(node))
            else:
                init_board.append(0)                       
        init_board = np.reshape(init_board, (n,n))
        return init_board
    
    def convert_to_cleaner_problem(self, board):
        rep = []
        for cell in board:
            if cell == []:
                rep.append(0)
            else:
                rep.append(cell[-1])
        return np.reshape(rep, (self.n, self.n))
    