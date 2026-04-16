import numpy as np
class Board():
    # list of all 4 directions on the board, as (x,y) offests    
    __directions = [(1,0),(0,-1),(-1,0),(0,1)]
    '''
    4/8/26 - There's an issue where the new source and destination is persistant throughout nextstates
    find a way to allow for memory of path travelled, without nextstate taking the new value.
    '''
    def __init__(self, source, destination, n=5):
        "Set up initial board configuration."
        self.n = n
        self.source = source
        self.dest = destination
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n    
        self.pieces[self.source[0]][self.source[1]]=1
        self.pieces[self.dest[0]][self.dest[1]]=1
        source_index = self.coord_to_index(self.source)
        dest_index = self.coord_to_index(self.dest)
        source_layer = np.full((self.n,self.n), source_index)
        dest_layer = np.full((self.n,self.n), dest_index)
        self.pieces = {
            "board"      : np.array([self.pieces,source_layer,dest_layer]), 
            "trav_source": self.source,
            "trav_dest"  : self.dest,
            "anchor"     : "trav_source",
            "target"     : "trav_dest",
            "turn_num"   : 2}
    

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces["board"][0][index]
    
    def index_to_coord(self, index):
        assert index <= self.n * self.n
        anchor_y = index%self.n
        anchor_x = int((index - anchor_y) / self.n)
        return (anchor_x, anchor_y)
    
    def coord_to_index(self, coord):
        return self.n*coord[0] + coord[1]
    
    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.        
        """
        moves = set()  # stores the legal moves.

        # Get all the empty squares (color==0)
        anchor = self.pieces["anchor"]
        anchor_index = self.pieces[anchor]
        for index, direction in enumerate(self.__directions):
            possible_move = self.add_tuples(direction, anchor_index)
            if(self._is_legal(possible_move)):
                moves.add(index)
        return list(moves)
    
    def add_tuples(self, a,b):
        total = np.add(np.array(a),np.array(b))
        return tuple(total.tolist())
    
    def has_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.        
        """
        # Get all the empty squares (color==0)
        anchor = self.pieces["anchor"]
        anchor_index = self.pieces[anchor]
        for direction in self.__directions:
            possible_move = self.add_tuples(direction, anchor_index)
            if self._is_legal(possible_move):
                return True
        return False
    
    def is_win(self, color):
        anchor = self.pieces["anchor"]
        target = self.pieces["target"]
        anchor_index = self.pieces[anchor]
        target_index = self.pieces[target]
        for direction in self.__directions:
            possible_move = self.add_tuples(direction, anchor_index)
            if(possible_move == target_index):
                return True
        return False
            
    def execute_move(self, action, color):
        anchor = self.pieces["anchor"]
        move = self.__directions[action]
        travel_loc = self.add_tuples(self.pieces[anchor], move)
        (x,y) = travel_loc
        assert self[x][y] == 0
        self[x][y] = 1 
        self.pieces[anchor] = travel_loc
        self.pieces["turn_num"] += 1

    
    def _is_legal(self, move):
        is_in_boundary = all([x < self.n and x >= 0 for x in move])
        if not is_in_boundary: 
            return False        
        return self[move[0]][move[1]] == 0 # check if space is empty