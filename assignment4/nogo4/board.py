"""
board.py
board.py
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller

Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move

The board uses a 1-dimensional representation with padding
"""

import numpy as np
from typing import List, Tuple

from board_base import (
    board_array_size,
    coord_to_point,
    is_black_white,
    is_black_white_empty,
    opponent,
    where1d,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    MAXSIZE,
    NO_POINT,
    GO_COLOR,
    GO_POINT,
)


"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.

The board is stored as a one-dimensional array of GO_POINT in self.board.
See GoBoardUtil.coord_to_point for explanations of the array encoding.
"""
class GoBoard(object):
    def __init__(self, size: int):
        """
        Creates a Go board of given size
        """
        assert 2 <= size <= MAXSIZE
        self.reset(size)

    def reset(self, size: int) -> None:
        """
        Creates a start state, an empty board with given size.
        """
        self.size: int = size
        self.NS: int = size + 1
        self.WE: int = 1
        self.current_player: GO_COLOR = BLACK
        self.maxpoint: int = board_array_size(size)
        self.board: np.ndarray[GO_POINT] = np.full(self.maxpoint, BORDER, dtype=GO_POINT)
        self.liberty_of: np.ndarray[GO_POINT] = np.full(self.maxpoint, NO_POINT, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self._initialize_neighbors()
        
        
    def copy(self) -> 'GoBoard':
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        return b

        
    def get_color(self, point: GO_POINT) -> GO_COLOR:
        return self.board[point]

    def pt(self, row: int, col: int) -> GO_POINT:
        return coord_to_point(row, col, self.size)

    def _fast_liberty_check(self, nb_point: GO_POINT) -> bool:
        lib = self.liberty_of[nb_point]
        if lib != NO_POINT and self.get_color(lib) == EMPTY:
            return True  # quick exit, block has a liberty
        if self._stone_has_liberty(nb_point):
            return True  # quick exit, no need to look at whole block
        return False

    def _detect_capture(self, nb_point: GO_POINT) -> bool:
        """
        Check whether opponent block on nb_point is captured.
        Returns boolean.
        """
        if self._fast_liberty_check(nb_point):
            return False
        opp_block = self._block_of(nb_point)
        return not self._has_liberty(opp_block)

    def _detect_captures(self, point: GO_POINT, opp_color: GO_COLOR) -> bool:
        """
        Did move on point capture something?
        """
        for nb in self.neighbors_of_color(point, opp_color):
            if self._detect_capture(nb):
                return True
        return False


    def _is_legal_simple(self, point: GO_POINT, color: GO_COLOR):
        assert self.pt(1, 1) <= point <= self.pt(self.size, self.size), "out of bound"
        #assert color==self.current_player, f"not current player {self.current_player}"
        if self.board[point] != EMPTY:
            return False
        return True

    def is_legal(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check whether it is legal for color to play on point
        """
        if not self._is_legal_simple(point,color):
            return False
        opp_color = opponent(color)
        self.board[point] = color
        legal = True
        has_capture = self._detect_captures(point, opp_color)
        if has_capture:
            return False
        if not self._stone_has_liberty(point):
            block = self._block_of(point)
            if not self._has_liberty(block, read_only=True):  # suicide
                legal = False
        self.board[point] = EMPTY
        return legal
           
    def get_empty_points(self) -> np.ndarray:
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def row_start(self, row: int) -> int:
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1
        
        
    def _initialize_empty_points(self, board_array: np.ndarray) -> None:
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start: int = self.row_start(row)
            board_array[start : start + self.size] = EMPTY

    def _on_board_neighbors(self, point: GO_POINT) -> List:
        nbs: List[GO_POINT] = []
        for nb in self._neighbors(point):
            if self.board[nb] != BORDER:
                nbs.append(nb)
        return nbs

    def _initialize_neighbors(self) -> None:
        """
        precompute neighbor array.
        For each point on the board, store its list of on-the-board neighbors
        """
        self.neighbors: List[List[GO_POINT]] = []
        for point in range(self.maxpoint):
            if self.board[point] == BORDER:
                self.neighbors.append([])
            else:
                self.neighbors.append(self._on_board_neighbors(GO_POINT(point)))

    def is_eye(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check if point is a simple eye for color
        """
        if not self._is_surrounded(point, color):
            return False
        # Eye-like shape. Check diagonals to detect false eye
        opp_color = opponent(color)
        false_count = 0
        at_edge = 0
        for d in self._diag_neighbors(point):
            if self.board[d] == BORDER:
                at_edge = 1
            elif self.board[d] == opp_color:
                false_count += 1
        return false_count <= 1 - at_edge  # 0 at edge, 1 in center
        
        
    def _is_surrounded(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        check whether empty point is surrounded by stones of color
        (or BORDER) neighbors
        """
        for nb in self._neighbors(point):
            nb_color = self.board[nb]
            if nb_color != BORDER and nb_color != color:
                return False
        return True

    def _stone_has_liberty(self, stone: GO_POINT) -> bool:
        lib = self.find_neighbor_of_color(stone, EMPTY)
        return lib != NO_POINT

    def _get_liberty(self, block: np.ndarray) -> GO_POINT:
        """
        Find any liberty of the given block.
        Returns NO_POINT in case there is no liberty.
        block is a numpy boolean array
        """
        for stone in where1d(block):
            lib: GO_POINT = self.find_neighbor_of_color(stone, EMPTY)
            if lib != NO_POINT:
                return lib
        return NO_POINT

    def _has_liberty(self, block: np.ndarray, read_only: bool = False) -> bool:
        """
        Check if the given block has any liberty.
        block is a numpy boolean array
        """
        lib = self._get_liberty(block)
        if lib == NO_POINT:
            return False
        assert self.get_color(lib) == EMPTY
        if not read_only:
            for stone in where1d(block):
                self.liberty_of[stone] = lib
        return True
        
        
    def _block_of(self, stone: GO_POINT) -> np.ndarray:
        """
        Find the block of given stone
        Returns a board of boolean markers which are set for
        all the points in the block 
        """
        color: GO_COLOR = self.get_color(stone)
        assert is_black_white(color)
        return self.connected_component(stone)

    def connected_component(self, point: GO_POINT) -> np.ndarray:
        """
        Find the connected component of the given point.
        """
        marker = np.full(self.maxpoint, False, dtype=np.bool_)
        pointstack = [point]
        color: GO_COLOR = self.get_color(point)
        assert is_black_white_empty(color)
        marker[point] = True
        while pointstack:
            p = pointstack.pop()
            neighbors = self.neighbors_of_color(p, color)
            for nb in neighbors:
                if not marker[nb]:
                    marker[nb] = True
                    pointstack.append(nb)
        return marker

    def play_move(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Play a move of color on point
        Returns whether move was legal
        """
        assert is_black_white(color)
        if not self._is_legal_simple(point,color):
            return False
        opp_color = opponent(color)
        self.board[point] = color
        if self._detect_captures(point, opp_color):
            self.board[point] = EMPTY
            return False
        if not self._stone_has_liberty(point):
            # check suicide of whole block
            block = self._block_of(point)
            if not self._has_liberty(block):  # undo suicide move
                self.board[point] = EMPTY
                return False
        self.current_player = opp_color
        return True
        


    def find_neighbor_of_color(self, point: GO_POINT, color: GO_COLOR) -> GO_POINT:
        """ Return one neighbor of point of given color, if exists
            returns NO_POINT otherwise.
        """
        for nb in self.neighbors[point]:
            if self.get_color(nb) == color:
                return nb
        return NO_POINT



    def move(self, action: GO_POINT, color: GO_COLOR) -> bool:
        """
        I defined this.

        Play a move of color on point
        Returns whether move was legal
        """
                    
        
        assert is_black_white(color)
        
        #if self.board[action] != EMPTY:
            #return False
        
        self.board[action] = color
        
        self.current_player = opponent(color)

        #return self.board

    def neighbors_of_color(self, point: GO_POINT, color: GO_COLOR) -> List:
        """ List of neighbors of point of given color """
        nbc: List[GO_POINT] = []
        for nb in self._neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def _neighbors(self, point: GO_POINT) -> List:
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point: GO_POINT) -> List:
        """ List of all four diagonal neighbors of point """
        return [point - self.NS - 1,
                point - self.NS + 1,
                point + self.NS - 1,
                point + self.NS + 1]

    def last_board_moves(self) -> List:
        """
        Get the list of last_move and second last move.
        Only include moves on the board (not NO_POINT, not PASS).
        """
        board_moves: List[GO_POINT] = []
        return board_moves
