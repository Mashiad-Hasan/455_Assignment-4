"""
mcts.py
Adopted from Cmput 455 sample code Go5

Implements a game tree for MCTS in class TreeNode, and the search itself in class MCTS
"""
import numpy as np
import sys
import time
from typing import Dict, Tuple, Set

from board_base import opponent, BLACK, GO_COLOR, GO_POINT, NO_POINT, coord_to_point
from board import GoBoard
from board_util import GoBoardUtil


def uct(child_wins: int, child_visits: int, parent_visits: int, exploration: float) -> float:
    return child_wins / child_visits + exploration * np.sqrt(np.log(parent_visits) / child_visits)


def uct_rave(child_wins: int,
             child_visits: int,
             parent_visits: int,
             exploration: float,
             amaf_wins: int,
             amaf_visits: int,
             k: int
             ) -> float:
    beta=np.sqrt(k/(3*parent_visits+k))
    return (1-beta)*(child_wins / child_visits) + beta*(amaf_wins/amaf_visits) +\
        exploration * np.sqrt(np.log(parent_visits) / child_visits)


def uct_rave_beta(child_wins: int,
             child_visits: int,
             parent_visits: int,
             exploration: float,
             amaf_wins: int,
             amaf_visits: int,
             b_squared: float
             ) -> float:
    if amaf_visits:
        beta=amaf_visits/(child_visits+amaf_visits+4*b_squared*child_visits*amaf_visits)
    return (1-beta)*(child_wins / child_visits) + beta*(amaf_wins/amaf_visits) +\
        exploration * np.sqrt(np.log(parent_visits) / child_visits)


class TreeNode:
    """
    A node in the MCTS tree
    """

    def __init__(self, color: GO_COLOR) -> None:
        self.move: GO_POINT = NO_POINT
        self.color: GO_COLOR = color
        self.n_visits: int = 0
        self.n_opp_wins: int = 0
        self.parent: 'TreeNode' = self
        self.children: Dict[TreeNode] = {}
        self.expanded: bool = False
        # rave para
        self.amaf_n_visits: int = 0
        self.amaf_n_opp_wins: int = 0

    def set_parent(self, parent: 'TreeNode') -> None:
        self.parent: 'TreeNode' = parent

    def expand(self, board: GoBoard, color: GO_COLOR) -> None:
        """
        Expands tree by creating new children.
        """
        opp_color = opponent(board.current_player)
        moves = board.get_empty_points()
        for move in moves:
            if board.is_legal(move, color) and not board.is_eye(move, color):
                node = TreeNode(opp_color)
                node.move = move
                node.set_parent(self)
                self.children[move] = node
                self.expanded = True

    def select_in_tree(self, exploration: float, rave:float) -> Tuple[GO_POINT, 'TreeNode']:
        """
        Select move among children that gives maximizes UCT.
        If number of visits are zero for a node, value for that node is infinite, so definitely will get selected

        It uses: argmax(child_num_wins/child_num_vists + C * sqrt( ln(parent_num_vists) / child_num_visits )
        Returns:
        A tuple of (move, next_node)
        """
        n_child = None
        n_uct_val = -1
        for move, child in self.children.items():
            if child.n_visits == 0:
                return child.move, child
            # uct_val = uct(child.n_opp_wins, child.n_visits, self.n_visits, exploration)
            uct_val=uct_rave_beta(child.n_opp_wins, child.n_visits, self.n_visits, exploration,
                             child.amaf_n_opp_wins,child.amaf_n_visits, rave)
            if uct_val > n_uct_val:
                n_uct_val = uct_val
                n_child = child
        return n_child.move, n_child

    def select_best_child(self) -> Tuple[GO_POINT, 'TreeNode']:
        _n_visits = -1
        best_child = None
        for move, child in self.children.items():
            if child.n_visits > _n_visits:
                _n_visits = child.n_visits
                best_child = child
        if best_child:
            return best_child.move, best_child
        else:
            return None, None

    def update(self, winner: GO_COLOR, moves:Set) -> None:
        self.n_opp_wins += self.color != winner
        self.n_visits += 1
        if self.move in moves:
            self.amaf_n_opp_wins += self.color != winner
            self.amaf_n_visits += 1
        if not self.is_root():
            self.parent.update(winner, moves)
        else:
            self.amaf_n_opp_wins += self.color != winner
            self.amaf_n_visits += 1

    def is_leaf(self) -> bool:
        """
        Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent == self


class MCTS:

    def __init__(self,weights) -> None:
        self.root: 'TreeNode' = TreeNode(BLACK)
        self.root.set_parent(self.root)
        self.toplay: GO_COLOR = BLACK
        self.weights = weights

    def search(self, board: GoBoard, color: GO_COLOR) -> None:
        """
        Run a single playout from the root to the given depth, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        board -- a copy of the board.
        color -- color to play
        """
        node = self.root
        # This will be True only once for the root
        if not node.expanded:
            node.expand(board, color)
        while not node.is_leaf():
            move, next_node = node.select_in_tree(self.exploration, self.rave)
            # assert board.play_move(move, color)
            board.fast_play_move(move,color)
            color = opponent(color)
            node = next_node
        if not node.expanded:
            node.expand(board, color)

        assert board.current_player == color
        winner,moves = self.rollout(board, color, move)
        node.update(winner,moves)

    def play_game(self, board, limit, vp: GO_COLOR, init_move) -> (GO_COLOR, Set):
        moves=set([init_move])
        for _ in range(limit):
            color = board.current_player
            move = GoBoardUtil.generate_random_move(board, color, True)
            if not move:
                move = GoBoardUtil.generate_random_move(board, color, False)
            if move:
                board.fast_play_move(move, color)
                if color==vp:
                    moves.add(move)
            else:
                break
        return opponent(board.current_player), moves

    def get_weight(self,board,move):
        positions = [
            move - board.NS - 1,
            move - board.NS,
            move - board.NS + 1,
            move - 1,
            move + 1,
            move + board.NS - 1,
            move + board.NS,
            move + board.NS + 1,
        ]
        dec = 0
        i = 0
        for pos in reversed(positions):
            dec += board.board[pos] * (4 ** i)  # takes state at the position
            i += 1
        try:
            weight=self.weights[dec]
            return weight
        except IndexError:
            return None

    def play_game_pattern(self, board, limit, vp: GO_COLOR, init_move) -> (GO_COLOR, Set):
        moves = set([init_move])
        for _ in range(limit):
            color = board.current_player
            legal_moves = GoBoardUtil.generate_legal_moves(board, color)
            if not legal_moves:
                return opponent(color), moves
            pattern_moves = {}
            total = 0
            for move in legal_moves:
                weight = self.get_weight(board, move)
                if weight:
                    if move in pattern_moves.keys():
                        break
                    total += weight
                    pattern_moves[move] = weight
            for key, value in pattern_moves.items():
                pattern_moves[key] = value / total

            selected_move = np.random.choice(a=list(pattern_moves.keys()),  p=list(pattern_moves.values()))
            board.fast_play_move(selected_move, board.current_player)
            if color==vp:
                moves.add(selected_move)
        return opponent(board.current_player), moves

    def rollout(self, board: GoBoard, color: GO_COLOR, init_move) -> (GO_COLOR, Set):
        """
        Use the rollout policy to play until the end of the game, returning the winner of the game
        +1 if black wins, +2 if white wins, 0 if it is a tie.
        Also, the set of moves.
        """
        # return self.play_game(board, self.limit, color, init_move)
        return self.play_game_pattern(board, self.limit, color, init_move)

    def get_move(
            self,
            board: GoBoard,
            color: GO_COLOR,
            limit: int,
            check_selfatari: bool,
            num_simulation: int,
            exploration: float,
            timelimit: int,
            rave: float
    ) -> GO_POINT:
        """
        Runs all playouts sequentially and returns the most visited move.
        """
        s_time=time.time_ns()

        if self.toplay != color:
            sys.stderr.write(f"Tree is for wrong color ({color}{self.toplay}) to play. Deleting.\n")
            sys.stderr.flush()
            self.toplay = color
            self.root = TreeNode(color)
        self.limit = limit
        self.check_selfatari = check_selfatari
        self.exploration = exploration
        self.rave = rave

        if not self.root.expanded:
            self.root.expand(board, color)

        for _ in range(num_simulation * len(self.root.children)):
            cboard = board.copy()
            self.search(cboard, color)
            if time.time_ns()-s_time >= timelimit*1e+9:
                break

        best_move, best_child = self.root.select_best_child()
        return best_move

    def update_with_move(self, last_move: GO_POINT) -> None:
        """
        Step forward in the tree, keeping everything we already know about the subtree, assuming
        that get_move() has been called already. Siblings of the new root will be garbage-collected.
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
        else:
            self.root = TreeNode(opponent(self.toplay))
        self.root.parent = self.root
        self.toplay = opponent(self.toplay)

    def print_pi(self, board: GoBoard):
        pi = np.full((board.size, board.size), 0)
        for r in range(board.size):
            for c in range(board.size):
                point = coord_to_point(r + 1, c + 1, board.size)
                if point in self.root.children:
                    pi[r][c] = self.root.children[point].n_visits
        pi = np.flipud(pi)
        for r in range(board.size):
            for c in range(board.size):
                s = "{:5}".format(pi[r, c])
                sys.stderr.write(s)
            sys.stderr.write("\n")
        sys.stderr.flush()
