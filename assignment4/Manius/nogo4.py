#!/usr/local/bin/python3
# /usr/bin/python3
# Set the path to your python3 above

#!/usr/bin/python3
# Set the path to your python3 above



from gtp_connection import GtpConnection
from board_base import DEFAULT_SIZE, GO_POINT, GO_COLOR,EMPTY
from board import GoBoard, opponent
from board_util import GoBoardUtil
from engine import GoSimulationEngine
from mcts import TreeNode,MCTS
from pathlib import Path
import numpy as np


def count_at_depth(node, depth, nodesAtDepth):
    if not node.expanded:
        return
    nodesAtDepth[depth] += 1
    for _, child in node.children.items():
        count_at_depth(child, depth + 1, nodesAtDepth)


def read_weights():
    script_dir = Path(__file__).parent.absolute()
    a=np.empty(65536)
    with open(script_dir/'weights.txt', 'r') as f:
        for line in f:
            ind,val=line.split()
            a[int(ind)]=float(val)
    return a


class NoGo:
    def __init__(self,
                 sim: int=1000,
                 check_selfatari: bool=True,
                 limit: int = 49,
                 exploration: float = 0.2,
                 timelimit: int = 26,
                 rave: float = 0.04,
                 op_max: int = 7
                 ):
        """
        Go player that selects moves randomly from the set of legal moves.
        Does not use the fill-eye filter.
        Passes only if there is no other legal move.

        Parameters
        ----------
        name : str
            name of the player (used by the GTP interface).
        version : float
            version number (used by the GTP interface).
        """
        GoSimulationEngine.__init__(self, "NoGo4", 1.0,
                                    sim, check_selfatari, limit, timelimit)
        self.weights = read_weights()
        self.MCTS = MCTS(self.weights)
        self.exploration = exploration
        self.rave=rave
        self.opening_flag=True
        self.openings=[[57,50,41,59],[9,18,25,11],[15,22,13,31],[63,54,61,47]]
        self.op_counter=0
        self.op_max=op_max

    def reset(self) -> None:
        self.MCTS = MCTS(self.weights)

    def update(self, move: GO_POINT) -> None:
        self.parent = self.MCTS.root
        self.MCTS.update_with_move(move)


    def get_move_opening(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        importance=[3,2,1,1]
        op_rects=[board.board[seq] for seq in self.openings]
        op_player=[rect==color for rect in op_rects]
        op_opp = [rect == opponent(color) for rect in op_rects]
        op_empty = [rect == EMPTY for rect in op_rects]
        op_val=[2*max(0,np.dot(importance,op_opp[i])-1.5*np.dot(importance,op_player[i]))
                +1.1*np.dot(importance,op_player[i])
                +np.dot(importance,op_empty[i]) for i in range(4)]
        rank=np.argsort(op_val)
        move=None
        for i in range(3,-1,-1):
            for j in range(4):
                if op_empty[rank[i]][j]:
                    if board.is_legal(self.openings[rank[i]][j],color):
                        move=self.openings[rank[i]][j]
                        break
            if move:
                self.op_counter+=1
                break
        if (not move) or self.op_counter >= self.op_max:
            self.opening_flag = False
        return move

    def get_move(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        move=None
        if self.opening_flag:
            move=self.get_move_opening(board,color)
        if not move:
            move = self.MCTS.get_move(
                board,
                color,
                limit=self.limit,
                check_selfatari=self.check_selfatari,
                num_simulation=self.sim,
                exploration=self.exploration,
                timelimit=self.timelimit,
                rave=self.rave
            )
        if not move:
            move = GoBoardUtil.generate_random_move(board,color,False)
        self.MCTS.print_pi(board)
        self.update(move)
        return move
        # return GoBoardUtil.generate_random_move(board, color,
        #                                         use_eye_filter=False)




    

def run() -> None:
    """
    start the gtp connection and wait for commands.
    """
    board: GoBoard = GoBoard(DEFAULT_SIZE)
    con: GtpConnection = GtpConnection(NoGo(), board)
    con.start_connection()


if __name__ == "__main__":
    run()
