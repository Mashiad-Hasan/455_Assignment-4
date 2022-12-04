#!/usr/local/bin/python3
# /usr/bin/python3
# Set the path to your python3 above

#!/usr/bin/python3
# Set the path to your python3 above



from gtp_connection import GtpConnection
from board_base import DEFAULT_SIZE, GO_POINT, GO_COLOR
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoSimulationEngine
from mcts import TreeNode,MCTS


def count_at_depth(node, depth, nodesAtDepth):
    if not node.expanded:
        return
    nodesAtDepth[depth] += 1
    for _, child in node.children.items():
        count_at_depth(child, depth + 1, nodesAtDepth)





class NoGo:
    def __init__(self,
                 sim: int=500,
                 check_selfatari: bool=True,
                 limit: int = 49,
                 exploration: float = 0.4,
                 timelimit: int = 27
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
        self.MCTS = MCTS()
        self.exploration = exploration

    def reset(self) -> None:
        self.MCTS = MCTS()


    def update(self, move: GO_POINT) -> None:
        self.parent = self.MCTS.root
        self.MCTS.update_with_move(move)

    def get_move(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        move = self.MCTS.get_move(
            board,
            color,
            limit=self.limit,
            check_selfatari=self.check_selfatari,
            num_simulation=self.sim,
            exploration=self.exploration,
            timelimit=self.timelimit,
            rave=1000
        )
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
