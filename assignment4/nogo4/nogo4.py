#!/usr/local/bin/python3
# /usr/bin/python3
# Set the path to your python3 above

#!/usr/bin/python3
# Set the path to your python3 above



from gtp_connection import GtpConnection
from board_base import DEFAULT_SIZE, GO_POINT, GO_COLOR
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoEngine

import numpy as np
from collections import defaultdict

class NoGo:
    def __init__(self):
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
        GoEngine.__init__(self, "NoGo4", 1.0)

    def get_move(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        return GoBoardUtil.generate_random_move(board, color,
                                                use_eye_filter=False)


class MCTSNode():

    def __init__(self, state, color, parent=None, parent_action=None):
        self.state = state
        self.color = color
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return
    
    def untried_actions(self, board, color):
        self._untried_actions = GoBoardUtil.generate_legal_moves(board,color)
        return self._untried_actions


    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses


    def n(self):
        return self._number_of_visits


    def expand(self):
        
        action = self._untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MCTSNode(
            next_state, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node 
    # From the present state, next state is generated depending on the action which is carried out. In this step all the possible child nodes corresponding to generated states are appended to the children array and the child_node is returned. The states which are possible from the present state are all generated and the child_node corresponding to this generated state is returned.

    def is_terminal_node(self, board, color):
        return self.state.is_game_over(board,color)
    # This is used to check if the current node is terminal or not. Terminal node is reached when the game is over.

    def is_game_over (board, color):
        moves = GoBoardUtil.generate_legal_moves (board, color)
        if len(moves) > 0:
            return False
        else:
            return True
    
    def game_result():
        pass

    def move(self, action):

        GoBoard.move(action, self.color) 

    def rollout(self, board, color):
        current_rollout_state = self.state
        
        while not current_rollout_state.is_game_over(board,color):
            
            possible_moves = GoBoardUtil.generate_legal_moves (board,color)
            
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()
    # From the current state, entire game is simulated till there is an outcome for the game. This outcome of the game is returned. For example if it results in a win, the outcome is 1. Otherwise it is -1 if it results in a loss. And it is 0 if it is a tie. If the entire game is randomly simulated, that is at each turn the move is randomly selected out of set of possible moves, it is called light playout.

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)
    # In this step all the statistics for the nodes are updated. Untill the parent node is reached, the number of visits for each node is incremented by 1. If the result is 1, that is it resulted in a win, then the win is incremented by 1. Otherwise if result is a loss, then loss is incremented by 1.

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    # All the actions are poped out of _untried_actions one by one. When it becomes empty, that is when the size is zero, it is fully expanded.

    def best_child(self, c_param=0.1):
        
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    # Once fully expanded, this function selects the best child out of the children array. The first term in the formula corresponds to exploitation and the second term corresponds to exploration.

    def rollout_policy(self, possible_moves):
        
        return possible_moves[np.random.randint(len(possible_moves))]

    # Randomly selects a move out of possible moves. This is an example of random playout.

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    # Selects node to run rollout.

    def best_action(self):
        simulation_no = 100
        
        
        for i in range(simulation_no):
            
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        
        return self.best_child(c_param=0.)

    

def run() -> None:
    """
    start the gtp connection and wait for commands.
    """
    board: GoBoard = GoBoard(DEFAULT_SIZE)
    con: GtpConnection = GtpConnection(NoGo(), board)
    con.start_connection()


if __name__ == "__main__":
    run()
