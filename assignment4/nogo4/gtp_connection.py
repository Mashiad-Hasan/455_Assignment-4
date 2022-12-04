"""
gtp_connection.py
Module for playing games of Go using GoTextProtocol

Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller.
Parts of this code were originally based on the gtp module
in the Deep-Go project by Isaac Henrion and Amos Storkey
at the University of Edinburgh.
"""
import traceback
import numpy as np
import re
from sys import stdin, stdout, stderr
from typing import Any, Callable, Dict, List, Tuple
from collections import defaultdict

from board_base import (
    is_black_white,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    GO_COLOR, GO_POINT,
    MAXSIZE,
    coord_to_point,
    opponent
)
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoEngine



class GtpConnection:
    def __init__(self, go_engine: GoEngine, board: GoBoard, debug_mode: bool = False) -> None:
        """
        Manage a GTP connection for a Go-playing engine

        Parameters
        ----------
        go_engine:
            a program that can reply to a set of GTP commandsbelow
        board:
            Represents the current board state.
        """
        self._debug_mode: bool = debug_mode
        self.go_engine = go_engine
        self.board: GoBoard = board
        self.commands: Dict[str, Callable[[List[str]], None]] = {
            "protocol_version": self.protocol_version_cmd,
            "quit": self.quit_cmd,
            "name": self.name_cmd,
            "boardsize": self.boardsize_cmd,
            "showboard": self.showboard_cmd,
            "clear_board": self.clear_board_cmd,
            "komi": self.komi_cmd,
            "version": self.version_cmd,
            "known_command": self.known_command_cmd,
            "genmove": self.genmove_cmd,
            "list_commands": self.list_commands_cmd,
            "play": self.play_cmd,
            "legal_moves": self.legal_moves_cmd,
            "gogui-rules_legal_moves": self.gogui_rules_legal_moves_cmd,
            "gogui-rules_final_result": self.gogui_rules_final_result_cmd,
        }

        # argmap is used for argument checking
        # values: (required number of arguments,
        #          error message on argnum failure)
        self.argmap: Dict[str, Tuple[int, str]] = {
            "boardsize": (1, "Usage: boardsize INT"),
            "komi": (1, "Usage: komi FLOAT"),
            "known_command": (1, "Usage: known_command CMD_NAME"),
            "genmove": (1, "Usage: genmove {w,b}"),
            "play": (2, "Usage: play {b,w} MOVE"),
            "legal_moves": (1, "Usage: legal_moves {w,b}"),
        }

    def write(self, data: str) -> None:
        stdout.write(data)

    def flush(self) -> None:
        stdout.flush()

    def start_connection(self) -> None:
        """
        Start a GTP connection.
        This function continuously monitors standard input for commands.
        """
        line = stdin.readline()
        while line:
            self.get_cmd(line)
            line = stdin.readline()

    def get_cmd(self, command: str) -> None:
        """
        Parse command string and execute it
        """
        if len(command.strip(" \r\t")) == 0:
            return
        if command[0] == "#":
            return
        # Strip leading numbers from regression tests
        if command[0].isdigit():
            command = re.sub("^\d+", "", command).lstrip()

        elements: List[str] = command.split()
        if not elements:
            return
        command_name: str = elements[0]
        args: List[str] = elements[1:]
        if self.has_arg_error(command_name, len(args)):
            return
        if command_name in self.commands:
            try:
                self.commands[command_name](args)
            except Exception as e:
                self.debug_msg("Error executing command {}\n".format(str(e)))
                self.debug_msg("Stack Trace:\n{}\n".format(traceback.format_exc()))
                raise e
        else:
            self.debug_msg("Unknown command: {}\n".format(command_name))
            self.error("Unknown command")
            stdout.flush()

    def has_arg_error(self, cmd: str, argnum: int) -> bool:
        """
        Verify the number of arguments of cmd.
        argnum is the number of parsed arguments
        """
        if cmd in self.argmap and self.argmap[cmd][0] != argnum:
            self.error(self.argmap[cmd][1])
            return True
        return False

    def debug_msg(self, msg: str) -> None:
        """ Write msg to the debug stream """
        if self._debug_mode:
            stderr.write(msg)
            stderr.flush()

    def error(self, error_msg: str) -> None:
        """ Send error msg to stdout """
        stdout.write("? {}\n\n".format(error_msg))
        stdout.flush()

    def respond(self, response: str = "") -> None:
        """ Send response to stdout """
        stdout.write("= {}\n\n".format(response))
        stdout.flush()

    def reset(self, size: int) -> None:
        """
        Reset the board to empty board of given size
        """
        self.board.reset(size)

    def board2d(self) -> str:
        return str(GoBoardUtil.get_twoD_board(self.board))

    def protocol_version_cmd(self, args: List[str]) -> None:
        """ Return the GTP protocol version being used (always 2) """
        self.respond("2")

    def quit_cmd(self, args: List[str]) -> None:
        """ Quit game and exit the GTP interface """
        self.respond()
        exit()

    def name_cmd(self, args: List[str]) -> None:
        """ Return the name of the Go engine """
        self.respond(self.go_engine.name)

    def version_cmd(self, args: List[str]) -> None:
        """ Return the version of the  Go engine """
        self.respond(str(self.go_engine.version))

    def clear_board_cmd(self, args: List[str]) -> None:
        """ clear the board """
        self.reset(self.board.size)
        self.respond()

    def boardsize_cmd(self, args: List[str]) -> None:
        """
        Reset the game with new boardsize args[0]
        """
        self.reset(int(args[0]))
        self.respond()

    def showboard_cmd(self, args: List[str]) -> None:
        self.respond("\n" + self.board2d())

    def komi_cmd(self, args: List[str]) -> None:
        """
        Set the engine's komi to args[0]
        """
        self.go_engine.komi = float(args[0])
        self.respond()

    def known_command_cmd(self, args: List[str]) -> None:
        """
        Check if command args[0] is known to the GTP interface
        """
        if args[0] in self.commands:
            self.respond("true")
        else:
            self.respond("false")

    def list_commands_cmd(self, args: List[str]) -> None:
        """ list all supported GTP commands """
        self.respond(" ".join(list(self.commands.keys())))



    def legal_moves_cmd(self, args: List[str]) -> None:
        """
        List legal moves for color args[0] in {'b','w'}
        """
        board_color: str = args[0].lower()
        color: GO_COLOR = color_to_int(board_color)
        moves: List[GO_POINT] = GoBoardUtil.generate_legal_moves(self.board, color)
        gtp_moves: List[str] = []
        for move in moves:
            coords: Tuple[int, int] = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = " ".join(sorted(gtp_moves))
        self.respond(sorted_moves)



    """
    ==========================================================================
    Assignment 4 - game-specific commands start here
    ==========================================================================
    """
    """
    ==========================================================================
    Assignment 4 - commands we already implemented for you
    ==========================================================================
    """



    def gogui_analyze_cmd(self, args):
        """ We already implemented this function for Assignment 4 """
        self.respond("pstring/Legal Moves For ToPlay/gogui-rules_legal_moves\n"
                     "pstring/Side to Play/gogui-rules_side_to_move\n"
                     "pstring/Final Result/gogui-rules_final_result\n"
                     "pstring/Board Size/gogui-rules_board_size\n"
                     "pstring/Rules GameID/gogui-rules_game_id\n"
                     "pstring/Show Board/gogui-rules_board\n"
                     )

    def gogui_rules_game_id_cmd(self, args):
        """ We already implemented this function for Assignment 4 """
        self.respond("NoGo")

    def gogui_rules_board_size_cmd(self, args):
        """ We already implemented this function for Assignment 4 """
        self.respond(str(self.board.size))

    def gogui_rules_side_to_move_cmd(self, args):
        """ We already implemented this function for Assignment 4 """
        color = "black" if self.board.current_player == BLACK else "white"
        self.respond(color)

    def gogui_rules_board_cmd(self, args):
        """ We already implemented this function for Assignment 4 """
        size = self.board.size
        str = ''
        for row in range(size-1, -1, -1):
            start = self.board.row_start(row + 1)
            for i in range(size):
                #str += '.'
                point = self.board.board[start + i]
                if point == BLACK:
                    str += 'X'
                elif point == WHITE:
                    str += 'O'
                elif point == EMPTY:
                    str += '.'
                else:
                    assert False
            str += '\n'
        self.respond(str)



    def gogui_rules_legal_moves_cmd(self, args):
        # get all the legal moves
        legal_moves = GoBoardUtil.generate_legal_moves(self.board, self.board.current_player)
        coords = [point_to_coord(move, self.board.size) for move in legal_moves]
        # convert to point strings
        point_strs  = [ chr(ord('a') + col - 1) + str(row) for row, col in coords]
        point_strs.sort()
        point_strs = ' '.join(point_strs).upper()
        self.respond(point_strs)
        return

    def gogui_rules_final_result_cmd(self, args):
        '''
        get the game result: unknown, white or black
        '''

        # get legal moves
        legal_moves = GoBoardUtil.generate_legal_moves(self.board, self.board.current_player)
        # undetermined yet
        if len(legal_moves) > 0:
            self.respond('unknown')
        # The current player is lost
        else:
            if self.board.current_player == BLACK:
                self.respond('white')
            else:
                self.respond('black')

    def play_cmd(self, args: List[str]) -> None:
        """
        play a move args[1] for given color args[0] in {'b','w'}
        """
        try:
            board_color = args[0].lower()
            board_move = args[1]
            color = color_to_int(board_color)

            coord = move_to_coord(args[1], self.board.size)
            if coord:
                move = coord_to_point(coord[0], coord[1], self.board.size)
            else:
                self.error(
                    "Error executing move {} converted from {}".format(move, args[1])
                )
                return

            success = self.board.play_move(move, color)
            if not success:
                self.respond('illegal move')
                return
            else:
                self.debug_msg(
                    "Move: {}\nBoard:\n{}\n".format(board_move, self.board2d())
                )
            self.respond()
        except Exception as e:
            self.respond("Error: {}".format(str(e)))

    """
    ==========================================================================
    Assignment 4 - game-specific commands you have to implement or modify
    ==========================================================================
    """



    def genmove_cmd(self, args: List[str]) -> None:
        # """ generate a move for color args[0] in {'b','w'} """
        # board_color = args[0].lower()
        # color = color_to_int(board_color)
        # move = self.go_engine.get_move(self.board, color)
        # if move is None:
        #     self.respond('unknown')
        #     return

        # move_coord = point_to_coord(move, self.board.size)
        # move_as_string = format_point(move_coord)
        # if self.board.is_legal(move, color):
        #     self.board.play_move(move, color)
        #     self.respond(move_as_string)
        # else:
        #     self.respond("Illegal move: {}".format(move_as_string))

        ## MY CODE

        root = self.MCTSNode(self.board)
        selected_node = self.MCTSNode.best_action(root)

        self.respond(selected_node.get_state())

    def time_limit_cmd(self, args):
        '''
        set time limit per move
        '''
        self.timelimit = int(args[0])
        self.respond()

    """
    ==========================================================================
    Assignment 4 - game-specific commands end here
    ==========================================================================
    """
    class MCTSNode():

        def __init__(self, state, parent=None, parent_action=None):
            self.state = state
            
            self.parent = parent
            self.parent_action = parent_action
            self.children = []
            self._number_of_visits = 0
            self._results = defaultdict(int)
            self._results[1] = 0
            self._results[-1] = 0
            self._untried_actions = None
            self._untried_actions = self.untried_actions(state, state.current_player)
            return None
        
        def get_state(self):
            return self.state
        
        def untried_actions(self, board, color):
            self._untried_actions = GoBoardUtil.generate_legal_moves(board,color)
            return self._untried_actions


        def q(self):
            wins = self._results[1]
            loses = self._results[-1]
            return wins - loses


        def n(self):
            return self._number_of_visits


        def move(self, state, action, color):

            state.move(action, color) 
            return state

        def expand(self, state):
            
            action = self._untried_actions.pop()
            next_state = self.move(state, action, state.current_player)
            child_node = GtpConnection.MCTSNode(
                next_state, parent=self, parent_action=action)

            self.children.append(child_node)
            return child_node.get_state()
        # From the present state, next state is generated depending on the action which is carried out. In this step all the possible child nodes corresponding to generated states are appended to the children array and the child_node is returned. The states which are possible from the present state are all generated and the child_node corresponding to this generated state is returned.

        def is_terminal_node(self, board, color):
            return self.is_game_over(board,color)
        # This is used to check if the current node is terminal or not. Terminal node is reached when the game is over.

        def is_game_over (self, board, color):
            moves = GoBoardUtil.generate_legal_moves (board, color)
            if len(moves) > 0:
                return False
            else:
                return True
        
        def game_result(self, current_rollout_state):

            # moves = GoBoardUtil.generate_legal_moves(state, self.color)
            if self.state.current_player == current_rollout_state.current_player:
                return 1
            else:
                return -1


        def rollout(self, state):
            
            current_rollout_state = state
            
            while not self.is_game_over(current_rollout_state, current_rollout_state.current_player):
                
                possible_moves = GoBoardUtil.generate_legal_moves (current_rollout_state,current_rollout_state.current_player)
                GtpConnection.respond(possible_moves)
                action = self.rollout_policy(possible_moves)
                current_rollout_state = self.move(current_rollout_state, action, current_rollout_state.current_player)
            return self.game_result(current_rollout_state)
        # From the current state, entire game is simulated till there is an outcome for the game. This outcome of the game is returned. For example if it results in a win, the outcome is 1. Otherwise it is -1 if it results in a loss. And it is 0 if it is a tie. If the entire game is randomly simulated, that is at each turn the move is randomly selected out of set of possible moves, it is called light playout.

        def backpropagate(self, v, result):
            self._number_of_visits += 1.
            self._results[result] += 1.
            if self.parent:
                self.parent.backpropagate(result)
        # In this step all the statistics for the nodes are updated. Untill the parent node is reached, the number of visits for each node is incremented by 1. If the result is 1, that is it resulted in a win, then the win is incremented by 1. Otherwise if result is a loss, then loss is incremented by 1.

        def is_fully_expanded(self):
            return len(self._untried_actions) == 0

        # All the actions are poped out of _untried_actions one by one. When it becomes empty, that is when the size is zero, it is fully expanded.

        def best_child(self, current_node, c_param=0.1):
            GtpConnection.respond(current_node.children)

            #choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in current_node.children]
            #return self.children[np.argmax(choices_weights)]

        # Once fully expanded, this function selects the best child out of the children array. The first term in the formula corresponds to exploitation and the second term corresponds to exploration.

        def rollout_policy(self, possible_moves):
            
            GtpConnection.respond(len(possible_moves))
            # return possible_moves[np.random.randint(len(possible_moves))]

        # Randomly selects a move out of possible moves. This is an example of random playout.

        def _tree_policy(self, state):

            current_node = state
            while not self.is_terminal_node(current_node, self.state.current_player):
                
                if not self.is_fully_expanded():
                    return self.expand(current_node)
                else:
                    current_node = self.best_child(current_node)
            return current_node
        # Selects node to run rollout.

        def best_action(node):
            simulation_no = 100
            state = node.get_state()
            
            for i in range(simulation_no):
                
                v = node._tree_policy(state)
                reward = node.rollout(v)
                node.backpropagate(v,reward)
            
            GtpConnection.respond(node.children)
            return node.best_child(node, c_param=0.1)


def point_to_coord(point: GO_POINT, boardsize: int) -> Tuple[int, int]:
    """
    Transform point given as board array index
    to (row, col) coordinate representation.
    """
    NS = boardsize + 1
    return divmod(point, NS)


def format_point(move: Tuple[int, int]) -> str:
    """
    Return move coordinates as a string such as 'A1'
    """
    assert MAXSIZE <= 25
    column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    row, col = move
    return column_letters[col - 1] + str(row)


def move_to_coord(point_str: str, board_size: int) -> Tuple[int, int]:
    """
    Convert a string point_str representing a point, as specified by GTP,
    to a pair of coordinates (row, col) in range 1 .. board_size.

    """
    s = point_str.lower()
    col_c = s[0]
    col = ord(col_c) - ord("a")
    if col_c < "i":
        col += 1
    row = int(s[1:])

    return row, col



def color_to_int(c: str) -> int:
    """convert character to the appropriate integer code"""
    color_to_int = {"b": BLACK, "w": WHITE, "e": EMPTY, "BORDER": BORDER}
    return color_to_int[c]
