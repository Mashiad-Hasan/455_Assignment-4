from board_base import GO_POINT, NO_POINT, GO_COLOR
from board import GoBoard

DEFAULT_KOMI = 6.5

class GoEngine:
    def __init__(self, name: str, version: float) -> None:
        """
        name : name of the player used by the GTP interface
        version : version number used by the GTP interface
        """
        self.name: str = name
        self.version: float = version
        self.komi: float = DEFAULT_KOMI

    def get_move(self, board: GoBoard, color: int) -> GO_POINT:
        """
        name : name of the player used by the GTP interface
        version : version number used by the GTP interface
        """
        pass


class GoSimulationEngine(GoEngine):
    def __init__(self, name: str, version: float,
                 sim: int, check_selfatari: bool, limit: int = 49, timelimit: int = 28) -> None:
        """
        Go player that selects moves by simulation.
        """
        GoEngine.__init__(self, name, version)
        self.sim = sim
        self.check_selfatari = check_selfatari
        self.limit = limit
        self.timelimit = timelimit

    def simulate(self, board: GoBoard, move: GO_POINT, toplay: GO_COLOR) -> GO_COLOR:
        """
        Run a simulated game for a given move.
        Must override
        """
        pass

    def simulateMove(self, board: GoBoard, move: GO_POINT, toplay: GO_COLOR) -> int:
        """
        Run self.sim simulations for a given move. Returns number of wins.
        """
        wins = 0
        for _ in range(self.sim):
            result = self.simulate(board, move, toplay)
            if result == toplay:
                wins += 1
        return wins