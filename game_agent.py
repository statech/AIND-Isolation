"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import itertools


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def blank_percentage(game):
    """Calculate the percentage of blank squares in the board

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    Returns
    -------
    float
        Percentage of blank squares in the board
    """
    num_all_squares = game.height * game.width
    num_all_blanks = len(game.get_blank_spaces())
    return num_all_blanks / num_all_squares

def is_on_edge(game, move):
    """Test whether the move is on the edge of the board

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.

    Returns
    -------
    boolean
        Whether the move is on the edge of the board
    """
    height = game.height
    width = game.width
    edges = [(row, 0) for row in range(height)] +\
            [(row, width - 1) for row in range(height)] +\
            [(0, col) for col in range(width)] +\
            [(height - 1, col) for col in range(width)]
    return move in edges

def is_in_corner(game, move):
    """Test whether the move is in the corner of the board

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.

    Returns
    -------
    boolean
        Whether the move is in the corner of the board
    """
    height = game.height
    width = game.width
    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    return move in corners

def no_left_move(game, move):
    """Test whether the player is able to move to the left

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.

    Returns
    -------
    boolean
        Whether the player is able to move to the left
    """
    row, col = move
    left_directions = [(-2, -1), (2, -1), (-1, -2), (1, -2)]
    left_moves = [
        (row+drow, col+dcol) for drow, dcol in left_directions\
        if game.move_is_legal((row+drow, col+dcol))
    ]
    return not left_moves

def no_right_move(game, move):
    """Test whether the player is able to move to the right

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.

    Returns
    -------
    boolean
        Whether the player is able to move to the right
    """
    row, col = move
    right_directions = [(-2, 1), (2, 1), (-1, 2), (1, 2)]
    right_moves = [
        (row+drow, col+dcol) for drow, dcol in right_directions\
        if game.move_is_legal((row+drow, col+dcol))
    ]
    return not right_moves

def no_up_move(game, move):
    """Test whether the player is able to move up

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.

    Returns
    -------
    boolean
        Whether the player is able to move up
    """
    row, col = move
    up_directions = [(-1, -2), (-2, -1), (-1, 2), (-2, 1)]
    up_moves = [
        (row+drow, col+dcol) for drow, dcol in up_directions\
        if game.move_is_legal((row+drow, col+dcol))
    ]
    return not up_moves

def no_down_move(game, move):
    """Test whether the player is able to move down

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.

    Returns
    -------
    boolean
        Whether the player is able to move down
    """
    row, col = move
    down_directions = [(1, -2), (2, -1), (1, 2), (2, 1)]
    down_moves = [
        (row+drow, col+dcol) for drow, dcol in down_directions\
        if game.move_is_legal((row+drow, col+dcol))
    ]
    return not down_moves

def num_blocked_directions(game, move):
    """Count how many directions are blocked that the player can go in

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.

    Returns
    -------
    int
        number of blocked directions
    """
    blocked_directions = [
        no_left_move(game, move), no_right_move(game, move),
        no_up_move(game, move), no_down_move(game, move)
    ]
    return sum(blocked_directions)

def move_quality_score_heuristic(game, player):
    """This heuristic function evaluates the quality of every legal move based
    on the following three criteria:
        * whether the move is on the edge of the boarder
        * whether the move is in the corner of the boarder
        * whether the move can only move in certain directions
    The move gets penalized if it is found to satisfy the above criteria. The
    summation of the move quality score is returned.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    Returns
    -------
    float
        Overall quality score
    """
    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return -float('inf')
    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    player_score = 0.0
    opponent_score = 0.0
    blank_percent = blank_percentage(game)
    corner_coeff, edge_coeff, side_coeff1, side_coeff2 = 1, 0.8, 0.7, 0.5
    for move in player_moves:
        if is_in_corner(game, move):
            if num_blocked_directions(game, move) == 3:
                if blank_percent >= 0.5:
                    player_score += 0
                elif blank_percent >= 0.15:
                    player_score += -30
                else:
                    player_score += -60
            else:
                if blank_percent >= 0.5:
                    player_score += 5
                elif blank_percent >= 0.15:
                    player_score += -20
                else:
                    player_score += -45
            player_score *= corner_coeff
        elif is_on_edge(game, move):
            if num_blocked_directions(game, move) == 3:
                if blank_percent >= 0.5:
                    player_score += 0
                elif blank_percent >= 0.15:
                    player_score += -25
                else:
                    player_score += -50
            elif num_blocked_directions(game, move) == 2:
                if blank_percent >= 0.5:
                    player_score += 5
                elif blank_percent >= 0.15:
                    player_score += -20
                else:
                    player_score += -40
            else:
                if blank_percent >= 0.5:
                    player_score += 10
                elif blank_percent >= 0.15:
                    player_score += -15
                else:
                    player_score += -30
            player_score *= edge_coeff
        elif num_blocked_directions(game, move) == 3:
            if blank_percent >= 0.5:
                player_score += 5
            elif blank_percent >= 0.15:
                player_score += -20
            else:
                player_score += -45
            player_score *= side_coeff1
        elif num_blocked_directions(game, move) == 2:
            if blank_percent >= 0.5:
                player_score += 10
            elif blank_percent >= 0.15:
                player_score += -10
            else:
                player_score += -25
            player_score *= side_coeff2
        else:
            player_score += 15
    for move in opponent_moves:
        if is_in_corner(game, move):
            if num_blocked_directions(game, move) == 3:
                if blank_percent >= 0.5:
                    opponent_score += 0
                elif blank_percent >= 0.15:
                    opponent_score += -30
                else:
                    opponent_score += -60
            else:
                if blank_percent >= 0.5:
                    opponent_score += 5
                elif blank_percent >= 0.15:
                    opponent_score += -20
                else:
                    opponent_score += -45
            opponent_score *= corner_coeff
        elif is_on_edge(game, move):
            if num_blocked_directions(game, move) == 3:
                if blank_percent >= 0.5:
                    opponent_score += 0
                elif blank_percent >= 0.15:
                    opponent_score += -25
                else:
                    opponent_score += -50
            elif num_blocked_directions(game, move) == 2:
                if blank_percent >= 0.5:
                    opponent_score += 5
                elif blank_percent >= 0.15:
                    opponent_score += -20
                else:
                    opponent_score += -40
            else:
                if blank_percent >= 0.5:
                    opponent_score += 10
                elif blank_percent >= 0.15:
                    opponent_score += -15
                else:
                    opponent_score += -30
            opponent_score *= edge_coeff
        elif num_blocked_directions(game, move) == 3:
            if blank_percent >= 0.5:
                opponent_score += 5
            elif blank_percent >= 0.15:
                opponent_score += -20
            else:
                opponent_score += -45
            opponent_score *= side_coeff1
        elif num_blocked_directions(game, move) == 2:
            if blank_percent >= 0.5:
                opponent_score += 10
            elif blank_percent >= 0.15:
                opponent_score += -10
            else:
                opponent_score += -25
            opponent_score *= side_coeff2
        else:
            opponent_score += 15
    return player_score - opponent_score

def open_area_heuristic(game, player):
    """Number of all nodes that are accessible to the player minus number of
    all nodes that are accessible to the opponent. This heuristic function
    provides better estimation of the 'goodness' of the board than
    `number_of_moves_heuristic()`.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    player_area = game.get_open_area(player)
    if not player_area and game.active_player == player:
        return -float('inf')
    opponent_area = game.get_open_area(game.get_opponent(player))
    if not player_area and not opponent_area:
        return -float('inf') if game.active_player == player else float('inf')
    elif not player_area:
        return -float('inf')
    elif not opponent_area:
        return float('inf')
    else:
        if not set.union(player_area, opponent_area):
            if len(player_area) > len(opponent_area):
                return float('inf')
            elif len(player_area) < len(opponent_area):
                return -float('inf')
            else:
                if game.active_player == player:
                    return -float('inf')
                else:
                    return float('inf')
        else:
            return float(len(player_area) - len(opponent_area))


def longest_path_length_heuristic(game, player):
    """Length of the longest path the player can make minus that the opponent
    can make. This heuristic is perfectly accurate when the open areas to both
    players are completely separate.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    player_lpl = game.longest_path_length(player)
    if player_lpl == 0 and game.active_player == player:
        return -float('inf')
    opponent_lpl = game.longest_path_length(game.get_opponent(player))
    if player_lpl == opponent_lpl:
        return -float('inf') if game.active_player == player else float('inf')
    else:
        return float('inf') if player_lpl > opponent_lpl else -float('inf')

def open_area_and_longest_path_length_heuristic(game, player):
    """Combine the open_area and longest path length heuristic

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    blank_percent = blank_percentage(game)
    if blank_percent >= 0.2:
        return open_area_heuristic(game, player)
    else:
        return longest_path_length_heuristic(game, player)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return move_quality_score_heuristic(game, player)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # if no more legal moves
        if not legal_moves:
            return (-1, -1)

        # if the game just began, pick the center position
        if game.move_count == 0:
            return (game.height // 2, game.width // 2)

        # initialize best score and best move
        score, move = -float('inf'), (-1, -1)

        # search method must be one of {'minimax', 'alphabeta'}
        assert self.method in ['minimax', 'alphabeta']
        search = self.minimax if self.method == 'minimax' else self.alphabeta

        try:
            if self.iterative:
                for depth in itertools.count(start=1):
                    score, move = search(game, depth, False)
                    if score == float('inf'):
                        break
            else:
                score, move = search(game, self.search_depth, False)
        except Timeout:
            # Handle any actions required at timeout, if necessary
            return move

        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves(game.active_player)
        # if there are no legal moves, return the end state score and (-1, -1)
        if not legal_moves:
            return self.score(game, self), (-1, -1)
        # if depth is zero but there are legal moves, return one of the legal
        # moves and the search branch score
        if depth == 0:
            return self.score(game, self), legal_moves[0]

        # for other cases, use recursion
        if maximizing_player:
            best_score, best_move = float("-inf"), None
            for move in legal_moves:
                res = self.minimax(game.forecast_move(move), depth - 1, False)
                score = res[0]
                if score > best_score:
                    best_score, best_move = score, move
            return best_score, best_move
        else:
            best_score, best_move = float("inf"), None
            for move in legal_moves:
                res = self.minimax(game.forecast_move(move), depth - 1, True)
                score = res[0]
                if score < best_score:
                    best_score, best_move = score, move
            return best_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves(game.active_player)
        # if there are no legal moves, return the end state score and (-1, -1)
        if not legal_moves:
            return self.score(game, self), (-1, -1)
        # if depth is zero but there are legal moves, return one of the legal
        # moves and the search branch score
        if depth == 0:
            return self.score(game, self), legal_moves[0]
        # for other cases, use recursion
        if maximizing_player:
            best_score, best_move = float("-inf"), None
            # estimates = [
            #     self.score(game.forecast_move(move), game.active_player)
            #     for move in legal_moves
            # ]
            # legal_moves = [
            #     move for (est, move) in \
            #     sorted(zip(estimates, legal_moves), reverse=True)
            # ]
            for move in legal_moves:
                res = self.alphabeta(
                    game.forecast_move(move), depth - 1, alpha, beta, False
                )
                score = res[0]
                if score > best_score:
                    best_score, best_move = score, move
                if best_score >= beta:
                    return best_score, best_move
                alpha = max(alpha, best_score, score)
            return best_score, best_move
        else:
            best_score, best_move = float("inf"), None
            # estimates = [
            #     self.score(game.forecast_move(move), game.inactive_player)
            #     for move in legal_moves
            # ]
            # legal_moves = [
            #     move for (est, move) in sorted(zip(estimates, legal_moves))
            # ]
            for move in legal_moves:
                res = self.alphabeta(
                    game.forecast_move(move), depth - 1, alpha, beta, True
                )
                score = res[0]
                if score < best_score:
                    best_score, best_move = score, move
                if best_score <= alpha:
                    return best_score, best_move
                beta = min(beta, best_score)
            return best_score, best_move
