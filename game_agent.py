
class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculates the heuristic value of a game state from the point of view
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

    # This function returns the score based on whether the legal moves of opponent
    # player coincides with that of current player. If it does then the total_score
    # will decrease as the possibility of opponent choosing that move will increase
    # thus our no. of moves will decrease in the next turn.

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    player_legal_moves = game.get_legal_moves(player)
    score = 0

    for op_move in game.get_legal_moves(game.get_opponent(player)):
        if op_move in player_legal_moves:
            score += 1

    total_score = float(len(game.get_legal_moves(player)) - score)

    return total_score


def custom_score_2(game, player):
    """Calculates the heuristic value of a game state from the point of view
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

    # A vertical division is done and no. of blank spaces left on the side of
    # player's location is returned as score.
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    blank_spaces = game.get_blank_spaces()

    left_spaces = []
    right_spaces = []
    for bs in blank_spaces:
        y, x = bs
        y += 1
        x += 1
        if x < 4:
            left_spaces.append(bs)
        else:
            right_spaces.append(bs)

    pl_y, pl_x = game.get_player_location(player)

    pl_x += 1

    if pl_x < 4:
        return float(len(left_spaces))
    else:
        return float(len(right_spaces))


def custom_score_3(game, player):
    """Calculates the heuristic value of a game state from the point of view
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
    # Evaluates on the basis of distance between the location of the player with
    # respect to the center of board. The more the distance the less the score.

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)

    total_score =  float(game.height * game.width - ((h - y)**2 + (w - x)**2))

    return total_score


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  DO NOT MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        # Initializing the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            best_move = (-1, -1)
        else:
            best_move = legal_moves[0]

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Depth-limited minimax search algorithm

        This is a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves.

        Notes
        -----
            (1) Only `self.score()` method should be used for board evaluation.

        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)

        # Minimax Search algorithm implemented with limited depth.
        def max_value(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = game.get_legal_moves()

            if depth == 0 or not legal_moves:
                return self.score(game, self)

            score = float("-inf")

            for m in legal_moves:
                score = max(score, min_value(game.forecast_move(m), depth - 1))

            return score

        def min_value(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = game.get_legal_moves()
            if depth == 0 or not legal_moves:
                return self.score(game, self)

            score = float("inf")

            for m in game.get_legal_moves():
                score = min(score, max_value(game.forecast_move(m), depth - 1))

            return score

        best_score = float("-inf")
        best_move = legal_moves[0]
        for m in legal_moves:
            score = min_value(game.forecast_move(m), depth - 1)
            if score > best_score:
                best_score = score
                best_move = m

        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning.
    """

    def get_move(self, game, time_left):
        """Searches for the best move from the available legal moves and returns a
        result before the time limit expires.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. We must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            best_move = (-1, -1)
        else:
            best_move = legal_moves[0]

        try:
            # Iterative deepening search is implemented here.
            depth = 1
            while True:
                best_move = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            pass

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implemented depth-limited minimax search with alpha-beta pruning.

        This is a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

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

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) Only `self.score()` method should be for board evaluation
                to pass the project tests.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return (-1, -1)

        # Alpha Beta Pruning is implemented here.
        def max_value(game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = game.get_legal_moves()

            if depth == 0 or not legal_moves:
                return self.score(game, self)

            score = float("-inf")

            for m in legal_moves:
                score = max(score, min_value(game.forecast_move(m), depth - 1, alpha, beta))
                if score >= beta :
                    return score
                alpha = max(alpha, score)

            return score

        def min_value(game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = game.get_legal_moves()

            if depth == 0 or not legal_moves:
                return self.score(game, self)

            score = float("inf")

            for m in legal_moves:
                score = min(score, max_value(game.forecast_move(m), depth - 1, alpha, beta))
                if score <= alpha :
                    return score
                beta = min(beta, score)

            return score

        best_score = float("-inf")
        best_move = legal_moves[0]

        for m in legal_moves:
            score = min_value(game.forecast_move(m), depth - 1, best_score, beta)
            if score > best_score:
                best_score = score
                best_move = m

        return best_move
