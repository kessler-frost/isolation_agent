# Run this if you(human) wants to play against the agent.

from random import randint

class HumanPlayer():
    """Player that chooses a move according to user's input."""

    def get_move(self, game, time_left):
        """
        Select a move from the available legal moves based on user input at the
        terminal.

        **********************************************************************
        NOTE: If testing with this player, remember to disable move timeout in
              the call to `Board.play()`.
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
        ----------
        (int, int)
            The move in the legal moves list selected by the user through the
            terminal prompt; automatically return (-1, -1) if there are no
            legal moves
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)

        print(game.to_string()) #display the board for the human player
        print(('\t'.join(['[%d] %s' % (i, str(move)) for i, move in enumerate(legal_moves)])))

        valid_choice = False
        while not valid_choice:
            try:
                index = int(input('Select move index:'))
                valid_choice = 0 <= index < len(legal_moves)

                if not valid_choice:
                    print('Illegal move! Try again.')

            except ValueError:
                print('Invalid index! Try again.')

        return legal_moves[index]


if __name__ == "__main__":
    from isolation import Board
    from game_agent import AlphaBetaPlayer

    # create an isolation board (by default 7x7)
    player1 = HumanPlayer()
    player2 = AlphaBetaPlayer()
    game = Board(player1, player2)

    initial_pos1 = (randint(0, 6), randint(0, 6))
    initial_pos2 = (randint(0, 6), randint(0, 6))
    
    while (initial_pos2 == initial_pos1):
        initial_pos2 = (randint(0, 6), randint(0, 6))
        
    game.apply_move(initial_pos1)
    game.apply_move(initial_pos2)

    # players take turns moving on the board, so player1 should be next to move
    assert(player1 == game.active_player)
    
    winner, history, outcome = game.play(ENABLE_TIMEOUT = False)
    print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
    print(game.to_string())
    print("Move history:\n{!s}".format(history))
