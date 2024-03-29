# Research Review

The file 'research_review.pdf' contains my review of [How to advance general game playing artificial intelligence by player modelling](https://arxiv.org/pdf/1606.00401.pdf) research paper.

# Isolation Playing Agent

![Example game of isolation](viz.gif)

## Synopsis

In this project, I developed an adversarial search agent to play the game "Isolation".  Isolation is a deterministic, two-player game of perfect information in which the players alternate turns moving a single piece from one cell to another on a board.  Whenever either player occupies a cell, that cell becomes blocked for the remainder of the game.  The first player with no remaining legal moves loses, and the opponent is declared the winner.  These rules are implemented in the `isolation.Board` class provided in the repository. 

This project uses a version of Isolation where each agent is restricted to L-shaped movements (like a knight in chess) on a rectangular grid (like a chess or checkerboard).  The agents can move to any open cell on the board that is 2-rows and 1-column or 2-columns and 1-row away from their current position on the board. Movements are blocked at the edges of the board (the board does not wrap around), however, the player can "jump" blocked or occupied spaces (just like a knight in chess).

Additionally, agents will have a fixed time limit each turn to search for the best move and respond.  If the time limit expires during a player's turn, that player forfeits the match, and the opponent wins.

## Instructions

If you want to play against the agent then simply run and follow the onscreen instructions by choosing each available move :-
```
python play_with_human.py
```

If you want to test the performance of my agent against other predefined agents then run :-
```
python tournament.py
```

If you simply want to see a sample of two different agents playing Isolation execute :-
```
python sample_players.py
```


### Tournament

The `tournament.py` script is used to evaluate the effectiveness of different custom heuristics(ways to measure how "good" or "bad" the current position of our player is).  The script measures relative performance of your agent (named "Student" in the tournament) in a round-robin tournament against several other pre-defined agents.  The Student agent uses time-limited Iterative Deepening along with different custom heuristics.

The performance of time-limited iterative deepening search is hardware dependent (faster hardware is expected to search deeper than slower hardware in the same amount of time).  The script controls for these effects by also measuring the baseline performance of an agent called "ID_Improved" that uses Iterative Deepening and the improved_score heuristic defined in `sample_players.py`.

The tournament opponents are listed below. (See also: sample heuristics and players defined in sample_players.py)

- Random: An agent that randomly chooses a move each turn.
- MM_Open: MinimaxPlayer agent using mini-max search algorithm with the open_move_score(only no. of available moves) heuristic with search depth 3
- MM_Center: MinimaxPlayer agent using mini-max search algorithm with the center_score(square of distance from center of the board to current player position) heuristic with search depth 3
- MM_Improved: MinimaxPlayer agent using mini-max search algorithm with the improved_score(difference in no. of available moves to both players) heuristic with search depth 3
- AB_Open: AlphaBetaPlayer using iterative deepening alpha-beta search and the open_move_score heuristic
- AB_Center: AlphaBetaPlayer using iterative deepening alpha-beta search and the center_score heuristic
- AB_Improved: AlphaBetaPlayer using iterative deepening alpha-beta search and the improved_score heuristic

## Game Visualization

The `isoviz` folder contains a modified version of chessboard.js that can animate games played on a 7x7 board.  In order to use the board, you must run a local webserver by running `python -m http.server 8000` from this project directory (you can replace 8000 with another port number if that one is unavailable), then open your browser to `http://localhost:8000` and navigate to the `/isoviz/display.html` page.  Enter the move history of an isolation match (i.e., the array returned by the Board.play() method) into the text area and run the match.  Refresh the page to run a different game.

## Heuristic Evaluations

The file 'heuristic_analysis.pdf' contains the results of my evaluation of all the three custom heuristics that I used to design my agent.