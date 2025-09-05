QUARK MK1

AChess AI with GUI built in Python.
It features a Tkinter-based interface, an Alpha-Beta pruning engine with optimizations, and optional Neural Network evaluation support.

✨ Features

Play human vs AI or let the AI play against itself.

Tkinter GUI with resizable chessboard, move list, evaluation bar, and light/dark themes.

Alpha-Beta pruning engine with:

Transposition tables

Move ordering (MVV-LVA, killer moves, history heuristic)

Null-move pruning and late move reductions

Quiescence search

Heuristic evaluation: material balance, piece centrality, pawn advancement, mobility.

Optional Neural Network evaluation (value_net.pt) if PyTorch and NumPy are available.

Interactive controls:

New game, undo, board flip

Adjustable search depth & time per move

Enable/disable neural net blending

Theme toggle (light/dark)

📦 Requirements

Python 3.8+

Required:

pip install python-chess


Optional (for NN evaluation):

pip install torch numpy

▶️ Usage

Run the app with:

python chessbot-mk1.py


Controls:

n → New game

u → Undo

f → Flip board

a → AI vs AI (single move)

s → Open settings

