# quantum-chessbot-mk1.py
import time
import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import chess
import chess.polyglot
import tkinter as tk
from tkinter import ttk, messagebox

# Optional dependencies
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    import numpy as np
except Exception:
    np = None

# ------------- Engine constants -------------
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
UNICODE_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}
INF = 10**9
MATE_SCORE = 100000
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# ------------- Optional NN -------------

def encode_board_planes(board: chess.Board, perspective_color: bool):
    if np is None:
        raise RuntimeError("NumPy required for NN features.")
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for i, pt in enumerate(order):
        for sq in board.pieces(pt, perspective_color):
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            planes[i, r, c] = 1.0
        for sq in board.pieces(pt, not perspective_color):
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            planes[6 + i, r, c] = 1.0
    return planes

# ------------- Heuristic Evaluation -------------
def centrality_bonus(file_idx: int, rank_idx: int) -> int:
    df = abs(file_idx - 3.5)
    dr = abs(rank_idx - 3.5)
    return int((3.5 - max(df, dr)) * 4) if max(df, dr) <= 3.5 else 0

def material_and_position_eval(board: chess.Board) -> int:
    if board.is_game_over():
        if board.is_checkmate():
            return -MATE_SCORE
        return 0
    score_white = 0
    score_black = 0
    for pt, val in PIECE_VALUES.items():
        wp = board.pieces(pt, chess.WHITE)
        bp = board.pieces(pt, chess.BLACK)
        score_white += len(wp) * val
        score_black += len(bp) * val
        if pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in wp:
                r = chess.square_rank(sq)
                f = chess.square_file(sq)
                score_white += centrality_bonus(f, r)
            for sq in bp:
                r = 7 - chess.square_rank(sq)
                f = chess.square_file(sq)
                score_black += centrality_bonus(f, r)
        elif pt == chess.PAWN:
            for sq in wp:
                r = chess.square_rank(sq)
                score_white += r * 3
            for sq in bp:
                r = 7 - chess.square_rank(sq)
                score_black += r * 3
    my_moves = len(list(board.legal_moves))
    opp = board.copy()
    opp.turn = not board.turn
    opp_moves = len(list(opp.legal_moves))
    mobility = my_moves - opp_moves
    if board.turn == chess.WHITE:
        score_white += mobility
    else:
        score_black += mobility
    base = score_white - score_black
    return base if board.turn == chess.WHITE else -base

# ------------- Engine (Alpha-Beta + extras) -------------
@dataclass
class TTEntry:
    depth: int
    flag: int
    value: int
    best_move: Optional[chess.Move]

class SearchTimeout(Exception):
    pass

class AlphaBetaEngine:
    def __init__(self, use_nn: bool = False, nn_weight: float = 0.5, model_path: str = "value_net.pt"):
        self.tt: Dict[int, TTEntry] = {}
        self.nodes = 0
        self.killers: Dict[int, List[Optional[chess.Move]]] = {}
        self.history: Dict[Tuple[bool, int, int], int] = {}
        self.use_nn = bool(use_nn and (torch is not None) )
        self.nn_weight = nn_weight
        self.model = None
        self.start_time = 0.0
        self.time_limit: Optional[float] = None
        self.stop = False

  

    def evaluate(self, board: chess.Board) -> int:
        heur = material_and_position_eval(board)
        if self.use_nn and self.model is not None and np is not None and torch is not None:
            with torch.no_grad():
                planes = encode_board_planes(board, perspective_color=board.turn)
                x = torch.from_numpy(planes).unsqueeze(0)
                val = self.model(x).item()
            nn_cp = int(val * 1000)
            return int((1 - self.nn_weight) * heur + self.nn_weight * nn_cp)
        return heur

    def choose_move(self, board: chess.Board, max_depth: int = 4, time_limit: Optional[float] = 2.0) -> Tuple[Optional[chess.Move], int]:
        self.nodes = 0
        self.start_time = time.time()
        self.time_limit = time_limit
        self.stop = False

        best_move = None
        best_score = -INF
        aspiration_window = 50

        try:
            prev_score = 0
            for depth in range(1, max_depth + 1):
                alpha, beta = -INF, INF
                if depth >= 3 and best_move is not None:
                    alpha = prev_score - aspiration_window
                    beta = prev_score + aspiration_window
                score, move = self._iter_depth_search(board, depth, alpha, beta)
                if move is not None:
                    best_move = move
                    best_score = score
                    prev_score = score
                if score <= alpha or score >= beta:
                    score, move = self._iter_depth_search(board, depth, -INF, INF)
                    if move is not None:
                        best_move = move
                        best_score = score
                        prev_score = score
        except SearchTimeout:
            pass
        return best_move, best_score

    def _iter_depth_search(self, board: chess.Board, depth: int, alpha: int, beta: int) -> Tuple[int, Optional[chess.Move]]:
        score = self._negamax(board, depth, alpha, beta, ply=0, null_allowed=True)
        key = chess.polyglot.zobrist_hash(board)
        tt = self.tt.get(key)
        best_move = tt.best_move if tt else None
        return score, best_move

    def _time_check(self):
        if self.time_limit is not None and (time.time() - self.start_time) >= self.time_limit:
            raise SearchTimeout()

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int, null_allowed: bool) -> int:
        self._time_check()
        self.nodes += 1

        alpha_orig = alpha
        alpha = max(alpha, -MATE_SCORE + ply)
        beta = min(beta, MATE_SCORE - ply)
        if alpha >= beta:
            return alpha

        key = chess.polyglot.zobrist_hash(board)
        tt_entry = self.tt.get(key)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_EXACT:
                return tt_entry.value
            elif tt_entry.flag == TT_LOWER:
                alpha = max(alpha, tt_entry.value)
            elif tt_entry.flag == TT_UPPER:
                beta = min(beta, tt_entry.value)
            if alpha >= beta:
                return tt_entry.value

        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
            return 0

        in_check = board.is_check()
        if in_check:
            depth += 1

        moves = list(board.legal_moves)

        # Null-move pruning
        if null_allowed and depth >= 3 and not in_check and len(moves) > 0:
            board.push(chess.Move.null())
            try:
                null_score = -self._negamax(board, depth - 1 - 2, -beta, -beta + 1, ply + 1, null_allowed=False)
            finally:
                board.pop()
            if null_score >= beta:
                return beta

        if not moves:
            return 0

        # Move ordering
        tt_move = tt_entry.best_move if tt_entry else None
        def mvv_lva_score(m: chess.Move) -> int:
            if board.is_capture(m):
                victim = board.piece_at(m.to_square)
                attacker = board.piece_at(m.from_square)
                v = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
                a = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
                return 10000 + v - a
            return 0
        killers = self.killers.get(ply, [None, None])
        def move_score(m: chess.Move) -> int:
            if tt_move and m == tt_move:
                return 1_000_000
            s = mvv_lva_score(m)
            if killers[0] and m == killers[0]:
                s += 9000
            elif killers[1] and m == killers[1]:
                s += 8000
            s += self.history.get((board.turn, m.from_square, m.to_square), 0)
            return s
        moves.sort(key=move_score, reverse=True)

        best_val = -INF
        best_move = None
        legal_index = 0

        for m in moves:
            self._time_check()
            legal_index += 1
            is_capture = board.is_capture(m)
            gives_check = board.gives_check(m)
            reduction = 0
            if depth >= 3 and legal_index > 4 and not is_capture and not gives_check and not in_check:
                reduction = 1
            board.push(m)
            try:
                val = -self._negamax(board, depth - 1 - reduction, -beta, -alpha, ply + 1, null_allowed=True)
                if reduction and val > alpha:
                    val = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, null_allowed=True)
            finally:
                board.pop()
            if val > best_val:
                best_val = val
                best_move = m
            if best_val > alpha:
                alpha = best_val
                if not is_capture:
                    key_h = (board.turn, m.from_square, m.to_square)
                    self.history[key_h] = self.history.get(key_h, 0) + depth * depth
                    if ply not in self.killers:
                        self.killers[ply] = [None, None]
                    if self.killers[ply][0] != m:
                        self.killers[ply][1] = self.killers[ply][0]
                        self.killers[ply][0] = m
            if alpha >= beta:
                break

        flag = TT_EXACT
        if best_val >= beta:
            flag = TT_LOWER
        elif best_val <= alpha_orig:
            flag = TT_UPPER
        self.tt[key] = TTEntry(depth=depth, flag=flag, value=best_val, best_move=best_move)
        return best_val

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self._time_check()
        stand_pat = self.evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        moves = [m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]
        def cap_score(m: chess.Move) -> int:
            victim = board.piece_at(m.to_square)
            attacker = board.piece_at(m.from_square)
            v = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
            a = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
            return v - a
        moves.sort(key=cap_score, reverse=True)
        for m in moves:
            self._time_check()
            board.push(m)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

# ------------- Tkinter UI (modern + minimal) -------------
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8

class ChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess AI")
        self.root.minsize(860, 560)

        # ttk theme
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", padding=6)
        style.configure("Toolbar.TFrame", background="#f4f5f7")
        style.configure("Status.TFrame", background="#f4f5f7")
        style.configure("Panel.TFrame", background="#ffffff")

        # Game state
        self.board = chess.Board()
        self.engine = AlphaBetaEngine()
        self.engine_score_cp = 0
        self.board_flipped = False
        self.human_color = chess.WHITE
        self.selected_sq: Optional[chess.Square] = None
        self.legal_targets: List[chess.Square] = []
        self.drag_piece_id = None
        self.drag_origin = None
        self.last_move: Optional[chess.Move] = None
        self.ai_thinking = False
        self.hover_sq: Optional[chess.Square] = None

        # Dynamic sizing
        self.board_size = 640           # current board canvas side in px
        self.square_size = self.board_size // 8

        # Engine controls (in Settings dialog)
        self.depth_var = tk.IntVar(value=4)
        self.time_var = tk.DoubleVar(value=2.0)
        self.nn_var = tk.BooleanVar(value=False)
        self.nn_weight_var = tk.DoubleVar(value=0.5)

        # Theme
        self.theme = "light"
        self._apply_theme(self.theme)

        # Layout grid weights (row 1 is the content row)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)  # center column (board) grows

        self._build_toolbar()
        self._build_content()
        self._build_statusbar()
        self._bind_shortcuts()

        self.draw_board()
        self.draw_pieces()
        self.draw_coords()
        self.update_eval_bar()
        self.update_move_list()

    # ---------- Theme (plain hex + stipple overlays) ----------
    def _apply_theme(self, name: str):
        if name == "dark":
            self.sq_light = "#b9c0c8"
            self.sq_dark = "#4e5963"
            self.hl_select = "#e5c558"
            self.hl_move = "#60d394"
            self.hl_last = "#7ab8ff"
            self.hl_check = "#ff6b6b"
            self.bg = "#1e2227"
            self.panel_bg = "#2a2f36"
            self.text_fg = "#e8eef5"
        else:
            self.sq_light = "#f0d9b5"
            self.sq_dark = "#b58863"
            self.hl_select = "#f7ec6a"
            self.hl_move = "#9de24a"
            self.hl_last = "#80c0ff"
            self.hl_check = "#ff6961"
            self.bg = "#f4f5f7"
            self.panel_bg = "#ffffff"
            self.text_fg = "#333333"
        self.root.configure(bg=self.bg)

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self._apply_theme(self.theme)
        # refresh
        self.draw_board()
        self.draw_pieces()
        self.draw_coords()
        self.update_eval_bar()
        self.move_panel.configure(background=self.panel_bg)
        self.moves_text.configure(background=self.panel_bg, fg=self.text_fg, insertbackground=self.text_fg)
        self.toolbar.configure(style="Toolbar.TFrame")
        self.statusbar.configure(style="Status.TFrame")

    # ---------- UI build ----------
    def _build_toolbar(self):
        self.toolbar = ttk.Frame(self.root, style="Toolbar.TFrame", padding=(12, 8))
        self.toolbar.grid(row=0, column=0, columnspan=3, sticky="ew")

        btn_new = ttk.Button(self.toolbar, text="New", command=self.new_game)
        btn_undo = ttk.Button(self.toolbar, text="Undo", command=self.undo_move)
        btn_flip = ttk.Button(self.toolbar, text="Flip", command=self.flip_board)
        btn_ai = ttk.Button(self.toolbar, text="AI ▶", command=self.ai_vs_ai_once)
        btn_settings = ttk.Button(self.toolbar, text="⚙ Settings", command=self.open_settings)
        btn_theme = ttk.Button(self.toolbar, text="🌙/☀️", command=self.toggle_theme)

        for w in (btn_new, btn_undo, btn_flip, btn_ai, btn_settings, btn_theme):
            w.pack(side="left", padx=(0, 8))

    def _build_content(self):
        # Left: Eval bar (height follows board)
        self.eval_canvas = tk.Canvas(self.root, width=40, height=self.board_size, bg="#333333", highlightthickness=0)
        self.eval_canvas.grid(row=1, column=0, padx=(16, 0), pady=8, sticky="ns")

        # Center: Board (resizable canvas)
        self.canvas = tk.Canvas(self.root, width=self.board_size, height=self.board_size, bg=self.panel_bg, highlightthickness=0)
        self.canvas.grid(row=1, column=1, padx=16, pady=8, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", lambda e: self._clear_hover())

        # Right: Moves panel
        self.move_panel = tk.Frame(self.root, bg=self.panel_bg)
        self.move_panel.grid(row=1, column=2, padx=(0, 16), pady=8, sticky="ns")

        lbl = tk.Label(self.move_panel, text="Moves", bg=self.panel_bg, fg=self.text_fg, font=("Segoe UI", 12, "bold"))
        lbl.pack(anchor="w", pady=(0, 6))

        # Text + scrollbar
        self.moves_text = tk.Text(self.move_panel, width=28, height=36, bg=self.panel_bg, fg=self.text_fg, bd=0, padx=8, pady=8, wrap="none")
        self.moves_text.configure(state="disabled")
        scroll = ttk.Scrollbar(self.move_panel, command=self.moves_text.yview)
        self.moves_text.configure(yscrollcommand=scroll.set)
        self.moves_text.pack(side="left", fill="y")
        scroll.pack(side="left", fill="y")

    def _build_statusbar(self):
        self.statusbar = ttk.Frame(self.root, style="Status.TFrame", padding=(12, 6))
        self.statusbar.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.status_var = tk.StringVar(value="Your move (White)")
        self.status_label = tk.Label(self.statusbar, textvariable=self.status_var, bg=self.bg, fg=self.text_fg)
        self.status_label.pack(anchor="w")

    def _bind_shortcuts(self):
        self.root.bind("n", lambda e: self.new_game())
        self.root.bind("u", lambda e: self.undo_move())
        self.root.bind("f", lambda e: self.flip_board())
        self.root.bind("a", lambda e: self.ai_vs_ai_once())
        self.root.bind("s", lambda e: self.open_settings())

    # ---------- Resizing ----------
    def on_canvas_resize(self, event):
        # Keep canvas square and recompute sizes
        new_side = max(320, min(event.width, event.height))
        if abs(new_side - self.board_size) < 2:
            return
        self.board_size = int(new_side)
        self.square_size = self.board_size // 8

        # Resize canvases
        self.canvas.config(width=self.board_size, height=self.board_size)
        self.eval_canvas.config(height=self.board_size)

        # Redraw everything
        self.draw_board()
        self.draw_pieces()
        self.draw_coords()
        self.update_eval_bar()

    # ---------- Board drawing ----------
    def draw_board(self):
        for tag in ("square", "lastmove", "check", "sel", "move", "hover", "coords", "piece"):
            self.canvas.delete(tag)

        for r in range(8):
            for c in range(8):
                x0 = c * self.square_size
                y0 = r * self.square_size
                x1 = x0 + self.square_size
                y1 = y0 + self.square_size
                color = self.sq_light if (r + c) % 2 == 0 else self.sq_dark
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color, tags="square")

        # Last move (stipple to simulate transparency)
        if self.last_move:
            fr, fc = self.square_to_rc(self.last_move.from_square)
            tr, tc = self.square_to_rc(self.last_move.to_square)
            self._highlight_square(fr, fc, self.hl_last, tag="lastmove", stipple="gray25")
            self._highlight_square(tr, tc, self.hl_last, tag="lastmove", stipple="gray25")

        # Check
        if self.board.is_check():
            ks = self.board.king(self.board.turn)
            if ks is not None:
                r, c = self.square_to_rc(ks)
                self._highlight_square(r, c, self.hl_check, tag="check", stipple="gray25")

    def draw_pieces(self):
        self.canvas.delete("piece")
        # selection + legal targets
        if self.selected_sq is not None:
            sr, sc = self.square_to_rc(self.selected_sq)
            self._highlight_square(sr, sc, self.hl_select, tag="sel", stipple="gray25")
            for t in self.legal_targets:
                tr, tc = self.square_to_rc(t)
                self._highlight_square(tr, tc, self.hl_move, tag="move", stipple="gray25")

        # piece font scales with square size
        piece_font = ("DejaVu Sans", max(18, int(self.square_size * 0.62)))
        for sq, piece in self.board.piece_map().items():
            if self.drag_piece_id and self.drag_origin == sq:
                continue
            r, c = self.square_to_rc(sq)
            x = c * self.square_size + self.square_size // 2
            y = r * self.square_size + self.square_size // 2
            glyph = UNICODE_PIECES[piece.symbol()]
            self.canvas.create_text(x, y, text=glyph, font=piece_font, tags="piece")

    def draw_coords(self):
        self.canvas.delete("coords")
        files = "abcdefgh"
        ranks_top_to_bottom = ["8","7","6","5","4","3","2","1"]
        if self.board_flipped:
            files = files[::-1]
            ranks_top_to_bottom = ranks_top_to_bottom[::-1]

        coord_font = ("Segoe UI", max(8, int(self.square_size * 0.18)))
        # files at bottom
        for c in range(8):
            x = c * self.square_size + self.square_size - (self.square_size // 8)
            y = self.board_size - (self.square_size // 10)
            self.canvas.create_text(x, y, text=files[c], fill="#888", font=coord_font, tags="coords")
        # ranks at left
        for r in range(8):
            x = (self.square_size // 8)
            y = r * self.square_size + (self.square_size // 8)
            self.canvas.create_text(x, y, text=ranks_top_to_bottom[r], fill="#888", font=coord_font, tags="coords")

    def _highlight_square(self, r, c, color, tag="hl", stipple=None):
        x0 = c * self.square_size
        y0 = r * self.square_size
        x1 = x0 + self.square_size
        y1 = y0 + self.square_size
        kwargs = {"fill": color, "outline": "", "tags": tag}
        if stipple:
            kwargs["stipple"] = stipple
        self.canvas.create_rectangle(x0, y0, x1, y1, **kwargs)

    def _clear_hover(self):
        self.hover_sq = None
        self.canvas.delete("hover")

    def on_mouse_move(self, event):
        if self.drag_piece_id:
            return
        sq = self.xy_to_square(event.x, event.y)
        if sq == self.hover_sq:
            return
        self.hover_sq = sq
        self.canvas.delete("hover")
        if sq is not None:
            r, c = self.square_to_rc(sq)
            hover_color = "#ffffff" if self.theme == "dark" else "#000000"
            self._highlight_square(r, c, hover_color, tag="hover", stipple="gray25")

    def update_eval_bar(self):
        # Resize bar height and redraw
        h = self.board_size
        self.eval_canvas.config(height=h)

        cp_side = self.engine_score_cp
        cp_white = cp_side if self.board.turn == chess.WHITE else -cp_side
        prob_white = 1.0 / (1.0 + math.exp(-cp_white / 200.0))
        h_white = int(prob_white * h)
        self.eval_canvas.delete("all")
        self.eval_canvas.create_rectangle(0, 0, 40, h - h_white, fill="#222", outline="")
        self.eval_canvas.create_rectangle(0, h - h_white, 40, h, fill="#EEE", outline="")
        self.eval_canvas.create_line(0, h - h_white, 40, h - h_white, fill="#666")

    # ---------- Move list ----------
    def update_move_list(self):
        tmp = chess.Board()
        lines = []
        pair = []
        move_no = 1
        for i, mv in enumerate(self.board.move_stack):
            san = tmp.san(mv)
            if i % 2 == 0:
                pair = [f"{move_no}.", san, ""]
            else:
                pair[2] = san
                lines.append("{:>4} {:<8} {:<8}".format(pair[0], pair[1], pair[2]))
                move_no += 1
            tmp.push(mv)
        if len(self.board.move_stack) % 2 == 1:
            lines.append("{:>4} {:<8} {:<8}".format(pair[0], pair[1], ""))

        self.moves_text.configure(state="normal")
        self.moves_text.delete("1.0", "end")
        self.moves_text.insert("end", "\n".join(lines))
        self.moves_text.configure(state="disabled")
        self.moves_text.see("end")

    # ---------- Coordinates helpers (use dynamic square_size) ----------
    def square_to_rc(self, sq: chess.Square) -> Tuple[int, int]:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        if not self.board_flipped:
            r = 7 - rank
            c = file
        else:
            r = rank
            c = 7 - file
        return r, c

    def rc_to_square(self, r: int, c: int) -> chess.Square:
        if not self.board_flipped:
            file = c
            rank = 7 - r
        else:
            file = 7 - c
            rank = r
        return chess.square(file, rank)

    def xy_to_square(self, x, y) -> Optional[chess.Square]:
        c = x // self.square_size
        r = y // self.square_size
        if c < 0 or c > 7 or r < 0 or r > 7:
            return None
        return self.rc_to_square(int(r), int(c))

    # ---------- Mouse handlers ----------
    def on_mouse_down(self, event):
        if self.ai_thinking or self.board.turn != self.human_color:
            return
        sq = self.xy_to_square(event.x, event.y)
        if sq is None:
            return
        piece = self.board.piece_at(sq)
        if piece is None or piece.color != self.board.turn:
            return
        self.selected_sq = sq
        self.legal_targets = [m.to_square for m in self.board.legal_moves if m.from_square == sq]
        glyph = UNICODE_PIECES[piece.symbol()]
        piece_font = ("DejaVu Sans", max(18, int(self.square_size * 0.62)))
        self.drag_piece_id = self.canvas.create_text(event.x, event.y, text=glyph, font=piece_font, tags="piece")
        self.drag_origin = sq
        self.draw_board()
        self.draw_pieces()

    def on_mouse_drag(self, event):
        if self.drag_piece_id:
            self.canvas.coords(self.drag_piece_id, event.x, event.y)

    def on_mouse_up(self, event):
        if not self.drag_piece_id or self.drag_origin is None:
            return
        to_sq = self.xy_to_square(event.x, event.y)
        from_sq = self.drag_origin

        self.canvas.delete(self.drag_piece_id)
        self.drag_piece_id = None
        self.drag_origin = None

        if to_sq is not None and to_sq in self.legal_targets:
            move = chess.Move(from_sq, to_sq)
            piece = self.board.piece_at(from_sq)
            if piece and piece.piece_type == chess.PAWN and (chess.square_rank(to_sq) in [0, 7]):
                promo = self.prompt_promotion(self.board.turn)
                move = chess.Move(from_sq, to_sq, promotion=promo)

            if move in self.board.legal_moves:
                self.board.push(move)
                self.last_move = move
                self.selected_sq = None
                self.legal_targets = []
                self.draw_board()
                self.draw_pieces()
                self.draw_coords()
                self.engine_score_cp = material_and_position_eval(self.board)
                self.update_eval_bar()
                self.update_move_list()

                if self.board.is_game_over():
                    self.show_game_over()
                    return

                if self.board.turn != self.human_color:
                    self.start_ai_turn()
                else:
                    side = "White" if self.board.turn == chess.WHITE else "Black"
                    self.status_var.set(f"Your move ({side})")
        else:
            self.selected_sq = None
            self.legal_targets = []
            self.draw_board()
            self.draw_pieces()

    # ---------- Promotion ----------
    def prompt_promotion(self, color: bool) -> int:
        top = tk.Toplevel(self.root)
        top.title("Promote to")
        top.transient(self.root)
        top.resizable(False, False)
        top.grab_set()
        frm = ttk.Frame(top, padding=12)
        frm.pack()
        ttk.Label(frm, text="Choose promotion:").pack(pady=(0, 6))
        choice = {"promo": chess.QUEEN}
        def set_p(p):
            choice["promo"] = p
            top.destroy()
        row = ttk.Frame(frm)
        row.pack()
        for text, ptype in [("Queen", chess.QUEEN), ("Rook", chess.ROOK), ("Bishop", chess.BISHOP), ("Knight", chess.KNIGHT)]:
            ttk.Button(row, text=text, command=lambda p=ptype: set_p(p)).pack(side="left", padx=6)
        self.root.wait_window(top)
        return choice["promo"]

    # ---------- AI ----------
    def start_ai_turn(self):
        if self.ai_thinking or self.board.is_game_over():
            return
        self.ai_thinking = True
        side = "White" if self.board.turn == chess.WHITE else "Black"
        self.status_var.set(f"AI thinking ({side})...")
        self.root.config(cursor="watch")

        def run_ai():
            try:
                self.engine.nn_weight = float(self.nn_weight_var.get())
                self.engine.use_nn = bool(self.nn_var.get() and self.engine.model )
                move, score = self.engine.choose_move(self.board, max_depth=int(self.depth_var.get()), time_limit=float(self.time_var.get()))
            except Exception as e:
                print("Engine error:", e)
                move, score = None, 0
            self.root.after(0, lambda: self.finish_ai_turn(move, score))

        threading.Thread(target=run_ai, daemon=True).start()

    def finish_ai_turn(self, move: Optional[chess.Move], score: int):
        self.ai_thinking = False
        self.root.config(cursor="")
        if move is not None:
            self.board.push(move)
            self.last_move = move
            self.engine_score_cp = score
            self.draw_board()
            self.draw_pieces()
            self.draw_coords()
            self.update_eval_bar()
            self.update_move_list()

            if self.board.is_game_over():
                self.show_game_over()
                return

            if self.human_color == self.board.turn:
                side = "White" if self.board.turn == chess.WHITE else "Black"
                self.status_var.set(f"Your move ({side})")
            else:
                self.start_ai_turn()
        else:
            self.status_var.set("No move found (draw?)")

    # ---------- Game actions ----------
    def new_game(self):
        if self.ai_thinking:
            messagebox.showinfo("Busy", "Wait for AI to finish the move.")
            return
        self.board.reset()
        self.selected_sq = None
        self.last_move = None
        self.human_color = chess.WHITE
        self.engine_score_cp = 0
        self.draw_board()
        self.draw_pieces()
        self.draw_coords()
        self.update_eval_bar()
        self.update_move_list()
        self.status_var.set("Your move (White)")

    def flip_board(self):
        self.board_flipped = not self.board_flipped
        self.draw_board()
        self.draw_pieces()
        self.draw_coords()

    def undo_move(self):
        if self.ai_thinking:
            messagebox.showinfo("Busy", "Wait for AI to finish the move.")
            return
        if len(self.board.move_stack) >= 1:
            self.board.pop()
        if len(self.board.move_stack) >= 1:
            self.board.pop()
        self.last_move = self.board.peek() if len(self.board.move_stack) else None
        self.selected_sq = None
        self.draw_board()
        self.draw_pieces()
        self.draw_coords()
        self.update_eval_bar()
        self.update_move_list()
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        self.status_var.set(f"Your move ({turn})")

    def ai_vs_ai_once(self):
        if self.ai_thinking:
            return
        self.start_ai_turn()

    def show_game_over(self):
        result = "Draw"
        if self.board.is_checkmate():
            result = "0-1" if self.board.turn == chess.WHITE else "1-0"
        messagebox.showinfo("Game Over", f"Result: {result}")
        self.status_var.set(f"Game Over: {result}")

    # ---------- Settings dialog ----------
    def open_settings(self):
        top = tk.Toplevel(self.root)
        top.title("Settings")
        top.transient(self.root)
        top.resizable(False, False)
        frm = ttk.Frame(top, padding=12)
        frm.pack(fill="both", expand=True)

        # Engine depth
        row1 = ttk.Frame(frm)
        row1.pack(fill="x", pady=4)
        lbl1 = ttk.Label(row1, text=f"Depth: {self.depth_var.get()}")
        lbl1.pack(side="left")
        def on_depth_change(v):
            lbl1.config(text=f"Depth: {int(float(v))}")
        depth_scale = ttk.Scale(row1, from_=2, to=6, orient="horizontal", variable=self.depth_var, command=on_depth_change)
        depth_scale.pack(side="right", fill="x", expand=True, padx=(12, 0))

        # Time
        row2 = ttk.Frame(frm)
        row2.pack(fill="x", pady=4)
        lbl2 = ttk.Label(row2, text=f"Time (s): {self.time_var.get():.1f}")
        lbl2.pack(side="left")
        def on_time_change(v):
            lbl2.config(text=f"Time (s): {float(v):.1f}")
        ttk.Scale(row2, from_=0.2, to=5.0, orient="horizontal", variable=self.time_var, command=on_time_change).pack(side="right", fill="x", expand=True, padx=(12, 0))

        # NN enable
        row3 = ttk.Frame(frm)
        row3.pack(fill="x", pady=4)
        ttk.Checkbutton(row3, text="Use Neural Net (requires value_net.pt)", variable=self.nn_var).pack(anchor="w")

        # NN weight
        row4 = ttk.Frame(frm)
        row4.pack(fill="x", pady=4)
        lbl4 = ttk.Label(row4, text=f"NN Weight: {self.nn_weight_var.get():.1f}")
        lbl4.pack(side="left")
        def on_nnw(v):
            lbl4.config(text=f"NN Weight: {float(v):.1f}")
        ttk.Scale(row4, from_=0.1, to=0.9, orient="horizontal", variable=self.nn_weight_var, command=on_nnw).pack(side="right", fill="x", expand=True, padx=(12, 0))

        # Buttons
        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="Close", command=top.destroy).pack(side="right")
# ------------- Main -------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ChessApp(root)
    root.mainloop()