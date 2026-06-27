# quark-chessbot-mk1.py
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

# Piece-Square Tables (from white's perspective)
# fmt: off
PST_PAWN = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0
]

PST_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

PST_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

PST_ROOK = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0
]

PST_QUEEN = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
]

PST_KING = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20
]
# fmt: on
INF = 10**9
MATE_SCORE = 100000
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
MAX_QS_PLY = 12

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
PST_MAP = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING,
}


def _pst_value(piece_type: int, square: chess.Square, color: bool) -> int:
    pst = PST_MAP[piece_type]
    return pst[square] if color == chess.WHITE else pst[chess.square_mirror(square)]


def _center_distance(square: chess.Square) -> int:
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    return min(abs(file - 3) + abs(rank - 3), abs(file - 4) + abs(rank - 4))


def _game_phase(board: chess.Board) -> float:
    phase = 0
    weights = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
    for piece_type, weight in weights.items():
        phase += weight * (
            len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
        )
    return min(1.0, phase / 24.0)


def _mobility_score(board: chess.Board, color: bool) -> int:
    work = board.copy(stack=False)
    work.turn = color
    if work.is_checkmate():
        return -80
    return len(list(work.legal_moves)) * 2


def _king_safety(board: chess.Board, color: bool, phase: float) -> int:
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    shield_rank = chess.square_rank(king_sq) + (1 if color == chess.WHITE else -1)
    shield = 0
    for file in (chess.square_file(king_sq) - 1, chess.square_file(king_sq), chess.square_file(king_sq) + 1):
        if 0 <= file <= 7 and 0 <= shield_rank <= 7:
            piece = board.piece_at(chess.square(file, shield_rank))
            if piece is not None and piece.color == color and piece.piece_type == chess.PAWN:
                shield += 14
    center_activity = max(0, 24 - _center_distance(king_sq) * 6)
    return int(shield * phase + center_activity * (1.0 - phase))


def _rook_file_score(board: chess.Board, color: bool) -> int:
    score = 0
    own_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    own_files = {chess.square_file(sq) for sq in own_pawns}
    enemy_files = {chess.square_file(sq) for sq in enemy_pawns}
    for rook_sq in board.pieces(chess.ROOK, color):
        file = chess.square_file(rook_sq)
        if file not in own_files and file not in enemy_files:
            score += 25
        elif file not in own_files:
            score += 12
    return score


def _development_score(board: chess.Board, color: bool) -> int:
    if not is_opening(board):
        return 0
    score = 0
    back_rank = 0 if color == chess.WHITE else 7
    for piece_type, bonus in ((chess.KNIGHT, 22), (chess.BISHOP, 18)):
        for sq in board.pieces(piece_type, color):
            if chess.square_rank(sq) != back_rank:
                score += bonus
    for queen_sq in board.pieces(chess.QUEEN, color):
        if chess.square_rank(queen_sq) != back_rank:
            score -= 45
    return score


def material_and_position_eval(board: chess.Board) -> int:
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    phase = _game_phase(board)
    white = 0
    black = 0

    for color in (chess.WHITE, chess.BLACK):
        score = 0
        for piece_type, value in PIECE_VALUES.items():
            pieces = board.pieces(piece_type, color)
            score += value * len(pieces)
            for sq in pieces:
                score += _pst_value(piece_type, sq, color)

        score += _mobility_score(board, color)
        score += _king_safety(board, color, phase)
        score += evaluate_pawns(board, color)
        score += evaluate_passed_pawns(board, color)
        score += _rook_file_score(board, color)
        score += _development_score(board, color)
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            score += 35

        for sq in (chess.D4, chess.E4, chess.D5, chess.E5):
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type == chess.PAWN:
                score += int(24 * phase + 10 * (1.0 - phase))

        if color == chess.WHITE:
            white = score
        else:
            black = score

    base = white - black
    return base if board.turn == chess.WHITE else -base

def evaluate_pawns(board, color):
    score = 0
    pawns = board.pieces(chess.PAWN, color)

    for sq in pawns:
        file = chess.square_file(sq)

        # Doubled pawn
        same_file = [p for p in pawns if chess.square_file(p) == file]
        if len(same_file) > 1:
            score -= 15

        # Isolated pawn
        adjacent_files = [file - 1, file + 1]
        isolated = True
        for p in pawns:
            if chess.square_file(p) in adjacent_files:
                isolated = False
        if isolated:
            score -= 20

    return score


def material_balance(board: chess.Board) -> int:
    bal = 0
    for piece_type, value in PIECE_VALUES.items():
        if piece_type == chess.KING:
            continue
        bal += value * (len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK)))
    return bal


def is_endgame(board: chess.Board) -> bool:
    # Rough heuristic: little non-pawn material left -> endgame
    non_pawn = 0
    for pt, v in PIECE_VALUES.items():
        if pt == chess.PAWN or pt == chess.KING:
            continue
        non_pawn += v * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
    return non_pawn <= 1300


def is_opening(board: chess.Board) -> bool:
    # Simple opening heuristic: few moves played and non-endgame
    return len(board.move_stack) < 14 and not is_endgame(board)


def evaluate_passed_pawns(board: chess.Board, color: bool) -> int:
    score = 0
    pawns = list(board.pieces(chess.PAWN, color))
    enemy = list(board.pieces(chess.PAWN, not color))
    enemy_files = {chess.square_file(sq) for sq in enemy}
    for sq in pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        passed = True
        # For white, enemy pawns on same or adjacent files ahead (higher rank)
        for ef in [file - 1, file, file + 1]:
            for ep in enemy:
                er = chess.square_rank(ep)
                efp = chess.square_file(ep)
                if efp == ef:
                    if color == chess.WHITE and er > rank:
                        passed = False
                    if color == chess.BLACK and er < rank:
                        passed = False
        if passed:
            # Bonus increases as pawn advances
            advance = (rank if color == chess.WHITE else (7 - rank))
            bonus = 20 + advance * 25
            # Extra if supported by a pawn behind on adjacent file
            support = 0
            back_rank = rank - 1 if color == chess.WHITE else rank + 1
            for df in [file - 1, file + 1]:
                if 0 <= df <= 7 and 0 <= back_rank <= 7:
                    sqb = chess.square(df, back_rank)
                    p = board.piece_at(sqb)
                    if p is not None and p.piece_type == chess.PAWN and p.color == color:
                        support += 15
            score += bonus + support
    return score



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

    def __init__(
        self,
        use_nn: bool = False,
        nn_weight: float = 0.5,
        model_path: str = "value_net.pt",
        book_path: Optional[str] = None,
        tt_size: int = 200_000,
    ):
        self.tt: Dict[int, TTEntry] = {}
        self.tt_size = tt_size
        self.nodes = 0
        self.killers: Dict[int, List[Optional[chess.Move]]] = {}
        self.history: Dict[Tuple[bool, int, int], int] = {}
        self.nn_weight = nn_weight
        self.model = None
        self.use_nn = False
        self.start_time = 0.0
        self.time_limit: Optional[float] = None
        self.stop = False
        self.book = None
        self.book_path = book_path
        self._load_book(book_path)

        # The old NN hook had no model architecture, so keep it opt-in only when a
        # caller supplies an already loaded model object in future work.
        if use_nn and model_path and torch is not None and np is not None:
            self.use_nn = bool(self.model)

    def _load_book(self, book_path: Optional[str]) -> None:
        self.book = None
        if not book_path:
            return
        try:
            self.book = chess.polyglot.open_reader(book_path)
        except Exception:
            self.book = None

    def set_book_path(self, book_path: Optional[str]) -> None:
        if self.book is not None:
            try:
                self.book.close()
            except Exception:
                pass
        self.book_path = book_path
        self._load_book(book_path)

    def clear_for_new_game(self) -> None:
        self.tt.clear()
        self.killers.clear()
        self.history.clear()
        self.nodes = 0

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

    def choose_move(
        self,
        board: chess.Board,
        max_depth: int = 6,
        time_limit: Optional[float] = 2.0,
    ) -> Tuple[Optional[chess.Move], int]:
        self.nodes = 0
        self.start_time = time.time()
        self.time_limit = max(0.01, time_limit) if time_limit is not None else None
        self.stop = False

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0

        if self.book is not None:
            try:
                entry = self.book.weighted_choice(board)
                if entry.move in legal_moves:
                    return entry.move, 0
            except (IndexError, KeyError):
                pass

        best_move = legal_moves[0]
        best_score = -INF
        prev_score = 0
        max_depth = max(1, int(max_depth))

        try:
            for depth in range(1, max_depth + 1):
                alpha, beta = -INF, INF
                if depth >= 4:
                    window = 50
                    alpha = prev_score - window
                    beta = prev_score + window

                score, move = self._search_root(board, depth, alpha, beta)
                if score <= alpha or score >= beta:
                    score, move = self._search_root(board, depth, -INF, INF)

                if move is not None:
                    best_move = move
                    best_score = score
                    prev_score = score
        except SearchTimeout:
            pass

        return best_move, best_score

    def _time_check(self):
        if self.stop:
            raise SearchTimeout()
        if self.time_limit is not None and (time.time() - self.start_time) >= self.time_limit:
            raise SearchTimeout()

    def _store_tt(self, key: int, entry: TTEntry) -> None:
        if len(self.tt) >= self.tt_size and key not in self.tt:
            try:
                self.tt.pop(next(iter(self.tt)))
            except StopIteration:
                pass
        self.tt[key] = entry

    def _capture_score(self, board: chess.Board, move: chess.Move) -> int:
        victim = board.piece_at(move.to_square)
        if victim is None and board.is_en_passant(move):
            victim_value = PIECE_VALUES[chess.PAWN]
        else:
            victim_value = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
        attacker = board.piece_at(move.from_square)
        attacker_value = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        return victim_value * 10 - attacker_value

    def _move_order_score(
        self,
        board: chess.Board,
        move: chess.Move,
        tt_move: Optional[chess.Move],
        ply: int,
    ) -> int:
        if tt_move and move == tt_move:
            return 1_000_000

        score = 0
        if move.promotion:
            score += PIECE_VALUES.get(move.promotion, 0) + 30_000
        if board.is_capture(move):
            score += 20_000 + self._capture_score(board, move)
        if board.gives_check(move):
            score += 12_000

        killers = self.killers.get(ply, [None, None])
        if killers[0] and move == killers[0]:
            score += 9_000
        elif killers[1] and move == killers[1]:
            score += 8_000

        score += self.history.get((board.turn, move.from_square, move.to_square), 0)

        if is_opening(board):
            piece = board.piece_at(move.from_square)
            if piece is not None and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                back_rank = 0 if piece.color == chess.WHITE else 7
                if chess.square_rank(move.from_square) == back_rank:
                    score += 350
            if piece is not None and piece.piece_type == chess.QUEEN:
                score -= 500

        return score

    def _ordered_moves(
        self,
        board: chess.Board,
        tt_move: Optional[chess.Move],
        ply: int,
    ) -> List[chess.Move]:
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: self._move_order_score(board, m, tt_move, ply), reverse=True)
        return moves

    def _search_root(self, board: chess.Board, depth: int, alpha: int, beta: int) -> Tuple[int, Optional[chess.Move]]:
        alpha_orig = alpha
        key = chess.polyglot.zobrist_hash(board)
        tt_entry = self.tt.get(key)
        tt_move = tt_entry.best_move if tt_entry else None
        moves = self._ordered_moves(board, tt_move, 0)
        best_score = -INF
        best_move = moves[0] if moves else None

        for move_index, move in enumerate(moves):
            self._time_check()
            is_capture = board.is_capture(move)
            gives_check = board.gives_check(move)
            board.push(move)
            try:
                reduction = 0
                if (
                    depth >= 4
                    and move_index >= 5
                    and not board.is_check()
                    and not is_capture
                    and not gives_check
                ):
                    reduction = 1
                score = -self._negamax(board, depth - 1 - reduction, -beta, -alpha, 1, True)
                if reduction and score > alpha:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, 1, True)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        flag = TT_EXACT
        if best_score >= beta:
            flag = TT_LOWER
        elif best_score <= alpha_orig:
            flag = TT_UPPER
        self._store_tt(key, TTEntry(depth=depth, flag=flag, value=best_score, best_move=best_move))
        return best_score, best_move

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int, null_allowed: bool) -> int:
        self._time_check()
        self.nodes += 1

        alpha_orig = alpha
        alpha = max(alpha, -MATE_SCORE + ply)
        beta = min(beta, MATE_SCORE - ply)
        if alpha >= beta:
            return alpha

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_fifty_moves()
            or board.can_claim_threefold_repetition()
        ):
            return 0

        key = chess.polyglot.zobrist_hash(board)
        tt_entry = self.tt.get(key)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_EXACT:
                return tt_entry.value
            if tt_entry.flag == TT_LOWER:
                alpha = max(alpha, tt_entry.value)
            elif tt_entry.flag == TT_UPPER:
                beta = min(beta, tt_entry.value)
            if alpha >= beta:
                return tt_entry.value

        in_check = board.is_check()
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)
        if in_check:
            depth += 1

        moves = self._ordered_moves(board, tt_entry.best_move if tt_entry else None, ply)
        if not moves:
            return -MATE_SCORE + ply if in_check else 0

        if (
            null_allowed
            and depth >= 3
            and not in_check
            and not is_endgame(board)
            and material_balance(board) != 0
        ):
            board.push(chess.Move.null())
            try:
                reduction = 3 if depth >= 6 else 2
                null_score = -self._negamax(board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, False)
            finally:
                board.pop()
            if null_score >= beta:
                return beta

        best_val = -INF
        best_move = moves[0]

        for move_index, move in enumerate(moves):
            self._time_check()
            is_capture = board.is_capture(move)
            gives_check = board.gives_check(move)
            reduction = 0
            if depth >= 3 and move_index >= 4 and not is_capture and not gives_check and not in_check:
                reduction = 1

            board.push(move)
            try:
                value = -self._negamax(board, depth - 1 - reduction, -beta, -alpha, ply + 1, True)
                if reduction and value > alpha:
                    value = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
            finally:
                board.pop()

            if value > best_val:
                best_val = value
                best_move = move

            if value > alpha:
                alpha = value
                if not is_capture:
                    key_h = (board.turn, move.from_square, move.to_square)
                    self.history[key_h] = min(1_000_000, self.history.get(key_h, 0) + depth * depth)
                    killers = self.killers.setdefault(ply, [None, None])
                    if killers[0] != move:
                        killers[1] = killers[0]
                        killers[0] = move

            if alpha >= beta:
                break

        flag = TT_EXACT
        if best_val >= beta:
            flag = TT_LOWER
        elif best_val <= alpha_orig:
            flag = TT_UPPER
        self._store_tt(key, TTEntry(depth=depth, flag=flag, value=best_val, best_move=best_move))
        return best_val

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self._time_check()
        if ply >= MAX_QS_PLY:
            return self.evaluate(board)

        stand_pat = self.evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        moves = [m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]
        moves.sort(key=lambda m: self._move_order_score(board, m, None, ply), reverse=True)

        for move in moves:
            self._time_check()
            if board.is_capture(move) and self._capture_score(board, move) < -250 and not board.gives_check(move):
                continue
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

# ------------- Tkinter UI (improved) -------------
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8

class ChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess AI")
        self.root.minsize(860, 560)

        # ttk theme / style
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure("TButton", padding=6)
        self.style.configure("Toolbar.TFrame", background="#f4f5f7")
        self.style.configure("Status.TFrame", background="#f4f5f7")
        self.style.configure("Panel.TFrame", background="#ffffff")
        self.style.configure("TNotebook", background="#ffffff")
        self.style.configure("TNotebook.Tab", padding=(8, 4))

        # Game state
        self.board = chess.Board()
        self.engine = AlphaBetaEngine()
        self.engine_score_cp = 0
        self.pv = []
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

        # Engine info panel vars
        self.info_eval_var = tk.StringVar(value="Eval: +0.00")
        self.info_side_var = tk.StringVar(value="Side to move: White")
        self.info_nodes_var = tk.StringVar(value="Nodes: 0")
        self.info_time_var = tk.StringVar(value="Search time: 0.00 s")
        self.info_depth_var = tk.StringVar(value=f"Depth: {self.depth_var.get()}")
        self.last_search_time = 0.0

        # Theme
        self.theme = "light"
        self._apply_theme(self.theme)

        # Layout grid weights (row 1 is the content row)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)  # center column (board) grows

        self._build_menubar()
        self._build_toolbar()
        self._build_content()
        self._build_statusbar()
        self._bind_shortcuts()

        self.draw_board()
        self.draw_pieces()
        self.draw_coords()
        self.update_eval_bar()
        self.update_move_list()
        self.update_engine_info_panel()

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
        # refresh board-related
        self.canvas.configure(bg=self.panel_bg)
        self.eval_canvas.configure(bg="#333333")
        self.draw_board()
        self.draw_pieces()
        self.draw_coords()
        self.update_eval_bar()
        # right panel and labels
        self.move_panel.configure(bg=self.panel_bg)
        self.side_header.configure(bg=self.panel_bg, fg=self.text_fg)
        self.tab_moves.configure(bg=self.panel_bg)
        self.tab_info.configure(bg=self.panel_bg)
        self.moves_text.configure(background=self.panel_bg,
                                  fg=self.text_fg,
                                  insertbackground=self.text_fg)
        self.info_side_label.configure(bg=self.panel_bg, fg=self.text_fg)
        self.info_eval_label.configure(bg=self.panel_bg, fg=self.text_fg)
        self.info_nodes_label.configure(bg=self.panel_bg, fg=self.text_fg)
        self.info_depth_label.configure(bg=self.panel_bg, fg=self.text_fg)
        self.info_time_label.configure(bg=self.panel_bg, fg=self.text_fg)
        # status / toolbar
        self.status_label.configure(bg=self.bg, fg=self.text_fg)
        self.toolbar.configure(style="Toolbar.TFrame")
        self.statusbar.configure(style="Status.TFrame")
        # notebook styling
        self.style.configure("TNotebook", background=self.panel_bg)
        self.style.configure("TNotebook.Tab", padding=(8, 4))

    # ---------- Menu bar ----------
    def _build_menubar(self):
        menubar = tk.Menu(self.root)

        game_menu = tk.Menu(menubar, tearoff=False)
        game_menu.add_command(label="New game\tN", command=self.new_game)
        game_menu.add_command(label="Undo last 2 moves\tU", command=self.undo_move)
        game_menu.add_separator()
        game_menu.add_command(label="Flip board\tF", command=self.flip_board)
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="Game", menu=game_menu)

        engine_menu = tk.Menu(menubar, tearoff=False)
        engine_menu.add_command(label="AI vs AI (one move)\tA", command=self.ai_vs_ai_once)
        engine_menu.add_command(label="Settings...\tS", command=self.open_settings)
        menubar.add_cascade(label="Engine", menu=engine_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_command(label="Toggle light/dark theme", command=self.toggle_theme)
        menubar.add_cascade(label="View", menu=view_menu)

        self.root.config(menu=menubar)

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
        self.eval_canvas = tk.Canvas(
            self.root,
            width=40,
            height=self.board_size,
            bg="#333333",
            highlightthickness=0,
        )
        self.eval_canvas.grid(row=1, column=0, padx=(16, 0), pady=8, sticky="ns")

        # Center: Board (resizable canvas)
        self.canvas = tk.Canvas(
            self.root,
            width=self.board_size,
            height=self.board_size,
            bg=self.panel_bg,
            highlightthickness=0,
        )
        self.canvas.grid(row=1, column=1, padx=16, pady=8, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", lambda e: self._clear_hover())

        # Right: side panel with tabs (Moves / Engine info)
        self.move_panel = tk.Frame(self.root, bg=self.panel_bg)
        self.move_panel.grid(row=1, column=2, padx=(0, 16), pady=8, sticky="ns")

        self.side_header = tk.Label(
            self.move_panel,
            text="Game / Engine",
            bg=self.panel_bg,
            fg=self.text_fg,
            font=("Segoe UI", 12, "bold"),
        )
        self.side_header.pack(anchor="w", pady=(0, 6))

        self.tabs = ttk.Notebook(self.move_panel)
        self.tab_moves = tk.Frame(self.tabs, bg=self.panel_bg)
        self.tab_info = tk.Frame(self.tabs, bg=self.panel_bg)
        self.tabs.add(self.tab_moves, text="Moves")
        self.tabs.add(self.tab_info, text="Engine")
        self.tabs.pack(fill="both", expand=True)

        # --- Moves tab ---
        moves_frame = tk.Frame(self.tab_moves, bg=self.panel_bg)
        moves_frame.pack(fill="both", expand=True)

        self.moves_text = tk.Text(
            moves_frame,
            width=28,
            height=36,
            bg=self.panel_bg,
            fg=self.text_fg,
            bd=0,
            padx=8,
            pady=8,
            wrap="none",
        )
        self.moves_text.configure(state="disabled")
        scroll = ttk.Scrollbar(moves_frame, command=self.moves_text.yview)
        self.moves_text.configure(yscrollcommand=scroll.set)
        self.moves_text.pack(side="left", fill="y")
        scroll.pack(side="left", fill="y")

        # --- Engine info tab ---
        info_inner = tk.Frame(self.tab_info, bg=self.panel_bg)
        info_inner.pack(fill="both", expand=True, padx=6, pady=6)

        self.info_side_label = tk.Label(
            info_inner,
            textvariable=self.info_side_var,
            bg=self.panel_bg,
            fg=self.text_fg,
            font=("Segoe UI", 11, "bold"),
        )
        self.info_side_label.pack(anchor="w", pady=(0, 4))

        self.info_eval_label = tk.Label(
            info_inner,
            textvariable=self.info_eval_var,
            bg=self.panel_bg,
            fg=self.text_fg,
            font=("Segoe UI", 10),
        )
        self.info_eval_label.pack(anchor="w", pady=(0, 4))

        self.info_nodes_label = tk.Label(
            info_inner,
            textvariable=self.info_nodes_var,
            bg=self.panel_bg,
            fg=self.text_fg,
            font=("Segoe UI", 10),
        )
        self.info_nodes_label.pack(anchor="w", pady=(0, 4))

        self.info_depth_label = tk.Label(
            info_inner,
            textvariable=self.info_depth_var,
            bg=self.panel_bg,
            fg=self.text_fg,
            font=("Segoe UI", 10),
        )
        self.info_depth_label.pack(anchor="w", pady=(0, 4))

        self.info_time_label = tk.Label(
            info_inner,
            textvariable=self.info_time_var,
            bg=self.panel_bg,
            fg=self.text_fg,
            font=("Segoe UI", 10),
        )
        self.info_time_label.pack(anchor="w", pady=(0, 4))

    def _build_statusbar(self):
        self.statusbar = ttk.Frame(self.root, style="Status.TFrame", padding=(12, 6))
        self.statusbar.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.status_var = tk.StringVar(value="Your move (White)")
        self.status_label = tk.Label(
            self.statusbar, textvariable=self.status_var, bg=self.bg, fg=self.text_fg
        )
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
                self.canvas.create_rectangle(
                    x0, y0, x1, y1, fill=color, outline=color, tags="square"
                )

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
        ranks_top_to_bottom = ["8", "7", "6", "5", "4", "3", "2", "1"]
        if self.board_flipped:
            files = files[::-1]
            ranks_top_to_bottom = ranks_top_to_bottom[::-1]

        coord_font = ("Segoe UI", max(8, int(self.square_size * 0.18)))
        # files at bottom
        for c in range(8):
            x = c * self.square_size + self.square_size - (self.square_size // 8)
            y = self.board_size - (self.square_size // 10)
            self.canvas.create_text(
                x, y, text=files[c], fill="#888", font=coord_font, tags="coords"
            )
        # ranks at left
        for r in range(8):
            x = self.square_size // 8
            y = r * self.square_size + (self.square_size // 8)
            self.canvas.create_text(
                x, y, text=ranks_top_to_bottom[r], fill="#888", font=coord_font, tags="coords"
            )

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

        # engine_score_cp is stored as white-perspective centipawns
        cp_white = self.engine_score_cp
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

    # ---------- Engine info panel helpers ----------
    def update_side_to_move_label(self):
        side = "White" if self.board.turn == chess.WHITE else "Black"
        self.info_side_var.set(f"Side to move: {side}")

    def update_engine_info_panel(self, search_time: Optional[float] = None):
        # Eval in pawns from side to move's perspective
        self.info_eval_var.set(f"Eval: {self.engine_score_cp/100.0:+.2f}")
        self.info_depth_var.set(f"Depth: {int(self.depth_var.get())}")
        if search_time is not None:
            self.last_search_time = search_time
        self.info_time_var.set(f"Search time: {self.last_search_time:.2f} s")
        self.info_nodes_var.set(f"Nodes: {self.engine.nodes}")
        self.update_side_to_move_label()

    # ---------- Coordinates helpers ----------
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
        self.drag_piece_id = self.canvas.create_text(
            event.x, event.y, text=glyph, font=piece_font, tags="piece"
        )
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
                try:
                    heur = self.engine.evaluate(self.board)
                except Exception:
                    heur = material_and_position_eval(self.board)
                self.engine_score_cp = heur if self.board.turn == chess.WHITE else -heur
                self.update_eval_bar()
                self.update_move_list()
                self.update_engine_info_panel()

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
        for text, ptype in [
            ("Queen", chess.QUEEN),
            ("Rook", chess.ROOK),
            ("Bishop", chess.BISHOP),
            ("Knight", chess.KNIGHT),
        ]:
            ttk.Button(row, text=text, command=lambda p=ptype: set_p(p)).pack(
                side="left", padx=6
            )
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
                self.engine.use_nn = bool(self.nn_var.get() and self.engine.model)
                t0 = time.time()
                move, score = self.engine.choose_move(
                    self.board,
                    max_depth=int(self.depth_var.get()),
                    time_limit=float(self.time_var.get()),
                )
                elapsed = time.time() - t0
            except Exception as e:
                print("Engine error:", e)
                move, score, elapsed = None, 0, 0.0
            self.root.after(0, lambda: self.finish_ai_turn(move, score, elapsed))

        threading.Thread(target=run_ai, daemon=True).start()

    def finish_ai_turn(self, move: Optional[chess.Move], score: int, elapsed: float):
        self.ai_thinking = False
        self.root.config(cursor="")
        if move is not None:
            self.board.push(move)
            self.last_move = move
            # Normalize engine evaluation to white-perspective (centipawns)
            try:
                heur = self.engine.evaluate(self.board)
            except Exception:
                heur = material_and_position_eval(self.board)
            # heur is from side-to-move perspective; convert to white-perspective
            self.engine_score_cp = heur if self.board.turn == chess.WHITE else -heur
            self.draw_board()
            self.draw_pieces()
            self.draw_coords()
            self.update_eval_bar()
            self.update_move_list()
            self.update_engine_info_panel(search_time=elapsed)

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
        self.engine.clear_for_new_game()
        self.selected_sq = None
        self.last_move = None
        self.human_color = chess.WHITE
        self.engine_score_cp = 0
        self.engine.nodes = 0
        self.last_search_time = 0.0
        self.draw_board()
        self.draw_pieces()
        self.draw_coords()
        self.update_eval_bar()
        self.update_move_list()
        self.update_engine_info_panel()
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
        try:
            heur = self.engine.evaluate(self.board)
        except Exception:
            heur = material_and_position_eval(self.board)
        self.engine_score_cp = heur if self.board.turn == chess.WHITE else -heur
        self.update_eval_bar()
        self.update_move_list()
        self.update_engine_info_panel()
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
            self.info_depth_var.set(f"Depth: {int(float(v))}")

        depth_scale = ttk.Scale(
            row1,
            from_=2,
            to=6,
            orient="horizontal",
            variable=self.depth_var,
            command=on_depth_change,
        )
        depth_scale.pack(side="right", fill="x", expand=True, padx=(12, 0))

        # Time
        row2 = ttk.Frame(frm)
        row2.pack(fill="x", pady=4)
        lbl2 = ttk.Label(row2, text=f"Time (s): {self.time_var.get():.1f}")
        lbl2.pack(side="left")

        def on_time_change(v):
            lbl2.config(text=f"Time (s): {float(v):.1f}")

        ttk.Scale(
            row2,
            from_=0.2,
            to=5.0,
            orient="horizontal",
            variable=self.time_var,
            command=on_time_change,
        ).pack(side="right", fill="x", expand=True, padx=(12, 0))

        # NN enable
        row3 = ttk.Frame(frm)
        row3.pack(fill="x", pady=4)
        ttk.Checkbutton(
            row3,
            text="Use Neural Net (requires value_net.pt)",
            variable=self.nn_var,
        ).pack(anchor="w")

        # NN weight
        row4 = ttk.Frame(frm)
        row4.pack(fill="x", pady=4)
        lbl4 = ttk.Label(row4, text=f"NN Weight: {self.nn_weight_var.get():.1f}")
        lbl4.pack(side="left")

        def on_nnw(v):
            lbl4.config(text=f"NN Weight: {float(v):.1f}")

        ttk.Scale(
            row4,
            from_=0.1,
            to=0.9,
            orient="horizontal",
            variable=self.nn_weight_var,
            command=on_nnw,
        ).pack(side="right", fill="x", expand=True, padx=(12, 0))

        # Buttons
        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="Close", command=top.destroy).pack(side="right")


# ------------- Main -------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ChessApp(root)
    root.mainloop()
