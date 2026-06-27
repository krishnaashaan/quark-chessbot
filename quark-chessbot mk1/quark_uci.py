import sys
from typing import Optional

import chess

from chessbot_mk1 import AlphaBetaEngine


ENGINE_NAME = "QuarkChessBot"
ENGINE_AUTHOR = "Krishnaashaan"

board = chess.Board()
engine = AlphaBetaEngine()

max_depth = 5
move_overhead_ms = 75


def send(message: str) -> None:
    print(message, flush=True)


def parse_int(tokens: list[str], name: str) -> Optional[int]:
    if name not in tokens:
        return None
    index = tokens.index(name)
    if index + 1 >= len(tokens):
        return None
    try:
        return int(tokens[index + 1])
    except ValueError:
        return None


def apply_moves(moves: list[str]) -> None:
    global board
    for move_text in moves:
        try:
            board.push_uci(move_text)
        except ValueError:
            send(f"info string ignored illegal move {move_text}")
            break


def handle_position(command: str) -> None:
    global board
    tokens = command.split()
    if len(tokens) < 2:
        return

    moves_index = tokens.index("moves") if "moves" in tokens else None

    try:
        if tokens[1] == "startpos":
            board = chess.Board()
        elif tokens[1] == "fen":
            fen_end = moves_index if moves_index is not None else len(tokens)
            fen = " ".join(tokens[2:fen_end])
            board = chess.Board(fen)
        else:
            send("info string unsupported position command")
            return
    except ValueError as exc:
        send(f"info string invalid position {exc}")
        board = chess.Board()
        return

    if moves_index is not None:
        apply_moves(tokens[moves_index + 1 :])


def time_budget_from_go(tokens: list[str]) -> tuple[int, float]:
    depth = parse_int(tokens, "depth") or max_depth
    depth = max(1, min(depth, max_depth))

    movetime = parse_int(tokens, "movetime")
    if movetime is not None:
        return depth, max(0.05, (movetime - move_overhead_ms) / 1000.0)

    side_time = parse_int(tokens, "wtime" if board.turn == chess.WHITE else "btime")
    side_inc = parse_int(tokens, "winc" if board.turn == chess.WHITE else "binc") or 0
    movestogo = parse_int(tokens, "movestogo")

    if side_time is None:
        return depth, 2.0

    remaining = max(1, side_time - move_overhead_ms)
    if movestogo and movestogo > 0:
        budget_ms = remaining / max(1, movestogo) + side_inc * 0.5
    else:
        budget_ms = remaining / 30 + side_inc * 0.75

    cap_ms = max(50, remaining * 0.25)
    budget_ms = min(max(50, budget_ms), cap_ms)
    return depth, budget_ms / 1000.0


def safe_bestmove(depth: int, time_limit: float) -> chess.Move | None:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    try:
        move, score = engine.choose_move(board, max_depth=depth, time_limit=time_limit)
        if move in legal_moves:
            send(f"info depth {depth} score cp {score} nodes {engine.nodes}")
            return move
    except Exception as exc:
        send(f"info string search error {exc}")

    return legal_moves[0]


def handle_go(command: str) -> None:
    tokens = command.split()
    depth, time_limit = time_budget_from_go(tokens)
    move = safe_bestmove(depth, time_limit)
    send(f"bestmove {move.uci() if move else '0000'}")


def handle_setoption(command: str) -> None:
    global max_depth, move_overhead_ms
    parts = command.split()
    if "name" not in parts:
        return

    name_start = parts.index("name") + 1
    value_start = parts.index("value") if "value" in parts else len(parts)
    name = " ".join(parts[name_start:value_start]).strip().lower()
    value = " ".join(parts[value_start + 1 :]).strip() if value_start < len(parts) else ""

    try:
        if name == "depth":
            max_depth = max(1, min(20, int(value)))
        elif name == "move overhead":
            move_overhead_ms = max(0, min(5_000, int(value)))
        elif name == "book path":
            engine.set_book_path(value or None)
    except ValueError:
        send(f"info string invalid option value for {name}")


def handle_uci() -> None:
    send(f"id name {ENGINE_NAME}")
    send(f"id author {ENGINE_AUTHOR}")
    send("option name Depth type spin default 5 min 1 max 20")
    send("option name Move Overhead type spin default 75 min 0 max 5000")
    send("option name Book Path type string default <empty>")
    send("uciok")


def main() -> None:
    global board

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        try:
            if line == "uci":
                handle_uci()
            elif line == "isready":
                send("readyok")
            elif line == "ucinewgame":
                board = chess.Board()
                engine.clear_for_new_game()
            elif line.startswith("setoption"):
                handle_setoption(line)
            elif line.startswith("position"):
                handle_position(line)
            elif line.startswith("go"):
                handle_go(line)
            elif line == "stop":
                engine.stop = True
            elif line == "quit":
                break
        except Exception as exc:
            send(f"info string error {exc}")


if __name__ == "__main__":
    main()
