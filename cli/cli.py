"""UBGI CLI - Run MiniChess AI vs AI or Human vs AI matches."""

import argparse
import os
import sys
import threading
import random

from cli.games.minichess import get_context as _minichess_ctx
from gui.ubgi_client import UBGIEngine

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CHESS_FAMILY = frozenset({"minichess"})
_SHOGI_FAMILY = frozenset()
_BOARD_GAMES = _CHESS_FAMILY | _SHOGI_FAMILY
_game_ctx: dict = {}

def _init_game(game_name: str, board_size: int | None = None) -> None:
    _game_ctx.update(_minichess_ctx())

def print_board(state) -> None:
    printer = _game_ctx.get("print_board")
    if printer is not None:
        printer(state, _game_ctx)

def format_search_info(info: dict | None) -> str:
    if not info: return ""
    parts: list[str] = []
    
    depth = info.get("depth")
    if depth is not None:
        parts.append(f"depth={depth}")
        
    score_cp = info.get("score_cp")
    if score_cp is not None: parts.append(f"score={score_cp / 100.0:+.2f}")
    elif info.get("score_mate") is not None: parts.append(f"mate={info['score_mate']}")
    if info.get("string") == "Random Opening": parts.append("Random Opening")
    return ", ".join(parts)

def format_move_display(move_or_uci, state=None) -> str:
    game_name = _game_ctx.get("name", "generic")
    if game_name in _BOARD_GAMES and not isinstance(move_or_uci, str):
        return _game_ctx["format_move"](move_or_uci)
    return str(move_or_uci)

def _init_game_state(game_name: str):
    if game_name == "generic": return None
    if "make_state" in _game_ctx: return _game_ctx["make_state"](_game_ctx.get("board_size", 15))
    return _game_ctx["state_class"].initial()

def _check_game_over(state, game_name: str, verbose: bool) -> str | None:
    if game_name == "generic": return None
    result, winner = _game_ctx.get("check_game_over")(state)
    first_label, second_label = ("Sente", "Gote") if game_name in _SHOGI_FAMILY else ("White", "Black")

    if result in ("win", "checkmate", "perpetual_check", "stalemate_loss"):
        winner_str = first_label if winner == 0 else second_label
        if verbose: print(f"  >> {winner_str} wins!", flush=True)
        return "white" if winner == 0 else "black"
    if result == "draw":
        if verbose: print("  >> Draw!", flush=True)
        return "draw"
    if result == "no_moves":
        if verbose: print(f"  >> {(first_label if winner == 1 else second_label)} has no legal moves!", flush=True)
        return "white" if winner == 0 else "black"
    return None

def _determine_side_to_move(state, game_name: str, uci_moves: list[str]) -> bool:
    if game_name in _BOARD_GAMES: return state.player == 0
    return len(uci_moves) % 2 == 0

def run_game(
    white_path: str, black_path: str, time_limit: int, white_algo: str, black_algo: str,
    verbose: bool = True, show_board: bool = True, game_num: int | None = None, total_games: int | None = None,
    depth: int = 0, params: list[str] | None = None, white_params: list[str] | None = None, black_params: list[str] | None = None,
    random_open: str = "0"
) -> str:
    game_name = _game_ctx.get("name", "generic")
    has_state = game_name != "generic"
    uci_moves: list[str] = []
    move_number = 0
    state = _init_game_state(game_name)

    # 處理隨機開局步數解析
    k_limit = 0
    if random_open:
        if "-" in random_open:
            try:
                low, high = map(int, random_open.split("-"))
                k_limit = random.randint(low, high)
            except ValueError: pass
        elif random_open.isdigit():
            k_limit = int(random_open)

    if verbose:
        print(f"=== {'Game ' + str(game_num) + '/' + str(total_games) if game_num else 'New Game'} ===", flush=True)
        print(f"  White: {'Human' if white_path == 'human' else white_algo}", flush=True)
        print(f"  Black: {'Human' if black_path == 'human' else black_algo}", flush=True)
        print(f"  Time limit: {time_limit}ms per move", flush=True)
        if k_limit > 0: print(f"  Random Opening: First {k_limit} moves", flush=True)
        if has_state and show_board: print_board(state)

    w_eng, b_eng = None, None
    try:
        if white_path != "human":
            w_opts = {"Algorithm": white_algo}
            for p in (params or []) + (white_params or []):
                if "=" in p: k, v = p.split("=", 1); w_opts[k] = v
            w_eng = UBGIEngine(os.path.abspath(white_path), initial_options=w_opts)
            w_eng.new_game()

        if black_path != "human":
            b_opts = {"Algorithm": black_algo}
            for p in (params or []) + (black_params or []):
                if "=" in p: k, v = p.split("=", 1); b_opts[k] = v
            b_eng = UBGIEngine(os.path.abspath(black_path), initial_options=b_opts)
            b_eng.new_game()

        while True:
            if has_state:
                over = _check_game_over(state, game_name, verbose)
                if over is not None: return over
                is_white = _determine_side_to_move(state, game_name, uci_moves)
            else:
                is_white = len(uci_moves) % 2 == 0

            side_name = "White" if is_white else "Black"
            if is_white: move_number += 1

            bestmove_uci, info = None, None

            if (white_path if is_white else black_path) == "human":
                bestmove_uci = input(f"  {side_name}'s turn. Enter move: ").strip()
            else:
                # 【新增】隨機開局攔截
                if len(uci_moves) < k_limit and has_state and state.legal_actions:
                    random_move = random.choice(state.legal_actions)
                    bestmove_uci = _game_ctx["move_to_uci"](random_move)
                    info = {"depth": 0, "score_cp": 0, "string": "Random Opening"}
                    time.sleep(0.01) # 微小延遲，避免瞬間印完
                else:
                    active_eng = w_eng if is_white else b_eng
                    active_eng.set_position(moves=uci_moves)

                    done_event = threading.Event()
                    move_res, last_info = {}, {}

                    def info_cb(inf): 
                        if 'depth' in inf: last_info.update(inf)
                    def done_cb(bm): 
                        move_res['bm'] = bm; done_event.set()

                    if depth > 0:
                        active_eng.go(depth=depth, info_callback=info_cb, done_callback=done_cb)
                        done_event.wait()
                    else:
                        active_eng.go(movetime=time_limit, info_callback=info_cb, done_callback=done_cb)
                        if not done_event.wait(timeout=(time_limit / 1000.0) + 30.0):
                            active_eng.stop_and_wait(timeout=2.0)

                    bestmove_uci = move_res.get('bm')
                    info = last_info

            if not bestmove_uci or bestmove_uci in ("none", "(none)", "0000"):
                if verbose: print(f"  >> {side_name} engine failed to return a move! {side_name} loses.", flush=True)
                return "black" if is_white else "white"

            uci_moves.append(bestmove_uci)

            if verbose:
                prefix = f"{move_number}." if is_white else f"{move_number}..."
                info_str = format_search_info(info)
                print(f"  {prefix} {side_name}: {format_move_display(bestmove_uci)} ({info_str})", flush=True)

            if has_state:
                apply_fn = _game_ctx.get("apply_move")
                if apply_fn is not None:
                    state, _ = apply_fn(state, bestmove_uci, _game_ctx)
            if verbose and has_state and show_board: print_board(state)

    finally:
        if w_eng: w_eng.quit()
        if b_eng: b_eng.quit()

def run_tournament(*args, **kwargs):
    pass 

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="minichess")
    parser.add_argument("--white", required=True)
    parser.add_argument("--black", required=True)
    parser.add_argument("--time", type=int, default=2000)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--white-algo", default="minimax")
    parser.add_argument("--black-algo", default="minimax")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-board", action="store_true") 
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--random-open", type=str, default="0") # [新增] 接收隨機開局參數
    parser.add_argument("--param", action="append", default=[])
    parser.add_argument("--white-param", action="append", default=[])
    parser.add_argument("--black-param", action="append", default=[])
    args = parser.parse_args()

    _init_game(args.game.lower())
    verbose = False if args.quiet else (True if args.verbose or args.games == 1 else False)
    show_board = not args.no_board

    if args.games > 1:
        run_tournament(args.white, args.black, args.time, args.white_algo, args.black_algo, args.games, verbose, show_board, args.depth, args.param, args.white_param, args.black_param)
        return

    try:
        result = run_game(
            args.white, args.black, args.time, args.white_algo, args.black_algo, 
            verbose, show_board, depth=args.depth, params=args.param, white_params=args.white_param, black_params=args.black_param,
            random_open=args.random_open
        )
        res_map = {'white': '1-0', 'black': '0-1', 'draw': '1/2-1/2'}
        print(f"Result: {res_map.get(result, '1/2-1/2')}", flush=True)
    except KeyboardInterrupt:
        print("\nGame aborted.", flush=True)

if __name__ == "__main__":
    main()