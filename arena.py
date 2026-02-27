"""モデル比較対局: 新旧モデルを対戦させて強さを測定する."""

import os
import sys
import torch

ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Annan-Shogi")
sys.path.insert(0, ENGINE_DIR)

from annan_shogi import Game, Color
from config import Config
from model import AnnanNet
from player import AIPlayer


def arena_match(player1: AIPlayer, player2: AIPlayer,
                num_games: int = 20) -> dict:
    """2つのプレイヤーを対戦させる.

    各プレイヤーが先手・後手を同数ずつ担当する。

    戻り値: {"player1_wins": int, "player2_wins": int, "draws": int}
    """
    results = {"player1_wins": 0, "player2_wins": 0, "draws": 0}

    for i in range(num_games):
        # 偶数回は player1 が先手、奇数回は player2 が先手
        if i % 2 == 0:
            black_player, white_player = player1, player2
            p1_color = "BLACK"
        else:
            black_player, white_player = player2, player1
            p1_color = "WHITE"

        game = Game()
        move_count = 0

        while game.result.value == "ONGOING" and move_count < 500:
            state = game.state
            current_player = black_player if game.turn is Color.BLACK else white_player
            move, _ = current_player.select_move(state, temperature=0.0)

            if move is None:
                break

            game.apply(move)
            move_count += 1

        result = game.result.value
        if result == "BLACK_WIN":
            winner = "BLACK"
        elif result == "WHITE_WIN":
            winner = "WHITE"
        else:
            winner = None

        if winner is None:
            results["draws"] += 1
            outcome = "引分"
        elif winner == p1_color:
            results["player1_wins"] += 1
            outcome = "P1勝"
        else:
            results["player2_wins"] += 1
            outcome = "P2勝"

        print(f"  対局 {i+1}/{num_games}: {move_count}手 → {outcome}")

    return results


def compare_models(model_path_1: str, model_path_2: str,
                   num_games: int = 20, config: Config = Config()):
    """2つのモデルを比較対局させる."""
    # 比較時は探索を軽くする
    config.num_simulations = 100

    print(f"モデル1: {model_path_1}")
    print(f"モデル2: {model_path_2}")
    print(f"対局数: {num_games}")

    player1 = AIPlayer.load(model_path_1, config)
    player2 = AIPlayer.load(model_path_2, config)

    results = arena_match(player1, player2, num_games)

    total = num_games
    p1_rate = results["player1_wins"] / total * 100
    p2_rate = results["player2_wins"] / total * 100
    draw_rate = results["draws"] / total * 100

    print(f"\n結果:")
    print(f"  モデル1: {results['player1_wins']}勝 ({p1_rate:.1f}%)")
    print(f"  モデル2: {results['player2_wins']}勝 ({p2_rate:.1f}%)")
    print(f"  引き分け: {results['draws']} ({draw_rate:.1f}%)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="モデル比較対局")
    parser.add_argument("model1", help="モデル1のパス")
    parser.add_argument("model2", help="モデル2のパス")
    parser.add_argument("--games", type=int, default=20, help="対局数")
    args = parser.parse_args()

    compare_models(args.model1, args.model2, args.games)
