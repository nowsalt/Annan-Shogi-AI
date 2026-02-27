"""自己対局: AIが自分自身と対局して学習データを生成する."""

import os
import sys
import json
import time
import numpy as np
import torch
from tqdm import tqdm

ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Annan-Shogi")
sys.path.insert(0, ENGINE_DIR)

from annan_shogi import Game, Color
from config import Config
from encoder import encode_state, move_to_index
from model import AnnanNet
from player import AIPlayer


def self_play_game(player: AIPlayer, config: Config) -> list[dict]:
    """1局の自己対局を実行し、学習データを返す.

    戻り値: [{"state": ndarray, "policy": ndarray, "value": float}, ...]
    """
    game = Game()
    history = []  # (盤面テンソル, 方策, 手番)

    move_count = 0

    while game.result.value == "ONGOING":
        state = game.state

        # 温度: 序盤は探索的、中終盤は貪欲
        if move_count < config.temperature_threshold:
            temperature = 1.0
        else:
            temperature = 0.1

        move, policy_dict = player.select_move(state, temperature)

        if move is None:
            break

        # 盤面テンソル
        state_tensor = encode_state(state)

        # 方策ベクトル (合法手のみ)
        policy_vec = np.zeros(config.policy_size, dtype=np.float32)
        for m, p in policy_dict.items():
            idx = move_to_index(m)
            policy_vec[idx] = p

        history.append({
            "state": state_tensor,
            "policy": policy_vec,
            "turn": str(state.turn),
        })

        game.apply(move)
        move_count += 1

        # 安全弁: 最大500手で打ち切り
        if move_count >= 500:
            break

    # 勝敗から報酬を設定
    result = game.result.value
    training_data = []

    for h in history:
        if result == "BLACK_WIN":
            value = 1.0 if h["turn"] == "BLACK" else -1.0
        elif result == "WHITE_WIN":
            value = 1.0 if h["turn"] == "WHITE" else -1.0
        else:
            value = 0.0  # 引き分け / 打ち切り

        training_data.append({
            "state": h["state"],
            "policy": h["policy"],
            "value": np.float32(value),
        })

    return training_data


def run_self_play(model: AnnanNet, config: Config, num_games: int = None):
    """複数局の自己対局を実行する.

    戻り値: 学習データのリスト
    """
    if num_games is None:
        num_games = config.num_self_play_games

    model.eval()
    player = AIPlayer(model, config)
    all_data = []

    print(f"  自己対局 {num_games}局開始...")
    with tqdm(total=num_games, desc="Self-Play Games") as pbar:
        for i in range(num_games):
            t0 = time.time()
            game_data = self_play_game(player, config)
            elapsed = time.time() - t0

            all_data.extend(game_data)
            pbar.set_postfix({"last_moves": len(game_data), "time": f"{elapsed:.1f}s"})
            pbar.update(1)

    return all_data


def save_training_data(data: list[dict], path: str) -> None:
    """学習データをファイルに保存する."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    states = np.array([d["state"] for d in data])
    policies = np.array([d["policy"] for d in data])
    values = np.array([d["value"] for d in data])
    np.savez_compressed(path, states=states, policies=policies, values=values)
    print(f"  保存: {path} ({len(data)}サンプル)")


def load_training_data(path: str) -> list[dict]:
    """学習データを読み込む."""
    npz = np.load(path)
    data = []
    for i in range(len(npz["states"])):
        data.append({
            "state": npz["states"][i],
            "policy": npz["policies"][i],
            "value": npz["values"][i],
        })
    return data
