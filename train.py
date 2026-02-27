"""学習スクリプト: 自己対局データでニューラルネットを学習する."""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Annan-Shogi")
sys.path.insert(0, ENGINE_DIR)

from config import Config
from model import AnnanNet
from self_play import run_self_play, save_training_data, load_training_data


class ShogiDataset(Dataset):
    """学習用データセット."""

    def __init__(self, data: list[dict]):
        self.states = torch.tensor(
            np.array([d["state"] for d in data]), dtype=torch.float32
        )
        self.policies = torch.tensor(
            np.array([d["policy"] for d in data]), dtype=torch.float32
        )
        self.values = torch.tensor(
            np.array([d["value"] for d in data]), dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


def train_epoch(model: AnnanNet, dataloader: DataLoader,
                optimizer: optim.Optimizer, device: str) -> dict:
    """1エポックの学習を実行する."""
    model.train()
    total_loss = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    num_batches = 0

    for states, policies, values in dataloader:
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)

        # 順伝播
        pred_policy, pred_value = model(states)

        # 方策損失 (クロスエントロピー)
        # policies は確率分布なので、ソフトターゲットのクロスエントロピー
        log_softmax_policy = torch.log_softmax(pred_policy, dim=1)
        policy_loss = -torch.sum(policies * log_softmax_policy, dim=1).mean()

        # 価値損失 (MSE)
        value_loss = nn.MSELoss()(pred_value, values)

        # 合計損失
        loss = policy_loss + value_loss

        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
        num_batches += 1

    return {
        "total_loss": total_loss / max(num_batches, 1),
        "policy_loss": policy_loss_sum / max(num_batches, 1),
        "value_loss": value_loss_sum / max(num_batches, 1),
    }


def training_loop(config: Config = Config(), num_iterations: int = 10):
    """自己対局 → 学習 を繰り返すメインループ.

    引数:
        num_iterations: イテレーション回数
    """
    device = config.device
    model = AnnanNet(config).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # データディレクトリ作成
    os.makedirs("data/games", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)

    # リプレイバッファ
    replay_buffer = []

    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*50}")
        print(f"イテレーション {iteration}/{num_iterations}")
        print(f"{'='*50}")

        # --- 自己対局 ---
        print(f"\n[自己対局] {config.num_self_play_games}局...")
        new_data = run_self_play(model, config)

        # データ保存
        save_training_data(
            new_data,
            f"data/games/iter_{iteration:04d}.npz"
        )

        # リプレイバッファに追加
        replay_buffer.extend(new_data)
        if len(replay_buffer) > config.replay_buffer_size:
            replay_buffer = replay_buffer[-config.replay_buffer_size:]

        print(f"  バッファサイズ: {len(replay_buffer)}")

        # --- 学習 ---
        print(f"\n[学習] {config.num_epochs}エポック...")
        dataset = ShogiDataset(replay_buffer)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        with tqdm(range(1, config.num_epochs + 1), desc="Training Epochs") as pbar:
            for epoch in pbar:
                losses = train_epoch(model, dataloader, optimizer, device)
                pbar.set_postfix({
                    "loss": f"{losses['total_loss']:.4f}", 
                    "pol": f"{losses['policy_loss']:.4f}", 
                    "val": f"{losses['value_loss']:.4f}"
                })

        # --- モデル保存 ---
        model_path = f"data/models/model_iter_{iteration:04d}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"  モデル保存: {model_path}")

    # 最終モデル保存
    torch.save(model.state_dict(), "data/models/best_model.pt")
    print(f"\n学習完了! 最終モデル: data/models/best_model.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="安南将棋AI 学習")
    parser.add_argument("--iterations", type=int, default=10, help="イテレーション回数")
    parser.add_argument("--games", type=int, default=None, help="1イテレーションの自己対局数")
    parser.add_argument("--simulations", type=int, default=None, help="MCTS探索回数")
    parser.add_argument("--device", type=str, default="cpu", help="デバイス (cpu/cuda)")
    args = parser.parse_args()

    config = Config()
    if args.games:
        config.num_self_play_games = args.games
    if args.simulations:
        config.num_simulations = args.simulations
    config.device = args.device

    training_loop(config, num_iterations=args.iterations)
