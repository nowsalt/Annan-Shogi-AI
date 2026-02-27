"""AIプレイヤー: MCTS+NNで最善手を返す."""

import numpy as np
import torch

from config import Config
from model import AnnanNet
from mcts import MCTS


class AIPlayer:
    """AlphaZero方式のAIプレイヤー."""

    def __init__(self, model_or_inferencer, config: Config = Config()):
        if isinstance(model_or_inferencer, AnnanNet):
            from inferencer import BatchInferencer
            self.model = model_or_inferencer
            self.inferencer = BatchInferencer(model_or_inferencer, config)
            self._owns_inferencer = True
        else:
            self.model = model_or_inferencer.model
            self.inferencer = model_or_inferencer
            self._owns_inferencer = False
            
        self.config = config
        self.mcts = MCTS(self.inferencer, config)

    def shutdown(self):
        """自分が生成したInferencerなら終了させる."""
        if self._owns_inferencer:
            self.inferencer.shutdown()

    def select_move(self, state, temperature: float = 0.0):
        """MCTS探索で最善手を選択する.

        引数:
            state: 現在のゲーム状態
            temperature: 温度パラメータ
                0.0: 最も訪問回数の多い手を選択 (貪欲)
                1.0: 訪問回数に比例した確率で選択 (探索的)

        戻り値:
            (選択した手, 方策確率分布)
        """
        visit_counts = self.mcts.search(state)

        if not visit_counts:
            return None, {}

        moves = list(visit_counts.keys())
        counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)

        if temperature == 0.0:
            # 貪欲選択
            best_idx = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[best_idx] = 1.0
        else:
            # 温度付き確率選択
            counts_temp = counts ** (1.0 / temperature)
            probs = counts_temp / np.sum(counts_temp)

        move_idx = np.random.choice(len(moves), p=probs)

        # 方策分布 (学習用)
        policy = {m: p for m, p in zip(moves, probs)}

        return moves[move_idx], policy

    @classmethod
    def load(cls, model_path: str, config: Config = Config()) -> "AIPlayer":
        """学習済みモデルを読み込んでプレイヤーを作成する."""
        model = AnnanNet(config)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()
        model.to(config.device)
        return cls(model, config)
