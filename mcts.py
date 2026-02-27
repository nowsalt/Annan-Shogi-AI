"""AlphaZero方式のモンテカルロ木探索 (MCTS).

PUCT選択 + ニューラルネットによる評価で探索を行う。
"""

import math
import numpy as np
import torch

from config import Config
from encoder import encode_state, move_to_index
from model import AnnanNet


class MCTSNode:
    """MCTSの探索ノード."""

    __slots__ = ("_state", "parent", "move", "children",
                 "visit_count", "value_sum", "prior")

    def __init__(self, state, parent=None, move=None, prior: float = 0.0):
        self._state = state
        self.parent = parent
        self.move = move          # このノードに至った手
        self.children = []        # 子ノードリスト
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior        # NNによる事前確率

    @property
    def state(self):
        """状態を遅延評価で生成する."""
        if self._state is None:
            self._state = self.parent.state.apply_move(self.move)
        return self._state

    @property
    def q_value(self) -> float:
        """平均価値."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """展開済みかどうか."""
        return len(self.children) > 0

    def select_child(self, c_puct: float) -> "MCTSNode":
        """PUCT選択: 探索と活用のバランスを取った子ノード選択."""
        total_visits = sum(c.visit_count for c in self.children)
        sqrt_total = math.sqrt(total_visits + 1)

        best_score = -float("inf")
        best_child = None

        for child in self.children:
            # Q値 + 探索ボーナス
            q = child.q_value
            u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_child = child

        return best_child


class MCTS:
    """モンテカルロ木探索."""

    def __init__(self, model: AnnanNet, config: Config = Config()):
        self.model = model
        self.config = config

    @torch.no_grad()
    def search(self, state) -> dict:
        """MCTS探索を実行し、各合法手の訪問回数を返す.

        戻り値: {Move: visit_count} の辞書
        """
        root = MCTSNode(state)
        self._expand(root)
        self._add_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            node = root

            # 選択: 葉ノードまで降りる
            while node.is_expanded() and node.children:
                node = node.select_child(self.config.c_puct)

            # 評価
            if node.state.result.value != "ONGOING":
                # 終局: 勝敗を評価
                value = self._terminal_value(node)
            else:
                # 展開 + NNで評価
                value = self._expand(node)

            # バックプロパゲーション
            self._backpropagate(node, value)

        # 訪問回数を返す
        visit_counts = {}
        for child in root.children:
            visit_counts[child.move] = child.visit_count

        return visit_counts

    def _expand(self, node: MCTSNode) -> float:
        """ノードを展開し、NNで評価値を返す."""
        state = node.state
        legal_moves = state.get_legal_moves()

        if not legal_moves:
            return -1.0  # 合法手なし = 負け

        # NNで方策と価値を取得
        state_tensor = torch.tensor(
            encode_state(state), dtype=torch.float32
        ).unsqueeze(0).to(self.config.device)

        policy_logits, value = self.model(state_tensor)
        policy_logits = policy_logits.squeeze(0).cpu().numpy()
        value = value.item()

        # 合法手のみの確率分布を作成
        legal_indices = [move_to_index(m) for m in legal_moves]
        legal_logits = policy_logits[legal_indices]

        # ソフトマックス
        max_logit = np.max(legal_logits)
        exp_logits = np.exp(legal_logits - max_logit)
        priors = exp_logits / np.sum(exp_logits)

        # 子ノードを作成
        for move, prior in zip(legal_moves, priors):
            # 状態は遅延評価するため None を渡す
            child = MCTSNode(None, parent=node, move=move, prior=float(prior))
            node.children.append(child)

        return value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """評価値を親に伝播する."""
        while node is not None:
            node.visit_count += 1
            # 手番が交互なので、1つ上のノードでは符号を反転
            node.value_sum += value
            value = -value
            node = node.parent

    def _terminal_value(self, node: MCTSNode) -> float:
        """終局ノードの評価値."""
        result = node.state.result.value
        parent_turn = node.parent.state.turn if node.parent else None

        if result == "BLACK_WIN":
            return 1.0 if str(parent_turn) == "BLACK" else -1.0
        elif result == "WHITE_WIN":
            return 1.0 if str(parent_turn) == "WHITE" else -1.0
        return 0.0  # 引き分け

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """ルートノードにDirichletノイズを追加して探索を多様化."""
        if not root.children:
            return
        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * len(root.children)
        )
        eps = self.config.dirichlet_epsilon
        for child, n in zip(root.children, noise):
            child.prior = (1 - eps) * child.prior + eps * n
