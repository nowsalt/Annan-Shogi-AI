"""ハイパーパラメータ設定."""


class Config:
    """AlphaZero学習・探索のハイパーパラメータ."""

    # --- ニューラルネット ---
    num_res_blocks: int = 10        # ResNetブロック数
    num_channels: int = 128         # 畳み込みチャンネル数
    input_channels: int = 44       # 入力チャンネル数 (盤面特徴量)

    # --- MCTS ---
    num_simulations: int = 50       # 1手あたりのシミュレーション回数 (CPU実用向けに400から変更)
    c_puct: float = 1.5             # PUCT探索定数
    dirichlet_alpha: float = 0.3    # ルートノードのDirichletノイズα
    dirichlet_epsilon: float = 0.25 # ノイズの混合比率

    # --- 自己対局 ---
    temperature_threshold: int = 30 # この手数まで温度=1.0, 以降は0に近づける
    num_self_play_games: int = 100  # 1イテレーションあたりの自己対局数

    # --- 学習 ---
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4      # L2正則化
    num_epochs: int = 10            # 1イテレーションあたりの学習エポック数
    replay_buffer_size: int = 50000 # リプレイバッファの最大サイズ

    # --- 方策の出力次元 ---
    # 盤上移動: 9×9 始点 × 9×9 終点 × 2 (成/不成) = 13,122
    # 駒打ち: 7 駒種 × 9×9 = 567
    # 合計: 13,689 (実際の合法手はこの一部)
    policy_size: int = 13_689

    # --- その他 ---
    board_size: int = 9
    device: str = "cpu"             # "cuda" if GPU available
