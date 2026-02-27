# 安南将棋AI (AlphaZero方式)

MCTS + ResNet による安南将棋の強い AI。

## 依存関係

```bash
pip install -r requirements.txt
```

## 使い方

### 学習の実行

```bash
# 軽量テスト (少ない局数・探索)
python3 train.py --iterations 3 --games 5 --simulations 50

# 本格学習
python3 train.py --iterations 100 --games 100 --simulations 400

# GPU使用
python3 train.py --iterations 100 --device cuda
```

### モデル比較

```bash
python3 arena.py data/models/model_iter_0001.pt data/models/model_iter_0010.pt --games 20
```

## 構成

| ファイル | 内容 |
|----------|------|
| `config.py` | ハイパーパラメータ |
| `encoder.py` | 盤面 → テンソル変換 |
| `model.py` | ResNet (方策+価値ヘッド) |
| `mcts.py` | MCTS探索 (PUCT選択) |
| `player.py` | AIプレイヤー |
| `self_play.py` | 自己対局データ生成 |
| `train.py` | 学習メインループ |
| `arena.py` | モデル比較対局 |
