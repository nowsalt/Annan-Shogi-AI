"""ResNetベースの方策+価値ネットワーク.

AlphaZero方式:
- 入力: 44ch × 9×9 の盤面特徴量
- 共通: ResNetの残差ブロック
- 方策ヘッド: 合法手の確率分布
- 価値ヘッド: 現局面の勝率 [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class ResBlock(nn.Module):
    """残差ブロック."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class AnnanNet(nn.Module):
    """安南将棋のAlphaZeroネットワーク."""

    def __init__(self, config: Config = Config()):
        super().__init__()
        c = config.num_channels

        # 入力層
        self.conv_in = nn.Conv2d(config.input_channels, c, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(c)

        # 残差ブロック
        self.res_blocks = nn.ModuleList([
            ResBlock(c) for _ in range(config.num_res_blocks)
        ])

        # 方策ヘッド
        self.policy_conv = nn.Conv2d(c, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 9 * 9, config.policy_size)

        # 価値ヘッド
        self.value_conv = nn.Conv2d(c, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(9 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """順伝播.

        引数:
            x: (batch, 44, 9, 9) の入力テンソル

        戻り値:
            policy: (batch, policy_size) の対数確率
            value:  (batch, 1) の勝率 [-1, 1]
        """
        # 共通層
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        # 方策ヘッド
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        # 合法手マスク前のlog_softmaxはMCTS側で行う

        # 価値ヘッド
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
