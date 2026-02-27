"""盤面エンコーダ: State → テンソル, Move → インデックス変換.

盤面を44チャンネル×9×9のテンソルに変換する。
指し手を方策ベクトルのインデックスに変換する。
"""

import sys
import os
import numpy as np

# エンジンのパスを追加
ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Annan-Shogi")
sys.path.insert(0, ENGINE_DIR)

from annan_shogi.core.color import Color
from annan_shogi.core.piece import PieceType, HAND_PIECE_TYPES
from annan_shogi.core.square import Square
from annan_shogi.core.move import Move
from annan_shogi.state import State
from annan_shogi.rules.annan_rule import get_effective_piece_type

# 駒種の一覧 (チャンネル順)
_PIECE_TYPES = [
    PieceType.FU, PieceType.KY, PieceType.KE, PieceType.GI,
    PieceType.KI, PieceType.KA, PieceType.HI, PieceType.OU,
    PieceType.TO, PieceType.NY, PieceType.NK, PieceType.NG,
    PieceType.UM, PieceType.RY,
]

_PIECE_TYPE_TO_IDX = {pt: i for i, pt in enumerate(_PIECE_TYPES)}


def encode_state(state: State) -> np.ndarray:
    """盤面状態を44ch×9×9のnumpy配列に変換する.

    チャンネル構成:
      0-13:  先手の各駒種 (14種)
      14-27: 後手の各駒種 (14種)
      28-34: 先手の持ち駒数 (7種, 全マス同じ値で正規化)
      35-41: 後手の持ち駒数 (7種)
      42:    手番 (先手=1, 後手=0)
      43:    安南ルール (実効駒種が変化しているマス=1)
    """
    planes = np.zeros((44, 9, 9), dtype=np.float32)

    # --- 盤上の駒 (ch 0-27) ---
    for rank in range(9):
        for file in range(1, 10):
            sq = Square(file, rank)
            piece = state.board[sq]
            if piece is None:
                continue
            col = 9 - file  # 配列のカラムindex
            pt_idx = _PIECE_TYPE_TO_IDX[piece.piece_type]
            if piece.color is Color.BLACK:
                planes[pt_idx, rank, col] = 1.0
            else:
                planes[14 + pt_idx, rank, col] = 1.0

    # --- 持ち駒 (ch 28-41) ---
    for i, pt in enumerate(HAND_PIECE_TYPES):
        black_count = state.stands[Color.BLACK].count(pt)
        white_count = state.stands[Color.WHITE].count(pt)
        # 正規化 (最大枚数で割る)
        max_count = _MAX_HAND_COUNT.get(pt, 1)
        if black_count > 0:
            planes[28 + i, :, :] = black_count / max_count
        if white_count > 0:
            planes[35 + i, :, :] = white_count / max_count

    # --- 手番 (ch 42) ---
    if state.turn is Color.BLACK:
        planes[42, :, :] = 1.0

    # --- 安南ルール情報 (ch 43) ---
    for rank in range(9):
        for file in range(1, 10):
            sq = Square(file, rank)
            piece = state.board[sq]
            if piece is None or piece.piece_type is PieceType.OU:
                continue
            try:
                eff = get_effective_piece_type(state.board, sq, piece.color)
                if eff != piece.piece_type:
                    col = 9 - file
                    planes[43, rank, col] = 1.0
            except Exception:
                pass

    return planes


# 持ち駒の最大枚数 (正規化用)
_MAX_HAND_COUNT = {
    PieceType.FU: 9, PieceType.KY: 2, PieceType.KE: 2,
    PieceType.GI: 2, PieceType.KI: 2, PieceType.KA: 1,
    PieceType.HI: 1,
}


# ===== 指し手 ↔ インデックス変換 =====

def move_to_index(move: Move) -> int:
    """指し手を方策ベクトルのインデックスに変換する.

    盤上移動: src_rank*9 + (9-src_file) を始点,
              dst_rank*9 + (9-dst_file) を終点として
              index = src*81*2 + dst*2 + promote
    駒打ち:   index = 81*81*2 + piece_idx*81 + dst_rank*9 + (9-dst_file)
    """
    if move.is_drop:
        assert move.drop is not None
        pt_idx = HAND_PIECE_TYPES.index(move.drop)
        dst_idx = move.dst.rank * 9 + (9 - move.dst.file)
        return 81 * 81 * 2 + pt_idx * 81 + dst_idx
    else:
        assert move.src is not None
        src_idx = move.src.rank * 9 + (9 - move.src.file)
        dst_idx = move.dst.rank * 9 + (9 - move.dst.file)
        promote = 1 if move.promote else 0
        return src_idx * 81 * 2 + dst_idx * 2 + promote


def index_to_move(index: int) -> Move:
    """方策ベクトルのインデックスを指し手に変換する."""
    drop_start = 81 * 81 * 2

    if index >= drop_start:
        # 駒打ち
        rem = index - drop_start
        pt_idx = rem // 81
        dst_idx = rem % 81
        dst_rank = dst_idx // 9
        dst_file = 9 - (dst_idx % 9)
        return Move(dst=Square(dst_file, dst_rank), drop=HAND_PIECE_TYPES[pt_idx])
    else:
        # 盤上移動
        promote = index % 2
        rem = index // 2
        dst_idx = rem % 81
        src_idx = rem // 81
        src_rank = src_idx // 9
        src_file = 9 - (src_idx % 9)
        dst_rank = dst_idx // 9
        dst_file = 9 - (dst_idx % 9)
        return Move(
            dst=Square(dst_file, dst_rank),
            src=Square(src_file, src_rank),
            promote=bool(promote),
        )
