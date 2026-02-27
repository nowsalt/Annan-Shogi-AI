"""バッチ推論器: 複数のMCTSスレッドからの推論要求を蓄積し、まとめてGPUで評価する."""

import threading
import time
import torch
import numpy as np
from typing import List, Tuple
from concurrent.futures import Future

from config import Config
from model import AnnanNet


class BatchInferencer:
    """複数スレッドからの推論をまとめてバッチ処理するクラス."""

    def __init__(self, model: AnnanNet, config: Config):
        self.model = model
        self.config = config
        self.device = config.device
        
        self.batch_size = getattr(config, "inference_batch_size", 32)
        self.timeout = getattr(config, "inference_timeout", 0.01)  # 10ms
        
        self.queue: List[Tuple[np.ndarray, Future]] = []
        self.lock = threading.Lock()
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("BatchInferencer worker thread started.")

    def predict(self, state_tensor: np.ndarray) -> Future:
        """推論リクエストをキューに追加し、Futureを返す."""
        future = Future()
        with self.lock:
            self.queue.append((state_tensor, future))
        return future

    def _worker_loop(self):
        """裏で継続的にキューを監視し、推論を実行する."""
        self.model.eval()
        while self.is_running or len(self.queue) > 0:
            batch = []
            
            with self.lock:
                if len(self.queue) >= self.batch_size:
                    batch = self.queue[:self.batch_size]
                    self.queue = self.queue[self.batch_size:]
                elif len(self.queue) > 0:
                    # Timeout処理は厳密にやると複雑なので、キューに要素があれば少々待って消化する簡易実装
                    batch = self.queue[:]
                    self.queue = []
            
            if not batch:
                time.sleep(self.timeout / 2.0)
                continue
                
            # バッチ推論の実行
            tensors = [item[0] for item in batch]
            futures = [item[1] for item in batch]
            
            x = torch.tensor(np.stack(tensors), dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                policy_logits, values = self.model(x)
                
            policy_logits = policy_logits.cpu().numpy()
            values = values.cpu().numpy()
            
            
            for i in range(len(futures)):
                # 結果をセット
                futures[i].set_result((policy_logits[i], float(values[i][0])))

    def shutdown(self):
        """ワーカースレッドを安全に終了させる."""
        self.is_running = False
        print("Waiting for BatchInferencer worker thread to join...")
        self.worker_thread.join()
        print("BatchInferencer worker thread joined.")
