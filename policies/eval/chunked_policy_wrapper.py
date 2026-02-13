from __future__ import annotations

import numpy as np
import torch

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict

from .flow_inference import sample_action_sequence, FlowSampleCfg

@dataclass
class ChunkExecCfg:
    pred_horizon: int = 16     # model predicts total 16
    exec_horizon: int = 8      # execute only 8, then replan
    action_dim: int = 10       # 10D absolute pose target
    proprio_dim: int = 10      # 10D state input


class ChunkedFlowPolicyWrapper:
    """
    Wrapper to adapt a flow-matching policy to chunked/receding-horizon execution.
    - keep a queue of planned actions
    - if queue empty or replan interval reached: run network and refill queue
    - return next action (absolute 10D target) each env step
    """
    def __init__(
        self,
        policy: torch.nn.Module,
        *,
        device: torch.device,
        exec_cfg: ChunkExecCfg,
        sample_cfg: FlowSampleCfg,
    ):
        self.policy = policy
        self.device = device
        self.exec_cfg = exec_cfg
        self.sample_cfg = sample_cfg

        self._queue: Deque[np.ndarray] = deque()
        self._since_replan: int = 0

    def reset(self):
        self._queue.clear()
        self._since_replan = 0

    @torch.no_grad()
    def _replan(self, batch: Dict[str, torch.Tensor]):
        """
        batch:
          "observation.image": (B,3,H,W) float in [0,1]
          "observation.state": (B,10)
        """
        img = batch["observation.image"].to(self.device)
        st = batch["observation.state"].to(self.device)

        # Encode obs (your policy API)
        obs_memory = self.policy.encode_obs(images=img, proprio=st)  # (B, memory_len, d_model)

        # Sample full predicted horizon (B, H, 10)
        seq = sample_action_sequence(
            policy=self.policy,
            obs_memory=obs_memory,
            pred_horizon=int(self.exec_cfg.pred_horizon),
            action_dim=int(self.exec_cfg.action_dim),
            cfg=self.sample_cfg, 
            device=self.device,
        )

        seq = seq.detach().float().cpu().numpy()  # (B,H,10)
        assert seq.shape[0] == 1, "Evaluation wrapper assumes batch size 1."

        # Push only the first exec_horizon actions (receding horizon).
        self._queue.clear()
        for i in range(int(self.exec_cfg.exec_horizon)):
            self._queue.append(seq[0, i].copy())

        self._since_replan = 0
        
    @torch.no_grad()
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns:
          (B,10) absolute target10 for this single env step
        """
        # Replan if needed
        if (len(self._queue) == 0) or (self._since_replan >= int(self.exec_cfg.exec_horizon)):
            self._replan(batch)

        a = self._queue.popleft()
        self._since_replan += 1

        a_t = torch.from_numpy(a).to(self.device).unsqueeze(0)  # (1,10)
        return a_t
