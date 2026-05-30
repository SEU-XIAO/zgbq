from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from battlefield_rl.config import AgentConfig
from battlefield_rl.rl.network import TacticalD3QN, epsilon_greedy, masked_q_values
from battlefield_rl.rl.replay import PrioritizedReplayBuffer, Transition


@dataclass
class LearnStats:
    loss: float
    td_abs_mean: float # TD 误差绝对值的平均数


class D3QNAgent:
    def __init__(self, cfg: AgentConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        # 创建两个一模一样的神经网络。Double DQN
        self.online = TacticalD3QN().to(self.device)
        self.target = TacticalD3QN().to(self.device)

        # 让 target 网络刚开始时复制一份 online 网络一模一样的随机初始参数
        self.target.load_state_dict(self.online.state_dict())
        # 把 target 网络设置成“测试模式”
        self.target.eval()

        # 创建 Adam 优化器，用来帮 online 网络更新卷积核和全连接层的数字
        self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.memory = PrioritizedReplayBuffer(cfg.memory_size, alpha=cfg.per_alpha)

        self.global_step = 0


    # 两个超参数线性插值
    def epsilon(self) -> float:
        t = min(1.0, self.global_step / max(1, self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + t * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def beta(self) -> float:
        t = min(1.0, self.global_step / max(1, self.cfg.per_beta_steps))
        return self.cfg.per_beta_start + t * (1.0 - self.cfg.per_beta_start)

    # 接收当前战场的观察状态和动作掩码，然后根据经典的epsilon贪婪策略决定下一步具体往哪走。
    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, epsilon_override: float | None = None) -> int:
        self.online.eval() # 开启评测模式
        # 加上batch这一个维度在前面
        x = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        m = torch.from_numpy(mask).unsqueeze(0).float().to(self.device)

        q = self.online(x)
        q_mask = masked_q_values(q, m)
        epsilon = self.epsilon() if epsilon_override is None else epsilon_override
        action = epsilon_greedy(q_mask[0], m[0], epsilon) # 剥离batch维度
        return action


    # 记忆存储
    def push_transition(self, tr: Transition) -> None:
        self.memory.add(tr)

    # 同步目标网络（Double DQN 核心），每隔target_sync_steps同步一次
    def maybe_sync_target(self) -> None:
        if self.global_step % self.cfg.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

    # 需要保存保存的状态
    def checkpoint_state(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optim.state_dict(),
            "global_step": self.global_step,
        }

    # 保存模型
    def save_checkpoint(self, path: str | Path, extra: dict | None = None) -> None:
        payload = self.checkpoint_state()
        if extra:
            payload.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    # 加载模型
    def load_checkpoint(self, path: str | Path) -> dict:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        if "optimizer" in checkpoint:
            self.optim.load_state_dict(checkpoint["optimizer"])
        self.global_step = int(checkpoint.get("global_step", 0))
        return checkpoint

    def learn_step(self) -> LearnStats | None:
        # 满足一个batch开始学习
        if len(self.memory) < self.cfg.batch_size:
            return None

        # online切换为训练模式
        self.online.train()
        # 索引，样本，修正统计偏差的重要性采样权重
        idxs, samples, weights = self.memory.sample(self.cfg.batch_size, beta=self.beta())

        # 遍历解包 samples 列表，把它们用 np.stack 或 np.array 拼成矩阵，然后全部搬运到 GPU（self.device）上转成 PyTorch 张量。
        s = torch.from_numpy(np.stack([x.state for x in samples])).float().to(self.device)
        a = torch.from_numpy(np.array([x.action for x in samples], dtype=np.int64)).to(self.device)
        r = torch.from_numpy(np.array([x.reward for x in samples], dtype=np.float32)).to(self.device)
        ns = torch.from_numpy(np.stack([x.next_state for x in samples])).float().to(self.device)
        d = torch.from_numpy(np.array([x.done for x in samples], dtype=np.float32)).to(self.device)
        m = torch.from_numpy(np.stack([x.mask for x in samples])).float().to(self.device)
        nm = torch.from_numpy(np.stack([x.next_mask for x in samples])).float().to(self.device)
        w = torch.from_numpy(weights).float().to(self.device)
        expert = torch.from_numpy(np.array([x.expert_action for x in samples], dtype=np.int64)).to(self.device)

        # 原始Q值
        q_raw = self.online(s)
        # Q(s)
        q = masked_q_values(q_raw, m)
        # Q(s,a)
        q_sa = q.gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # 当前网络输出动作
            next_online = masked_q_values(self.online(ns), nm)
            next_actions = torch.argmax(next_online, dim=1)
            # target网络输出评分
            next_target = masked_q_values(self.target(ns), nm)
            next_q = next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = r + (1.0 - d) * self.cfg.gamma * next_q

        # TD误差
        td = q_sa - target
        # 强化学习损失
        td_loss = (w * F.smooth_l1_loss(q_sa, target, reduction="none")).mean()

        expert_mask = expert >= 0 # -1是没有专家引导
        if expert_mask.any() and self.cfg.imitation_loss_weight > 0:
            # 把主网络输出的当前状态所有动作的分数大表q当作分类概率，把专家的动作当作标准答案（Label），算交叉熵损失
            imitation_loss = F.cross_entropy(q[expert_mask], expert[expert_mask])
        else:
            imitation_loss = q_sa.new_tensor(0.0)
        loss = td_loss + self.cfg.imitation_loss_weight * imitation_loss

        # 清空旧梯度
        self.optim.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optim.step()

        # 更新优先级
        prios = td.detach().abs().cpu().numpy() + 1e-6
        
        self.memory.update_priorities(idxs, prios)

        return LearnStats(loss=float(loss.item()), td_abs_mean=float(np.mean(prios)))
