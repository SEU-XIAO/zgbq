# ZGBQ - 基于强化学习的战场栅格化路径规划系统

## 项目概述

ZGBQ 是一个**基于强化学习的战场栅格化路径规划系统**，核心目标是在有敌人驻守的战场地图上，为智能体规划出一条从起点到终点的战术路径——既能抵达目标，又能规避敌人的视野和威胁。

系统包含两个核心任务模块：
1. **路径规划**（`battlefield_rl`）— 在有敌人的地图上找到安全路径
2. **阵地选择**（`select_field`）— 为炮兵选择最佳射击阵地/瞭望点

---

## 架构设计：三级决策体系

项目采用**全局规划 + 局部执行 + 几何回退**的三层架构，确保在各种复杂场景下都能找到可行路径。

```
┌─────────────────────────────────────────────────────┐
│                   全局规划层                          │
│   JPS/A* 搜索 → 威胁感知代价 → 路径压缩 → 航路点插值  │
└──────────────────────┬──────────────────────────────┘
                       │ waypoints
┌──────────────────────▼──────────────────────────────┐
│                   局部执行层                          │
│   D3QN 模型 → 逐段执行(45步) → 振荡检测 → 失败恢复    │
└──────────────────────┬──────────────────────────────┘
                       │ 失败时
┌──────────────────────▼──────────────────────────────┐
│                   几何回退层                          │
│   沿规划路径硬走 → 脱困后重新接管 RL                    │
└─────────────────────────────────────────────────────┘
```

### 第一层：全局规划器

**文件位置**：`battlefield_rl/planner/global_planner.py`

- **JPS（Jump Point Search）**：首选的全局路径搜索算法，是 A* 的加速变体。通过"强迫邻居"剪枝和跳跃探测，跳过大量中间格子，只保留关键跳点
- **A* 降级**：JPS 失败时自动降级为标准 A* 算法
- **威胁感知代价**：搜索过程中将敌人威胁纳入边权，使规划路径天然绕开高危区域
- **路径后处理**：
  - `compress_jump_points()`：提取拐点，压缩路径表示
  - `interpolate_waypoints()`：按固定间隔（默认22格）插值生成航路点序列

### 第二层：局部 RL 执行器

**文件位置**：`battlefield_rl/rl/` + `scripts/eval_hierarchical_executor.py`

- **网络结构**：Dueling Double DQN (D3QN)，用 `TacticalD3QN` 实现
- **执行模式**：每次只执行一小段（`SEGMENT_STEPS = 45`步），目标是走到下一个航路点
- **防振荡机制**：Q值反向惩罚 + 滑动窗口检测
- **失败恢复**：振荡/卡住/超时 → 缩短航路间隔重规划 → 几何回退

### 第三层：几何回退机制

**文件位置**：`scripts/eval_hierarchical_executor.py`

这是系统的核心保底机制，确保在 RL 模型卡壳时仍能继续前进。放弃神经网络，直接沿 JPS/A* 规划的几何路径一步一步走，脱困后重新交给 RL 模型。

---

## 一、全局规划算法

**文件**：`battlefield_rl/planner/global_planner.py`

### 1.1 威胁感知 A*

标准 A* 搜索，但在边权计算中加入了敌人威胁代价：

```python
# global_planner.py 第 122-166 行
step_cost = euclidean_distance(current, nxt)
if enemies and threat_weight > 0:
    step_cost += _threat_cost(grid, nxt, enemies, threat_weight, threat_decay)
```

**威胁代价函数** `_threat_cost()`（第 48-78 行）：
对每个敌人，依次检查三个条件：
1. **距离**：目标点是否在敌人最大射程内
2. **视角**：目标点是否在敌人 FOV 扇形内（半角检测）
3. **视线**：用 Bresenham 画线算法检查两点间是否有障碍物遮挡

三个条件都满足时，威胁值 = `threat_weight × (1.0 - distance / max_range)`，线性衰减。多个敌人取最大值。

### 1.2 Jump Point Search (JPS)

JPS 是 A* 的加速变体，核心思想是**跳过中间节点，只保留关键跳点**。

**强迫邻居检测** `_has_forced_neighbor()`（第 190-221 行）：
- 对角移动：检查左右/上下方向是否有障碍物产生"强迫邻居"
- 直线移动：检查垂直方向是否有障碍物产生"强迫邻居"

**跳跃探测** `_jump()`（第 224-247 行）：
沿指定方向递归前进，直到遇到：
- 障碍物（停止）
- 目标点（返回）
- 强迫邻居（返回，这是一个关键跳点）

**方向剪枝** `_pruned_directions()`（第 250-294 行）：
根据父节点方向和周围障碍物，只保留必须探索的方向，大幅减少搜索空间。

**路径展开** `_expand_jump_path()`（第 307-331 行）：
将跳点序列还原为完整的格子路径，填充中间格子。

**降级策略** `plan_global_path()`（第 456-480 行）：
```python
path = jps(...) if use_jps else []
if not path:
    path = astar(...)  # JPS 失败时降级为 A*
```

### 1.3 路径后处理

**路径压缩** `compress_jump_points()`（第 396-411 行）：
从完整路径中提取方向变化点（拐点），去掉直线中间的冗余点。

**航路点插值** `interpolate_waypoints()`（第 415-453 行）：
两种模式：
1. **距离模式**（默认）：累计欧几里得距离，每隔 `interval`（默认22）单位插入一个航路点
2. **切比雪夫模式**：每隔 N 个格子步插入一个航路点

---

## 二、强化学习算法

### 2.1 网络架构：Dueling Double DQN

**文件**：`battlefield_rl/rl/network.py`

**算法选型理由**：
- **Double DQN**：分离动作选择和价值评估，减少 Q 值过估计问题
- **Dueling 架构**：分离状态价值 V(s) 和动作优势 A(s,a)，提升学习效率
- **CNN 特征提取**：栅格地图天然是二维图像，用卷积提取空间特征

**网络结构** `TacticalD3QN`（第 9-42 行）：

```
输入: (batch, 5, 25, 25)  — 5通道25x25局部观测
    │
    ├─ Conv2d(5→32, 3×3, stride=1, pad=1) → ReLU → (batch, 32, 25, 25)
    ├─ Conv2d(32→64, 3×3, stride=2, pad=1) → ReLU → (batch, 64, 13, 13)
    ├─ Conv2d(64→128, 3×3, stride=2, pad=1) → ReLU → (batch, 128, 7, 7)
    │
    ├─ Flatten → (batch, 6272)
    │
    ├─ 价值头 V(s):
    │   Linear(6272→256) → ReLU → Linear(256→1)
    │
    └─ 优势头 A(s,a):
        Linear(6272→256) → ReLU → Linear(256→8)

输出: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
```

**Q 值掩码**（第 46-47 行）：
无效动作（碰撞墙壁/越界）的 Q 值设为 -1e9，确保不会被选中：
```python
def masked_q_values(q, mask):
    return q + (1.0 - mask) * (-1e9)
```

### 2.2 智能体：D3QNAgent

**文件**：`battlefield_rl/rl/agent.py`

**Double DQN 更新**（第 118-122 行）：
```python
# 用 online 网络选动作
next_online = masked_q_values(self.online(ns), nm)
next_actions = torch.argmax(next_online, dim=1)
# 用 target 网络评价值
next_target = masked_q_values(self.target(ns), nm)
next_q = next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
target = r + (1.0 - d) * self.cfg.gamma * next_q
```

**损失函数**（第 125-132 行）：
由两部分组成：

1. **TD 损失**：Smooth L1 (Huber) 损失，用 PER 重要性采样权重修正
   ```python
   td_loss = (w * F.smooth_l1_loss(q_sa, target, reduction="none")).mean()
   ```

2. **模仿损失**：对有专家引导的样本，额外计算交叉熵损失，加速初期学习
   ```python
   imitation_loss = F.cross_entropy(q[expert_mask], expert[expert_mask])
   ```

总损失 = TD 损失 + `imitation_loss_weight` × 模仿损失

**目标网络同步**：每 2000 步将 online 网络参数复制到 target 网络。

**梯度裁剪**：max_norm=10.0，防止梯度爆炸。

### 2.3 优先经验回放 (PER)

**文件**：`battlefield_rl/rl/replay.py`

**核心思想**：TD 误差越大的样本越重要，被采样的概率越高。

**实现细节** `PrioritizedReplayBuffer`（第 23-62 行）：
- 容量：200,000 条转移
- **优先级指数** α=0.6：控制优先化程度，α=0 时退化为均匀采样
- **新样本**：以当前最大优先级插入，保证至少被采样一次
- **采样概率**：`p_i = priority^α / Σ(priority^α)`
- **重要性采样权重**：`w_i = (N × p_i)^(-β)`，β 从 0.4 线性增长到 1.0（250,000 步），用于修正采样偏差
- **优先级更新**：每次学习后用 `|TD error| + 1e-6` 更新该样本的优先级

**Transition 数据结构**（第 11-20 行）：
```python
@dataclass
class Transition:
    state: np.ndarray        # 当前观测
    action: int              # 执行的动作
    reward: float            # 获得的奖励
    next_state: np.ndarray   # 下一观测
    done: bool               # 是否终止
    mask: np.ndarray         # 当前动作掩码
    next_mask: np.ndarray    # 下一动作掩码
    guided: bool             # 是否由专家引导
    expert_action: int       # 专家选择的动作
```

### 2.4 探索策略：ε-贪心

**ε 衰减**（agent.py 第 43-45 行）：
- 初始值：0.4（高探索）
- 终止值：0.05（低探索）
- 衰减方式：线性，持续 150,000 步

执行阶段使用 `greedy_action()`（eval_hierarchical_executor.py 第 113-130 行），ε=0，纯贪心，但带有**反向惩罚**：
```python
# 对上一步的反方向施加惩罚，抑制振荡
if prev_action is not None:
    reverse_action = REVERSE_MAP[prev_action]
    q[0, reverse_action] -= return_penalty  # 默认 2.0
```

---

## 三、环境设计

**文件**：`battlefield_rl/env/tactical_env.py`

### 3.1 状态空间（观测空间）

5 通道的 25×25 局部地图，以智能体为中心裁剪：

| 通道 | 含义 | 值域 |
|------|------|------|
| ch_free | 可通行区域 | 1.0 = 可通行 |
| ch_block | 障碍物 | 1.0 = 阻挡 |
| ch_threat | 敌人威胁热力图 | 0.0~1.0，线性衰减 |
| ch_wp | 航路点标记 | 1.0 = 航路点位置 |
| ch_visit | 足迹记忆 | 衰减系数 0.97/步 |

观测形状：`(5, 25, 25)`

**足迹层设计**：每次访问一个格子，该位置的 visit 值设为 1.0，之后每步乘以 0.97 衰减。这为模型提供了"最近去过哪里"的记忆信号。

### 3.2 动作空间

8 个离散动作，对应 8 个方向的移动：

```python
ACTIONS_8 = [
    (-1, 0),  # 0: 上
    ( 1, 0),  # 1: 下
    ( 0,-1),  # 2: 左
    ( 0, 1),  # 3: 右
    (-1,-1),  # 4: 左上
    (-1, 1),  # 5: 右上
    ( 1,-1),  # 6: 左下
    ( 1, 1),  # 7: 右下
]
```

反向动作映射（用于振荡抑制）：
```python
REVERSE_MAP = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 5: 6, 6: 5, 7: 4}
```

### 3.3 奖励函数设计

**奖励计算**（tactical_env.py 第 178-247 行）：

每步的总奖励 = 以下各项之和：

```python
reward = step_penalty                          # 时间惩罚
       + progress_reward_scale * Δdistance     # 推进奖励
       + exposure_penalty_scale * threat_scale * threat       # 威胁惩罚
       + visible_event_penalty * threat_scale * visible       # 可见性惩罚
       + revisit_penalty_scale * visited_value                # 重复访问惩罚
       + (-0.1 if 无效动作 else 0)                           # 碰撞惩罚
       + waypoint_reward  (到达航路点时)                       # 航路点奖励
       + goal_reward      (到达终点时)                         # 目标奖励
       + timeout_penalty  (超时或卡住时)                       # 失败惩罚
```

**各项详解**：

| 奖励项 | 公式 | 默认值 | 触发条件 |
|--------|------|--------|----------|
| 时间惩罚 | 每步固定 | -0.1 | 每步都有 |
| 推进奖励 | `0.3 × (old_dist - new_dist)` | 系数 0.3 | 距离缩短为正，远离为负 |
| 威胁惩罚 | `-0.15 × threat_scale × threat` | 系数 -0.15 | 处于敌人威胁范围内 |
| 可见性惩罚 | `-0.5 × threat_scale × visible` | 系数 -0.5 | 被敌人直接看到 |
| 重复访问惩罚 | `-0.05 × visited_value` | 系数 -0.05 | 走回头路 |
| 碰撞惩罚 | 固定 -0.1 | -0.1 | 无效动作（撞墙/越界） |
| 航路点奖励 | 到达时 +1.0 | +1.0 | pos == waypoint |
| 目标奖励 | 到达时 +50.0 | +50.0 | pos == goal |
| 超时惩罚 | 超时或卡住 -60.0 | -60.0 | 步数耗尽或连续无进展 |

**推进奖励计算**（tactical_env.py 第 341-344 行）：
```python
def _progress_reward(self, old_pos, new_pos):
    d0 = euclidean(old_pos, self.goal)
    d1 = euclidean(new_pos, self.goal)
    return self.cfg.progress_reward_scale * (d0 - d1)  # 0.3 × 距离变化
```

**威胁/可见性惩罚中的 threat_scale**：
- Stage 1：`threat_scale = 0.0`，威胁和可见性惩罚不生效
- Stage 2：`threat_scale = 0.5`，惩罚减半
- Stage 3：`threat_scale = 1.0`，完整惩罚

**多敌人威胁融合**：多个敌人的威胁用概率 OR 方式合并：
```python
combined = 1.0 - (1.0 - existing_threat) * (1.0 - new_threat)
```

### 3.4 威胁计算

**文件**：`battlefield_rl/env/tactical_env.py` 第 281-330 行

对每个敌人，依次检查：
1. **距离检查**：目标点在敌人 `max_range` 范围内
2. **视角检查**：目标点在敌人 FOV 扇形内（半角检测）
3. **视线检查**：用 Bresenham 画线算法，检查两点间是否有障碍物遮挡

三个条件都满足时，威胁值 = `max(0.0, 1.0 - distance / max_range)`，线性衰减。

**Bresenham 画线算法**（第 49-70 行）：
标准的 Bresenham 直线算法，用于视线检测（Line-of-Sight）。从敌人位置向目标点画线，如果线上的格子有障碍物，则视线被遮挡。

### 3.5 终止条件

| 条件 | 判定 | 结果 |
|------|------|------|
| 到达目标 | `pos == goal` | +50 奖励，done |
| 超时 | `steps >= timeout_steps (50)` | -60 惩罚，done |
| 卡住 | `no_progress_steps >= no_progress_limit (10)` | -60 惩罚，done |

**无进展检测**（第 222-228 行）：
如果连续 10 步到目标的欧几里得距离都没有减小，判定为"卡住"。

### 3.6 专家策略（启发式引导）

**文件**：`battlefield_rl/env/tactical_env.py` 第 145-176 行

`heuristic_action()` 是一个手写的贪心评分函数，用于训练初期的专家引导（DAgger 风格）：

```python
score = distance_weight × progress      # 距离权重 1.0
      - threat_weight × threat           # 威胁权重 1.5
      - revisit_weight × revisit         # 重复惩罚 0.2
```

对每个候选动作计算得分，选最高分的动作。训练初期有一定概率用专家动作代替 RL 决策，帮助模型快速学到基本走法。

---

## 四、训练系统

### 4.1 分层训练器

**文件**：`battlefield_rl/train/trainer.py`

**单集流程**（第 192-336 行）：
1. 采样训练场景（程序化生成 / 真实地图裁剪）
2. 运行全局规划器获取航路点
3. 创建环境，设置敌人和航路点
4. 循环执行（最多 50 步）：
   - 计算专家动作 `heuristic_action()`
   - 以概率 `guide_prob` 使用专家动作（DAgger），否则用 ε-贪心
   - 存储转移、学习、同步目标网络
5. 如果 `should_replan`（卡住），提前终止

**DAgger 风格的专家引导**：
不是纯模仿学习（Behavioral Cloning），而是让专家在 RL 策略产生的状态分布下提供动作标签，避免分布偏移问题。

### 4.2 课程学习

**文件**：`battlefield_rl/train/curriculum.py`

**三阶段渐进式训练**：

| 阶段 | 威胁等级 | 敌人数量 | 引导概率衰减 | 升阶条件 |
|------|---------|---------|-------------|---------|
| Stage 1 | 0（无威胁）| 0 | 1.00→0.70 (3000集) | 成功率≥85%，超时率≤15%，路径效率≤1.80 |
| Stage 2 | 0.5（弱威胁）| 1 | 0.50→0.05 (1200集) | 成功率≥80%，超时率≤20%，路径效率≤2.00 |
| Stage 3 | 1.0（全威胁）| 1~3 | 0.20→0.00 (1500集) | 无自动升阶，持续训练 |

**升阶判定**：每 50 集评估一次，跑 50 集评估集（ε=0.02），检查滚动指标是否达标。

**设计思路**：
- Stage 1：先学会基本的路径规划能力（不考虑敌人）
- Stage 2：引入弱威胁，学会简单的威胁规避
- Stage 3：全威胁、多敌人，精细化战术决策

### 4.3 训练场景采样

**文件**：`battlefield_rl/map/scene_sampler.py`

**LocalSceneSampler**（第 27-215 行）：

生成 35×35 的训练场景，两种来源：
- **程序化场景**（70%）：随机矩形建筑（2-6个）+ 随机短墙（带门洞）
- **真实地图裁剪**（30%）：从真实战场地图随机裁剪 35×35 窗口

**目标点采样**：
- 曼哈顿距离 8~22 格
- 必须 BFS 可达（不能是孤立区域）

**敌人放置**（受课程阶段控制）：
- Stage 2：70% 在窗口内、30% 无敌人
- Stage 3：40% 窗口内、40% 窗口外带、20% 无敌人
- 敌人 FOV=90°，射程=18 格，朝向加 ±12° 随机抖动

---

## 五、分层执行器（部署核心）

**文件**：`scripts/eval_hierarchical_executor.py`

### 5.1 执行流程

`plan_and_execute()` 函数（第 312-571 行）实现完整的三级决策：

```
1. 全局规划: JPS/A* → 完整路径 → 压缩 → 航路点插值
2. 逐段执行: for waypoint in waypoints:
     ├── run_segment(): RL 模型执行最多 45 步
     ├── 成功 → 继续下一段
     ├── 失败(振荡/卡住/超时) → 重规划(缩短航路间隔)
     └── 连续失败 ≥2次 → 几何回退
3. 统计输出: 可见性比例、重复率、路径效率
```

### 5.2 关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| WAYPOINT_INTERVAL | 18 | 航路点间距 |
| SEGMENT_STEPS | 45 | 每段最大执行步数 |
| WAYPOINT_TOLERANCE | 3 | 中间航路点曼哈顿容忍度 |
| GOAL_TOLERANCE | 0 | 最终目标精确匹配 |
| MAX_REPLANS_PER_TRAP | 2 | 触发回退前的最大重规划次数 |
| FALLBACK_ESCAPE_STEPS | 20 | 每次回退的最大步数 |
| MAX_FALLBACK_EVENTS | 20 | 整个任务最大回退次数 |
| MAX_FALLBACK_STEPS | 2000 | 回退总步数上限 |
| MAX_TOTAL_STEPS | 12000 | 任务总步数上限 |
| OSCILLATION_WINDOW | 10 | 振荡检测滑动窗口大小 |
| OSCILLATION_UNIQUE_LIMIT | 3 | 窗口内最少去重坐标数 |
| OSCILLATION_MIN_STEPS | 8 | 振荡检测生效的最少步数 |

### 5.3 段执行与振荡检测

`run_segment()` 函数（第 178-215 行）：

```python
for _ in range(max_steps):
    # 1. 检查是否到达航路点（容忍度内）
    if manhattan(env.pos, waypoint) <= tolerance:
        return env.pos, trace, True, "reached"

    # 2. RL 模型决策（带反向惩罚）
    action = greedy_action(model, obs, env.action_mask(), device, prev_action)
    prev_action = action
    step = env.step(action)

    # 3. 再次检查到达
    if manhattan(env.pos, waypoint) <= tolerance:
        return env.pos, trace, True, "reached"

    # 4. 振荡检测：最近10步内去重坐标 ≤ 3
    if len(trace) >= OSCILLATION_MIN_STEPS:
        recent = trace[-OSCILLATION_WINDOW:]
        if len(set(recent)) <= OSCILLATION_UNIQUE_LIMIT:
            return env.pos, trace, False, "oscillation"

    # 5. 环境终止检测（卡住/超时）
    if step.done:
        if step.info.get("stuck"):
            return env.pos, trace, False, "stuck"
        if step.info.get("timeout"):
            return env.pos, trace, False, "timeout"
```

**中间航路点容忍度**：曼哈顿距离 ≤ 3 即算到达，不要求精确匹配。只有最终目标使用 `GOAL_TOLERANCE = 0`（精确到达）。这是有意设计——中间航路点只是引导方向的路标。

### 5.4 防振荡机制

**双层防抖**：

1. **Q 值反向惩罚**（第 113-130 行）：
   在 `greedy_action()` 中，对上一步的反方向 Q 值减去 2.0，从决策层面抑制回头行为。

2. **滑动窗口检测**（第 204-207 行）：
   如果最近 10 步中只出现了 3 个或更少的不同坐标，判定为振荡，立即终止当前段。

### 5.5 重规划策略

当一段执行失败但还没达到回退阈值时（第 507-525 行）：
1. 递增 `trap_replans` 计数器
2. **缩短航路间隔**：`replan_interval = max(4, WAYPOINT_INTERVAL / (trap_replans + 1))`
   - 第1次重规划：间隔 18→9
   - 第2次重规划：间隔 9→6
3. 从当前位置重新运行全局规划器，生成更密集的航路点

**设计意图**：航路点越密集，RL 模型每段需要走的距离越短，越容易成功。相当于把一个"走不过去"的长路段拆成多个"能走过去"的短路段。

### 5.6 几何回退机制

当 `trap_replans >= MAX_REPLANS_PER_TRAP (2)` 时触发（第 442-501 行）：

```
触发条件：同一位置连续重规划 2 次仍然失败
    │
    ▼
执行: follow_geometric_path()
    ├── 沿 JPS/A* 规划的几何路径一步一步走
    ├── 不经过神经网络决策
    ├── 每次最多走 20 步
    └── 总预算: 2000 步 / 20 次
    │
    ▼
恢复: 从新位置重新规划 → RL 模型接管
```

**`follow_geometric_path()`**（第 218-256 行）：
- 取规划路径中当前位置之后的路径段
- 逐步验证每一步是否可通行且相邻
- 直接移动到目标格子，不经过 RL 决策

**安全限制**：
- `MAX_FALLBACK_EVENTS = 20`：整个任务最多触发 20 次回退
- `MAX_FALLBACK_STEPS = 2000`：回退总步数不超过 2000 步
- `MAX_TOTAL_STEPS = 12000`：整个任务总步数不超过 12000 步

### 5.7 可见性刷新

当航路点距离当前位置过远时（切比雪夫距离 > 10），触发路径刷新（第 377-397 行）：
```python
if chebyshev(pos, waypoints[0]) > WAYPOINT_MAX_CHEBYSHEV:
    refresh = build_plan(grid, pos, goal, ...)
    waypoints = make_waypoints(refresh, pos, goal, ...)
```

**设计意图**：RL 模型执行过程中可能偏离规划路径较远，此时继续按原航路点走可能不合理。从当前位置重新规划，保证航路点始终是合理的。

---

## 六、地图系统

**文件**：`battlefield_rl/map/loader.py`

### 6.1 地图格式

文本文件，每个格子用 `(高度,类型)` 表示：
```
(0,0) (0,0) (0,1) (0,0) ...
(0,0) (5,1) (5,1) (0,0) ...
```

- 类型 0：可通行
- 类型 1：静态障碍（建筑物）
- 类型 2：另一种障碍

### 6.2 BattlefieldMap 类（第 14-34 行）

存储两个 numpy 数组：
- `heights`：int16，高程数据
- `types`：int8，格子类型

提供方法：`is_static_blocked()`、`in_bounds()`、`passable()` 等。

---

## 七、部署接口

### 7.1 HTTP API（`interfaces/path_planning_api.py`）

接收 JSON 请求，调用 `plan_and_execute`。

**请求格式**：
```json
{
  "task": "path_planning",
  "map": "MyPath_Data417.txt",
  "model": "episode_8000.pt",
  "start": [20, 20],
  "goal": [73, 97],
  "enemies": [
    {
      "row": 50, "col": 50,
      "facing_deg": 225,
      "fov_deg": 90,
      "range": 20
    }
  ],
  "use_fallback": true
}
```

**响应格式**：
```json
{
  "success": true,
  "path": [[20,20], [21,21], ...],
  "total_steps": 150,
  "visible_ratio": 0.15,
  "path_efficiency": 1.2,
  "geometric_fallback_count": 2,
  "geometric_fallback_steps": 40
}
```

### 7.2 UDP Server（`interfaces/udp_path_server.py`）

二进制协议，支持坐标系转换（xy ↔ row/col），同时服务路径规划和阵地选择两个任务。

**协议格式**：
- 消息头：2 字节消息类型 + 20 字节消息 ID
- 消息体：UTF-8 编码的 JSON 字符串

### 7.3 阵地选择模块（`select_field/`）

独立的战术分析模块，纯 Python 实现，无第三方依赖。通过可视域分析和多因子评分，在栅格地图上为炮兵选择最佳射击阵地或瞭望点。

**文件结构**：

| 文件 | 职责 |
|------|------|
| `models.py` | 数据结构：`Point`、`Rect`、`TargetSpec`、`CandidateScore` |
| `map_loader.py` | 地图加载，`MapGrid` 类 |
| `geometry.py` | 欧氏距离、Bresenham 直线算法 |
| `visibility.py` | 可视域缓存、视线判断、暴露度计算 |
| `scoring.py` | 全部评分算法（核心） |
| `service.py` | `TacticalAnalyzer` 门面类 |
| `cli.py` | 命令行接口 |

#### 7.3.1 两种工作模式

**Lookout 模式** — 在两个区域间选择最佳瞭望点：

```bash
python -m select_field --map MyPath_Data417.txt lookout \
  --region-a 0 0 100 100 --region-b 200 200 100 100 --top-k 5
```

**Position 模式** — 为特定目标选择射击阵地：

```bash
python -m select_field --map MyPath_Data417.txt position \
  --search-region 0 0 500 500 --target 80 80 ZM --target 120 120 JM --patch-size 5
```

目标类型说明：
- **ZM（直接射击）**：要求阵地能看到目标，距离越近越好
- **JM（间接射击）**：要求阵地隐蔽不被敌人发现，与目标保持在最佳距离区间内

#### 7.3.2 可视域分析

**文件**：`select_field/visibility.py`

核心类 `VisibilityCache`，封装视线判断逻辑并提供缓存。

**视线判断** `line_of_sight()`（第 56-60 行）：
```python
def line_of_sight(self, start: Point, end: Point) -> bool:
    for cell in bresenham_line(start, end)[1:-1]:  # 跳过首尾端点
        if not self.grid.is_passable(cell.x, cell.y):
            return False
    return True
```

用 Bresenham 算法获取两点间所有中间格子，逐一检查是否可通行。任何一个中间格子被障碍物阻挡则视线被遮断。

**暴露度评分** `exposure_score()`（第 46-51 行）：
对每个能看到该点的敌人累加 `1.0 / (distance + 1.0)`，距离越近暴露度越高。

**缓存机制**：以 `(enemy_index, x, y)` 为键缓存所有视线判断结果，避免重复计算 Bresenham 直线。

#### 7.3.3 Lookout 模式算法

**文件**：`select_field/scoring.py` 第 63-116 行

```
区域A 采样候选点(≤1200) → 过滤不可通行
区域B 采样候选点(≤1200) → 过滤不可通行
区域A 采样目标点(≤1500) → 过滤不可通行
区域B 采样目标点(≤1500) → 过滤不可通行

对区域A每个候选点：计算对区域B目标的视线覆盖率 → 排序取top_k
对区域B每个候选点：计算对区域A目标的视线覆盖率 → 排序取top_k

合并两组top_k → 全局排序 → 最终top_k
```

**覆盖率评分** `_coverage_score()`（第 48-60 行）：
```python
visible = sum(1 for point in target_points if visibility.line_of_sight(source, point))
ratio = visible / total  # 可见目标比例
```

#### 7.3.4 Position 模式算法（粗筛 + 精筛 + 精细评分）

**文件**：`select_field/scoring.py`

这是模块中最复杂的算法，采用三级搜索策略：

**阶段一：构建二值积分图**（第 119-126 行）

构建二维前缀和数组，可通行格子值为 1，不可行为 0。用于 O(1) 计算任意矩形区域内的可通行格子数。

**阶段二：粗筛**（第 216-243 行）

```
自适应步长: ceil(sqrt(面积 / 2500))
遍历搜索区域中所有 patch 起点
    ├── 过滤: clearance_ratio < 0.8 → 淘汰
    ├── 种子评分: 0.8×通畅率 + 0.4×开阔度 + 距离分
    └── 最小堆保留前 300 个
```

**通畅率** `_patch_clearance_ratio()`（第 138-140 行）：候选 patch 内可通行格子比例。低于 0.8 直接淘汰。

**开阔度** `_ring_openness_ratio()`（第 143-152 行）：patch 外扩 1 格形成的"环"的通畅率，衡量阵地周围开阔程度。

**距离带评分** `_distance_band_score()`（第 155-161 行）：
```python
if distance < low:   return distance / low              # 低于下限，线性增长
if distance > high:  return max(0, 1 - (d-high)/high)   # 超过上限，线性衰减
return 1.0                                               # 最佳区间内，满分
```

**阶段三：精筛**（第 246-276 行）

对每个粗筛候选，在半径=2 的邻域（5×5=25 个位置）内搜索更优位置，再次检查通畅率 ≥ 0.8。

**阶段四：精细评分**（第 279-365 行）

对每个候选 patch 分别计算 ZM 和 JM 得分：

**ZM 评分** `_zm_score()`（第 279-304 行）：

| 评分项 | 权重 | 说明 |
|--------|------|------|
| 视线可见性 | +2.5（可见）/ -3.0（不可见） | 核心条件，不对称奖惩 |
| 距离衰减 | `1.0 / (distance + 1)` | 越近越好 |
| 通畅率 | 1.0 | 阵地内部通行性 |
| 开阔度 | 0.5 | 阵地周围开阔程度 |

**JM 评分** `_jm_score()`（第 307-334 行）：

| 评分项 | 权重 | 说明 |
|--------|------|------|
| 隐蔽性 | 1.8 | 不被任何敌人看到时 +1.8 |
| 距离评分 | 1.2 | 12~24 格为最佳区间，过近或过远均衰减 |
| 通畅率 | 0.8 | 阵地内部通行性 |
| 开阔度 | 0.5 | 阵地周围开阔程度 |

**合并**：`combined = 基础分(0.8×通畅率 + 0.4×开阔度) + ZM分 + JM分`

**最终输出**：
- `best_combined`：综合最优
- `best_zm`：ZM 类型最优（可能为空）
- `best_jm`：JM 类型最优（可能为空）
- `top_k`：前 K 个候选列表

#### 7.3.5 阵地选择 API 请求格式

```json
{
  "map": "MyPath_Data417.txt",
  "mode": "position",
  "search_region": {"x": 0, "y": 0, "w": 500, "h": 500},
  "targets": [
    {"enemy": [80, 80], "type": "ZM"},
    {"enemy": [120, 120], "type": "JM"}
  ],
  "patch_size": 5,
  "top_k": 5
}
```

**响应格式**：
```json
{
  "success": true,
  "task": "site_selection",
  "mode": "position",
  "result": {
    "best_combined": {"rect": {...}, "score": 8.5, "details": {...}},
    "best_zm": {"rect": {...}, "score": 6.2, "details": {...}},
    "best_jm": {"rect": {...}, "score": 7.1, "details": {...}},
    "top_k": [...]
  }
}
```

#### 7.3.6 UDP 适配层

`interfaces/udp_path_server.py` 中的阵地选择适配处理（第 196-290 行）：

- **模式别名**：`"阵地选择"` → `"position"`，`"瞭望点"` → `"lookout"`
- **字段别名**：`searchRegion` / `regionA` / `regionB` / `ZCArea`
- **目标来源**：`targets` / `enemyInfo` / `enemy` 三种字段名均兼容
- **响应别名**：`best_zm` → `best_direct`，`best_jm` → `best_indirect`
- **默认值**：搜索区域为空时使用全图，patch_size 默认 5，top_k 默认 5

---

## 八、测试工具体系

| 文件 | 功能 |
|------|------|
| `tests/detect_oscillation.py` | 路径振荡全面诊断：滑窗/折返/循环/位移效率/严重度分级 |
| `tests/test_recency_masking.py` | 条件式禁行掩码 A/B 对比测试 |
| `tests/compare_global_models.py` | 多 checkpoint 完整路径规划任务对比 |
| `tests/compare_local_executor_models.py` | 多模型局部执行段对比 |
| `tests/compare_visibility.py` | 敌人可见性回避能力对比 |
| `tests/compare_visibility_realmap.py` | 真实地图可见性对比 |
| `tests/compare_visibility_tendency.py` | 可见性变化趋势分析 |
| `tests/benchmark_path_algorithms.py` | 路径算法性能基准测试 |
| `tests/visualize_model_vs_classic.py` | 模型路径 vs 经典算法路径可视化对比 |

### 使用示例

```bash
# 振荡检测
python tests/detect_oscillation.py --model episode_8000.pt --cases 100 --stage 3

# 模型对比
python tests/compare_global_models.py --model episode_6000.pt --model episode_8000.pt --cases 50

# 可见性测试
python tests/compare_visibility.py --model episode_6000.pt --model episode_8000.pt --n-enemies 3

# Recency Masking 测试
python tests/test_recency_masking.py --model episode_8000.pt --cases 100
```

---

## 九、配置参数速查

### EnvConfig（`battlefield_rl/config/settings.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| window_size | 25 | 局部视野窗口大小 |
| step_penalty | -0.1 | 每步时间惩罚 |
| progress_reward_scale | 0.3 | 推进奖励系数 |
| exposure_penalty_scale | -0.15 | 威胁暴露惩罚系数 |
| visible_event_penalty | -0.5 | 可见性惩罚系数 |
| revisit_penalty_scale | -0.05 | 重复访问惩罚系数 |
| waypoint_reward | 1.0 | 航路点到达奖励 |
| goal_reward | 50.0 | 目标到达奖励 |
| timeout_penalty | -60.0 | 超时/卡住惩罚 |
| timeout_steps | 50 | 单段最大步数 |
| no_progress_limit | 10 | 无进展步数上限 |

### AgentConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| gamma | 0.99 | 折扣因子 |
| lr | 1e-4 | 学习率 |
| epsilon_start | 0.4 | 初始探索率 |
| epsilon_end | 0.05 | 最终探索率 |
| epsilon_decay_steps | 150,000 | 探索率衰减步数 |
| target_sync_steps | 2,000 | 目标网络同步步数 |
| batch_size | 64 | 批次大小 |
| memory_size | 200,000 | 经验回放缓冲区大小 |
| per_alpha | 0.6 | PER 优先级指数 |
| per_beta_start | 0.4 | PER IS 权重起始值 |
| per_beta_steps | 250,000 | PER β 衰减步数 |
| imitation_loss_weight | 1.0 | 模仿损失权重 |

### PlannerConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| waypoint_interval | 22 | 航路点插值间隔 |
| allow_diagonal | True | 允许对角移动 |
| threat_weight | 5.0 | 规划时威胁代价权重 |
| threat_decay | 0.5 | 威胁距离衰减系数 |

### CurriculumConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_stage | 3 | 最大课程阶段 |
| enemy_fov_deg | 90.0 | 敌人视角 |
| enemy_range | 18 | 敌人射程 |
| guide_start / guide_end | 见课程表 | 各阶段引导概率范围 |

---

## 十、项目结构

```
ZGBQ/
├── battlefield_rl/              # 核心强化学习系统
│   ├── config/                  # 配置定义
│   │   └── settings.py          # 所有配置类
│   ├── map/                     # 地图解析与数据结构
│   │   ├── loader.py            # 地图加载器
│   │   └── scene_sampler.py     # 训练场景采样器
│   ├── planner/                 # 全局路径规划
│   │   └── global_planner.py    # JPS/A* 规划器
│   ├── env/                     # 局部战术环境
│   │   └── tactical_env.py      # 环境实现
│   ├── rl/                      # 强化学习组件
│   │   ├── network.py           # D3QN 网络
│   │   ├── agent.py             # 智能体
│   │   └── replay.py            # 优先经验回放
│   └── train/                   # 训练系统
│       ├── trainer.py           # 分层训练器
│       └── curriculum.py        # 课程学习
├── interfaces/                  # 部署接口
│   ├── path_planning_api.py     # HTTP API
│   ├── udp_path_server.py       # UDP 服务器
│   └── api.py                   # 统一分发器
├── select_field/                # 阵地选择模块
│   ├── models.py                # 数据结构: Point, Rect, TargetSpec
│   ├── map_loader.py            # 地图加载, MapGrid 类
│   ├── geometry.py              # 欧氏距离, Bresenham 直线
│   ├── visibility.py            # 可视域缓存, 视线判断
│   ├── scoring.py               # 评分算法: lookout/position 两种模式
│   ├── service.py               # TacticalAnalyzer 门面类
│   └── cli.py                   # 命令行接口
├── scripts/                     # 运行脚本
│   ├── eval_hierarchical_executor.py  # 分层执行评估
│   ├── train_local_executor.py  # 训练脚本
│   ├── run_demo.py              # 演示脚本
│   └── stress_test_executor.py  # 压力测试
├── tests/                       # 测试工具
├── MyPath_Data417.txt           # 示例地图文件
└── episode_*.pt                 # 模型检查点
```

---

## 快速开始

### 环境要求

- Python 3.12+
- PyTorch 2.0+
- NumPy

### 安装

```bash
pip install torch numpy
```

### 运行演示

```bash
# 基础演示
python scripts/run_demo.py --map MyPath_Data417.txt --start 20,20 --goal 420,420

# 完整训练
python scripts/train_local_executor.py --map MyPath_Data417.txt --episodes 10000

# 路径规划 API
python interfaces/path_planning_api.py --map MyPath_Data417.txt --start 20,20 --goal 73,97

# UDP 服务器
python -m interfaces.udp_path_server --port 12345 --map MyPath_Data417.txt

# 阵地选择 (position 模式)
python -m select_field --map MyPath_Data417.txt position --search-region 0 0 500 500 --target 80 80 ZM --target 120 120 JM

# 阵地选择 (lookout 模式)
python -m select_field --map MyPath_Data417.txt lookout --region-a 0 0 100 100 --region-b 200 200 100 100
```

---

## 参考资料

- **JPS 算法**：Harabor & Grastien, "Jump Point Search - Fast Optimal Pathfinding on Grids", 2011
- **Dueling DQN**：Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", 2016
- **Double DQN**：Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", 2016
- **优先经验回放**：Schaul et al., "Prioritized Experience Replay", 2016
- **课程学习**：Bengio et al., "Curriculum Learning", 2009
- **DAgger**：Ross et al., "Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning", 2011

---

## 许可证

本项目为内部开发项目，仅供学习和研究使用。
