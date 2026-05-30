from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """强化学习环境配置。"""

    # ── 观测空间 ──────────────────────────────────────────
    window_size: int = 25          # 局部观测窗口边长（格），即智能体视野为 25×25 网格
    max_height_diff: int = 0       # 最大允许高差（格），超过则视为不可通行，当前没有做这个，但是保留了接口，函数中也有相应的扩展，后续可以增加这个需求
    los_height_margin: float = 0.0  # 视线高差余量，用于通视判断时的容差

    # ── 奖励函数 ──────────────────────────────────────────
    step_penalty: float = -0.1              # 每步基础惩罚，鼓励尽快到达目标
    progress_reward_scale: float = 0.3      # 前进奖励系数，与朝目标方向的位移成正比
    exposure_penalty_scale: float = -0.15   # 暴露惩罚系数，被敌人看见时按距离加权扣分
    visible_event_penalty: float = -0.5     # 进入敌人视野事件的一次性惩罚
    revisit_penalty_scale: float = -0.05    # 重复访问惩罚系数，抑制原地绕圈

    waypoint_reward: float = 1.0   # 到达航路点的奖励
    goal_reward: float = 50.0      # 到达终点的奖励
    timeout_penalty: float = -60.0 # 超时惩罚，步数用尽未到终点时扣分

    # ── 终止条件 ──────────────────────────────────────────
    timeout_steps: int = 50        # 单个片段（segment）最大步数，超过则判定超时
    no_progress_limit: int = 10    # 连续无前进步数上限，超过则提前终止


@dataclass(frozen=True)
class PlannerConfig:
    """全局路径规划器配置。"""

    waypoint_interval: int = 22    # 航路点间距（格），全局路径每隔 N 格插入一个航路点
    allow_diagonal: bool = True    # 是否允许对角线移动（8 连通 vs 4 连通）
    threat_weight: float = 1.0     # 威胁权重，A*/JPS 搜索时敌人威胁对路径代价的影响系数
    threat_decay: float = 0.5      # 威胁衰减系数，距离敌人越远威胁按此系数线性衰减


@dataclass(frozen=True)
class AgentConfig:
    """D3QN 智能体训练配置。"""

    # ── 折扣与学习率 ──────────────────────────────────────
    gamma: float = 0.96            # 折扣因子，越大越重视长期回报,这里没有必要看100步，25步最好，50步超时了
    lr: float = 1e-4               # Adam 学习率

    # ── ε-贪心探索 ────────────────────────────────────────
    epsilon_start: float = 0.4     # 初始探索率
    epsilon_end: float = 0.05      # 最终探索率
    epsilon_decay_steps: int = 150_000  # 探索率从 start 线性衰减到 end 的总步数

    # ── 网络同步 ──────────────────────────────────────────
    target_sync_steps: int = 2_000 # 目标网络同步频率（步），每隔 N 步将在线网络权重复制到目标网络

    # ── 经验回放 ──────────────────────────────────────────
    batch_size: int = 64           # 每次训练采样的 batch 大小
    memory_size: int = 200_000     # 优先经验回放缓冲区容量

    # ── 优先经验回放（PER）──────────────────────────────
    per_alpha: float = 0.6         # 优先级指数，控制优先级对采样概率的影响程度
    per_beta_start: float = 0.4    # 重要性采样权重初始值，用于修正采样偏差
    per_beta_steps: int = 250_000  # beta 从 start 线性增长到 1.0 的总步数

    # ── 模仿学习 ──────────────────────────────────────────
    imitation_loss_weight: float = 1.0  # DAgger 模仿损失权重，与 TD 损失加权求和


@dataclass(frozen=True)
class CurriculumThreshold:
    """课程学习阶段晋升阈值。"""

    window_size: int = 200         # 滑动窗口大小（episode），用于计算晋升指标的滚动均值
    eval_every_episodes: int = 50  # 每隔多少 episode 做一次晋升评估
    eval_episodes: int = 50        # 每次评估运行的 episode 数
    eval_epsilon: float = 0.02     # 评估时的探索率（接近贪心）

    # ── Stage 1 → 2 晋升条件（无障碍无威胁）────────────
    stage1_success_rate: float = 0.85    # 成功率阈值
    stage1_timeout_rate: float = 0.15    # 超时率上限
    stage1_path_efficiency: float = 1.80 # 路径效率上限（实际步数 / 曼哈顿距离）

    # ── Stage 2 → 3 晋升条件（弱威胁）──────────────────
    stage2_success_rate: float = 0.80
    stage2_timeout_rate: float = 0.20
    stage2_path_efficiency: float = 2.00

    def for_stage(self, stage: int) -> tuple[float, float, float]:
        """返回指定阶段的 (success_rate, timeout_rate, path_efficiency) 阈值。"""
        if stage <= 1:
            return self.stage1_success_rate, self.stage1_timeout_rate, self.stage1_path_efficiency
        return self.stage2_success_rate, self.stage2_timeout_rate, self.stage2_path_efficiency


@dataclass(frozen=True)
class CurriculumConfig:
    """课程学习整体配置，控制各阶段的威胁强度和专家指导策略。"""

    # ── 阶段数量 ──────────────────────────────────────────
    max_stage: int = 3             # 最大课程阶段数（1=无障碍, 2=弱威胁, 3=全威胁）

    # ── 各阶段威胁缩放 ───────────────────────────────────
    threat_scale_stage1: float = 0.0  # 阶段 1 威胁权重（0 = 无威胁）
    threat_scale_stage2: float = 0.5  # 阶段 2 威胁权重（半强度）
    threat_scale_stage3: float = 1.0  # 阶段 3 威胁权重（全强度）

    # ── 各阶段敌人数量 ───────────────────────────────────
    stage2_enemy_count_min: int = 1   # 阶段 2 最少敌人数
    stage2_enemy_count_max: int = 1   # 阶段 2 最多敌人数
    stage3_enemy_count_min: int = 1   # 阶段 3 最少敌人数
    stage3_enemy_count_max: int = 3   # 阶段 3 最多敌人数

    # ── 场景采样区域 ──────────────────────────────────────
    stage2_inner_ratio: float = 0.8   # 阶段 2 起终点落在建筑群内部的比例
    stage3_inner_ratio: float = 0.5   # 阶段 3 起终点落在建筑群内部的比例

    # ── 敌人参数范围 ──────────────────────────────────────
    enemy_fov_min_deg: float = 90.0   # 敌人视场角最小值（度）
    enemy_fov_max_deg: float = 90.0   # 敌人视场角最大值（度）
    enemy_range_min: int = 18         # 敌人探测距离最小值（格）
    enemy_range_max: int = 18         # 敌人探测距离最大值（格）

    outer_band_extra: int = 5         # 建筑群外圈采样带额外宽度（格）

    # ── 专家启发式引导（DAgger）──────────────────────────
    heuristic_guidance: bool = True   # 是否启用专家启发式引导

    # 阶段 1 引导：从 100% 线性衰减到 70%，持续 3000 episode
    stage1_guide_start: float = 1.00
    stage1_guide_end: float = 0.70
    stage1_guide_decay_episodes: int = 3000

    # 阶段 2 引导：从 50% 线性衰减到 5%，持续 1200 episode
    stage2_guide_start: float = 0.50
    stage2_guide_end: float = 0.05
    stage2_guide_decay_episodes: int = 1200

    # 阶段 3 引导：从 20% 线性衰减到 0%，持续 1500 episode
    stage3_guide_start: float = 0.20
    stage3_guide_end: float = 0.00
    stage3_guide_decay_episodes: int = 1500

    # ── 启发式动作评分权重 ────────────────────────────────
    heuristic_distance_weight: float = 1.0   # 距离权重（朝目标方向的前进量）
    heuristic_threat_weight: float = 1.5     # 威胁权重（被敌人发现的风险）
    heuristic_revisit_weight: float = 0.2    # 重复访问权重（回到已访问格子的惩罚）


@dataclass(frozen=True)
class TrainingSceneConfig:
    """训练场景生成配置。"""

    # ── 基本设置 ──────────────────────────────────────────
    enabled: bool = True           # 是否启用局部训练场景（False 则只用真实地图裁剪）
    scene_size: int = 35           # 局部场景边长（格），即 35×35 的训练地图
    procedural_ratio: float = 0.70 # 程序化生成场景占比（剩余从真实地图随机裁剪）

    # ── 敌人参数 ──────────────────────────────────────────
    enemy_fov_deg: float = 90.0          # 敌人视场角（度）
    enemy_range: int = 18                # 敌人探测距离（格）
    enemy_facing_jitter_deg: float = 12.0 # 敌人朝向随机抖动范围（±度）

    # ── 各阶段起终点分布 ─────────────────────────────────
    # 阶段 2：70% 在建筑群内、30% 无敌人
    stage2_inside_ratio: float = 0.70
    stage2_none_ratio: float = 0.30

    # 阶段 3：40% 在建筑群内、40% 在外圈、20% 无敌人
    stage3_inside_ratio: float = 0.40
    stage3_outer_ratio: float = 0.40
    stage3_none_ratio: float = 0.20

    # ── 目标采样 ──────────────────────────────────────────
    goal_min_distance: int = 8      # 起终点最小曼哈顿距离（格）
    goal_max_distance: int = 22     # 起终点最大曼哈顿距离（格）
    max_sample_attempts: int = 100  # 采样可达目标的最大尝试次数
