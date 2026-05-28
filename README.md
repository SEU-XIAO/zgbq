# ZGBQ - 基于强化学习的战场栅格化路径规划系统

## 项目概述

ZGBQ 是一个**基于强化学习的战场栅格化路径规划系统**，核心目标是在有敌人驻守的战场地图上，为智能体规划出一条从起点到终点的战术路径——既能抵达目标，又能规避敌人的视野和威胁。

系统包含两个核心任务模块：
1. **路径规划**（`battlefield_rl`）— 在有敌人的地图上找到安全路径
2. **阵地选择**（`select_field`）— 为炮兵选择最佳射击阵地/瞭望点

---

## 架构设计：三级决策体系

项目采用**全局规划 + 局部执行 + 几何回退**的三层架构，确保在各种复杂场景下都能找到可行路径。

### 第一层：全局规划器

**文件位置**：`battlefield_rl/planner/global_planner.py`

- **JPS（Jump Point Search）**：首选的全局路径搜索算法，是 A* 的加速变体。通过"强迫邻居"剪枝和跳跃探测，跳过大量中间格子，只保留关键跳点
- **A* 降级**：JPS 失败时自动降级为标准 A* 算法
- **路径后处理**：
  - `compress_jump_points()`：提取拐点，压缩路径表示
  - `interpolate_waypoints()`：按固定间隔（默认18格）插值生成航路点序列（waypoints）

### 第二层：局部RL执行器

**文件位置**：`battlefield_rl/rl/`

- **网络结构**：Dueling Double DQN (D3QN)，用 `TacticalD3QN` 实现
  - 3层卷积提取局部视野特征（25x25窗口 → 13x13 → 7x7 → 128通道）
  - Dueling 架构：分成 V(s) 价值头 + A(s,a) 优势头，Q = V + (A - mean(A))
- **执行模式**：每次只执行一小段（`SEGMENT_STEPS = 45`步），目标是走到下一个航路点
- **观测空间**：5通道的 25x25 局部地图
  - 通道1：可通行层
  - 通道2：障碍层
  - 通道3：敌人威胁层
  - 通道4：路标层
  - 通道5：足迹层
- **经验回放**：优先经验回放（PER），配合重要性采样修正

### 第三层：几何回退机制

**文件位置**：`scripts/eval_hierarchical_executor.py`

这是系统的核心保底机制，确保在RL模型卡壳时仍能继续前进。

---

## 核心模块详解

### 1. 地图系统 (`battlefield_rl/map/`)

**文件**：
- `loader.py`：地图加载和数据结构
- `scene_sampler.py`：训练场景采样器

**功能**：
- 从文本文件加载栅格化地图（高度+类型）
- 支持程序化生成训练场景（块状建筑、短墙、门洞）
- 从真实地图裁剪训练片段

**地图格式**：
```
(高度,类型) (高度,类型) ...
```
- 类型0：可通行
- 类型1：静态障碍
- 类型2：另一种障碍

### 2. 环境系统 (`battlefield_rl/env/`)

**文件**：`tactical_env.py`

**核心类**：`TacticalBattlefieldEnv`

**功能**：
- 8方向移动动作（上下左右+4个斜角）
- 敌人威胁场渲染（基于距离、角度、视线）
- 可见性检测（Bresenham画线算法）
- 足迹衰减机制（防止重复访问）
- 多维度奖励函数：
  - 时间惩罚：`step_penalty = -0.1`
  - 推进奖励：基于欧几里得距离变化
  - 威胁惩罚：处于敌人威胁范围内
  - 可见性惩罚：被敌人直接看到
  - 重复访问惩罚：基于足迹地图
  - 路标奖励：到达中间路标
  - 目标奖励：到达终点 `goal_reward = 50.0`
  - 超时惩罚：`timeout_penalty = -60.0`

### 3. 强化学习系统 (`battlefield_rl/rl/`)

**文件**：
- `network.py`：神经网络定义
- `agent.py`：D3QN智能体
- `replay.py`：优先经验回放缓冲区

**核心特性**：
- **Double DQN**：分离动作选择和价值评估，减少Q值过估计
- **Dueling 架构**：分离状态价值和动作优势，提升学习效率
- **优先经验回放（PER）**：重要样本更高概率被采样
- **ε-贪心策略**：探索与利用的平衡，线性衰减

### 4. 训练系统 (`battlefield_rl/train/`)

**文件**：
- `trainer.py`：分层训练器
- `curriculum.py`：课程学习门控

**课程学习策略**：

| 阶段 | 威胁等级 | 敌人数量 | 启发式引导概率 | 升阶条件 |
|------|---------|---------|--------------|---------|
| Stage 1 | 0（无威胁）| 0 | 100% → 70% | 成功率≥85%，超时率≤15%，路径效率≤1.80 |
| Stage 2 | 0.5（弱威胁）| 1 | 50% → 5% | 成功率≥80%，超时率≤20%，路径效率≤2.00 |
| Stage 3 | 1.0（全威胁）| 1-3 | 20% → 0% | 无自动升阶，持续训练 |

**启发式引导**：
训练初期，有一定概率用 `heuristic_action()` 代替RL模型决策（手动写死的贪心策略：距离权重1.0 + 威胁权重1.5 + 重复惩罚0.2），帮助模型快速学到基本走法。随训练推进逐步降低引导概率。

### 5. 全局规划器 (`battlefield_rl/planner/`)

**文件**：`global_planner.py`

**算法**：
- **JPS（Jump Point Search）**：A* 的加速变体
  - 强迫邻居检测：识别必须探索的关键点
  - 跳跃探测：沿方向跳过中间节点
  - 方向剪枝：减少不必要的探索
- **A***：标准启发式搜索，作为JPS的降级方案
- **路径压缩**：保留起点、终点和拐点
- **航路点插值**：按距离或步数间隔生成中间目标

---

## 回退机制详解（核心设计）

### 触发条件

当RL执行器**连续失败**（振荡/卡住/超时）且**重规划次数用尽**时触发：

```
trap_replans >= MAX_REPLANS_PER_TRAP（默认2次）
```

**触发流程**：
1. 一个航路点执行失败
2. 重新规划路径，提取新航路点
3. 再次尝试执行，仍然失败
4. 第3次重规划后仍然失败
5. **触发几何回退**

### 回退执行

`follow_geometric_path()` 函数做的事情是：
- 放弃RL模型，直接沿着A*/JPS规划的几何路径前进
- 一步一步走，不经过神经网络决策
- 每次最多走 `FALLBACK_ESCAPE_STEPS = 20` 步（避免一次性走太远）
- 走完20步后，**重新用RL模型接管**（从当前位置重新规划全局路径）

### 安全限制

- `MAX_FALLBACK_EVENTS = 20`：整个任务中最多触发20次回退
- `MAX_FALLBACK_STEPS = 2000`：回退总步数不超过2000步
- `MAX_TOTAL_STEPS = 12000`：整个任务总步数不超过12000步

### 回退后的恢复

回退走完后：
1. 从当前位置重新调用 `build_plan()` 生成新的全局路径
2. 重新提取航路点
3. 重置 `trap_replans = 0`
4. 继续用RL模型执行

**设计保证**：RL模型能处理大部分正常路段，但遇到模型"卡壳"的地方，用几何路径硬走一段脱困，再还给模型。

---

## 振荡检测与应对

### 运行时检测（`eval_hierarchical_executor.py`）

- `run_segment()` 中：最近10步内去重坐标 ≤ 3 → 判定为振荡，立即终止该段
- 触发重规划或几何回退

### 分析工具（`tests/detect_oscillation.py`）

多维度振荡分析：
- **滑动窗口检测**：窗口内去重坐标数
- **即时折返检测**：A→B→A 模式
- **循环模式检测**：长度2-6的循环
- **位移效率分析**：净位移/累计距离
- **热点格子统计**：高频重复访问的位置
- **推进窗口分析**：每段路径对目标的推进

**严重程度分级**：
- `none`：无振荡
- `mild`：轻微振荡
- `moderate`：中度振荡
- `severe`：严重振荡

### 条件式 Recency Masking（`tests/test_recency_masking.py`）

**实验性回退机制**：
- 只在检测到振荡时，封锁最近走过的3个格子（禁止再进入）
- 如果封锁导致所有方向都走不通，则回退到原始mask（防止卡死）
- 对比实验表明这能打破部分振荡模式

---

## 测试工具体系

### 测试文件说明

| 文件 | 功能 | 用途 |
|------|------|------|
| `tests/detect_oscillation.py` | 路径振荡的全面诊断工具 | 分析模型执行路径中的振荡问题 |
| `tests/test_recency_masking.py` | 条件式禁行掩码的A/B对比测试 | 验证振荡应对策略的有效性 |
| `tests/compare_global_models.py` | 多个checkpoint在完整路径规划任务上的对比 | 模型版本间性能对比 |
| `tests/compare_local_executor_models.py` | 多个模型在局部执行段上的对比 | 局部执行能力评估 |
| `tests/compare_visibility.py` | 敌人可见性回避能力的对比 | 可见性规避效果评估 |
| `tests/compare_visibility_realmap.py` | 真实地图上的可见性对比 | 真实场景性能验证 |
| `tests/compare_visibility_tendency.py` | 可见性变化趋势分析 | 训练过程监控 |
| `tests/benchmark_path_algorithms.py` | 路径算法性能基准测试 | 算法效率评估 |
| `tests/benchmark_realmap.py` | 真实地图性能基准 | 真实场景性能测试 |

### 使用示例

**振荡检测**：
```bash
python tests/detect_oscillation.py --model episode_8000.pt --cases 100 --stage 3
```

**模型对比**：
```bash
python tests/compare_global_models.py --model episode_6000.pt --model episode_8000.pt --cases 50
```

**可见性测试**：
```bash
python tests/compare_visibility.py --model episode_6000.pt --model episode_8000.pt --n-enemies 3 --enemy-range 60
```

**Recency Masking测试**：
```bash
python tests/test_recency_masking.py --model episode_8000.pt --cases 100
```

---

## 部署接口

### HTTP API（`interfaces/path_planning_api.py`）

**功能**：接收JSON请求，调用 `plan_and_execute`

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
      "row": 50,
      "col": 50,
      "facing_deg": 225,
      "fov_deg": 90,
      "range": 20
    }
  ],
  "use_fallback": true,
  "output_dir": "outputs"
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

### UDP Server（`interfaces/udp_path_server.py`）

**功能**：二进制协议包装，支持坐标系转换（xy↔row/col），同时服务路径规划和阵地选择两个任务

**协议格式**：
- 消息头：2字节消息类型 + 20字节消息ID
- 消息体：UTF-8编码的JSON字符串

**支持的坐标系**：
- 输入：x,y坐标系（笛卡尔坐标系）
- 内部：row,col坐标系（矩阵坐标系）
- 输出：x,y坐标系

### 阵地选择接口（`select_field/`）

**功能**：独立模块，用可视域分析为炮兵选阵地

**模式**：
- `position`：阵地选择，为特定目标选择最佳射击位置
- `lookout`：瞭望点选择，在两个区域间选择最佳观察位置

---

## 快速开始

### 环境要求

- Python 3.12+
- PyTorch 2.0+
- NumPy

### 安装依赖

```bash
pip install torch numpy
```

### 运行演示

**基础演示**：
```bash
python scripts/run_demo.py --map MyPath_Data417.txt --start 20,20 --goal 420,420
```

**路径规划API**：
```bash
python -c "
from interfaces.path_planning_api import handle_path_planning
result = handle_path_planning({
    'map': 'MyPath_Data417.txt',
    'start': [20, 20],
    'goal': [73, 97],
    'enemies': [{'row': 50, 'col': 50, 'facing_deg': 225, 'fov_deg': 90, 'range': 20}],
    'use_fallback': True
})
print(result)
"
```

**UDP服务器**：
```bash
python -m interfaces.udp_path_server --port 12345 --map MyPath_Data417.txt
```

**阵地选择**：
```bash
python -m select_field --map MyPath_Data417.txt --mode position --enemy 80,80 --type ZM
```

---

## 配置说明

### 环境配置（`battlefield_rl/config/settings.py`）

**EnvConfig**：
- `window_size = 25`：局部视野窗口大小
- `step_penalty = -0.1`：每步时间惩罚
- `progress_reward_scale = 0.3`：推进奖励系数
- `goal_reward = 50.0`：到达目标奖励
- `timeout_penalty = -60.0`：超时惩罚
- `timeout_steps = 50`：最大步数
- `no_progress_limit = 10`：无进展步数限制

**AgentConfig**：
- `gamma = 0.99`：折扣因子
- `lr = 1e-4`：学习率
- `epsilon_start = 0.4`：初始探索率
- `epsilon_end = 0.05`：最终探索率
- `epsilon_decay_steps = 150_000`：探索率衰减步数
- `target_sync_steps = 2_000`：目标网络同步步数
- `batch_size = 64`：批次大小
- `memory_size = 200_000`：经验回放缓冲区大小

**CurriculumConfig**：
- `max_stage = 3`：最大课程阶段
- `threat_scale_stage1/2/3`：各阶段威胁等级
- `stage2/3_enemy_count_min/max`：各阶段敌人数量范围
- `heuristic_guidance = True`：是否启用启发式引导
- `stage1/2/3_guide_start/end`：各阶段引导概率范围

### 全局规划配置

**PlannerConfig**：
- `waypoint_interval = 22`：航路点插值间隔
- `allow_diagonal = True`：是否允许对角移动

**关键常量**（`eval_hierarchical_executor.py`）：
- `WAYPOINT_INTERVAL = 18`：航路点间隔
- `SEGMENT_STEPS = 45`：每段最大执行步数
- `MAX_REPLANS_PER_TRAP = 2`：最大重规划次数
- `FALLBACK_ESCAPE_STEPS = 20`：回退逃脱步数
- `MAX_FALLBACK_EVENTS = 20`：最大回退事件数
- `MAX_FALLBACK_STEPS = 2000`：最大回退步数
- `MAX_TOTAL_STEPS = 12000`：最大总步数

---

## 项目结构

```
ZGBQ/
├── battlefield_rl/              # 核心强化学习系统
│   ├── config/                  # 配置定义
│   │   ├── __init__.py
│   │   └── settings.py          # 所有配置类
│   ├── map/                     # 地图解析与数据结构
│   │   ├── __init__.py
│   │   ├── loader.py            # 地图加载器
│   │   └── scene_sampler.py     # 训练场景采样器
│   ├── planner/                 # 全局路径规划
│   │   ├── __init__.py
│   │   └── global_planner.py    # JPS/A* 规划器
│   ├── env/                     # 局部战术环境
│   │   ├── __init__.py
│   │   └── tactical_env.py      # 环境实现
│   ├── rl/                      # 强化学习组件
│   │   ├── __init__.py
│   │   ├── network.py           # D3QN 网络
│   │   ├── agent.py             # 智能体
│   │   └── replay.py            # 优先经验回放
│   ├── train/                   # 训练系统
│   │   ├── __init__.py
│   │   ├── trainer.py           # 分层训练器
│   │   └── curriculum.py        # 课程学习
│   └── __init__.py
├── interfaces/                  # 部署接口
│   ├── path_planning_api.py     # HTTP API
│   ├── udp_path_server.py       # UDP 服务器
│   └── site_selection_api.py    # 阵地选择 API
├── select_field/                # 阵地选择模块
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                   # 命令行接口
│   ├── models.py                # 数据模型
│   ├── map_loader.py            # 地图加载
│   ├── geometry.py              # 几何计算
│   ├── visibility.py            # 可视域分析
│   ├── scoring.py               # 评分系统
│   └── service.py               # 服务层
├── scripts/                     # 运行脚本
│   ├── run_demo.py              # 演示脚本
│   └── eval_hierarchical_executor.py  # 分层执行评估
├── tests/                       # 测试工具
│   ├── detect_oscillation.py    # 振荡检测
│   ├── test_recency_masking.py  # Recency Masking 测试
│   ├── compare_global_models.py # 全局模型对比
│   ├── compare_local_executor_models.py  # 局部执行对比
│   ├── compare_visibility.py    # 可见性对比
│   ├── compare_visibility_realmap.py  # 真实地图可见性
│   ├── compare_visibility_tendency.py # 可见性趋势
│   ├── benchmark_path_algorithms.py   # 路径算法基准
│   └── benchmark_realmap.py     # 真实地图基准
├── MyPath_Data417.txt           # 示例地图文件
├── episode_*.pt                 # 模型检查点
├── request.json                 # 示例请求
├── request_lookout.json         # 瞭望点请求示例
├── request_site.json            # 阵地选择请求示例
├── history.txt                  # 开发历史
├── 强化学习.txt                  # 学习笔记
└── README.md                    # 项目文档
```

---

## 开发历史

### 关键里程碑

1. **基础架构搭建**：完成地图加载、环境定义、全局规划器
2. **强化学习集成**：实现D3QN网络、经验回放、训练循环
3. **课程学习实现**：分阶段训练，逐步增加难度
4. **回退机制开发**：几何回退保底，确保鲁棒性
5. **振荡检测系统**：多维度振荡分析和应对策略
6. **部署接口完善**：HTTP API和UDP服务器
7. **阵地选择模块**：独立的可视域分析系统

### 已知问题

- 振荡问题：部分场景下模型仍会陷入振荡
- 路径效率：复杂地形下路径效率有待提升
- 可见性规避：敌人密集区域的规避策略需要优化

### 未来扩展

- 更严格的 JPS 强迫邻居剪枝
- 地形阴影视线建模
- 动态重规划策略与多敌人协同威胁
- 多智能体协同路径规划
- 实时地图更新和动态障碍物处理

---

## 参考资料

- **JPS算法**：Jump Point Search - Fast Optimal Pathfinding on Grids
- **Dueling DQN**：Dueling Network Architectures for Deep Reinforcement Learning
- **Double DQN**：Deep Reinforcement Learning with Double Q-learning
- **优先经验回放**：Prioritized Experience Replay
- **课程学习**：Curriculum Learning

---

## 许可证

本项目为内部开发项目，仅供学习和研究使用。

---

## 联系方式

如有问题或建议，请联系项目维护者。