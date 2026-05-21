# 基于 JPS + Tactical D3QN 的栅格化战场路径规划

## 项目结构

- `battlefield_rl/config`: 配置定义
- `battlefield_rl/map`: 地图解析与数据结构
- `battlefield_rl/planner`: 全局路径规划（A* / JPS风格跳点）与路标插值
- `battlefield_rl/env`: 局部战术环境、威胁场渲染、动作掩码
- `battlefield_rl/rl`: D3QN 网络、PER 回放池、Agent
- `battlefield_rl/train`: 训练器
- `scripts`: 运行脚本

## 快速开始

```bash
python scripts/run_demo.py --map MyPath_Data417.txt --start 20,20 --goal 420,420
```

## 说明

当前版本优先搭建工业化模块骨架与可运行闭环，后续可继续扩展：
- 更严格的 JPS 强迫邻居剪枝
- 地形阴影视线建模
- 动态重规划策略与多敌人协同威胁
