# 寻路算法对比测试

这个目录只放独立测试脚本，不修改现有训练、环境、接口代码。

## 对比对象

- `bfs`: 静态地图上的 8 邻域最短路，不考虑敌人视野。
- `heuristic`: 使用 `TacticalBattlefieldEnv.heuristic_action()` 的贪心执行器。
- `current`: 当前模型执行链路，调用 `scripts.eval_hierarchical_executor.plan_and_execute()`。

## 示例

指定单个起终点：

```bash
python tests/benchmark_path_algorithms.py --map MyPath_Data417.txt --start 20,20 --goal 30,30
```

随机抽样 3 个短距离 case：

```bash
python tests/benchmark_path_algorithms.py --map MyPath_Data417.txt --cases 3 --bucket short
```

带敌人：

```bash
python tests/benchmark_path_algorithms.py --map MyPath_Data417.txt --start 20,20 --goal 80,80 --enemy 50,50,225,90,20
```

只快速测试 BFS 和启发式，不加载模型：

```bash
python tests/benchmark_path_algorithms.py --map MyPath_Data417.txt --cases 3 --skip-current
```

结果会写入 `outputs/benchmark_path_algorithms_*.json`，同时在终端输出压缩 JSON。
