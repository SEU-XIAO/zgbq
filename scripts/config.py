"""scripts 模块的默认配置。

训练、压测、demo 等脚本的启动参数默认值集中管理于此。
修改此文件即可调整默认行为，无需手动传参。
"""

from interfaces.config import DEFAULT_MAP_PATH  # noqa: F401 — 地图路径统一从 interfaces 读取

# ── 训练默认参数 ──────────────────────────────────────────
DEFAULT_EPISODES = 500            # 训练总 episode 数
DEFAULT_SEED = 42                 # 随机种子
DEFAULT_DEVICE = "auto"           # 训练设备："auto" / "cuda" / "cpu"
DEFAULT_SAVE_DIR = "checkpoints"  # 模型 checkpoint 输出目录
DEFAULT_SAVE_EVERY = 100          # 每隔 N 个 episode 保存一次 checkpoint
