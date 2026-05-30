"""interfaces 模块的默认配置。

修改此文件即可调整 UDP 服务器的默认参数，无需手动传参。
"""

# ── 网络配置 ──────────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9000

# ── 协议配置 ──────────────────────────────────────────────
# UDP 报文结构: [MsgType(2B)] [MsgID(20B)] [JSON Body(变长)]
# 寻路和选址目前复用同一个消息类型，服务端根据 payload 内容自动区分
MSG_TYPE_PATH_PLANNING = 0x3311   # 消息类型标识（USHORT），寻路请求
MSG_TYPE_SITE_SELECTION = 0x3311  # 消息类型标识（USHORT），选址请求
MSG_ID_SIZE = 20                  # 消息 ID 字段长度（字节），不足补 \x00
HEADER_SIZE = 2 + MSG_ID_SIZE     # 固定头部长度 = MsgType(2) + MsgID(20) = 22 字节
ENDIAN = "little"                 # 字节序，"little" 或 "big"，需与调用方一致

# ── 地图配置 ──────────────────────────────────────────────
# 换地图只需把新 txt 放到根目录，然后改这里的文件名
DEFAULT_MAP_PATH = "MyPath_Data417.txt"

# ── 模型配置 ──────────────────────────────────────────────
# 默认使用的 D3QN 模型 checkpoint 文件名（放在项目根目录下）
DEFAULT_MODEL_PATH = "episode_8000.pt"

# ── 寻路默认参数 ──────────────────────────────────────────
DEFAULT_FOV_DEG = 90.0
DEFAULT_ENEMY_RANGE = 20

# ── UDP 收发配置 ──────────────────────────────────────────
DEFAULT_BUFFER_SIZE = 65535
