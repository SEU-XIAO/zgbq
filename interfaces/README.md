# 接口使用说明

所有外部的调用均使用json进行输入与输出

Run:

```bash
python -m interfaces.api --input request.json
```

如果 `--input` 没有, api就从终端输入流当中读取json

## 寻路

坐标使用（行，列） `[row, col]`，访问数组时通常是 grid[row][col].

```json
{
  # 任务名
  "task": "path_planning",
  # 地图txt文件对于项目根目录的相对路径
  "map": "MyPath_Data417.txt",
  # 起点
  "start": [20, 20],
  # 终点
  "goal": [220, 220],
  # 敌人信息，五元卒
  "enemies": [
    {
      "row": 80,
      "col": 80,
      "facing_deg": 225,
      "fov_deg": 90,
      "range": 20
    }
  ],
  # 输出目录
  "output_dir": "outputs"
}
```

Fields:

- `task`: string, required, must be `path_planning`.
- `map`: string, required, txt map path.
- `start`: `[int, int]`, required, start coordinate as `[row, col]`.
- `goal`: `[int, int]`, required, goal coordinate as `[row, col]`.
- `enemies`: array, optional, each enemy is `{row:int,col:int,facing_deg:float,fov_deg:float,range:int}`.
- `output_dir`: string, optional, writes the same JSON result into this directory.

## 选址：瞭望点（lookout）

坐标使用 `[x, y]` and 矩形表示使用 `{x, y, w, h}`,访问数组时仍然要换回 grid[y][x].

```json
{
  "task": "site_selection",
  "mode": "lookout",
  "map": "MyPath_Data417.txt",
  "region_a": {"x": 10, "y": 10, "w": 80, "h": 80},
  "region_b": {"x": 180, "y": 180, "w": 80, "h": 80},
  "top_k": 5
}
```

Fields:

- `task`: string, required, must be `site_selection`.
- `mode`: string, required, must be `lookout`.
- `map`: string, required, txt map path.
- `region_a`: object, required, `{x:int,y:int,w:int,h:int}`.
- `region_b`: object, required, `{x:int,y:int,w:int,h:int}`.
- `top_k`: int, optional, default `5`.

## Site Selection: Position

```json
{
  "task": "site_selection",
  "mode": "position",
  "map": "MyPath_Data417.txt",
  "search_region": {"x": 10, "y": 10, "w": 120, "h": 120},
  "targets": [
    {"enemy": [80, 80], "type": "ZM"},
    {"enemy": [120, 110], "type": "JM"}
  ],
  "patch_size": 5,
  "top_k": 5
}
```

Fields:

- `task`: string, required, must be `site_selection`.
- `mode`: string, required, must be `position`.
- `map`: string, required, txt map path.
- `search_region`: object, required, `{x:int,y:int,w:int,h:int}`.
- `targets`: array, optional, each target is `{enemy:[x:int,y:int],type:string}`.
- `patch_size`: int, optional, default `5`.
- `top_k`: int, optional, default `5`.
