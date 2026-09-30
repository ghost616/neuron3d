# n3d_viz 测试报告（R23）—— 第二轮修复复验：离朱 R22 的 3 类缺陷

- 测试对象：`n3d_viz/core.py`（P1/P2/P3 修复）、`n3d_viz/verify_viz.py`（新增 5 项回归断言）、`n3d_viz/README.md` + `.module_agent/n3d_viz/current_spec.md`（文档同步）
- 复验基准：上一轮（R22）报告的 3 类缺陷
- FC 产物：`checkpoints/n3d_shape/full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.pt`（fc_dim=-1，H=825，seed=42）
- 执行环境：Windows / Python 3.12.10 / Node v25.2.1 / Playwright 1.63.0（Chromium headless）
- **结论：3 类缺陷全部修复到位，全量回归零破坏，产物逐字节不变。本轮未发现新的功能/产物缺陷。**
  另有 **1 项纯文档不一致（P4，非代码缺陷）** 详见第五节。

## 一、测试概览

| 测试类型 | 用例数 | 通过 | 失败 | 结论 |
|---|---|---|---|---|
| 编译/静态检查（compileall + 3 × node --check） | 4 | 4 | 0 | 通过 |
| **修复复验 + 全量回归**（独立测试程序 `.lizhu_env/r23/t_r23.py`） | **132** | **132** | **0** | 通过 |
| E2E 真实浏览器（Playwright + Chromium） | 40 | 40 | 0 | 通过 |
| GUI（tkinter withdraw） | 1（含 12 项子断言） | 1 | 0 | 通过 |
| 模块自带 `verify_viz.py` | 490 | 490 | 0（跳过 1） | 通过，退出码 0（`[2f]` 由 83 → **88**） |
| 渲染器逻辑冒烟（Node + DOM 桩） | 2 轮 | 2 | 0 | 无 FC 13/13、有 FC 20/20 |
| **合计** | **669** | **668** | **0** | 唯一 1 项为 `[2c]` 对照用 MLP 基线的明确跳过（非缺陷） |

被测源码快照（SHA256）：

| 文件 | SHA256（前 16） | 说明 |
|---|---|---|
| `n3d_viz/core.py` | `3296accb3130e41f` | 本轮修复（上一轮为 `aae9472e562f6087`） |
| `n3d_viz/verify_viz.py` | `56bdfc777c05efa7` | 本轮新增断言 |
| `n3d_viz/export_geometry.py` | `0a8b475c2e482256` | 未改动 |
| `n3d_viz/render_html.py` | `7cd491d919b25906` | 未改动 |
| `n3d_viz/__main__.py` | `a80173fe627eccb7` | 未改动 |
| `n3d_viz/gui.py` | `3114ea0cca7961c6` | 未改动 |
| `n3d_viz/assets/viewer_fc.js` | `27eb7d608a9d0402` | 未改动 |
| `n3d_viz/assets/viewer_smoke.js` | `71e4b2d06ec58733` | 未改动 |
| `n3d_viz/assets/viewer.js` | `53753ae80aafc41e` | 与 git HEAD 逐字节一致 |
| `n3d_viz/assets/viewer.html` | `60cdb020df4ea74e` | 与 git HEAD 逐字节一致 |

修复范围（用本机模块备份 `1790511882253.bak` 的**修复前** `core.py` 与当前版本逐行 diff 得出）：
**仅 8 个 hunk，全部位于 `validate_fc_top_k`（P1/P2）、`_fc_panel_geometry`（P3）的入口校验与其 docstring，以及 `extract_fc` docstring 的一行**。
**几何计算主体（`_place_units_in_panel` / `_grid_layout` / `_sample_top_k` / 抽样式与产物写出路径）一行未改** —— 这正是 FC 产物得以逐字节不变的结构性原因。

## 二、P1 / P2 复验：`core.validate_fc_top_k` 不再静默截断、异常类型统一

新契约：必须是 `int` 且不是 `bool`；所有非法输入一律 `ValueError`。

| 输入 | 期望 | 实测 |
|---|---|---|
| `1` / `3` / `8` | 正常返回同一 `int` | 返回且 `type is int` ✅ |
| `True` / `False` | 拒绝（bool 特判） | `ValueError: fc_top_k 必须是整数，不接受布尔值（当前为 True）。` ✅ |
| `2.5` / `1.5` / `8.7` / `3.0` / `2.0` / `-1.0` | 拒绝（含「看似整数的浮点」） | `ValueError: … 不接受 float 类型（当前为 2.5）；非整数一律报错，不做静默截断。` ✅ |
| `"abc"` / `"3"` / `"3.0"` | 拒绝（`str` 不隐式转换） | `ValueError: … 不接受 str 类型` ✅ |
| `None` | `ValueError`（**不再是裸 `TypeError`**） | `ValueError: … 不接受 NoneType 类型（当前为 None）` ✅ |
| `[1]` / `(1,)` / `{"k": 1}` | 拒绝（容器） | `ValueError: … 不接受 list/tuple/dict 类型` ✅ |
| `0` / `9` / `-1` / `100` | 拒绝（越界） | `ValueError: fc_top_k 必须在 [1, 8] 区间内，当前为 0；越界值一律报错，不做静默截断。` ✅ |

- **异常类型集合 == `{"ValueError"}`**（22 类非法输入逐个断言）✅
- 上一轮的 4 个缺陷点 `3.0` / `True` / `False` / `None` 现均被拒绝 ✅
- 错误信息可读（含合法区间与当前值、含类型名）✅

调用方回归：

| 调用方 | 实测 |
|---|---|
| CLI `--fc-top-k 1 / 8` | 退出码 0，OBJ `l` 行 == 2588 + (582+588)×k ✅ |
| CLI `--fc-top-k 0 / 9 / -1 / abc / 1.5` | 退出码非 0、无 traceback、**不产生任何产物** ✅ |
| CLI `--fc-top-k 3.0` | argparse `type=int` 在入口拦下，退出码 2（未进入新校验）✅ |
| GUI Spinbox `"3"/"8"/"1"` | 通过校验 ✅ |
| GUI Spinbox `"0"/"9"/"abc"/"2.5"/""` | 全部被拒绝（`ValueError`），`gui.start()` 走 `except (ValueError, TypeError)` 分支给出可读状态 ✅ |

> 附注（信息性，非缺陷）：`isinstance(k, int)` 口径下 `numpy.int64/int32` **不被接受**（抛 `ValueError`），而 `int` 的**子类**会被接受（`isinstance` 语义）。已核对真实调用链：`extract_topology` 经 `_as_int_list` 产出的是原生 `int`（`int(v)`），CLI 是 `argparse type=int`，GUI 是 `int(str)`，默认值来自模块常量；`config` 取自 `torch.load` 的 Python 对象，故 `fc_width` 也是原生 `int` —— **无 numpy 整型进入该函数的真实路径**，不影响功能。

## 三、P3 复验：`_fc_panel_geometry` 入口校验提前

| 输入 | 上一轮行为 | 本轮实测 |
|---|---|---|
| `H=0` | 裸 `ZeroDivisionError` | `CheckpointSchemaError: 全连接层面板要求宽度 H > 0，实际得到 H=0。` ✅ |
| `H=-1` | 裸 `ValueError: math domain error` | `CheckpointSchemaError: … H > 0，实际得到 H=-1。` ✅ |
| `H=True` | 未覆盖 | `CheckpointSchemaError: … 要求宽度 H 为整数，实际得到 bool（True）。` ✅ |
| `H=2.0` / `H=None` | 未覆盖 | `CheckpointSchemaError: … 要求宽度 H 为整数，实际得到 float/NoneType` ✅ |
| `axis='w'` | `CheckpointSchemaError` | `未知的流向轴 'w'；合法取值为 ['x', 'y', 'z']。` ✅ |
| 云流向轴跨度为 0 | `CheckpointSchemaError` | `神经元云在流向轴 'z' 上的跨度为 0（实测 1.0）…` ✅ |
| **合法边界** `H=1` | — | 正常构造：`cols=1 rows=1 interval=(-1.4, -1.2) cloud=(-1.0,1.0)`，面板与云不相交 ✅ |
| **合法边界** 共线点云（平面内两轴跨度均为 0） | — | 走兜底间距正常构造：`cell_size=2.0 > 0` ✅ |
| 入口校验优先级 | — | `H=0` **且**云跨度也为 0 时，先报 H 校验错（`H > 0`），顺序正确 ✅ |
| 单点云 `H=1`（span=0） | — | 报跨度为 0 的可读错 ✅ |
| 主链路 `extract_fc`（H=0 张量） | `CheckpointSchemaError` | 仍为 `checkpoint 的 fc_dim != 0 但有效宽度 H=0 <= 0。` ✅（未被新校验破坏） |

**新增的 5 项 `verify_viz` 回归断言**（`core.py` 之外的本轮 5 项）：

| 行号 | 断言 |
|---|---|
| 2255 | `[2f] core.validate_fc_top_k 接受 1 与 8` |
| 2260 | `[2f] core.validate_fc_top_k 拒绝非整数浮点与 bool（不静默截断）` |
| 2263 | `[2f] core.validate_fc_top_k 越界与非法类型都抛 ValueError（非裸异常）` |
| 2289 | `[2f] FC 几何入口 H<=0 / 类型非法 -> 可读 CheckpointSchemaError` |
| 2296 | `[2f] FC 几何入口 H=1 与共线云（两轴跨度均为 0）仍可用` |

`[2f]` 用例数 83 → **88**，与待测说明一致；运行结果 5 项全 PASS。

## 四、回归验证（全部通过）

### 4.1 三态判定

| 项目 | 实测 |
|---|---|
| `config` 无 `fc_dim`（二期 `model.pt`） | `is_fc_enabled=False`、`data.fc is None`、不报错 ✅ |
| 三期产物 `full_shapesphere_N256_..._s42.pt` | config 无 `fc_dim`、`data.fc is None`、不报错 ✅ |
| `fc_dim == 0`（FC 张量齐全） | `extract_fc` 返回 `None`，不报错 ✅ |
| `fc_dim == -1` | **启用**（宽度跟随 N），抽取到 `fc` 段 ✅ |
| `fc_dim == 1` | 启用 ✅ |
| 缺 `proj_weight` / `fc_out_weight` / `fc_in_weight` / `fc_out_bias`（逐个） | `CheckpointSchemaError` ✅ |
| CLI 走「缺键」产物 | 退出码非 0、无 traceback、**不产生任何产物**（原子性）✅ |

### 4.2 top-k 抽样

| 项目 | 期望 | 实测 |
|---|---|---|
| H / \|S_in\| / \|S_out\| | 825 / 582 / 588 | 一致 ✅ |
| `k=3` 抽样条数 | `\|S_in\|×k + \|S_out\|×k` = 3510 | 3510 ✅ |
| **`load_topology(fc_top_k=2)`** | **2340** | **2340**，且每神经元恰好 2 条 ✅ |
| 输入侧 / 输出侧口径 | 行内 / 列内 `\|w\|` top-k | 各核对 100 个神经元，逐位一致 ✅ |
| 每个 S_in / S_out 神经元 | 至少一条 | 无缺失（且恰好 k 条）✅ |
| `k=1` / `k=8` | 1170 / 9360 且产物不同、`k1 ⊂ k8` | 一致 ✅ |
| 同一产物两次运行 | 逐字节相同 | HTML/PLY/OBJ 三件套 SHA256 全等 ✅ |

### 4.3 零回归（硬约束）

| 项目 | 期望 | 实测 |
|---|---|---|
| 无 FC 重渲 HTML | SHA256 `15A80EBB…` / 88,521 字节 | 完全一致 ✅ |
| 无 FC 重渲 PLY | SHA256 `9A097D16…` / 4,120 字节 | 完全一致 ✅ |
| 无 FC 重渲 OBJ | SHA256 `1F594ECF…` / 15,695 字节 | 完全一致 ✅ |
| 磁盘锚点 `viz_model.{html,ply,obj}` | 同上 | 一致 ✅ |
| 三期产物重渲 | 与磁盘交付件一致：88,634 / 4,177 / 15,752 字节 | 三个字节数全部一致 ✅ |
| `assets/viewer.js` / `viewer.html` | 与 git HEAD 逐字节一致 | 一致 ✅ |
| 声明未改动的 `export_geometry.py` / `render_html.py` / `__main__.py` / `gui.py` / `viewer_fc.js` / `viewer_smoke.js` | 存在且 SHA256 与上一轮一致 | 一致 ✅ |

### 4.4 FC 产物逐字节不变（关键证据：修复前 vs 修复后）

用本机模块备份中的**修复前** `core.py`（`aae9472e562f6087`，即 R22 被测版本）与**修复后** `core.py`（`3296accb3130e41f`）在同一环境、同一 checkpoint、同一参数下分别渲染，再与磁盘交付件三方比对：

| 产物 | 修复前 SHA256 | 修复后 SHA256 | 磁盘交付件 | 字节数 | 结论 |
|---|---|---|---|---|---|
| HTML | `34cddb922807787d…` | `34cddb922807787d…` | `34cddb922807787d…` | 591,709 | **三方逐字节相同** ✅ |
| PLY | `25c1008c7c412f0a…` | `25c1008c7c412f0a…` | `25c1008c7c412f0a…` | 81,681 | **三方逐字节相同** ✅ |
| OBJ | `4e0c2a2f125f7c4f…` | `4e0c2a2f125f7c4f…` | `4e0c2a2f125f7c4f…` | 154,372 | **三方逐字节相同** ✅ |

字节数与待测说明声明的 **HTML 591,709 / PLY 81,681 / OBJ 154,372** 完全一致。
（上一轮脚本首轮比对时因传入**绝对路径** `-o` 导致 HTML meta 记录路径不同而误报，本轮统一使用与基线相同的相对路径风格。）

### 4.5 产物与声明

| 项目 | 实测 |
|---|---|
| PLY | `element vertex 825`（FC 点不混入）+ `element fc_node 1652 == 2×H+2`（含 `property uchar kind`）+ `element fc_edge 3510`；元素顺序 `vertex → fc_node → fc_edge`；头部含 `NOT all connections` 与两侧参数量 480150/485100 ✅ |
| OBJ | 独立 group `n3d_viz_fc_sampled_edges` 与 `n3d_viz_core_edges` 分离；注释含「非全部连接」+ `k=3` + `480,150 条 + 485,100 条` + `declared total would be 965250 links` ✅ |
| HTML | `meta.fcDeclaration` 含「非全部连接」、`meta.fcNotAllConnections == true`、`fcTopK==3`、`fcWidth==825`、`fcSampleEdges==3510 == len(payload.fc.edges)`；负载含 panels(2)/blocks(2)/edges/矩阵形状 ✅ |
| 参数量 vs 抽样条数 | 480150 / 485100 与 3510 **分别标注、未混用** ✅ |
| 声明三处齐备 | HTML meta / PLY 头部注释 / OBJ 注释 ✅ |
| 自包含 | 无 `<script src>`、无 `syn_dist` ✅ |

### 4.6 面板几何与「区间 / 中心」口径

独立复算（与待测说明的澄清一致）：

| 量 | 实测 |
|---|---|
| 输入面板**区间** `flow_interval` | `[-1.2938, -1.2801]` ✅ |
| 输出面板**区间** | `[1.2801, 1.2938]` ✅ |
| 输入/输出面板**中心** `flow` | `-1.2869` / `1.2869` ✅（与区间明确区分） |
| 面板厚度 | `0.013654475`（∈ [1e-3, span×0.10]）✅ |
| 云区间 | `[-0.9899, 0.9899]`，与两面板区间**不相交** ✅ |
| 面板点 | `2×H = 1650` ✅ |
| 边界块中心 z / 尺寸 | `∓1.5511` / `1.1879 × 1.1879 × 0.2772` ✅ |
| 网格 | `29 × 29 = ceil(sqrt(825))` 列 ✅ |

### 4.7 渲染器（含真实浏览器 E2E）

| 场景 | 实测 |
|---|---|
| `node --check` × 3 | 退出码 0 ✅ |
| 无 FC 冒烟 / 有 FC 冒烟 | 13/13 、 20/20 ✅ |
| E2E 无 FC 产物 | 无 JS 报错；负载无 `fc` 段；`#view` 真实绘制（非透明采样 40960/40960）；**未创建** `#view-fc`；**无** `#fc-stats` ✅ |
| E2E FC k=3 | 叠加画布 `#view-fc` 与基础画布同投影尺寸 1280×800；`z-index:20`；`pointer-events:none`；**确实画出内容**（非透明 3096/40960）；`#fc-stats` 含 `H=825 top-k=3 proj_weight=582×825 480150 条 fc_out_weight=825×588 485100 条 抽样连线=3510 条` + 抽样声明；图例 `data-fc=1` ✅ |
| E2E FC k=1 | `fc.edges==1170`；叠加层非透明 3003/40960 ✅ |
| E2E 交互回归 | 阈值拖到 0 →「保留 2588 / 2588 条边」；悬停 → hover-label 正确；视口 1280×800 → 900×620 后叠加画布尺寸跟随（900×620）且仍有内容、**全程无 JS 报错** ✅ |
| k 影响渲染 | k=3（3096）≠ k=1（3003）绘制像素数 ✅ |
| E2E 总计 | **40 / 40 通过** ✅ |

### 4.8 模块自带验证脚本

```
python n3d_viz/verify_viz.py --report .lizhu_env/r23/verify_report.md
→ 汇总：通过 490 / 失败 0 / 跳过 1（共 490），退出码 0
```

- `[2f]` FC 组 88 项（上一轮 83 项）全部 PASS，含本轮新增 5 项
- `[2d]` 零回归锚点：磁盘产物与重渲产物 SHA256 与字节数逐项一致，临时产物已清理（0 字节残留）
- 跳过项唯一：`[2c]` 中 18 个 `*_mlp.pt` 对照用 MLP 基线缺 N3D 拓扑键（明确跳过，非缺陷）

## 五、本轮唯一未通过项：文档不一致（P4，非代码缺陷）

`current_spec.md` 中仍保留上一轮（几何常量不同时期）的**陈旧实测值**，与 `README.md` 及磁盘交付件**自相矛盾**：

| 位置 | 文档写的值 | 独立实测值（本次与 README:403-407 / current_spec.md:418-419 一致） |
|---|---|---|
| `current_spec.md:387-388` | 输入区间 `[-1.2944, -1.2794]`、输出 `[1.2794, 1.2944]`、边界块中心 `∓1.5518` | `[-1.2938, -1.2801]`、`[1.2801, 1.2938]`、`∓1.5511` |
| `current_spec.md:274`、`389`、`394` | 三件套字节 `587,796 / 81,681 / 153,748`；SHA256 `C2F15D3D…` / `5FFF5F6F…` / `E1F6397A…` | 字节 `591,709 / 81,681 / 154,372`；SHA256 `34CDDB92…` / `25C1008C…` / `4E0C2A2F…` |
| `README.md:615` | `[2f]` 行写面板区间 `[-1.2944, -1.2794]` / `[1.2794, 1.2944]` | 同上（与 README:403-404 自相矛盾） |

补充证据：`current_spec.md` 引用的三个 SHA256（`C2F15D3D` / `5FFF5F6F` / `E1F6397A`）**在 `checkpoints/n3d_viz/` 下的 21 个 `viz_*` 文件中一个都找不到**；而磁盘上与本文其余各处一致的值是 `34CDDB92…`（591,709）、`25C1008C…`（81,681）、`4E0C2A2F…`（154,372）。

- **影响**：纯文档/规格描述问题，不影响任何代码行为与产物（产物本身经三方逐字节比对确认正确）。
- **修复建议**：刷新 `current_spec.md:274/387/388/389/394` 与 `README.md:615` 为上述实测值；建议在文档中为该类「实测值表」加注**采集时的源码 SHA256 或版本标记**，避免后续再出现同类脱耦（本轮已实测两次因几何常量演进导致的引用漂移）。

## 六、结论与建议

1. **P1 / P2 / P3 三项修复全部验证通过**（含合法边界 `H=1`、共线点云、`H` 类型非法、`None`、越界等），异常类型契约与 docstring 现已一致。
2. **全量回归零破坏**：三态判定、top-k 口径（含 `fc_top_k=2 → 2340`）、零回归锚点、FC 产物、CLI、GUI、渲染器（含真实浏览器）全部通过。
3. **FC 产物逐字节不变**已用「修复前 vs 修复后 vs 磁盘交付件」三方比对证实，且修复 diff 仅 8 个 hunk、全在入口校验与 docstring，几何与写出路径零改动。
4. 建议按第五节刷新文档中的陈旧实测值（P4），并在实测值表中标注采集时的源码版本/哈希。

## 七、复现命令

```powershell
# 编译 / 静态检查
python -m compileall -q n3d_viz
node --check n3d_viz/assets/viewer.js ; node --check n3d_viz/assets/viewer_fc.js ; node --check n3d_viz/assets/viewer_smoke.js

# 模块自带验证（490 / 0 / 1，退出码 0）
python -X utf8 n3d_viz/verify_viz.py --report .lizhu_env/r23/verify_report.md

# 本轮独立复验 + 全量回归（132 项，结果写 result_r23.json）
python -X utf8 .lizhu_env/r23/t_r23.py

# 真实浏览器 E2E（40 项；先在 .lizhu_env/r22_e2e 内 npm install playwright）
node .lizhu_env/r22_e2e/e2e_fc.mjs `
  .lizhu_env/r23/e2e/no_fc/viz_model.html `
  .lizhu_env/r23/e2e/fc_k3/viz_full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.html `
  .lizhu_env/r23/e2e/fc_k1/viz_full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.html
```

## 八、测试产物索引

- 模块自带验证报告：`.lizhu_env/r23/verify_report.md`（490/0/1）
- 独立复验结构化结果：`result_r23.json`（132 项，路径见运行输出 `work dir`）
- E2E 结果：`.lizhu_env/r23/e2e/e2e_result.json`（40 项）
- 修复前快照（用于逐字节比对）：`.lizhu_env/r23/oldpkg/n3d_viz/core.py`（来自 `.module_agent/n3d_viz/backups/99d9e681d17a4e373c3eb1c008a089c1/1790511882253.bak`）
- 测试脚本：`.lizhu_env/r23/t_r23.py`；E2E 脚本沿用 `.lizhu_env/r22_e2e/e2e_fc.mjs`（Playwright 环境未重复安装）
- 报告全文：`test_reports/lizhu_n3d_viz_fc_r23_report.md`
