# n3d_viz 几何无关化（层配色同源 + K>9 可扩展）—— 离朱测试报告

- 被测功能说明来源：`read_test_specs`（父会话 6d3c2b99）
- 被测模块：`n3d_viz`（core.py / export_geometry.py / verify_viz.py / __main__.py / gui.py / README.md）
- 测试时间：2026-09-25（本会话）
- 环境：Windows / Python 3.12.10 / torch 2.14.0+cpu / numpy 2.5.3 / Node v25.2.1 / Tk 8.6 /
  Playwright 1.63.0（npm，Chromium）+ Chromium 已安装；**pytest 未安装**，故独立单元测试用标准库 `unittest`
- 测试资产（均在 `.lizhu_env/lizhu_viz_r15/`）：`test_palette_unit.py`、`test_zero_regression.py`、
  `test_cli_interface.py`、`test_gui_e2e.py`、`test_static_and_docs.py`、`e2e_viewer.js`、
  `baseline/n3d_viz_base/`（从 `git HEAD` 解包的**改动前**实现，用作独立对照）

## 1. 测试概览

| # | 测试类型 | 用例/断言数 | 通过 | 失败 | 跳过 | 退出码 |
|---|---|---|---|---|---|---|
| 1 | 编译测试（`python -m compileall -q n3d_viz`） | 1 | 1 | 0 | 0 | 0 |
| 2 | JS 语法检查（`node --check` × viewer.js / viewer_smoke.js） | 2 | 2 | 0 | 0 | 0 |
| 3 | 项目自带验证脚本 `verify_viz.py`（全量硬断言） | 157 | 157 | 0 | 0 | 0 |
| 4 | 独立单元测试（层配色函数边界/反向/等价） | 31 | 31 | 0 | 0 | 0 |
| 5 | 零回归 + 修复证据（旧实现 vs 新实现对照） | 23 | 23 | 0 | 0 | 0 |
| 6 | 接口 / CLI / GUI 契约测试 | 38 | 38 | 0 | 0 | 0 |
| 7 | GUI 端到端测试（真实 tkinter 后台线程） | 24 | 24 | 0 | 0 | 0 |
| 8 | E2E（Playwright + Chromium 真实浏览器交互） | 29 | 29 | 0 | 0 | 0 |
| 9 | 静态扫描 / 归属纪律 / 渲染器冒烟 / 文档一致性 | 37 | 36 | 1 | 0 | 1 |
| **合计** | | **342** | **341** | **1** | **0** | — |

唯一失败项为**文档级问题**（README 行号引用不实，见 §5.1），**无功能性失败**。

规格书列出的 5 条复现命令全部执行成功：

```
python -m compileall -q n3d_viz                                   -> exit 0
python n3d_viz/verify_viz.py --report checkpoints/n3d_viz/_verify/verify_report.md -> exit 0（157/0/0）
node --check n3d_viz/assets/viewer.js                             -> exit 0
node --check n3d_viz/assets/viewer_smoke.js                       -> exit 0
python -m n3d_viz -c checkpoints/n3d_shape/full_shapecylinder_a2_..._s42.pt \
    --out-dir checkpoints/n3d_viz/_verify/_lizhu                   -> exit 0（K=15，三件套已落盘）
```

## 2. 编译测试

| 检查 | 实测 |
|---|---|
| `python -m compileall -q n3d_viz` | exit 0，无 SyntaxError |
| `node --check n3d_viz/assets/viewer.js` | exit 0 |
| `node --check n3d_viz/assets/viewer_smoke.js` | exit 0 |

收尾复跑（在全部测试结束后）仍为 exit 0。

## 3. 新增层配色函数的独立单元测试（31/31 通过）

对 `core.hex_to_rgb` / `core.layer_palette_hex` / `core.layer_palette_rgb` /
`export_geometry.LAYER_COLORS` / `neuron_colors` / `build_html_payload` 独立编写，
**独立预言机**取自 `git HEAD` 改动前源码里那份内联 9 色数组与 9 色 RGB 常量。

正向覆盖：

- `hex_to_rgb("#4e8cff") == (78,140,255)`；`"4e8cff"`（无 `#`）同值；`"#4E8CFF"` 同值；
  `"  #4e8cff  "`（含空白）同值；`#000000` / `#ffffff` 边界值正确；9 个基础色全部往返一致。
- `layer_palette_hex(K)`：`len()==K`、去重数 `==K` 对 **K=1..64 全部成立**；
  `K=1` / `9` / `10` / `15` / `33` / `64` 全部为合法小写 `#rrggbb`。
- `K<=9` 与改动前内联 9 色数组**逐字节相同**（K=1..9 全部比对）。
- `layer_palette_rgb(K) == [hex_to_rgb(h) for h in layer_palette_hex(K)]` 对 K=0..64 与 100 全部成立。
- `export_geometry.LAYER_COLORS == (78,140,255),…,(155,89,182)`（等于改动前 RGB 常量），
  且 `== tuple(core.hex_to_rgb(c) for c in core.LEVEL_PALETTE_BASE)`（同源派生）。
- `neuron_colors(data)` 在合成 TopologyData 上：长度==N、去重数==K（K∈{1,2,9,10,15,33,64}），
  且每个神经元颜色与其所在层色一一对应（逐层校验）。
- `build_html_payload(data)["layers"][k]["color"] == layer_palette_hex(K)[k]`（K∈{1,9,10,15,33,64}），
  层色去重数==K；HTML 层色 hex→RGB 后与 PLY 色板逐位相等（**同源硬不变量**）。
- 幂等/确定性：同一 K 两次调用结果完全相同；返回的是新列表（改写返回值不污染 `LEVEL_PALETTE_BASE`）。

反向覆盖：

- `"#xyz"` / `"xyz"` / `"#4e8cf"`（5 位）/ `"#4e8cfff"`（7 位）/ `""` / `"#"` / `"4e8cff0"` /
  `"0x4e8cff"` / `"#4e8cfg"` / `"#zzzzzz"` 全部抛 `ValueError`（规格强制项全部满足）。
- `layer_palette_hex(0) == []`，且 `-1` / `-9` / `-1000` 同样返回 `[]`（规格边界）。

边界值与极限值：

- 规格承诺域 K∈[1,64] 全通过；**超出承诺域的说明性探测**：K=100 / 256 / 512 / 1024 / 2048
  的去重数仍恒等于 K（K=2048 耗时 11.4 ms），色相扩展的确定性搜索未触及 `_PALETTE_SEARCH_LIMIT` 失败分支。

发现的实现细节（非规格强制项，见 §5.2）：`hex_to_rgb` 对「长度恰好为 6 但含符号/空格」的文本
不会抛错，且可能返回越界分量。

## 4. 零回归与修复证据（旧实现 vs 新实现直接对照，23/23 通过）

方法：`git archive HEAD n3d_viz` 解包为包名 `n3d_viz_base`（**改动前**实现），与工作区 `n3d_viz`
并列导入，对同一 checkpoint 各自完整渲染三件套后逐字节比对。这比「与常量比对」更强：它证明
**新实现与改动前实现产物逐字节相同**，而不是只与自己记录的常量相同。

K=9 基准产物 `checkpoints/n3d_sphere/model.pt`（seed=42）：

| 比对 | 结果 |
|---|---|
| `viz_model.html`：旧实现 SHA == 新实现 SHA == 磁盘产物 == 常量锚点 | **15A80EBBF2FD586BFB3C4C41E25F79F0E3E8F0C19D3B60AE22BE3F296EDA518C** ✓ |
| `viz_model.ply`：同上四方一致 | **9A097D16306160F95C9E15826CB11ABED57C6809F2904660B700573889398801** ✓ |
| `viz_model.obj`：同上四方一致 | **1F594ECF466E28F751C174A85A8A4E5459A304973F6266B3E11C567C25A09FED** ✓ |
| 整份 HTML 文件逐字节相同（含内联 JS + 负载） | 88,521 == 88,521 字节 ✓ |
| `build_html_payload` 的 JSON 文本逐字节相同 | len=66,737 == 66,737 ✓ |
| K=9 层色序列 == 既有 9 色原序 | `#4e8cff #00b7c2 #2ecc71 #a3d977 #f7d154 #f39c12 #e8734a #d94f70 #9b59b6` ✓ |
| CLI 实际生成的 K=9 三件套 SHA == 三个锚点 | 三处全部一致 ✓ |

> 关键方法学提醒：`meta.checkpoint` 会把**调用时给出的路径文本**写进 HTML 负载，因此字节级锚点
> 只在「同一调用形式」下可比（锚点产物写的是 `checkpoints\n3d_sphere\model.pt`；用绝对路径调用会
> 多 14 字节）。本测试因此改用相对路径调用以与锚点同形式，这也是我第一轮出现 3 个红色告警的原因，
> **不是代码缺陷**。

K=15 修复证据（`checkpoints/n3d_shape/full_shapecylinder_a2_…_s42.pt`，seed=42，K=15）：

| 断言 | 改动前实现 | 现在实现 |
|---|---|---|
| HTML 负载层色去重数 | **9**（缺陷复现） | **15** ✓ |
| PLY 顶点色去重数 | **9**（缺陷复现） | **15** ✓ |
| 第 9..14 层与第 0..5 层同色（k%9 周期性撞色） | `[9,10,11,12,13,14]` 全部撞色 | `[]`（无撞色）✓ |
| HTML 层色 hex→RGB 与 PLY 色板 | — | 逐一相等 ✓ |
| K=15 三件套与 `checkpoints/n3d_viz/` 磁盘现有产物 | — | **逐字节相同**（HTML SHA `FA5D5426…`）✓ |
| 与几何无关的指标不受影响 | PLY 顶点 256 / OBJ `l` 行 717 | 256 / 717 ✓ |

## 5. 接口 / CLI / GUI 契约测试（38/38 通过）

参数校验（退出码 2）：

| 用例 | 实测 |
|---|---|
| 未知参数 `--definitely-not-a-flag` | exit 2 |
| `--threshold not-a-number` | exit 2 |
| 只给 `--out-dir` 不给 `-c`（不静默启动 GUI） | exit 2 |
| `-t 0.3 -t 0.5`（重复参数取末值） | exit 0，负载 `meta.threshold == 0.5` |

checkpoint 错误路径（退出码 3、无 traceback、不产生任何产物）：

| 用例 | 实测 |
|---|---|
| 路径不存在 | exit 3，文案含「不存在」 |
| 损坏文件（非 torch 序列化） | exit 3，文案含「损坏或无法反序列化」，无 traceback |
| 路径是目录 | exit 3 |
| 一期产物 `checkpoints/n3d_model_full.pt` | exit 3，**列出全部 6 个缺失键**（edge_src / edge_dst / edge_weight / in_scope_mask / out_scope_mask / level_node_reach），无 traceback，零产物 → **schema 边界未放松** ✓ |
| `in_scope_mask` 长度 N-1（形状不符） | exit 3，报错含「长度」 |
| `topo_index[5]=-1` / `edge_dst[11]=256`（越界负例） | exit 3，报错**定位到下标与值**，无 traceback，零产物 ✓ |

开关类参数（每个布尔开关都必须有产物级可观测差异）：

| 用例 | 实测 |
|---|---|
| `--no-plan-planes` | HTML 内容不同，`meta.showPlanes=false`（默认 `true`） |
| `--ply-ascii` | PLY 头部 `format ascii 1.0` |
| `--with-ply-edges` | PLY 头含 `element edge 736` |
| `-o custom.html` | PLY / OBJ 与它同目录同名（`custom.ply` / `custom.obj`） |
| `--quiet` | 抑制 `[数据]` 过程日志，保留 `[完成]` 摘要 |

正向与命名派生：`-c checkpoints/n3d_sphere/model.pt` → exit 0，落盘 `viz_model.{html,ply,obj}`
（88,521 / 4,120 / 15,695 字节），摘要含实测 `N=256 E=736`；K=15 产物 exit 0，
负载 `layers=15`、层色去重 15、PLY 顶点色去重 15、neurons=256 / edges=717、
OBJ 写出、摘要打印 `K=15`。

GUI（tkinter）契约与端到端：

| 用例 | 实测 |
|---|---|
| `gui.default_checkpoint_text()` | 返回 `checkpoints/n3d_sphere/model.pt`（存在则预填） |
| `build_smoke_window()` 构造 | 成功，顶层子控件 7 个，可销毁 |
| 点击「开始生成」 | 后台线程启动、`开始生成` 置灰 |
| 生成期间再次点击 | 提示「已有生成任务在运行」且**不并发**（worker 对象不变） |
| 完成后 | 状态栏「完成：…（88443 字节），PLY 256 项点，OBJ 717 条 l 行。」；`打开 HTML` / `打开输出文件夹` 恢复可用；`开始生成` 恢复 |
| 同名产物再生成 | 状态栏「完成（已覆盖同名产物：html, obj, ply）」**与**日志「以下同名产物此前已存在并被覆盖」同时提示（不静默覆盖） |
| 不存在的 checkpoint | 状态栏「失败：checkpoint 路径不存在。」+ 日志错误行，不启动线程、按钮保持可用 |
| 一期产物 | 线程路径回到失败态（不崩溃），日志含缺失键错误，`showerror` 被调用，按钮恢复可用 |

> 说明：我第一轮接口测试中 GUI 段曾有 4 条红色，原因是**我的测试脚本**轮询写法有缺陷
> （`done` 经 `queue` → `root.after` 回传，worker 线程刚结束时会话尚未刷新）以及 ttk `state`
> 未做 `str()` 转换；修正脚本后同一被测代码 38/38、独立 GUI 脚本 24/24 全绿。已在 §7 记录。

## 6. E2E 测试（Playwright + Chromium，29/29 通过）

用 Playwright 1.63.0 以 Chromium 1280×800 真实打开产物（`file://`，**UI 交互未用单元测试模拟**）：

| 断言 | 实测 |
|---|---|
| 内联负载 N=256 / E=717 / K=15 / neurons=256 / edges=717 | 一致 |
| 负载层色去重数 == 15 | 15 |
| 图例色块数 == K+3 == 18，前 15 个层色块两两不同 | 18 / 15 |
| 画布非空白（采样到大量非黑像素）与尺寸初始化 | nonBlank=10557，1280×800 |
| 阈值滑块真实交互 | 拖到 0 → 标签 `0.00`、「保留 717 / 717 条边」；0.30 时 383 条（过滤生效）；像素变化 |
| 层平面复选框 | 初始态 == `meta.showPlanes`(true)；切换后画布重绘 |
| 神经元开关 | 关闭后画布重绘 |
| 悬停命中 | 详情标签填充「神经元 #245 层 L14（z = 0.8485）入度 2 / 出度 0 S_in: 是 / S_out: 是」 |
| 左键拖拽旋转 / 滚轮缩放 / 重置视角 | 像素均发生变化、重置后仍正常绘制 |
| 离线自包含 | **仅 1 个 `file://` 请求，零 http(s) 请求** |
| console 错误 / 未捕获异常 | 0 / 0 |
| K=9 既有产物 | 图例 12 块、层色 == 既有 9 色原序、无 JS 异常 |
| `--no-plan-planes` 产物 | 复选框默认未勾选、`meta.showPlanes == false` |

另有 Node + DOM 桩渲染器冒烟独立复跑：K=15 产物 **12/0，图例 18 块**；K=9 产物 **12/0，图例 12 块**
（与 README §2 的声称一致）。

## 7. 静态扫描 / 归属纪律 / 文档一致性（36/37，1 项为文档问题）

零依赖与跨模块隔离：

| 断言 | 实测 |
|---|---|
| `n3d_viz` 全部顶层 import | `__future__ argparse collections colorsys dataclasses hashlib json math numpy os pathlib queue re shutil struct subprocess sys threading tkinter torch typing webbrowser` + 自身包名 → 仅标准库 + torch/numpy |
| 新引入的标准库 | `colorsys` / `math` / `hashlib`（与说明一致） |
| 第三方绘图/几何依赖（matplotlib/plotly/pyvista/tkinterdnd2/scipy/PIL/pandas/open3d） | 0 命中 |
| `^\s*(from|import) n3d_(sphere\|proto\|shape)` 正则扫描 | 0 命中（独立实现，另加 AST 级顶层 import 扫描同样 0 命中） |
| `requirements.txt` | 仍仅 `torch>=2.2.0` / `torchvision>=0.17.0` / `numpy>=1.26.0`；`git diff --name-only HEAD -- requirements.txt` 为空 |

归属纪律（他人模块零改动）：

| 断言 | 实测 |
|---|---|
| `git status --porcelain -- n3d_sphere n3d_proto n3d_shape` | 空（无改动） |
| 他人模块源码与 `.pt` 产物 mtime | 最新他人文件 `n3d_shape/config.py` @ 19:45:52 **早于**最早改动的 `n3d_viz` 源码 @ 21:07:12 → 本次改动未重写任何他人源码或产物 |

文档一致性（README）：

| 断言 | 实测 |
|---|---|
| 断言清单总项数 == 157 | 157 ✓ |
| 新增 80 项构成 == [2a]3 + [2b]42 + [2c]32 + [2d]3 | 3/42/32/3 ✓ |
| 报告第 125 行 == 渲染器逻辑冒烟（12/0） | ✓ |
| 报告第 71/72 行 == 负例临时产物清理断言 | **✗（见 §7.1）** |
| README 记载 `viz_model.{html,ply,obj}` = 88,521 / 4,120 / 15,695 字节 | ✓ |
| README 记载 K=15 三件套 = 88,429 / 4,182 / 15,613 字节 | ✓ |
| README 列出的 K=15 十五个层色 == 真实产物 | 逐个一致 ✓ |
| README 记载的 3 个锚点 SHA == verify 常量 | ✓ |
| README §6.1 实测数字可复算 | 层规模 / 层 z / 层入边数 / 入度直方图 / 出度直方图 / \|w\| min-median-max / S_in 193 / S_out 187 / `input_isolated_mask` 605 **全部复算一致** ✓ |
| `SYNTH_SEED = 20250925`（README 与源码一致） | ✓ |

### 7.1 【失败·文档级】README 行号引用不实

`n3d_viz/README.md` §6.1 表格末行称：

> 验证汇总 | 通过 **157** / 失败 0 / 跳过 0（退出码 0）；其中报告断言清单第 125 行为渲染器逻辑冒烟
> （内含 12 条子断言，均通过），**第 71/72 行为负例临时产物清理断言**

按规格书复现命令重新生成 `checkpoints/n3d_viz/_verify/verify_report.md` 后，实际行号为：

- 第 **125** 行 = `内联渲染器可执行且投影正确`（声称正确 ✓）
- 第 **71** 行 = `[2b] one_per_layer: 产物名由 ckpt 名派生`；第 **72** 行 = `[2b] single_layer: PLY 顶点数 == N(40)`
- 真正的临时产物清理断言在：第 **88/89** 行（`[2b] 合成组临时产物已清理` / `残留体积 == 0 字节`）
  与第 **151/152** 行（`负例产物已清理（_bad_index/ 无文件残留）` / `负例残留体积 == 0 字节`）

影响：纯文档引用错误，不影响任何功能与断言；但会误导验收方按行号核对。**建议把「第 71/72 行」
改为「第 151/152 行（负例清理）与第 88/89 行（合成组清理）」**，或改为不依赖行号的文字描述。
（注意：新增断言插入位置一旦变动，行号会再次漂移，故不写行号更稳。）

## 8. 其它发现与修复建议

### 8.1 【低危】`core.hex_to_rgb` 校验松散，可返回越界分量

规格强制项全部满足（`"#xyz"`、长度≠6 均抛 `ValueError`），但实现只校验「长度 == 6」再分段
`int(seg, 16)`，而 CPython 的 `int(..., 16)` 接受前导符号与空白：

| 输入 | 实测返回 | 期望 |
|---|---|---|
| `"#-12345"` | `(-1, 35, 69)`（**负分量**） | 抛 `ValueError` |
| `"#+12345"` | `(1, 35, 69)` | 抛 `ValueError` |
| `"#4e8c f"` | `(78, 140, 15)` | 抛 `ValueError` |
| `"# 4e8cf"` | `(4, 232, 207)` | 抛 `ValueError` |

与 docstring 契约相悖：docstring 声明「文本不是 6 位十六进制 → 抛 `ValueError`」、
「Returns: `(r, g, b)`，各分量取值 0..255」。

影响面：`hex_to_rgb` 目前仅被模块内部以 `LEVEL_PALETTE_BASE` 常量与
`layer_palette_hex` 的产物调用，**不存在外部输入路径**，因此不影响任何现有产物（K=9 零回归、
K>9 去重数均已实测成立）。属「契约与实现的缝隙」。

建议（可选，一行收紧）：

```python
import re
_HEX_RE = re.compile(r"[0-9a-fA-F]{6}\Z")
text = color.strip().lstrip("#")
if not _HEX_RE.fullmatch(text):
    raise ValueError(f"颜色文本必须是 6 位十六进制：{color!r}")
```

### 8.2 【信息】色相扩展色板在远超规格上限时依然稳健

规格只承诺 K∈[1,64]，实测 K=100/256/512/1024/2048 的去重数仍恒等于 K，
K=2048 仅 11.4 ms，未触发 `_PALETTE_SEARCH_LIMIT` 失败分支（`raise ValueError` 分支未覆盖到，
属正常）。

### 8.3 【信息·测试侧】我第一轮脚本自身的三处缺陷（非产品问题，已修正）

为避免与真实缺陷混淆，记录如下（均在测试脚本侧修正后复跑全绿）：

1. **Tk 事件泵轮询**：`done` 经 `queue` → `root.after(100, …)` 回传主线程，
   第一版「worker 线程一结束就跳出循环」会在会话刷新前读到旧状态（表现为 4 条 GUI 假红）；
   必须持续 `root.update()` 直到 `last_html` 有值且状态进入「完成」。
2. **态式复用**：以 `status.startswith("失败")` 作等待判据会被上一次的失败态直接满足 → 改为以
   日志新内容（`缺少二期拓扑必需键`）作单调判据，且**绝不在 worker 存活时再调 `start()`**
   （否则弹模态框阻塞自动化）。GUI 端到端脚本还把 `gui.messagebox` 替换为记录桩，
   以便既断言弹窗路径又不阻塞。
3. **字符串/类型比对**：日志原文是「以下同名产物此前已存在并**被覆盖**」（不是「已覆盖同名产物」）；
   ttk `Button["state"]` 需 `str()` 转换后才能与 `"normal"` 比较。

## 9. 结论

- 本轮变更的**功能目标全部达成且证据充分**：
  - 层配色唯一同源（`core.layer_palette_hex` / `layer_palette_rgb`），HTML 与 PLY 层色逐位一致；
  - `K<=9` 与改动前**逐字节相同**（用 `git HEAD` 旧实现重新渲染对照，不只是常量自证）；
  - `K>9` 层色去重数恒等于 K，K=15 缺陷（旧 9 → 新 15）修复证据齐全；
  - schema 边界未放松（一期产物、形状不符、越界索引均 exit 3 且无 traceback、零产物）；
  - CLI 退出码契约 0/2/3 不变，GUI 行为不变且新增 `default_checkpoint_text()` 正确；
  - 零新依赖、零跨模块 import、他人模块源码与产物零改动。
- `verify_viz.py` 全绿 157/0/0（退出码 0），与规格声称一致。
- **无功能性失败**。唯一失败项为 README 一处行号引用不实（文档级，建议改为 151/152 或不写行号）；
  另有一处低危契约缝隙（`hex_to_rgb` 对含符号/空格的 6 字符文本不报错、可返回负分量），
  无外部输入路径，可选收紧。

### 复现全部测试

```bash
python -m compileall -q n3d_viz
python n3d_viz/verify_viz.py --report checkpoints/n3d_viz/_verify/verify_report.md
node --check n3d_viz/assets/viewer.js && node --check n3d_viz/assets/viewer_smoke.js
python .lizhu_env/lizhu_viz_r15/test_palette_unit.py
python .lizhu_env/lizhu_viz_r15/test_zero_regression.py
python .lizhu_env/lizhu_viz_r15/test_cli_interface.py
python .lizhu_env/lizhu_viz_r15/test_gui_e2e.py
python .lizhu_env/lizhu_viz_r15/test_static_and_docs.py   # 除 README 行号引用外全绿
# Playwright E2E（NODE_PATH 指向 npx 缓存的 playwright 包）
set NODE_PATH=%LOCALAPPDATA%\npm-cache\_npx\e41f203b7505f1fb\node_modules
node .lizhu_env/lizhu_viz_r15/e2e_viewer.js ^
  checkpoints/n3d_viz/viz_full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.html ^
  checkpoints/n3d_viz/viz_model.html
```

### 环境问题说明

- **无因环境原因被跳过的测试**：Playwright/Chromium 可用（E2E 已真实执行），Tk 可用（GUI 冒烟与端到端已真实执行），
  Node 可用（渲染器冒烟已真实执行）。
- `pytest` 未安装（`No module named pytest`），因此独立单元测试改用标准库 `unittest` 运行（31 个测试方法全部 OK）。
- 磁盘 `E:` 为 **exFAT**，不支持硬链接，导致本会话的文件写入工具（原子替换实现）在 `E:` 上不可用，
  测试脚本改用 PowerShell 落盘；这只影响我的测试资产写入方式，**不影响被测代码**。
