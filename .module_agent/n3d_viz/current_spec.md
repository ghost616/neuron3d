# n3d_viz 功能说明

## 项目定位

`n3d_viz` 是**任意 N3D 拓扑产物的可视化工具模块，对几何零假设**：把已训练产物中的三维拓扑渲染成可交互的三维视图。图形**完全由 `model_state_dict` 的 `neuron_pos` 决定**，不假设球 / 立方体 / 圆柱，也不假设晶格（FCC）或分层规整性；随机点云、任意曲面、非晶格、非均匀分层都走同一套渲染逻辑，模块内**不存在任何「形状类型」字段或分支**。

- **零新依赖**：只使用 `torch` / `numpy` 与 Python 标准库（`tkinter`、`json`、`struct`、`colorsys`、`threading`、`queue`）；不引入 matplotlib / plotly / pyvista 等任何绘图或 GUI 第三方库。
- **自包含**：只读取 checkpoint 的 `state_dict`，**不 import** `n3d_sphere` / `n3d_proto` / `n3d_shape` 的任何代码；被可视化模块的源码与产物**零改动**。
- **边界**：「几何零假设」不等于放弃 schema 校验——产物仍须含 `REQUIRED_KEYS` 全部 12 个拓扑键且形状符合契约；不符合者明确报错并退出**码 3**（一期 `n3d_proto` 产物行为不变），不静默降级、不臆造图形。
- **「层」的口径**：**层 = 同时计算的神经元分组**，由 `level_node_reach` 给出的 `topo_index` 半开区间定义；层参考平面取该层神经元沿流向轴位置的**均值**（`layer_z = mean(zs)`），该平面必然包含这一组神经元，与几何是否规整无关。不做最小二乘拟合平面，也不做「层结构退化」特判。
- **真三维**：神经元分布在三维空间、流向轴 `z` 有多个分层取值；单张二维投影会丢掉大量结构（默认规模实测沿 z 投影把 256 个神经元压成 74 个唯一位置、736 条边只剩 255 条可分辨线段），因此产物必须可旋转缩放而非静态二维图。
## 数据来源与契约

唯一输入是一个二期训练产物 `.pt`，其 `model_state_dict` 必须包含：`neuron_pos [N,3]`、`edge_src [E]`、`edge_dst [E]`、`edge_weight [E]`、`edge_dist [E]`、`input_syn_pos [N*y_in,3]`、`output_syn_pos [N*y_out,3]`、`in_scope_mask [N]`、`out_scope_mask [N]`、`topo_index [N]`、`level_node_reach [K,2]`、`level_edge_reach [K,2]`、`in_degree [N]`、`out_degree [N]`、`neuron_bias [N]`。

- **一期产物不可用**：`n3d_proto` 产物缺少 `edge_src/edge_dst/edge_weight/in_scope_mask/out_scope_mask/level_node_reach`（只有突触级 `edge_index`），模块必须**明确报错并退出非 0**，不得静默降级或臆造图形。
- `syn_dist [N*y_out, N*y_in]` 体积巨大（默认规模约 16.8 MB），**不得嵌入 HTML**。
### 索引取值域契约（抽取阶段强制）

除长度校验外，`extract_topology` 还强制 `topo_index` / `edge_src` / `edge_dst` 的全部元素落在 `[0, N)`，违规抛 `CheckpointSchemaError`，错误信息附**越界值与其下标**（如 `topo_index[5]=-1`、`edge_dst[11]=256`）。

两条真实风险（均已修复并加断言）：

- `topo_index` 含**负值**时 `pos[i]` 会按 Python 负索引**静默**取到错误神经元，使 `layer_groups` / `layer_z` 静默错误——图是错的却不报错，比崩溃更危险；
- 含**越界值**时抛 `IndexError` 而非 `CheckpointSchemaError`，而 CLI 只捕获 `CheckpointError`，导致损坏产物以 traceback 崩溃、退出码 1，破坏「非二期产物给可读错误」的契约。

现在两类的实测行为均为：退出码 **3**、错误可读且含越界值与下标、无 traceback、不产生任何产物。回归断言见 `verify_viz.py` 的 `[6b]` 组（4 个负例 × 4 项断言）。

### 数值口径（唯一化）

- **`|edge_weight|` 的 median**：全模块统一为「升序排列后取上中位」`abs_w[n // 2]`（`E=736` 时为 `sort(|w|)[368]`），实测 **0.307065486907959**。不取两中位均值（`0.30649110674858093`），也不使用 `torch.median`（返回下中位 `0.3059167265892029`）；`TopologyData.weight_extremes()` 与 README 使用同一口径。
- **内联 HTML 浮点保留位数**：`build_html_payload` 的 `pos_digits=None` / `weight_digits=None` 表示**取默认位宽**（`DEFAULT_POS_DIGITS = 6` 位小数、`DEFAULT_WEIGHT_DIGITS = 8` 位小数），**不是**「不截断」。该语义与模块内私有工具 `_round_floats(values, digits=None)`（None = 不截断）不同，两者 docstring 均已写明。实测默认位宽下内联坐标与 `neuron_pos` 的 `max|diff| = 4.510e-07 < 1e-6`，内联 `|edge_weight|` 的 `max|diff| = 4.991e-09`。
## 展示元素

| 元素 | 说明 |
|---|---|
| 神经元节点 | 全部 N 个神经元的三维位置，按**所在层**着色（K 由 `level_node_reach` 决定、可任意大；默认规模实测 K=9，层尺寸 13/24/37/35/39/34/37/24/13） |
| 层配色 | K 个层色两两不同：`K <= 9` 取既有 9 色表前 K 个（逐字节不变），`K > 9` 按均匀色相扩展（标准库 `colorsys`）；HTML 与 PLY **同源**（`core.layer_palette_hex` / `core.layer_palette_rgb`），模块内不存在第二份色表 |
| 连接边 | E 条神经元级连接，粗细/颜色映射学到的权重绝对值 `\|edge_weight\|` |
| S_in 神经元 | `in_scope_mask` 为真的神经元（与输入层连接），独立颜色高亮 |
| S_out 神经元 | `out_scope_mask` 为真的神经元（与输出层连接），独立颜色高亮 |
| 分层参考平面 | 依据 `level_node_reach` / `level_edge_reach` 给出的 K 层，绘制层参考平面（平面高度 = 该层神经元位置均值） |

### 层配色随 K 可扩展（几何无关化的核心修复）

`core.py` 与 `export_geometry.py` 曾各自持有 **9 色 + `k % 9` 循环**的色表：实测 `checkpoints/n3d_shape/full_shapecylinder_a2_*.pt`（seed=42，非均匀分层，K=15）的产物**只有 9 种唯一层色，6 个层与前面的层重复**，分层着色失去可分辨性。修复为共用色板函数（唯一实现位于 `core.py`）：

- `K <= 9`：`LEVEL_PALETTE_BASE` 的前 K 个，输出**逐字节不变**（现有产物含二期 `viz_model.*` 为零回归锚点）；
- `K > 9`：按均匀色相生成 K 个颜色，**去重数恒等于 K**（`K ∈ [1, 64]` 已断言）。

## 交付产物

一条命令（或一次窗口操作）产生三个文件，默认写入 `checkpoints/n3d_viz/`：

1. `viz_<ckpt名>.html` —— 自包含交互式三维视图（数据与渲染器全部内联，**零外部 URL 引用**，断网可打开）
2. `viz_<ckpt名>.ply` —— 点云（神经元位置）
3. `viz_<ckpt名>.obj` —— 线框（神经元之间的连接，用 `l` 行表示）

**产物命名**：由 checkpoint 文件名派生，天然不撞名；不得使用固定名互相覆盖（对齐历史教训：同名覆盖导致取证失效）。

## 交互能力

- 旋转 / 缩放 / 平移（鼠标拖拽、滚轮、右键）
- 悬停神经元显示详情（id、所在层、入度、出度、是否 S_in / S_out）
- 图层开关（神经元 / 连接 / 层平面分别显示隐藏）
- 边权重阈值过滤（只显示 `|edge_weight|` 不小于阈值的边）

## 入口形式

- **GUI**：tkinter 窗口，含 `.pt` 文件选择（浏览按钮）、输出文件夹选择（浏览按钮）、开始按钮与状态提示、日志区、一键打开产物或输出文件夹。长任务在后台线程执行，界面不得卡死。**不做拖拽**（拖拽需 `tkinterdnd2`，与零依赖冲突）。
- **CLI**：`python -m n3d_viz --checkpoint <pt> --out-dir <dir> [...]`；不带参数时启动 GUI。
- 核心绘图逻辑必须与 GUI 解耦：`core.py` 等纯逻辑层可被脚本直接调用与断言，`gui.py` 只做薄封装。

## 验收标准（硬断言）


### 全连接层支持轮 · 首轮实测记录（含后续两轮修复）

> **行文口径提示**：本节是最初一轮的留档，其中「新增 FC 交付件」的 SHA256 与字节数是**面板尺度
> 调整之前**的产物（587,796 / 81,681 / 153,748、`C2F15D3D…` / `5FFF5F6F…` / `E1F6397A…`）。
> **现行权威口径见「输入/输出全连接层的三维展示」章节的实测值表**（591,709 / 81,681 / 154,372、
> `34CDDB92…` / `25C1008C…` / `4E0C2A2F…`，采集时 `core.py` = `3296accb3130e41f`）。

`python n3d_viz/verify_viz.py --report checkpoints/n3d_viz/_verify/verify_report.md`：
**通过 490 / 失败 0 / 跳过 1，退出码 0**。构成：既有 173 项 + `[2c]` 扩展 **+229**（32 → 261）+
新增 `[2f]` **88** 项（含离朱 R22 修复轮补入的 5 项契约回归防线）= 490。
唯一 SKIP 是 `[2c]` 的「非 N3D 拓扑产物已明确跳过（逐个计入报告）」说明行。

- **`[2c]` 为何扩展 229 项**：`checkpoints/n3d_shape/` 本轮新增了 MLP 基线（18 个，`config.arch == "mlp"`，
  只有 4 个键、无 N3D 拓扑）与 10 个 `fc_align` 产物。旧口径会成片误报（实测一次运行 122 项 FAIL，
  其中 108 项是 `KeyError: 'level_node_reach'`）。现改为 `_non_topology_reason` **预分拣**：
  非拓扑产物**明确 SKIP 并逐个计入报告**（不静默跳过），拓扑产物跑完整泛化不变量；
  「OBJ 核心边行数 == E」改为**按 group 计数**，另加「OBJ 总 l 行数 == E + FC 抽样条数」与
  （FC 产物）「面板点数 == 2×H+2」两条不变量。实测分类：拓扑 **9** + 非拓扑 **18** == **27** 个 `.pt`。
- **渲染器逻辑冒烟**：无 FC 产物 **13/13**，有 FC 产物 **20/20**。
- `python -m compileall -q n3d_viz` 与三个 `node --check`（`viewer.js` / `viewer_fc.js` / `viewer_smoke.js`）
  均退出码 0。
- **零回归（实测）**：`checkpoints/n3d_viz/viz_model.{html,ply,obj}` SHA256 仍为
  `15A80EBB…` / `9A097D16…` / `1F594ECF…`；生成 FC 交付件前后，目录下既有 **18 个** `viz_*` 交付件的
  SHA256 **逐位不变**（改动/删除行数 0）。
- 一期产物 `checkpoints/n3d_model_full.pt` 仍退出码 **3**、报错列出 10 个缺失键、**不产生任何产物**。
- `requirements.txt` 与 `n3d_sphere` / `n3d_proto` / `n3d_shape` **零改动**（git status 干净）。
- **本轮修复的自检缺陷（留档）**：① `_read_ply_colors` 与 `parse_ply_vertices` 原本不按
  element 归属解析，FC 产物在 `vertex` 之后多出 `fc_node` / `fc_edge` 会让顶点记录步长算错
  → 已改为按 `element` 归属收集 `property`（实测修复前会把顶点色与坐标解析成垃圾）；
  ② `viewer_smoke.js` 的 DOM 桩三处「比实现宽松」导致假绿灯 —— `getElementById` 会顺手创建
  不存在的元素（真实 DOM 返回 `null`，使渲染器的存在性判断永远为假、被守卫的代码块被静默跳过，
  实测 FC 统计面板因此从未创建而其他断言仍全绿）、`createElement` 返回共享元素（id 赋值后被覆盖）、
  缺 `document.body` / `document.createTextNode`；③ FC 的 `requestAnimationFrame` 刷新循环在
  「同步 RAF」桩下会同栈无限递归（实测 `RangeError` / 堆内存耗尽）→ 已加深度防护；
  ④ PLY `fc_node` 元素含 `2×H` 面板单元 **加 2 个边界块中心**，断言口径为 `2×H + 2`。
- **修复轮的实际改动范围（离朱 R23 复核）**：`core.py` 的 FC 几何/抽取修复 diff 仅 **8 个 hunk**，
  全在 `validate_fc_top_k` / `_fc_panel_geometry` 的入口校验与 docstring，**几何计算与写出路径一行未改**；
  离朱以修复前（`aae9472e562f6087`）与修复后（`3296accb3130e41f`）两份 `core.py` 分别渲染并与磁盘
  交付件三方比对，结果**三者完全相同**。

### 历史留档：回归补充 · 开关类参数必须真的改变产物（离朱 LZ-VIZ-01 / LZ-VIZ-02）

教训：`--no-plan-planes` / `--with-ply-edges` 曾出现「参数被 argparse 解析、但未接入实现」的
静默空操作，而当时全部断言都只走默认路径，因此完全漏检。`verify_viz.py` 因此增设
**[5c] 开关类参数的产物可观测差异**（8 项），对每个布尔开关都要求「开/关两态产物不同」：

- `include_planes=True/False` 的 HTML 内容必须不同；开关状态必须写入负载 `meta.showPlanes`；
- `PLY with_edges=True` 必须含 `element edge` 且声明边数 == E；`with_edges=False` 必须为 0；两者字节不同；
- CLI 实测：`--no-plan-planes` 产物含 `"showPlanes":false`；`--with-ply-edges` 产物 PLY 含 E 条 `element edge`；
- 渲染器侧同步断言：`#cb-planes` 复选框初始态必须等于 `meta.showPlanes`（根因是渲染器曾硬编码
  `showPlanes: true` 并反向覆盖模板意图）。

本轮 FC 的 `--fc-top-k` 沿用同一防线：`[2f]` 组断言 **k=1 与 k=8 的 HTML 与 OBJ 都必须不同**
（开关类参数不得是空操作）。

### 历史留档：零回归锚点的两层设计与参数集一致性

- **`[2d]` 两层断言**：层 1 = 磁盘锚点完整性（存在 + SHA256 + 字节数，6 项）；层 2 = 用**当前代码**
  经共享入口 `core.render_default` 把 `ANCHOR_RERENDER_CKPT`（相对路径，**不跟随**命令行
  `--checkpoint`）重渲到 `_verify/_anchor_rerender/`，与**同一组锚点常量**比对（6 项）+ 清理（2 项）。
  拒绝证明（实测留档）：对调 `core.LEVEL_PALETTE_BASE` 前两项后，层 2 的 html / ply 两条 FAIL、
  退出码 1，而 `[2a]` 5/5 与层 1 6/6 仍 PASS —— 证明该类代码回归只能由层 2 拦住。
- **`[2e]` 参数集一致性（3 项）**：`argparse` 默认值映射 == `core.DEFAULT_WRITE_OPTIONS`
  （现行含 `fc_top_k=3`）；白盒捕获 `render_default` 传给 `write_outputs` 的实参 == 该常量；
  CLI 选项表面 == 冻结清单 `CLI_OPTION_STRINGS`（现行 **17** 个 option string）。
  三种注入拒绝证明（实测留档）：`--no-plan-planes` 默认改 True、`--threshold` 默认改 0.5、
  新增一个 CLI 开关，分别使对应条目 FAIL 且退出码 1，恢复后源文件 SHA256 逐字节相同。
- **字节级复现口径**：HTML 内嵌 `meta.checkpoint` 记录**调用时给出的路径字符串**，故字节级复核
  只在同一调用形式下成立；锚点与交付件均用**相对仓库根**的路径产出。实测相对 → 绝对路径会让
  HTML 仅因该字段多 14 字节（PLY / OBJ 无色无路径字段，仍逐字节相同）。
- **静态扫描约束**：`verify_viz.py` 扫描 `^\s*(from|import) n3d_(sphere|proto|shape)`，实测 **0 命中**；
  顶层 import 仅 `torch` / `numpy` / 标准库；`requirements.txt` 无新增条目。
- **索引取值域契约**：`topo_index` / `edge_src` / `edge_dst` 越界（含负值）一律抛
  `CheckpointSchemaError` 并附「下标=值」，CLI 退出码 3、无 traceback、不产生任何产物；
  报错最多列 `_OUT_OF_RANGE_LIMIT`(=6) 项，措辞为「等至少 N 项（仅列前 N 项）」。
- **数值口径（唯一化）**：`|edge_weight|` 的 median 统一为「升序后取上中位」`abs_w[n // 2]`
  （默认规模实测 0.307065486907959）；内联 HTML 的 `pos_digits` / `weight_digits` 的 `None`
  表示**取默认位宽**（6 / 8 位小数），不是不截断。

### 历史留档：异常路径（必须报错，不得静默降级）

一期产物 `checkpoints/n3d_model_full.pt`（`n3d_proto`）缺 10 个二期拓扑键，实测退出码 **3**、
报错可读并列出全部缺失键名、**不产生任何产物**；路径不存在同样退出码 3 且报错含「路径不存在」。
本轮新增的 `fc_dim != 0` 但缺 FC 键走同一口径（退出码 3、无 traceback、零产物）。

## 引用纪律

凡报告中出现的实测数字（E、层尺寸、S_in/S_out、参数量等）必须标注其来源产物与 `seed`；坐标与层结构由 FCC 决定与 seed 无关，而边集、突触位置随 seed 变化，故图形与边数必须标注 seed。
## 输入/输出全连接层的三维展示

`n3d_shape` 第三轮引入的 `fc_dim` 把 N3D 核心**夹在两个全连接层之间**，本模块据此额外展示
「全连接输入层 / 投影到 S_in / 从 S_out 收集 / 线性输出」这套结构。

> **本节的实测数字均带采集时的源码版本标记**（离朱 R23 的 P4 建议）。同一指标在开发过程中
> 因面板尺度调整而变过两次，凡改动 `core.py` 的 FC 几何/抽取路径后，**必须重采本节所有数字并更新标记**，
> 否则会出现「文档自相矛盾」的漂移。本节标记：**`core.py` = `3296accb3130e41f`（离朱 R23 复核快照）**。

### 触发判定（三态，**只看 `config.fc_dim != 0`**）

| 产物状态 | 判定 | 行为 |
|---|---|---|
| `config` 无 `fc_dim` 键 | 无 FC | 走既有展示，**不报错** |
| `config.fc_dim == 0` | 无 FC | 走既有展示，**不报错** |
| `config.fc_dim != 0` 且 FC 键齐全 | **有 FC** | 展示 FC 层 |
| `config.fc_dim != 0` 但缺 FC 键 | **产物损坏/不完整** | **报错退出非 0**，不静默降级为无 FC 展示 |

实测（2026-09-27）：二期 `checkpoints/n3d_sphere/model.pt` 与三期未启用产物
`checkpoints/n3d_shape/full_shapesphere_N256_..._s42.pt` 的 `config` 里**都没有** `fc_dim` 键；
`*_fc_align.pt` 有 `fc_dim = -1`（**表示宽度跟随 N，是启用而非关闭**），
`*_fc_align_mlp.pt` 有 `fc_dim = 0`（MLP 基线，只有 4 个键、无 N3D 拓扑）。
判定**只用 `fc_dim != 0`**，不依赖「键是否存在」来启用——否则「产物损坏」会被误判成「无 FC」。

FC 键完整性由 `core.FC_REQUIRED_KEYS` 定义：`proj_weight` / `fc_out_weight` /
`fc_in_weight` / `fc_out_bias`，缺任一即 `CheckpointSchemaError`（CLI 退出码 3、无产物）。
有效宽度 `H` 由 FC 张量的**实际形状**推出（`proj_weight` 列数 == `fc_out_weight` 行数 ==
`fc_in_weight` 行数 == `config.fc_width`，四者不自洽即报错）；**不能**假设
`config.hidden_dim` 等于 `H`——实测该 FC 产物 `hidden_dim = 2048` 而 `H = 825`。

### 几何与不重叠判据（硬断言）

```
[输入边界块 784] ──▶ [H 单元面板·输入侧] ──(抽样连线)──▶ S_in 神经元
                                                       │ N3D 核心（既有展示不变）
[输出边界块 10] ◀── [H 单元面板·输出侧] ◀──(抽样连线)── S_out 神经元
```

- 两片面板**垂直于流向轴**（`config.flow_axis`），分别置于云流向轴跨度两端外侧；面板内按
  `ceil(sqrt(H))` 列做网格排布（`H=825` → `29×29`）；
- 面板在平面内的跨度 = 云在对应轴上的跨度 × `FC_PANEL_SPAN_RATIO`(1.0)，即面板与云**同尺度**；
- 间隙 = 云跨度 × `FC_PANEL_GAP_RATIO`(0.15)；面板流向轴厚度 = 单元中心间距 ×
  `FC_PANEL_THICKNESS_RATIO`(0.20)，夹在
  `[FC_PANEL_MIN_THICKNESS, 云跨度 × FC_PANEL_MAX_THICKNESS_RATIO(0.10)]` 之间；
- **不重叠硬断言（构造期执行）**：`panel_in.flow_interval[1] < cloud_lo` 且
  `cloud_hi < panel_out.flow_interval[0]`，即「面板节点的流向轴坐标区间 ∩ 神经元云流向轴区间 = ∅」。
  厚度下界的作用是让「不相交」不退化成「不接触」；
- **入口校验先于任何算术**：`H` 必须是正整数（且非 `bool`），否则抛可读
  `CheckpointSchemaError`（原实现会先算 `max(cols, rows)` / `math.sqrt(H)`，导致 `H=0` 抛裸
  `ZeroDivisionError`、`H=-1` 抛裸 `math domain error`）；流向轴非法、云流向轴跨度为 0 同为
  `CheckpointSchemaError`。**合法边界不得误报**：`H=1` 与「两轴跨度均为 0 的共线点云」正常构造；
- 边界块（输入 784 / 输出 10，块中心 + 三轴尺寸）置于面板外侧，以聚合箭头与面板块相连；
- 面板单元着色按该单元的权重范数映射（输入侧 `fc_in_weight` 行范数、输出侧 `fc_out_weight` 行范数），
  配色函数**复用唯一实现** `export_geometry.weight_color_rgb`，`assets/viewer_fc.js` 的 `normToRgb`
  是同式 JS 镜像；层色仍由 `core.layer_palette_hex` / `layer_palette_rgb` 提供，**不存在第二份色表**。

### 抽样口径（必须显式声明「非全部连接」）

- **输入侧**：对每个 `S_in` 神经元取 `proj_weight` 该**行**内 `|w|` 最大的 top-k；
- **输出侧**：对每个 `S_out` 神经元取 `fc_out_weight` 该**列**内 `|w|` 最大的 top-k；
- `k` 由 CLI `--fc-top-k` / GUI「全连接层抽样 k」控制，范围 `[1, 8]`（`MIN_FC_TOP_K` /
  `MAX_FC_TOP_K`），默认 `DEFAULT_FC_TOP_K = 3`；
- **`validate_fc_top_k` 要求 `int` 且非 `bool`**：`True`/`False`、`float`（**含 `3.0`**）、
  `str`、容器、`None`、越界值一律抛**可读 `ValueError`**（**绝不抛裸 `TypeError`**、
  **绝不静默截断或强转**；CLI 退出码 2）。原实现 `int(k)` 会静默接受 `2.5→2` / `True→1`；
- 抽样实现 `_sample_top_k` 用 `heapq.nlargest` 且排序键为 `(|w|, -下标)` 后按下标升序，
  **完全确定、无随机性**，产物可逐字节复现；top-k 相对全局阈值的关键优势是
  **保证每个 S_in / S_out 神经元都至少有一条连线**（实测 `|w| >= 0.30` 仅保留 1,308 条、
  `>= 0.20` 跳到 12,688 条，阈值极敏感且会静默丢掉整个神经元）；
- **声明文本**（`core.FC_NOT_ALL_CONNECTIONS_TEXT` + 两侧参数量，由 `FcData.declared_statement()` 给出）
  必须同时写进 **HTML 负载 `meta.fcDeclaration`**、**PLY 头部注释**、**OBJ 伴随注释**与 README。

**口径区分（两组，引用时必须写清）**：

1. **「区间」vs「中心」**：面板在流向轴上的位置有 `panels[].flow`（**中心**，单点）与
   `panels[].flow_interval`（**区间** = `[中心 − 厚度/2, 中心 + 厚度/2]`）两种口径；
   不重叠判据用**区间**。实测中心 `∓1.2869`、区间 `[-1.2938, -1.2801]` / `[1.2801, 1.2938]`，
   两者不可混引（曾出现把中心当区间端点核对的情况）。
2. **「参数量」vs「抽样条数」**：`projCount` / `fcOutCount` 是矩阵**全部**元素数
   （480,150 + 485,100 = 965,250），`sampleEdges` 是**渲染出来的**抽样条数（3,510）；
   产物 meta、CLI 日志、OBJ 注释与文档均**分别标注、不得混用**。

### 产物差异（有 FC vs 无 FC）

| 产物 | 无 FC | 有 FC |
|---|---|---|
| HTML | 既有形态，**逐字节不变** | 追加内联 FC 叠加渲染器 + 负载含 `"fc"` 段与 FC meta 字段（只在有 FC 时新增键，无 FC 时**不出现** `hasFc`/`fcTopK` 等键） |
| PLY | 既有形态，**逐字节不变** | `vertex` 后追加 `fc_node`（`2×H+2` 点，`uchar kind`：0=输入面板/1=输出面板/2=输入边界块/3=输出边界块）与 `fc_edge`（抽样连线）两个元素 |
| OBJ | 既有形态，**逐字节不变** | 追加 `2×H+2` 个 `v` 行 + 抽样 `l` 行，用 `g`/`o` 分组区分 `n3d_viz_core_edges` 与 `n3d_viz_fc_sampled_edges`，附抽样口径注释 |

**零回归是硬约束**：无 FC 产物三件套必须逐字节不变。由于 `assets/viewer.html` 与
`assets/viewer.js` 都被**逐字内联**进每一份 HTML，改动它们任何一个字节都会破坏该约束，
因此 FC 渲染被完整隔离在**独立的** `assets/viewer_fc.js` 中，`core.build_html` 只在
`data.fc` 非 None 时把它作为**追加的一段脚本块**内联进「内联数据」块（从而在 DOM 中排在
`viewer.js` 之前执行）。该叠加渲染器与基础渲染器共用同一套相机口径，画在
`pointer-events:none` 的透明叠加 canvas 上，3D 交互（旋转/缩放/平移/悬停）与既有元素完全一致；
自动取景把面板与边界块一并纳入半径（否则 FC 元素落到视野之外，表现为「开关打开但什么都看不见」）。

### 有 FC 产物的实测值（来源与 seed 可追溯）

来源 `checkpoints/n3d_shape/full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.pt`
（**seed=42**，`config.fc_dim = -1`，`test_acc = 0.9853`）；采集时源码 **`core.py` = `3296accb3130e41f`**。

| 指标 | 实测值 |
|---|---|
| `N` / `E` / `K` | 825 / 2,588 / 15 |
| `\|S_in\|` / `\|S_out\|` | 582 / 588 |
| `fc_dim` / `H`（=`fc_width`） | -1 / **825**（注意 `config.hidden_dim = 2048`，**不等于** `H`） |
| `fc_in_weight` 形状 / 参数量 | `[825, 784]` / 646,800 |
| `proj_weight` 形状 / 参数量 | `[582, 825]` / **480,150** |
| `fc_out_weight` 形状 / 参数量 | `[825, 588]` / **485,100** |
| k=3 抽样条数 | 1,746 + 1,764 = **3,510** |
| k=1 / k=8 抽样条数 | 1,170 / 9,360 |
| 面板点数 | `2×825 = 1,650`（PLY `fc_node` 另含 2 个边界块中心 → **1,652 = 2×H+2**） |
| 面板网格 | 29 × 29（`ceil(sqrt(825)) = 29`），单元中心间距 **0.06827** |
| 面板横向跨度 | ±**0.9558** |
| 云流向轴（z）**区间** | `[-0.9899, 0.9899]`（跨度 1.9799） |
| 输入面板流向轴**区间** / **中心** | `[-1.2938, -1.2801]` / `-1.2869`（厚度 **0.0136545**；与云区间**不相交**） |
| 输出面板流向轴**区间** / **中心** | `[1.2801, 1.2938]` / `1.2869`（与云区间**不相交**） |
| 边界块中心 z / 尺寸 | `∓1.5511` / `1.1879 × 1.1879 × 0.2772` |
| 三件套字节数 | HTML **591,709** / PLY **81,681** / OBJ **154,372** |
| 三件套 SHA256 | HTML `34CDDB92…` / PLY `25C1008C…` / OBJ `4E0C2A2F…` |

### 产物与交付件

FC 产物的三件套已生成到
`checkpoints/n3d_viz/viz_full_shapesphere_N825_..._fc-1_s42_fc_align.{html,ply,obj}`，
SHA256 为 `34CDDB92…` / `25C1008C…` / `4E0C2A2F…`（与上表同源，采集时 `core.py` = `3296accb3130e41f`）。
生成与后续两轮修复（浏览器可视性修复、契约缺陷修复）前后，`checkpoints/n3d_viz/` 下既有
**18 个** `viz_*` 交付件的 SHA256 **逐位不变**（含二期锚点 `15A80EBB…` / `9A097D16…` / `1F594ECF…`），
新增的 3 个 FC 文件之外无任何覆盖。离朱 R23 另以**修复前** `core.py`（`aae9472e562f6087`）与
**修复后**（`3296accb3130e41f`）分别渲染并与磁盘交付件三方比对，结果**三者完全相同**，
证明契约修复未触碰几何与写出路径。

### 真实浏览器视觉验证与 4 个「桩测不出来」的缺陷（实测留档）

`checkpoints/n3d_viz/_verify/viewer_fc_screenshot.png`（1440×900，547,213 字节）是
Chromium（`bun x playwright screenshot`，一次性图形验证工具、未引入任何依赖）打开 FC 产物
得到的证据：两片 29×29 面板（色彩 = 单元权重范数）、输入 784 / 输出 10 边界块（线框方盒 +
标签）、面板到 S_in / S_out 的抽样连线、独立 FC 统计面板、含 FC 条目的图例与
「全连接层（抽样）」开关。

**该验证暴露了 4 个 Node/DOM 桩无法发现的缺陷**（桩只验证「函数被调用」，不验证「画到了
可见区域」）：

| # | 缺陷 | 症状 | 根因与修复 |
|---|---|---|---|
| 1 | 叠加层 `z-index` 不够 | 改动 FC 绘制颜色后**截图字节数完全不变** → FC 一个像素都看不见 | `#view` 是 `position:absolute; inset:0` 且 z-index 为 auto，叠加层 `z-index:5` 仍排在它之后。改为 `z-index:20` |
| 2 | `draw()` 里的**正反馈环** | 面板被缩到几乎不可见（叠加层 `cam.dist` 由正确的 6.7565 涨到 12.2015 = 滚轮 clamp 上限 `radius*40`） | 原实现每帧「用上一帧 dist 反推用户缩放倍率」：第 1 帧归一到 `radiusWithFc*3.6`，第 2 帧又把它当已缩放距离再乘 `radiusWithFc/radius`，逐帧放大。改为自带 `userZoom` 倍率（默认 1），滚轮/重置按钮经捕获阶段事件同步 |
| 3 | 面板相对云太小 | 面板只占云横向约 40%，读不出「两端包裹」 | 新增 `FC_PANEL_SPAN_RATIO = 1.0`：单元间距改为「面板目标跨度 / 网格边长」，面板与云同尺度 |
| 4 | 抽样连线密度压过面板 | alpha 0.95 时输出侧面板糊成一片青蓝雾 | 连线降为 `rgba(150,240,255,0.16)`、线宽 0.6px、虚线间隔拉大；面板单元改不透明填充 + 亮边框并按 0.88 收缩留缝 |

第 1、2 条是**功能性缺陷**（FC 结构在真实浏览器里根本看不见），由「截图字节数异常不变」
与「页面内注入诊断盒读回叠加层 `cam.dist`」两条独立手段定位。

### 独立测试（离朱）暴露的 3 类契约缺陷与修复

离朱 R22 以 **731 项独立断言**复核后报告 **3 类真实缺陷**（均属「文档/规范声明了明确报错，
实际是未包装异常或静默截断」，**无功能性错误、无产物错误**）：

| 优先级 | 缺陷 | 实测症状 | 修复 |
|---|---|---|---|
| P1 | `core.validate_fc_top_k` 对**非整数浮点静默截断** | 原实现 `value = int(k)`，`2.5→2`、`1.5→1`、`8.7→8`、`3.0→3`、`True→1` 全部被**接受**，与「非整数一律报错、不静默截断」相悖。CLI 不受影响（argparse `type=int` 已在入口拦下），受影响的是 Python API 调用方 | 改为**严格要求 `int` 且非 `bool`**：`bool` / `float` / `str` / 容器一律抛可读 `ValueError`（含 `3.0` 这类「值合法但类型是 float」的输入，不静默强转） |
| P2 | 同函数 `validate_fc_top_k(None)` 抛**裸 `TypeError`** | docstring 只声明 `ValueError`，按契约只捕获 `ValueError` 的调用方（CLI / GUI）会**漏接** | `None` 与其它非法类型统一抛 `ValueError` |
| P3 | `core._fc_panel_geometry(H<=0)` **逃逸未包装异常** | 原实现在 `_place_units_in_panel` 的 `cell <= 0` 哨兵**之前**先算 `max(cols, rows)` / `math.sqrt(H)`，于是 `H=0` 抛裸 `ZeroDivisionError`、`H=-1` 抛裸 `ValueError: math domain error`。**主链路不受影响**（`extract_fc` 经 `_fc_hidden_width` 已用 `H<=0` 兜住），仅直接调用该几何函数时违约 | 把「`H` 为正整数」「流向轴合法」等校验**提到任何算术之前**，与既有的「云跨度为 0 / 面板相交」统一为 `CheckpointSchemaError` |

**回归防线（5 项，`[2f]` 组）**：

- `core.validate_fc_top_k 拒绝非整数浮点与 bool（不静默截断）` —— 13 类非法输入全部被拒；
- `core.validate_fc_top_k 越界与非法类型都抛 ValueError（非裸异常）` —— 异常类型集合恒为 `{"ValueError"}`；
- `FC 几何入口 H<=0 / 类型非法 -> 可读 CheckpointSchemaError` —— `H=0` / `H=-1` / `H=True`；
- `FC 几何入口 流向轴非法 / 云跨度为 0 -> 可读 CheckpointSchemaError`；
- `FC 几何入口 H=1 与共线云（两轴跨度均为 0）仍可用` —— **合法边界不得误报**（走兜底间距）。

**修复后实测**：`verify_viz.py` **通过 490 / 失败 0 / 跳过 1**，退出码 0
（`[2f]` 由 83 → **88** 项）；渲染器冒烟无 FC 13/13、有 FC 20/20；
无 FC 零回归成立（二期锚点与三期产物均逐字节不变）；FC 三件套**逐字节不变**。

**离朱 R23 复验结论**：**3 类缺陷全部修复到位、全量回归零破坏、FC 产物逐字节不变、无新的
功能/产物缺陷**。独立复验 132/132、E2E 40/40、`verify_viz.py` 490/0/1、编译与静态检查全 0，
合计 669 项断言 0 失败（唯一 1 项为 `[2c]` 对照 MLP 基线的明确跳过）。修复 diff 仅 8 个 hunk，
全在 `validate_fc_top_k` / `_fc_panel_geometry` 的入口校验与 docstring，**几何计算与写出路径一行未改**。
R23 唯一未通过项为 **纯文档漂移（P4）**：本节曾残留面板尺度调整前的旧值
（区间 `[-1.2944,-1.2794]`、字节 `587,796/81,681/153,748`、旧 SHA256）——已在本轮刷新为
上表数值并加注源码版本标记。
