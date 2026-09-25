# n3d_viz 功能说明

## 项目定位

`n3d_viz` 是 N3D 二期（`n3d_sphere`）的**配套可视化工具模块**，把已训练产物中的三维拓扑渲染成可交互的三维视图。

- **零新依赖**：只使用 `torch` / `numpy` 与 Python 标准库（`tkinter`、`json`、`struct`、`threading`、`queue`）；不引入 matplotlib / plotly / pyvista 等任何绘图或 GUI 第三方库。
- **自包含**：只读取 checkpoint 的 `state_dict`，**不 import** `n3d_sphere` / `n3d_proto` 的任何代码；被可视化模块的源码与产物**零改动**。
- **真三维**：神经元分布在三维空间、流向轴 `z` 有 9 个分层取值；单张二维投影会丢掉大量结构（实测沿 z 投影把 256 个神经元压成 74 个唯一位置、736 条边只剩 255 条可分辨线段），因此产物必须可旋转缩放而非静态二维图。

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
| 神经元节点 | 全部 N 个神经元的三维位置，按 `z` 分层着色（默认规模 9 层，层尺寸 13/24/37/35/39/34/37/24/13） |
| 连接边 | E 条神经元级连接，粗细/颜色映射学到的权重绝对值 `\|edge_weight\|` |
| S_in 神经元 | `in_scope_mask` 为真的神经元（与输入层连接），独立颜色高亮 |
| S_out 神经元 | `out_scope_mask` 为真的神经元（与输出层连接），独立颜色高亮 |
| 分层参考平面 | 依据 `level_node_reach` / `level_edge_reach` 给出的 K 层，绘制层参考平面 |

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

- PLY 顶点数 == N（默认 256）；OBJ `l` 行数 == E（默认 736）
- 产物中的神经元坐标与 checkpoint 的 `neuron_pos` 逐位相等（容差 1e-6）
- 层着色组数 == K == 9，各层神经元数 == 13/24/37/35/39/34/37/24/13
- S_in 高亮数 == 193、S_out 高亮数 == 187（默认规模 seed=42 实测值）
- 边权重阈值 0.30 时保留边数 == 379 / 736（默认规模 seed=42 实测值）
- HTML 体积 < 2 MB，且字符串中不含 `syn_dist`
- HTML 中不存在外部 URL 引用（无 `http://` / `https://` / 协议相对引用）
- 传入一期产物 → 退出码非 0，错误信息包含具体缺失键名
- `n3d_sphere` / `n3d_proto` 零改动
以上断言由 `n3d_viz/verify_viz.py` 逐条执行并汇总退出码（0 = 全部通过）。实测结果：**通过 50 / 失败 0 / 跳过 0，退出码 0**（基准 `checkpoints/n3d_sphere/model.pt`，seed=42，N=256，y=8x8；运行环境 Python 3.12.10 / torch 2.14.0+cpu / numpy 2.5.3）。

补充口径与实测值（均为真实执行产物）：

- 层着色同时以两种独立口径校验：`level_node_reach` 切片得到的层规模 == 13/24/37/35/39/34/37/24/13，以及 PLY 顶点颜色分组数 == 9 且各颜色计数排序后一致。
- 坐标一致性的最大绝对偏差实测：PLY `max|diff| = 0.000e+00`（binary float32 原值写出），OBJ `max|diff| = 4.846e-10`（ascii 9 位有效数字），内联 HTML `max|diff| = 4.510e-07`（坐标保留 6 位小数），三者在 1e-6 容差内。
- 内联数据规模自洽：neurons == 256、edges == 736、layers == 9；内联阈值统计 == {0.05: 681, 0.10: 620, 0.20: 517, 0.30: 379, 0.50: 140}；内联边权与 checkpoint 的 `|edge_weight|` 最大偏差 4.991e-09。
- 产物字节数与写出报告一致（HTML / PLY / OBJ 实测 88,292 / 4,120 / 15,695 字节）；PLY 用 `newline=""` 写出以避免 Windows 换行转换导致报告值与磁盘不一致。
- 一期产物异常路径实测退出码 **3**，错误信息含全部 6 个缺失键名（`edge_src` / `edge_dst` / `edge_weight` / `in_scope_mask` / `out_scope_mask` / `level_node_reach`）；路径不存在同样退出码 3 且报错含「路径不存在」。
- 静态扫描断言：`n3d_viz` 源码中 `^\s*(from|import) n3d_(sphere|proto)` 命中 0 处；全部顶层 import 仅 `torch` / `numpy` / 标准库；`requirements.txt` 无新增条目。
- GUI 冒烟：`gui` 可导入，`withdraw()` 状态下构造 Tk 窗口成功（顶层子控件 7 个）并销毁，不进入 mainloop。
- 渲染器逻辑冒烟（`assets/viewer_smoke.js`，Node + 最小 DOM/Canvas 桩，真实执行 HTML 中内联的 `viewer.js`）：通过 11 / 失败 0，包括圆形总数 == N + S_in + S_out == 636、S_in 高亮圈数 == 193、S_out 高亮圈数 == 187、投影 bbox 有限且落在画布附近、投影质心 (450.4, 310.3) 接近画布中心 (450, 310)、阈值 0 时保留全部 736 条边、悬停命中神经元 #27 并填充详情、图例色块数 == K+3 == 12。`node` 缺失时该条标记 SKIP，不影响其它断言。
### 回归补充：开关类参数必须真的改变产物（离朱 LZ-VIZ-01 / LZ-VIZ-02 修复后的防线）

教训：`--no-plan-planes` / `--with-ply-edges` 曾出现「参数被 argparse 解析、但未接入实现」的静默空操作，而当时全部断言都只走默认路径，因此完全漏检。现在 `verify_viz.py` 增设 **[5c] 开关类参数的产物可观测差异** 一组断言（8 项），对每个布尔开关都要求「开/关两态产物不同」：

- `include_planes=True/False` 的 HTML 内容必须不同；开关状态必须写入负载 `meta.showPlanes`（`"showPlanes":true` / `"showPlanes":false`）。
- `PLY with_edges=True` 必须含 `element edge` 且声明边数 == E == 736；`with_edges=False` 必须为 0；两者产物字节必须不同。
- CLI 实测：`--no-plan-planes` 产物含 `"showPlanes":false`；`--with-ply-edges` 产物 PLY 含 736 条 `element edge`。
- 渲染器侧同步补断言：`#cb-planes` 复选框初始态必须等于 `meta.showPlanes`（`assets/viewer_smoke.js`），根因是渲染器曾硬编码 `showPlanes: true` 并反向覆盖模板意图。

### 最终实测（修复后重新全量执行）

- `python n3d_viz/verify_viz.py`：**通过 58 / 失败 0 / 跳过 0，退出码 0**（原 50 项 + 新增 8 项开关差异断言）。
- 渲染器逻辑冒烟：**通过 12 / 失败 0**（新增「复选框初始态 == meta.showPlanes」；投影质心断言改为由桩尺寸推导，不再硬编码 450/310）。
- 三件套实测大小：HTML 88,521 / PLY 4,120 / OBJ 15,695 字节（来源 `checkpoints/n3d_sphere/model.pt`，seed=42）。
- 离朱独立测试（213 项）：修复前 206 通过 / 7 失败，失败项全部归因于上述 2 个缺陷。
### 收尾轮补充（文案与临时产物纪律）

- **精度口径文案一致性**：`build_html_payload` 的 docstring 首段必须与 Args 段、实现及 README §5 一致，即「坐标 / 权重默认按 6 / 8 位小数保留（`DEFAULT_POS_DIGITS` / `DEFAULT_WEIGHT_DIGITS`），不是不截断」。任何一处再写成「保留原始双精度值」即为回归。
- **越界项计数措辞**：`_out_of_range` 在第 `_OUT_OF_RANGE_LIMIT`（= 6）项处提前返回，故报错只能写「等至少 N 项（仅列前 N 项）」，不得写「共 N 项」。
- **负例临时产物纪律**：`verify_viz.py` 的负例 checkpoint 只保留 `core.REQUIRED_KEYS` 契约键（丢弃 `syn_dist` 等巨型张量，单个约 30KB），且负例跑完立即 `unlink`。断言：`_bad_index/` 无文件残留、残留体积 == 0 字节。（清理前实测残留 4 文件共 68.9MB，清理后 0 文件 0 字节，`_verify/` 目录总体积约 70MB → 0.67MB。）
- **README §6 表格**：表头为两列「指标 / 实测值」，全表每行必须均为 2 个单元格；`|edge_weight|` 的 median 口径指引指向 §5「数值口径（唯一化）」。

### 收尾轮最终实测

- `python n3d_viz/verify_viz.py --report checkpoints/n3d_viz/_verify/verify_report.md`：**通过 77 / 失败 0 / 跳过 0，退出码 0**。项数由 75 增至 77，原因：新增 2 条「负例临时产物已清理」断言（无文件残留、残留体积 0 字节），其余断言项未变。
- 正式产物 `checkpoints/n3d_viz/viz_model.{html,ply,obj}` 的 SHA256 在验证前后**完全一致**：`15A80EBBF2FD586B…` / `9A097D16306160F9…` / `1F594ECF466E28F7…`，字节 88,521 / 4,120 / 15,695 未变。
- 一期产物退出码 3；4 个越界负例（`topo_index[5]=-1` / `topo_index[7]=999` / `edge_src[3]=-9` / `edge_dst[11]=256`）均退出码 3、报错含「下标=值」文本、无 traceback。
- `python -m compileall -q n3d_viz` 与两个 `node --check` 均退出码 0；`requirements.txt` 与 `n3d_sphere` / `n3d_proto` 零改动。
- **如实记录**：第二轮复审的 warning#1 修复曾使「渲染器逻辑冒烟」在报告中的行号由 44 顺移为 45（原因：新增 1 项 CLI `returncode` 断言），该行号变更正确且必要，README 已同步。其余 info 项（`viewer.js` 死代码、`core.py` 的 `_MissingKeys` / `_round_floats`、`gui.py` 未使用的 `Callable` 导入、`__init__.py` 的 `__all__` 含 `"gui"`、README 的 `syn_dist`「未进入内存」措辞、`_read_ply_colors` 不按 element 归属解析、验证产物目录与仓库约定差异）仍保留原样。
## 引用纪律

凡报告中出现的实测数字（E、层尺寸、S_in/S_out、参数量等）必须标注其来源产物与 `seed`；坐标与层结构由 FCC 决定与 seed 无关，而边集、突触位置随 seed 变化，故图形与边数必须标注 seed。
