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
### 几何无关化回归防线（本轮新增 80 项，汇总 77 → 157）

- **参数化口径**：既有 `EXPECTED_N=256` / `EXPECTED_E=736` / `EXPECTED_LAYER_COUNTS=[13,24,37,35,39,34,37,24,13]` **保留在「二期基准组」内作原验收不变**；新增各组一律**从 checkpoint 读实际 N / E / K / 层规模**，不写死任何值。
- **[2a] 层配色可扩展性（3 项）**：`K ∈ [1,64]` 层色去重数 == K；`K <= 9` 层色 == 既有 9 色前 K 个（逐字节不变）；HTML 层色与 PLY 层色同源。
- **[2b] 非规整几何合成组（42 项 = 5 类样本 × 8 项 + 2 项清理）**：就地构造符合 schema 但几何完全不规整的 state_dict（只保留 `REQUIRED_KEYS`），**不依赖任何既有产物、永久有效**。5 类样本：`random_cloud`（随机均匀点云 + 随机边，破除晶格/FCC 假设）、`helix`（螺旋线，破除凸包/中心对称假设）、`one_per_layer`（每层仅 1 个神经元，破除层内并行度假设）、`single_layer`（K=1，破除多层假设）、`k33`（K=33，破除 K ≤ 9 假设）。逐项断言：顶点数 == N、`l` 行数 == E、层数 == 实际 K、层色去重数 == K、PLY 顶点色去重数 == K、坐标与 `neuron_pos` 逐位一致（容差 1e-6）、payload 的 neurons/edges/layers 长度自洽、产物名由 ckpt 名派生。样本随机种子 `SYNTH_SEED = 20250925`；临时 checkpoint 与产物跑完即删（有 2 条残留断言把守）。
- **[2c] 真实异构几何组（32 项 = 5 产物 × 6 项 + 2 项）**：用 `checkpoints/n3d_shape/` 的 5 个真实非球几何产物（存在则跑，缺失则明确 SKIP 并计入报告，不静默跳过），断言泛化不变量——顶点数 == N、`l` 行数 == E、层数 == `level_node_reach.shape[0]`、层色去重数 == K、坐标与 `neuron_pos` 逐位一致、产物名由 ckpt 名派生，外加「15 个产物名两两不同」与「K=15 产物层色去重数 == 15（修复前为 9）」。**不得断言任何形状标签。**
- **[2d] 零回归锚点（3 项）**：`checkpoints/n3d_viz/viz_model.{html,ply,obj}` 的 SHA256 与几何无关化改动前逐位相同，记录在 `verify_viz.py` 的 `PHASE2_ANCHOR_SHA256`。

### 几何无关化实测（来源与 seed 可追溯）

- `python n3d_viz/verify_viz.py --report checkpoints/n3d_viz/_verify/verify_report.md`：**通过 157 / 失败 0 / 跳过 0，退出码 0**（原 77 项 + 新增 80 项）。
- 非规整几何合成组 5 类样本全部通过（N/E/K：120/200/6、96/150/8、24/40/24、40/60/1、64/90/33；坐标 max|diff| 均为 0.000e+00）。
- 真实异构几何组 5 个产物全部通过（`checkpoints/n3d_shape/*.pt`，均 seed=42）：N 均为 256，E = 736 / 713 / 679 / 705 / 717，K = 9 / 9 / 5 / 9 / 15，层色去重数 == K 全部满足。
- **K=15 层色去重数由 9 → 15**（来源 `full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt`，seed=42；层色为 `#f25c5c #f2985c #f2d45c #d4f25c #98f25c #5cf25c #5cf298 #5cf2d4 #5cd4f2 #5c98f2 #5c5cf2 #985cf2 #d45cf2 #f25cd4 #f25c98`）。
- 5 个异构产物三件套已用新代码重新渲染到 `checkpoints/n3d_viz/`（覆盖此前同名文件），HTML/PLY/OBJ 字节：sphere 88,634 / 4,177 / 15,752；cube 87,755 / 4,175 / 15,888；cylinder λ=0.5 85,507 / 4,184 / 15,341；λ=1 87,173 / 4,182 / 15,561；λ=2 88,429 / 4,182 / 15,613。六套产物名（含二期 `viz_model.*`）两两不同。
- 渲染器逻辑冒烟（Node + DOM 桩真实执行内联 `viewer.js`）：二期产物通过 12 / 失败 0；K=15 异构产物同样通过 12 / 失败 0（图例色块数 == K+3 == 18）。
- 二期正式产物 SHA256 在改动前后**逐位相同**：`15A80EBBF2FD586B…` / `9A097D16306160F9…` / `1F594ECF466E28F7…`（字节 88,521 / 4,120 / 15,695 未变）。
- `python -m compileall -q n3d_viz`、`node --check`（`viewer.js` / `viewer_smoke.js`）均退出码 0；`requirements.txt` 与 `n3d_sphere` / `n3d_proto` / `n3d_shape` 零改动（git status 干净）。
### 静态扫描约束同步扩展

`verify_viz.py` 的源码扫描规则由 `^\s*(from|import) n3d_(sphere|proto)` 扩展为
`^\s*(from|import) n3d_(sphere|proto|shape)`：`n3d_shape` 的产物在 `[2c]` 组中只作为
**输入数据**被 `torch.load` 读取，其源码同样不得被 import（与 `n3d_sphere` / `n3d_proto`
同等约束，实测 0 命中）。
### 断言项数更正（77 → 158）

上文「几何无关化回归防线（本轮新增 80 项，汇总 77 → 157）」中的项数已被**本轮修订取代**：
`[2a]` 由 3 项增至 **4 项**（新增 `[2a] hex_to_rgb 输入契约（非法抛 ValueError、分量 0..255）`，
覆盖正向 6 项、反向 15 项含 `#-12345` / `#+12345` / `#4e8c f` / `# 4e8cf` / `##4e8cff` / `###4e8cff`，
以及 `K=1..64` 全部层色分量落在 [0,255]）。**现行汇总：通过 158 / 失败 0 / 跳过 0，退出码 0**
（既有 77 项 + 本轮新增 **81** 项 = 158 项；构成 `[2a]` 4 + `[2b]` 42 + `[2c]` 32 + `[2d]` 3）。
`verify_report.md` 断言清单序号随之为：**126** = 渲染器逻辑冒烟；**89/90** = `[2b]` 合成组临时产物清理；
**152/153** = 负例临时产物清理（序号由插入顺序决定，新增断言会使其顺移，**以断言名称为准**）。

`core.hex_to_rgb` 的输入契约相应收紧为：只剥离**至多一个**前导 `#`（原 `str.lstrip("#")` 是字符集合语义，
会静默接受 `"##4e8cff"`），剥离后逐字符校验 6 位十六进制（原 `int(seg, 16)` 接受前导正负号，
`"#-12345"` 会静默返回负分量 `(-1, 35, 69)`）；非法输入一律抛 `ValueError`，返回值分量恒落在 0..255。
该函数无外部输入路径（仅由 `LEVEL_PALETTE_BASE` 常量与 `layer_palette_hex` 产物调用），故不影响任何产物。
### 修复轮（皋陶审查 1 warning + 2 info）：锚点承重化、回退分支覆盖、复现口径

**W1（warning）—— `[2d]` 零回归锚点改为承重断言**。旧版 `[2d]` 只把磁盘上已有的
`viz_model.{html,ply,obj}` 与常量比对、**不重渲**，对「代码侧的 `K <= 9` 回归」完全无感：
实测把 `core.LEVEL_PALETTE_BASE` 前两项对调（`core.py` 一行改动）后，`[2a]` 四项与
`[2d]` 磁盘比对**全部 PASS**，而同命令行重渲出的 HTML / PLY 已与锚点不同。现拆为**两层**：

- **层 1 产物完整性**（6 项）：磁盘锚点存在、SHA256 与字节数 == `PHASE2_ANCHOR_SHA256` /
  `PHASE2_ANCHOR_BYTES`；
- **层 2 代码回归（承重，6 项）**：用**当前代码**把 `ANCHOR_RERENDER_CKPT`
  （= `checkpoints/n3d_sphere/model.pt`，**相对路径**）重渲到 `_verify/_anchor_rerender/`，
  三件套的 SHA256 与字节数同样 == **同一组锚点常量**；比对后清理临时产物并断言
  「无文件残留 / 残留体积 0 字节」（2 项）。
- 层 2 固定使用相对路径常量、**不跟随**命令行 `--checkpoint`。

**拒绝证明（实测留档）**：注入 `LEVEL_PALETTE_BASE` 前两项对调后 → `[2a]` 5/5 PASS、
`[2d][磁盘]` 6/6 PASS、**`[2d][重渲]` html FAIL**（期望 `15A80EBB…`，实测 `4E147DA5…`）、
**`[2d][重渲]` ply FAIL**（期望 `9A097D16…`，实测 `B7449AB4…`）、obj PASS（无色）、
`verify_viz.py` **退出码 1**；`core.py` SHA256 `2770BFCA…` → 注入 `21B5CCBC…` → 恢复
`2770BFCA…`（**逐字节相同**），恢复后全绿。

**I1（info）—— `_hue_palette_rgb` 撞色回退分支的覆盖**。在断言覆盖的 `K ∈ [10, 64]` 区间内
纯色相扩展撞色数实测为 **0**，故「色相微移 + 明度微降 + `_PALETTE_SEARCH_LIMIT`」的回退分支
**不可达**。新增断言 `[2a] _hue_palette_rgb 撞色回退分支（必撞色构造）`：以
`COLLISION_PROBE_K = 1536`（色相间隔 1/1536，小于 8 位量化步长）主动构造必然撞色，三重把守 ——
① 前提成立（纯色相扩展撞色数 > 0，实测 **636** 个）；② 回退真的被执行（对
`_hsv_to_rgb_bytes` 挂计数钩子，实测候选调用 **2,172** 次 = K + 回退 **636** 次，无回退时应恰好
== K）；③ 结果正确（`_hue_palette_rgb` 与公开入口 `layer_palette_hex` 的去重数均 == K）。
覆盖口径同时写入 `core._hue_palette_rgb` 的 docstring（含 `ValueError` 分支的实测不可达范围）。

**I2（info）—— 字节级复现口径**。HTML 内嵌 `meta.checkpoint` 记录**调用时给出的路径字符串**，
故字节级复核只在同一调用形式下成立：交付件与锚点均用**相对路径**产出，改用绝对路径重渲会让
HTML **仅因该字段多 14 字节**（PLY / OBJ 无色无路径字段，逐字节相同）。实测对照（相对 → 绝对）：
`model.pt` 88,521 → 88,535；sphere 88,634 → 88,648；cube 87,755 → 87,769；
cylinder λ=0.5 85,507 → 85,521；λ=1 87,173 → 87,187；λ=2 88,429 → 88,443。
该口径写进 `verify_viz.py` 的 `PHASE2_ANCHOR_*` / `ANCHOR_RERENDER_CKPT` 注释与 README §2。

**项数与序号**：`verify_viz.py` 由 **158 → 170** 项（本轮 +12 = `[2a]` 撞色回退 1 项 +
`[2d]` 层 2 的 6 项 + 重渲清理 2 项 + 层 1 的字节数 3 项），退出码 0；断言清单序号随之更新为
14（撞色回退）、138（渲染器冒烟）、90/91（`[2b]` 清理）、136/137（`[2d]` 重渲清理）、
164/165（负例清理）。本轮**未改动任何产物**：`viz_model.{html,ply,obj}` 与 5 套异构交付件的
SHA256 逐位不变。
### 收尾轮（皋陶维护性 info #1）：共享渲染入口 + 参数集一致性

**问题**：`[2d][重渲]` 曾用 `core.write_outputs(..., ply_binary=True, include_planes=True, threshold=THRESHOLD)`
**手写复刻** CLI 的默认参数集。锚点的语义本是「**CLI 默认形式**的产物」，一旦 `__main__` 的默认值变化
（新增/变更布尔开关、阈值、`ply_binary`），会出现两种坏结局之一：锚点断言**无故 FAIL**（被误判为回归），
或为了让断言变绿而两边一起改、**锚点悄悄漂移成「另一套默认形式」的产物**。

**修复（同时落实建议 a 与 b）**

1. **共享渲染入口**：`core.DEFAULT_WRITE_OPTIONS` 成为「CLI 默认形式」的**唯一事实来源**
   （`threshold=DEFAULT_THRESHOLD`、`ply_binary=True`、`with_ply_edges=False`、`include_planes=True`）；
   新增 `core.render_default(ckpt_path, out_dir, out, log, *, data=None)`，自身不手写任何参数、
   只展开该常量。`__main__` 的**默认路径**（`write_options_from_args(parse_args([])) == DEFAULT_WRITE_OPTIONS`）
   改调 `core.render_default(..., data=data)`；`verify_viz` 的 `[2d][重渲]` 同样改调它。
   新增 `__main__.write_options_from_args(args)`：以 `DEFAULT_WRITE_OPTIONS` 为起点按开关覆盖，
   因此默认值只在 core 定义一次。**CLI 渲染输出逐字节不变**（重构）。
2. **`[2e]` 参数集一致性承重断言（3 项）**：
   - `argparse` 默认值映射 == `core.DEFAULT_WRITE_OPTIONS`；
   - `render_default` 实际传给 `write_outputs` 的参数集 == `core.DEFAULT_WRITE_OPTIONS`
     （白盒：临时替换 `write_outputs` 为记录实参的桩，`finally` 恢复，**不落盘**）；
   - CLI 选项表面 == 冻结清单 `CLI_OPTION_STRINGS`（15 个 option string），
     使「新增/删除 CLI 开关」必须被显式处理。

**拒绝证明（三种注入，实测留档）**：`__main__.py` SHA256 三次注入前均为
`2E015E4EEDB72CF34ABF4732A6B1D3EB80563D904F2DFBF513484CAA8EEB569C`（7,352 字节），
每次恢复后逐字节相同，全量自检回到 173/0/0：

| 注入 | 命中条目 | 结果 |
|---|---|---|
| `--no-plan-planes` 的 argparse `default` 改为 `True` | `[2e]` 序号 138 | FAIL（`include_planes: False != True`）、退出码 1；`[2a]` 5/5 与 `[2d]` 14/14 仍全 PASS |
| `--threshold` 默认值改为 `0.50` | `[2e]` 序号 138 | FAIL（`threshold: 0.5 != 0.3`）、退出码 1；`[2d]` 14/14 仍全 PASS |
| 新增 CLI 开关 `--dummy-new-flag` | `[2e]` 序号 140 | FAIL（选项表面 16 != 15）、退出码 1 |

**重构零回归证明（实测）**：用重构后的 CLI 以**相对路径**渲染 `checkpoints/n3d_sphere/model.pt`
与 5 套异构产物，共 **18 个文件**（6 ckpt × 3 件套）的 SHA256 与重构前交付件**逐字节相同**；
CLI 退出码契约不变（二期 0 / 一期 3 且列出 6/6 缺失键、无 traceback / 非法参数 2）。
本轮**未改动任何产物**。

**项数与序号**：`verify_viz.py` 由 **170 → 173** 项（`[2e]` +3），退出码 0；断言清单序号更新为
138/139/140（`[2e]`）、141（渲染器冒烟）、167/168（负例清理）；`[2a]` 5 + `[2b]` 42 + `[2c]` 32 +
`[2d]` 14 + `[2e]` 3 = 96，加既有 77 项 = 173。

## 引用纪律

凡报告中出现的实测数字（E、层尺寸、S_in/S_out、参数量等）必须标注其来源产物与 `seed`；坐标与层结构由 FCC 决定与 seed 无关，而边集、突触位置随 seed 变化，故图形与边数必须标注 seed。
