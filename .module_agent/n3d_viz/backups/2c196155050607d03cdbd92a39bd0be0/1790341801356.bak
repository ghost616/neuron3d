# n3d_viz —— N3D 二期拓扑三维可视化工具

把已训练 checkpoint 中的**神经元位置**与**神经元级连接**渲染成三件套：
**自包含交互式三维 HTML** + **点云 PLY** + **线框 OBJ**。

- **零新依赖**：只用 `torch` / `numpy` 与 Python 标准库（`tkinter` 属标准库）。没有 matplotlib / plotly / pyvista / tkinterdnd2，`requirements.txt` 无任何改动。
- **自包含**：只读取 checkpoint 文件的 `state_dict`，**不 import** `n3d_sphere` / `n3d_proto`；这两个模块的源码与产物零改动。
- **单实现**：GUI 与 CLI 共用 `core` 层同一套逻辑，模块内不存在第二份绘图实现。
- **产物不撞名**：文件名由 checkpoint 名派生（`viz_<ckpt名>.html/.ply/.obj`），同名已存在时明确提示，不静默覆盖。

---

## 1. 用法

### 1.1 命令行（CLI）

```bash
# 一条命令产出三件套，默认写入 checkpoints/n3d_viz/
python -m n3d_viz --checkpoint checkpoints/n3d_sphere/model.pt

# 指定输出目录 / 初始权重阈值 / 默认关闭层平面 / PLY 用 ascii
python -m n3d_viz --checkpoint checkpoints/n3d_sphere/model.pt \
    --out-dir checkpoints/n3d_viz --threshold 0.30 --no-plan-planes --ply-ascii

# 显式指定 HTML 路径（PLY / OBJ 与其同目录同名）
python -m n3d_viz --checkpoint checkpoints/n3d_sphere/model.pt --out D:\tmp\viz_model.html
```

| 参数 | 说明 |
|---|---|
| `--checkpoint`, `-c` | 二期训练产物 `.pt` 路径；**省略时启动 GUI** |
| `--out-dir`, `-d` | 输出目录，不存在则创建（默认 `checkpoints/n3d_viz`） |
| `--out`, `-o` | 显式 HTML 路径；PLY / OBJ 与其同目录同名 |
| `--threshold`, `-t` | HTML 初始边权重阈值（默认 0.30） |
| `--no-plan-planes` | 默认不显示层参考平面（写入负载 `meta.showPlanes`，渲染器据此初始化复选框与绘制状态） |
| `--ply-ascii` | PLY 用 ascii 写出（默认 `binary_little_endian`） |
| `--with-ply-edges` | PLY 中附加 `edge` 元素（顶点索引对 + 权重），已经 CLI 透传至 `write_ply` |
| `--quiet`, `-q` | 只输出最终摘要 |

退出码：`0` 成功 / `2` 参数错误或无法启动 GUI / `3` checkpoint 相关错误（路径不存在、文件损坏、非二期产物缺拓扑键）。

### 1.2 图形界面（GUI）

```bash
python -m n3d_viz          # 不带参数即弹出窗口
python n3d_viz/gui.py      # 等价入口
```

窗口包含：

- `.pt` 文件输入框 + **浏览…**（`filedialog.askopenfilename`）
- 输出文件夹输入框 + **浏览…**（`filedialog.askdirectory`）
- 边权重阈值滑块 + **默认显示层平面** 勾选框
- **开始生成** 按钮、状态提示文字、可滚动日志区
- **打开 HTML**（默认浏览器）与 **打开输出文件夹** 按钮

长任务在**后台线程**执行，日志通过 `queue.Queue` + `root.after()` 回传主线程刷新，界面不卡死。
**不实现拖拽**——拖拽需要第三方 `tkinterdnd2`，与零依赖约束冲突。

### 1.3 其他可直跑脚本

```bash
python n3d_viz/render_html.py --checkpoint <pt> --out <html>           # 只生成 HTML 并做自包含校验
python n3d_viz/export_geometry.py --checkpoint <pt> --out-dir <dir>    # 只导出 PLY / OBJ
python n3d_viz/verify_viz.py                                           # 全部硬断言验证（见第 5 节）
node n3d_viz/assets/viewer_smoke.js <viewer.js> <data.json>            # 单独跑渲染器逻辑冒烟
```

---

## 2. 三件套产物

以 `--checkpoint checkpoints/n3d_sphere/model.pt` 为例，默认写入 `checkpoints/n3d_viz/`：

| 产物 | 命名规则 | 实测大小（来源见第 6 节） | 内容 |
|---|---|---|---|
| `viz_model.html` | `viz_<ckpt名>.html` | 88,521 字节 | 数据与渲染器全部内联，**零外部引用**，断网可打开 |
| `viz_model.ply` | `viz_<ckpt名>.ply` | 4,120 字节 | 256 个顶点（`x,y,z` float32 + `red,green,blue` uchar），`binary_little_endian` |
| `viz_model.obj` | `viz_<ckpt名>.obj` | 15,695 字节 | 256 个 `v` 行 + 736 个 `l` 行，索引从 1 开始 |

`<ckpt名>` 指 checkpoint 文件名去掉扩展名，例如 `checkpoints/n3d_sphere/model.pt` → `viz_model.*`；
传入 `.../full_N256_y8x8_..._s42.pt` 则得到 `viz_full_N256_y8x8_..._s42.*`，天然不撞名。

### HTML 自包含校验口径

- 文本中不存在 `http://` / `https://`，也不存在协议相对引用（`src="//`、`href="//`、`url(//`）；
- 不含 `syn_dist`（默认规模 `[2048,2048]` float32 = 16,777,216 字节，约 16.8MB，**严禁嵌入**）；
- 体积 < 2MB（实测 88,521 字节）；
- 无 `<script src=...>` 与 `<link ...>`。

---

## 3. 交互能力（HTML 视图）

| 操作 | 效果 |
|---|---|
| 左键拖拽 | 旋转（yaw 绕屏幕竖直轴，pitch 俯仰带限幅） |
| 滚轮 | 缩放（距离按 `exp(Δy·0.0012)` 变化，并夹在 `0.35R ~ 40R`） |
| 右键拖拽 | 平移（视图矩阵平移列叠加） |
| 悬停神经元 | 弹出详情：id、所在层、z 坐标、入度 / 出度、是否 S_in / S_out |
| 图层开关 | 神经元 / 连接 / 层平面分别显示隐藏 |
| 阈值滑块 | 只显示 `\|edge_weight\| ≥` 阈值的边，实时刷新保留边数 |
| 重置视角 | 恢复初始 yaw / pitch / 距离 / 平移 |

配色约定：神经元按 `z` 分层着色（9 色循环，与 PLY 顶点色一致）；**S_in** 以品红 `#ff5ec7` 同心小圆叠加（半径 0.55r），**S_out** 以金黄 `#ffd84d` 同心叠加（仅 S_out 时 0.55r，兼属两者时 0.26r 作为内圈），因此即便被高亮也能看出其所属层色；边按 `|w|` 归一化映射粗细（0.6~3.2 px）与颜色（弱=青蓝 → 强=橙红）；每层绘制一张水平参考平面并标注 `L1..L9`。

### 渲染原理（手写，无第三方库）

`n3d_viz/assets/viewer.js` 用 Canvas 2D 实现三维渲染：

1. **视图变换**：标准轨道相机。世界点 `p` → 相机坐标 `p_cam = R·(p − center) + (0, 0, −dist)`，其中 `R = RotX(pitch)·RotY(yaw)`，`center` 取点云质心。相机位于目标 +Z 方向 `dist` 处朝 −Z 观察，故目标点满足 `z_cam = −dist < 0`。
   > 实现要点：矩阵按**行主序**（`index = row*4+col`）存放。若把 `A[r][k]` 误读成 `a[k*4+r]`，得到的 `(A·B)ᵀ` 会把整个点云翻到相机背后，投影全部被剔除、画面只剩背景。
2. **透视投影**：`x_screen = W/2 + f·x_cam/(−z_cam)`，`y_screen = H/2 − f·y_cam/(−z_cam)`，焦距 `f = max(0.9H, 300)`；`z_cam > −0.001` 的点被剔除（位于相机之后）。
3. **深度排序（画家算法）**：把层平面、边（深度取两端点均值）、神经元（点深度）合并成一个列表，按 `depth = −z_cam` 升序（远 → 近）绘制，后画的覆盖先画的，从而在二维画布上得到正确遮挡关系。

### 渲染器逻辑冒烟（零第三方测试依赖）

`n3d_viz/assets/viewer_smoke.js` 用最小 DOM / Canvas 桩**真实执行** HTML 里内联的 `viewer.js`，断言：
圆形总数 == N + S_in + S_out、绘制线段数 ≥ 阈值内边数、投影 bbox 有限且落在画布附近、投影质心接近画布中心、
阈值滑块联动（阈值 0 时保留全部 736 条边）、**S_in 高亮圈数 == 193**、**S_out 高亮圈数 == 187**、悬停命中并填充详情、图例色块数 == K+3。
由 `verify_viz.py` 自动调用（需要 `node`，缺失时该条标记 SKIP，不影响其它断言）。

### 真实浏览器截图（视觉证据）

在 Chromium（Playwright，仅作为一次性图形验证工具，未引入任何依赖）中以 1280x760 视口打开产物得到：

```
bun x playwright screenshot --browser chromium --viewport-size "1280,760" \
    --wait-for-timeout 2500 file:///.../viz_model.html viewer_screenshot.png
```

结果保存为 `checkpoints/n3d_viz/_verify/viewer_screenshot.png`（233,076 字节），可见：
9 层参考平面（标注 L1..L9）、层色神经元、S_in/S_out 同心高亮、
按 `|w|` 映射粗细与颜色的连接（阈值 0.30 下保留 379 条）、叠加在左上的控件面板与右上的统计面板、
底部图例与操作提示。该截图不作为自动化断言，
自动化断言以 `assets/viewer_smoke.js`（Node + DOM 桩）为准。

---

## 4. 模块结构

```
n3d_viz/
├── __init__.py          # 包声明、零依赖与自包含约束
├── core.py              # checkpoint 加载 / 二期键校验 / 拓扑抽取 / 产物命名派生 / HTML 数据负载
├── export_geometry.py   # 零依赖 PLY（二进制/ascii）与 OBJ 线框写出器 + 回读解析器
├── render_html.py       # 单文件 HTML 生成 + 自包含断言
├── gui.py               # tkinter 界面（后台线程 + queue + after 轮询）
├── __main__.py          # CLI 入口；不带参数启动 GUI
├── verify_viz.py        # 全部硬断言验证脚本（真实执行，末尾汇总退出码）
├── assets/
│   ├── viewer.html      # HTML 模板（含两个占位标记）
│   ├── viewer.js        # 三维交互渲染器（构建时内联进产物）
│   └── viewer_smoke.js  # 渲染器逻辑冒烟脚本（Node + DOM 桩）
└── README.md
```

构建时 `render_html` 读取 `assets/viewer.html` 与 `assets/viewer.js`，把 `/*__N3D_DATA_JSON__*/` 替换为 JSON 负载、`/*__N3D_VIEWER_JS__*/` 替换为渲染器源码，因此产物不含任何外部引用。

---

## 5. 验收标准与断言口径

```bash
python n3d_viz/verify_viz.py --report checkpoints/n3d_viz/_verify/verify_report.md
# 实测：通过 77 / 失败 0 / 跳过 0，退出码 0
```

| 断言 | 口径 | 实测 |
|---|---|---|
| PLY 顶点数 | == N | 256 |
| OBJ `l` 行数 | == E | 736 |
| OBJ `v` 行数 | == N | 256 |
| 产物坐标 vs `neuron_pos` | 逐位一致，容差 1e-6 | PLY max\|diff\| = 0.000e+00；OBJ max\|diff\| = 4.846e-10 |
| 产物文件大小 | 与 `write_outputs` 报告一致 | HTML 88,521 / PLY 4,120 / OBJ 15,695 字节 |
| 层着色组数 | == K | 9（PLY 顶点色分组数同时为 9） |
| 各层神经元数 | == 13/24/37/35/39/34/37/24/13 | 一致（PLY 颜色计数排序后一致） |
| S_in 高亮数 | == `in_scope_mask` 为真计数 | 193 |
| S_out 高亮数 | == `out_scope_mask` 为真计数 | 187 |
| 阈值 0.30 保留边数 | == \|w\|≥0.30 实测值 | 379 / 736 |
| HTML 体积 | < 2MB | 88,521 字节 |
| HTML 不含 `syn_dist` | 文本不含该词 | 通过 |
| HTML 外部引用 | 无 `http://` / `https://` / 协议相对 | 通过 |
| 内联数据规模 | neurons==N、edges==E、layers==K | 256 / 736 / 9 |
| 内联阈值统计 | == {0.05:681, 0.10:620, 0.20:517, 0.30:379, 0.50:140} | 一致 |
| 一期产物异常路径 | 退出码非 0 且含缺失键名 | 退出码 3，含 `edge_src` / `edge_dst` / `edge_weight` / `in_scope_mask` / `out_scope_mask` / `level_node_reach` |
| 路径不存在 | 退出码非 0 且报错可读 | 退出码 3，输出含"路径不存在" |
| `n3d_viz` 不 import 业务模块 | 源码正则扫描 `^\s*(from\|import) n3d_(sphere\|proto)` | 0 命中 |
| 无第三方依赖 | 扫描全部顶层 import 与 `requirements.txt` | 仅 `torch` / `numpy` / 标准库 |
| GUI 冒烟 | `gui` 可导入，`withdraw()` 状态下构造并销毁 Tk 窗口（不进 mainloop） | 构造成功，顶层子控件 7 个 |
| 开关类参数产物差异 | 每个布尔开关开/关两态产物必须不同 | `include_planes` True/False 产物不同；`--no-plan-planes` 产物 `showPlanes == false`；`--with-ply-edges` 产物 PLY 含 736 条 `element edge` |
| 渲染器逻辑 | Node + DOM 桩执行内联 `viewer.js` | 通过 12 / 失败 0（汇总行进入报告第 45 行） |
| 负例临时产物清理 | 负例跑完 `_bad_index/` 无文件残留、残留体积 == 0 字节 | 残留 0 个文件 / 0 字节（清理前曾残留 4 个共 ~69MB 的 `bad_*.pt`） |

### 数值口径（唯一化，避免同一指标出现多个无法判断口径的数字）

**`|edge_weight|` 的 median 口径（全模块唯一）**：把 `|edge_weight|` **升序排列后取上中位**，即下标 `n // 2`
（`E=736` 时为 `sort(|w|)[368]`）。不取两中位均值（那会得到 `0.30649110674858093`），
也不使用 `torch.median`（它返回下中位 `0.3059167265892029`）。`TopologyData.weight_extremes()` 与
本 README 使用同一口径，实测均为 **0.307065486907959**（来源 `checkpoints/n3d_sphere/model.pt`，seed=42）。

**内联 HTML 的浮点保留位数**：`build_html_payload(pos_digits=None, weight_digits=None)` 中 `None`
表示**取默认位宽**（坐标 `DEFAULT_POS_DIGITS = 6` 位小数、权重 `DEFAULT_WEIGHT_DIGITS = 8` 位小数），
**不是**"不截断"；传入整数则按该位数四舍五入。实测该默认位宽下内联坐标与 `neuron_pos` 的
`max|diff| = 4.510e-07 < 1e-6`，内联 `|edge_weight|` 与 checkpoint 的 `max|diff| = 4.991e-09`。
（模块内私有工具 `_round_floats(values, digits=None)` 的 `None` 语义为"不截断"，与上面的负载函数
**不同**，已在各自 docstring 里写明。）

### 索引取值域契约

`extract_topology` 除长度校验外，还强制 `topo_index` / `edge_src` / `edge_dst` 全部落在 `[0, N)`：

- 若 `topo_index` 含**负值**，`pos[i]` 会按 Python 负索引**静默**取到错误神经元，使 `layer_groups` /
  `layer_z` 静默错误（图是错的却不报错）；
- 若含**越界值**，会抛 `IndexError` 而非 `CheckpointSchemaError`，而 CLI 只捕获 `CheckpointError`，
  于是损坏产物以 traceback 崩溃、退出码 1，破坏"非二期产物给可读错误"的契约。

现在两种情况都抛 `CheckpointSchemaError`，错误信息附**越界值与其下标**，进程退出码 **3**、无 traceback、
不产生任何产物。verify 脚本以 4 个负例（`topo_index[5]=-1` / `topo_index[7]=999` / `edge_src[3]=-9` /
`edge_dst[11]=256`）各断言 4 项（退出码 3、报错含越界文本、无 traceback、无产物）。

报错中最多列出 6 个越界项（`_OUT_OF_RANGE_LIMIT`），措辞为「等至少 N 项（仅列前 N 项）」——
扫描在第 6 项处提前返回，故该数字只是「已列出的项数」而非真实总数。

负例所用的临时 checkpoint 只保留契约键（丢弃 `syn_dist` 等巨型张量，单个从 ~18MB 降到 ~30KB），
且负例跑完立即 `unlink` 删除，因此 `_bad_index/` 不残留任何文件（verify 对「无文件残留」与
「残留体积 == 0 字节」各有一条断言；清理前实测曾残留 4 个共 ~69MB 的 `bad_*.pt`）。

### 异常路径（必须报错，不得静默降级）

一期产物 `checkpoints/n3d_model_full.pt`（`n3d_proto`）**缺少**二期拓扑键：

```
$ python -m n3d_viz --checkpoint checkpoints/n3d_model_full.pt
[错误] checkpoint 不是 N3D 二期产物，缺少二期拓扑必需键：edge_src, edge_dst, edge_weight,
in_scope_mask, out_scope_mask, topo_index, level_node_reach, level_edge_reach, in_degree,
out_degree（文件：...；完整必需键列表：neuron_pos, edge_src, ...）。
该文件可能是一期（n3d_proto）产物，其仅有突触级 edge_index，不含神经元级连接与作用域掩码，
无法可视化；请改用二期产物。
$ echo $LASTEXITCODE
3
```

一期状态字典实际含 `W_conn_sparse / tau_raw / neuron_threshold / W_in / W_out / neuron_pos /
input_syn_pos / output_syn_pos / dist / mask / edge_index / edge_dist / scatter_out_to_neuron /
broadcast_neuron_to_in / neuron_of_output_syn / neuron_of_input_syn / ln_s_in.weight / ln_s_in.bias`
——只有**突触级** `edge_index`，没有神经元级连接与 S_in / S_out 掩码，因此无法画出本模块要求的三维拓扑图。

---

## 6. 实测数字与引用纪律

**本节所有数字均为真实执行产物，未运行的一律不写。**

- **来源 checkpoint**：`checkpoints/n3d_sphere/model.pt`
- **seed = 42**（取自该产物的 `config.seed`；该产物记录 `test_acc = 0.9759`）
- **配置**：`N=256, y_in=y_out=8x8, H=0.1, D=0.1, flow_axis=z, placement=fcc, input_scope/readout_scope=any_isolated`
- **运行环境**：Python 3.12.10 / torch 2.14.0+cpu / numpy 2.5.3 / Tk 8.6（Windows）

| 指标 | 实测值 |
|---|---|
| 神经元数 N | 256 |
| 神经元级连接数 E | 736 |
| 连接密度 `E/(N·N)` | 0.01123046875 |
| 分层数 K | 9 |
| 层神经元数 | 13 / 24 / 37 / 35 / 39 / 34 / 37 / 24 / 13（合计 256） |
| 层 z 取值 | −0.565685 / −0.424264 / −0.282843 / −0.141421 / 0.0 / 0.141421 / 0.282843 / 0.424264 / 0.565685 |
| 层入边数 | 0 / 46 / 88 / 116 / 115 / 122 / 114 / 88 / 47（第 1 层为 0） |
| S_in（`in_scope_mask` 为真） | 193 |
| S_out（`out_scope_mask` 为真） | 187 |
| `input_isolated_mask` 为真 | 605 |
| 入度直方图 | 0:13, 1:35, 2:37, 3:57, 4:114（合计 256） |
| 出度直方图 | 0:14, 1:27, 2:37, 3:77, 4:101（合计 256） |
| `\|edge_weight\|`（min / median / max） | 0.002693684 / 0.307065487 / 0.845361531（median 口径见 §5「数值口径（唯一化）」） |
| 阈值过滤保留边数 | ≥0.05 → 681；≥0.10 → 620；≥0.20 → 517；**≥0.30 → 379**；≥0.50 → 140 |
| `syn_dist` 体积 | 16,777,216 字节（未进入内存、未嵌入 HTML） |
| HTML / PLY / OBJ 大小 | 88,521 / 4,120 / 15,695 字节 |
| 验证汇总 | 通过 77 / 失败 0 / 跳过 0（退出码 0）；其中第 45 行为渲染器逻辑冒烟（内含 12 条子断言，均通过），第 71/72 行为负例临时产物清理断言 |

**引用纪律**：

- **边集（E）、边权重、突触位置随 seed 变化**：任何边级数字（E=736、阈值保留边数、`|w|` 统计、层入边数）都必须连同其来源 checkpoint 与 `seed=42` 一起引用，换 seed 后这些数字会变。
- **神经元 FCC 位置与分层结构由排布参数（`placement=fcc`、`H`、`D`、`flow_axis`、`N`）决定，与 seed 无关**：N=256、K=9、层规模 13/24/37/35/39/34/37/24/13、层 z 取值属于几何常量，可在不同 seed 的同类产物间复用。
- 引用本 README 的数字时，请注明"来源 `checkpoints/n3d_sphere/model.pt`，seed=42"。
- `S_in` / `S_out` 计数（193 / 187）依赖 `input_scope=any_isolated` 与 `readout_scope=any_isolated` 及随机采样，**随 seed 变化**。

---

## 7. 注意事项

- **张量必须含二期拓扑键**：`REQUIRED_KEYS` 定义在 `n3d_viz/core.py`，缺任一键即抛 `CheckpointSchemaError` 并携带全部缺失键名。
- **不读 `syn_dist`**：`core._syn_dist_bytes` 只记录其字节数，从不把该张量转成 Python 对象。
- **产物覆盖**：同名产物存在时 CLI 打印提示、GUI 在日志与状态栏提示，verify 脚本打印 `existed` 标记；不做静默覆盖。
- **`verify_viz.py` 默认把产物写到 `checkpoints/n3d_viz/_verify/`**（含 `verify_report.md` 报告）**，避免与正式产物互相污染（对齐"验证类命令一律写入 `_verify/`"的产物纪律）。
- **平台**：Windows 上"打开输出文件夹"用 `os.startfile`；Linux 用 `xdg-open`；macOS 用 `open`。
- **无图形环境**：`--skip-gui` 可跳过 GUI 冒烟断言；CLI 入口不受影响。
