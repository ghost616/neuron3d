# n3d_viz 测试报告 —— 两端全连接层三维展示（方案 1）

- 测试对象：`n3d_viz` 本轮变更（core / export_geometry / render_html / __main__ / gui / verify_viz / assets/viewer_fc.js / assets/viewer_smoke.js）
- 基准产物：`checkpoints/n3d_sphere/model.pt`（N=256，无 FC）；FC 产物 `checkpoints/n3d_shape/full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.pt`（fc_dim=-1，H=825，seed=42）
- 执行环境：Windows / Python 3.12.10 / Node v25.2.1 / Playwright 1.63.0（npm，Chromium headless）
- 结论：**功能主体正确，零回归成立；发现 3 类健壮性缺陷（均为「文档/规范声明了明确报错，实际是未包装异常或静默截断」），无功能性错误、无产物错误。**

## 一、测试概览

| 测试类型 | 用例数 | 通过 | 失败 | 结论 |
|---|---|---|---|---|
| 编译/静态检查 | 4 | 4 | 0 | 通过 |
| 单元测试（FC 抽取 / 三态判定 / top-k / 面板几何 / 零回归 / 产物结构） | 178 | 171 | 7 | 7 项失败中 **4 项为首轮脚本自身断言错误（已定向复验通过）**，**3 项为真实缺陷** |
| 定向复验（修正脚本断言后重跑） | 19 | 11（OK） | 8（DEFECT） | 确认真实缺陷 3 类 |
| 渲染器逻辑冒烟（Node + DOM 桩） | 2 轮 | 2 | 0 | 无 FC 13/13、有 FC 20/20 |
| E2E（真实 Chromium，Playwright） | 40 | 40 | 0 | 通过 |
| GUI（tkinter withdraw 状态） | 3 | 3 | 0 | 通过 |
| 模块自带验证脚本 `verify_viz.py` | 485 | 485 | 0（跳过 1） | 通过，退出码 0 |
| **合计（去重后独立断言）** | **731** | **724** | **7** | 其中 4 项已证明为脚本误报；**真实缺陷 3 类** |

被测源码 SHA256（本轮快照）：

| 文件 | SHA256 |
|---|---|
| `n3d_viz/core.py` | `aae9472e562f60876a2a901d0ee931d90c27a0cd00a2a3eee5e4091924457d3f` |
| `n3d_viz/export_geometry.py` | `0a8b475c2e4822569c32ba23a3bbe61a0f38733649345e7a65b9b0eb73c8e25c` |
| `n3d_viz/render_html.py` | `7cd491d919b259062b5444d1d5d282cb72697b477163f898f127276ae25ce686` |
| `n3d_viz/__main__.py` | `a80173fe627eccb71ebee66908a3ae8c8b5a72f149264d6925ee00d72f3e6d1b` |
| `n3d_viz/gui.py` | `3114ea0cca7961c6367d78972786b0d2443114ae3873b9437ee5d8263a30f04f` |
| `n3d_viz/verify_viz.py` | `27a667f32310d68420ab0ba0de718741d20ba4d1b280d2126240f281359b0e02` |
| `n3d_viz/assets/viewer_fc.js`（新增） | `27eb7d608a9d0402ceff342815b5ab2081f672ba285ad08da7ece48d4169d742` |
| `n3d_viz/assets/viewer_smoke.js` | `71e4b2d06ec58733f12cd7a8bb50ec3e4212440096edb6ba3ee9c933c840c024` |
| `n3d_viz/assets/viewer.js`（未改动） | `53753ae80aafc41eb38c75f93e48a056c431ae9ff049f8b17df90a474a237990` |
| `n3d_viz/assets/viewer.html`（未改动） | `60cdb020df4ea74eb03fd0b137b2c75eaf4e9b3425f8c50a6e4cc0811792d2e5` |

## 二、编译 / 静态检查（全部通过）

| 命令 | 退出码 | 实测 |
|---|---|---|
| `python -m compileall -q n3d_viz` | 0 | 全部模块编译通过 |
| `node --check n3d_viz/assets/viewer.js` | 0 | 通过 |
| `node --check n3d_viz/assets/viewer_fc.js` | 0 | 通过 |
| `node --check n3d_viz/assets/viewer_smoke.js` | 0 | 通过 |

## 三、单元测试详细结果

### 3.1 触发判定三态 `is_fc_enabled`（9/9 通过）

| 输入 | 期望 | 实测 |
|---|---|---|
| `config` 无 `fc_dim` 键 | False | False ✅ |
| `fc_dim == 0` / `"0"` | False | False ✅ |
| `fc_dim == -1` | **True（启用，非关闭）** | True ✅ |
| `fc_dim == 1` / `2048` | True | True ✅ |
| `config is None` / `{}` | False | False ✅ |
| 仅 `{"fc_dim": -1}`（无任何 FC 键） | True（判定不依赖键存在性） | True ✅ |
| `fc_dim == "abc"` | False 且不抛异常 | False ✅ |

另：二期产物（无 `fc_dim`）`load_topology` 不报错且 `data.fc is None` ✅。

### 3.2 FC 抽取口径与几何（FC 产物，全部通过）

| 项目 | 期望 | 实测 |
|---|---|---|
| `config.fc_dim` | -1 | -1 ✅ |
| H / \|S_in\| / \|S_out\| | 825 / 582 / 588 | 825 / 582 / 588 ✅ |
| proj_weight / fc_out_weight 形状 | (582,825) / (825,588) | 一致 ✅ |
| 面板点 == 2×H | 1650 | 1650 ✅ |
| 默认 k | 3 | 3 ✅ |
| 抽样条数 | 3510 == (582+588)×3 | 3510 ✅ |
| 每 S_in/S_out 神经元连线数 | 恰好 k 条（≥1） | 582×3 + 588×3，无缺失 ✅ |
| 输入侧口径 | 每神经元取 `proj_weight` 该**行** `\|w\|` top-k | 逐位核对 120 个神经元，全部一致 ✅ |
| 输出侧口径 | 每神经元取 `fc_out_weight` 该**列** `\|w\|` top-k | 逐位核对 120 个神经元，全部一致 ✅ |
| 权重值 | 与状态字典逐位一致 | 前 2000 条 max\|diff\| = 0 ✅ |
| 参数量 | 480150 / 485100（与抽样条数分别标注） | 一致 ✅ |

面板几何（硬断言）：

- 面板流向轴区间 ∩ 云区间 = ∅：输入 `[-1.2938,-1.2801]`、输出 `[1.2801,1.2938]`、云 `[-0.9899,0.9899]` ✅（**与待测说明给的基线数值完全一致**；由本次真实执行独立复算确认）
- 网格 `ceil(sqrt(825)) = 29` 列 ✅；`cols*rows >= H` ✅
- 间隙 == 云跨度 × 0.15（实测 max\|diff\| = 5.55e-17）✅
- 面板厚度 ∈ `[1e-3, 云跨度×0.10]`（实测 0.0136545）✅，两片相等 ✅
- 面板单元 z 坐标恒定（垂直于流向轴）✅
- 边界块 输入784/输出10 置于面板**外侧**（-1.551148 < -1.293762；1.551148 > 1.293762）✅，流向轴尺寸 == 云跨度×0.14 ✅

`k` 生效性与确定性：

| 项目 | 实测 |
|---|---|
| k=1 | 1170 条，top_k=1 ✅ |
| k=8 | 9360 条，top_k=8 ✅ |
| k=1 与 k=8 产物不同 | 是（1170 vs 9360 条，HTML SHA256 不同）✅ |
| k=1 ⊂ k=8（top-k 单调性） | 成立 ✅ |
| 两次独立抽取 edges 相同 | 相同 ✅ |
| 两次独立写出 HTML/PLY/OBJ | **逐字节相同** ✅ |
| `with_fc_top_k(same_k)` | 原样返回入参对象（零开销）✅ |
| `with_fc_top_k(new_k)` | 重抽正确且不修改原对象 ✅ |

### 3.3 三态边界与原子性（通过）

- `fc_dim != 0` 且缺任一 FC 键（`proj_weight` / `fc_out_weight` / `fc_in_weight` / `fc_out_bias`）→ `CheckpointSchemaError`，错误信息列出缺键 ✅
- 四键全缺 → 报错并列出全部 4 个缺键 ✅
- CLI 走「缺键」产物：退出码 3、**无 traceback**、**不产生任何产物**（原子性成立）✅
- `fc_dim == 0` 但 FC 张量齐全：退出码 0、产出三件套、PLY 无 `fc_node`/`fc_edge`、OBJ 无 FC 分组与声明、HTML 负载无 `fc` 键且 meta 无 `fcDeclaration` ✅

### 3.4 零回归（全部通过）

| 项目 | 实测 |
|---|---|
| 无 FC 默认产物重渲 HTML | SHA256 `15a80ebbf2fd586b…` == 锚点 ✅ |
| 无 FC 默认产物重渲 PLY | SHA256 `9a097d16306160f9…` == 锚点 ✅ |
| 无 FC 默认产物重渲 OBJ | SHA256 `1f594ecf466e28f7…` == 锚点 ✅ |
| `assets/viewer.js` 与 git HEAD | 逐字节一致 ✅ |
| `assets/viewer.html` 与 git HEAD | 逐字节一致 ✅ |
| 新增 `assets/viewer_fc.js` | 存在且通过 `node --check` ✅ |

> 说明：首轮脚本用**绝对路径** `-o` 重渲导致 HTML 不等（产物会把传入路径写进 meta）；改用与锚点相同的相对路径风格重渲后逐字节一致。**这是脚本参数差异，不是零回归缺陷**，`verify_viz.py` 的 `[2d]` 锚点重渲同样通过。

### 3.5 产物差异与显式声明（全部通过）

PLY：

- `element vertex 825` 仍为 N（FC 点未混入顶点）✅
- `element fc_node 1652 == 2×H+2`，含 `property uchar kind` ✅
- `element fc_edge 3510` ✅
- 元素顺序 `vertex → fc_node → fc_edge` ✅
- 头部注释含 `NOT all connections`、两侧参数量 480150/485100、`H=825`、`top-k=3` ✅
- 回读解析器：`parse_ply_vertices` 825、`parse_ply_fc_nodes` (1652,3510)、顶点与 neuron_pos 一致 ✅
- 文件长度 == 头 + 825×15 + 1652×16 + 3510×12 ✅

OBJ：

- `v` 行 2477 == N + (2H+2) ✅；`l` 行 6098 == E(2588) + 抽样(3510) ✅
- 分组 `o/g n3d_viz_core_edges` 与 `o/g n3d_viz_fc_sampled_edges` 均在 ✅
- 注释含「非全部连接」、`k=3`、`480,150 条 + 485,100 条`、`NOT all connections; declared total would be 965250 links` ✅
- 索引范围合法、抽样 `l` 行只引用 FC 单元点（≥ N+1）✅；换行全为 LF ✅

HTML：

- `meta.fcDeclaration` 存在、`meta.fcNotAllConnections == true`、`meta.fcTopK==3`、`meta.fcWidth==825`、`meta.fcSampleEdges==3510 == len(payload.fc.edges)` ✅
- 负载 `fc` 段含 panels(2) / blocks(2) / edges(3510) / 矩阵形状 / 参数量 / 抽样期望值 ✅
- 参数量（480150）与抽样条数（3510）**分别标注、未混用** ✅
- 自包含（无 `<script src>`、无 `syn_dist`、无外部 URL）✅
- 抽样声明同时出现在 **HTML meta / PLY 头部注释 / OBJ 注释** 三处 ✅

### 3.6 真实浏览器 E2E（Playwright + Chromium，40/40 通过）

| 场景 | 断言 |
|---|---|
| 无 FC 产物 | 页面无 JS 报错 ✅；负载无 `fc` 段 ✅；`#view` 真的画了内容（非透明采样 40960/40960）✅；**未创建** `#view-fc` ✅；**无** `#fc-stats` ✅ |
| FC 产物（k=3） | 无 JS 报错 ✅；`fc.edges==3510==meta.fcSampleEdges` ✅；`fcTopK==3`、`fcWidth==825`、`panels=2/blocks=2` ✅；`#view-fc` 已创建且与基础画布同投影尺寸 1280×800 ✅；**叠加层真的画上了内容**（非透明 3096/40960）✅；`pointer-events:none`、`z-index:20`（严格大于基础画布，未被遮挡）✅；`#fc-stats` 含 `H=825 top-k=3 proj_weight=582×825 480150 条 fc_out_weight=825×588 485100 条 抽样连线=3510 条` 与抽样声明 ✅；图例 `data-fc=1` ✅ |
| FC 产物（k=1） | 同上，`fc.edges==1170`，叠加层非透明 3003/40960 ✅ |
| 交互回归 | 阈值滑块拖到 0 → 「保留 2588 / 2588 条边」✅；鼠标悬停 → hover-label 正确显示神经元信息 ✅；视口 1280×800 → 900×620 后叠加画布尺寸跟随基础画布（900×620），叠加层仍有内容且全程无 JS 报错 ✅ |
| k 影响渲染 | k=3（3096）与 k=1（3003）叠加层绘制像素数不同 ✅ |

### 3.7 GUI（通过）

- `VizApp` 在 `withdraw()` 状态下可构造、可销毁（8 行控件），无异常 ✅
- 「全连接层抽样 k」`ttk.Spinbox` 存在：默认值 `3`，范围 `1..8`，标签与说明文字齐全 ✅
- 输入口径：`'3'`→k=3 接受；`'0'`/`'9'`→被拒绝并给出可读信息；`'abc'`/`''`→被拒绝 ✅
- FC 信息面板：`_run_job` 按 `data.fc` 输出 fc_dim / H / 4 个矩阵形状与参数量 / 抽样条数（静态核对 + 浏览器端统计面板内容核对）✅

## 四、失败用例分析

### 4.1 真实缺陷（3 类，共 8 个复现点）

#### 缺陷 1（中）：`core.validate_fc_top_k` 对非整数浮点**静默截断**，与规范「不静默截断」冲突

- 位置：`n3d_viz/core.py:1098` `validate_fc_top_k`（`value = int(k)` 直接整型化）
- 规范要求：`k` 范围 `[1,8]`；越界（0 / 9 / -1 / **非整数** / 非数字）一律报错、**不静默截断**
- 实测：
  - `validate_fc_top_k(2.5)` → 接受并返回 **2**
  - `validate_fc_top_k(1.5)` → 接受并返回 **1**
  - `validate_fc_top_k(8.7)` → 接受并返回 **8**
  - `validate_fc_top_k(3.0)` → 接受并返回 3
  - `validate_fc_top_k(True)` → 接受并返回 1
- 影响面：
  - CLI **不受影响**（`argparse type=int` 在入口就拒绝 `--fc-top-k 1.9/2.5/1.0`，退出码 2，实测通过）
  - 受影响的是 **Python API 调用方**：任何把 `k` 以 `float`（或 `Decimal`/`numpy` 浮点）传入的调用方都会被静默截断，与规范声明不一致
  - GUI 为 `gui.py:247` 的 `core.validate_fc_top_k(int(self.fc_top_k_var.get().strip()))`：Spinbox 手输 `2.5` 时 `int("2.5")` 先抛 `ValueError` 而被拒绝（实测），因此 GUI 路径表现为「报错」而非「截断」——但 GUI 依赖的是 `int()` 的字符串解析而非校验函数本身
- 建议：`validate_fc_top_k` 内显式拒绝非整数（例如 `isinstance(k, bool)` → 拒绝、`float(k) != int(k)` → raise `ValueError`），并保持 CLI/GUI 复用同一入口

#### 缺陷 2（中）：`core.validate_fc_top_k(None)` 抛裸 `TypeError`，与 docstring 声明不符

- 位置：同 `core.py:1098`（`int(None)` → `TypeError`）
- docstring 仅声明 `Raises: ValueError`；GUI 的 `except (ValueError, TypeError)` 虽已兜住，但按契约只捕获 `ValueError` 的调用方会漏接
- 建议：函数内先做类型/可转换性判定，统一为 `ValueError`

#### 缺陷 3（低-中）：`core._fc_panel_geometry(H <= 0)` 逃逸未包装异常

- 位置：`n3d_viz/core.py:1119`；在 `_place_units_in_panel` 的 `if cell <= 0.0: raise CheckpointSchemaError` 哨兵**之前**，先执行了 `_grid_layout(h)` 与 `grid_side` 除法
- 规范要求：「异常：`H <= 0` … → 明确报错」；docstring `Raises: CheckpointSchemaError`
- 实测：
  - `_fc_panel_geometry(..., h=0, ...)` → 裸 **`ZeroDivisionError: integer division or modulo by zero`**
  - `_fc_panel_geometry(..., h=-1, ...)` → 裸 **`ValueError: math domain error`**（`math.sqrt(-1)`）
- 影响面：**主链路不受影响** —— `extract_fc` 经 `_fc_hidden_width` 已用 `H <= 0` 校验兜住；实测把 `proj_weight` 列数与 `fc_out_weight` 行数改为 0 后得到规范错误 `checkpoint 的 fc_dim != 0 但有效宽度 H=0 <= 0。` ✅。缺陷仅出现在直接调用该几何函数的场景（库内复用），属契约违反
- 建议：在 `_fc_panel_geometry` 入口加 `if h <= 0: raise CheckpointSchemaError(...)`，使 `H<=0` 与「流向轴非法」「云跨度为 0」三个异常口径一致

### 4.2 首轮脚本自身断言错误（已定向复验，均通过，非产品缺陷）

| 首轮失败项 | 真实原因 | 复验结果 |
|---|---|---|
| `fc_dim=-1 + 无 FC 张量 -> must fail` | 脚本 `want_none` 判据写反 | 实现正确抛 `CheckpointSchemaError`（不静默降级）✅ |
| `no-FC re-rendered html 与锚点不一致` | 脚本用**绝对路径** `-o` 重渲（路径被写进 meta） | 用相对路径风格重渲 → SHA256 与锚点**完全一致** ✅ |
| `OBJ comment carries param counts` | 脚本按纯数字 `480150` 匹配，实际文案为 `480,150`（千分位） | OBJ 注释确实含 `480,150 条 + 485,100 条` ✅ |
| GUI construct+destroy | 脚本在 `app.destroy()` 后又调 `root.destroy()`（Tk 已销毁） | 仅调 `app.destroy()` → 正常 ✅ |

## 五、模块自带验证脚本结果

```
python n3d_viz/verify_viz.py --report <report.md>
→ 汇总：通过 485 / 失败 0 / 跳过 1（共 485），退出码 0
```

- 跳过项：`[2c]` 中 18 个 `*_mlp.pt` 产物缺 N3D 拓扑键（对照用 MLP 基线，非缺陷）
- `[2c]` 异构产物组已扩展到 FC 产物（含 `fc-1` 系列 9 个 seed），FC 面板点 == 2×H+2 全部通过
- `[2d]` 零回归锚点：HTML/PLY/OBJ 的 SHA256 与字节数均与锚点一致
- `[2e]` 一致性：`argparse` 默认值映射 == `core.DEFAULT_WRITE_OPTIONS` == `render_default` 实际参数；CLI 选项清单与冻结清单一致（17 项）
- `[2f]` FC 组 83 项：含三态判定、缺键三类报错（不产生产物）、k=1/8 边界、PLY/OBJ/HTML 产物逐项核对、面板不重叠、每神经元 ≥1 连线、参数量 vs 抽样量分别标注、渲染器冒烟 20/20
- 报告与本次独立测试结论一致：**主链路 FC 功能与零回归均成立**

## 六、环境问题说明

无环境阻塞。以下为与环境的适配说明：

1. **本会话不提供 `bash` 工具**，全部命令（编译、CLI、E2E）改用 `pwsh` 执行；环境构建（`npm install playwright`）按要求在 `.lizhu_env/` 下子目录内完成。
2. 首轮 `t_fc.py` 为一次性脚本，其中 `_fc_panel_geometry` 异常用例与 GUI 用例含脚本自身缺陷（见 4.2），已用定向复验脚本 `round2b.py` 修正并重跑。
3. Playwright：`module_agent_testing(check_playwright)` 报告 npm 侧已安装（1.63.0），但全局 Node 环境无 `playwright` 模块；已在 `.lizhu_env/r22_e2e` 内 `npm install playwright`（added 2 packages，退出码 0），复用已存在的 ms-playwright Chromium 浏览器，E2E 正常执行。

## 七、修复建议（按优先级）

1. **P1｜`validate_fc_top_k` 严格化**（`core.py:1098`）：显式拒绝非整数与非数值输入，统一抛 `ValueError`，使 docstring / 规范 / 行为三者一致；CLI 与 GUI 继续复用该入口。
2. **P2｜`_fc_panel_geometry` 补 `H <= 0` 入口校验**（`core.py:1119`）：把 `H <= 0` 与「流向轴非法」「云跨度为 0」并列为同一口径的 `CheckpointSchemaError`，避免裸 `ZeroDivisionError` / `math domain error` 逃逸到库外。
3. **P3｜（文档）待测说明中的基线数值建议补充口径**：面板 z 区间 `[-1.2938,-1.2801]` / `[1.2801,1.2938]` 与云区间 `[-0.9899,0.9899]` 经本次真实执行独立复算**完全一致**；而中途某次渲染冒烟输出过 `[-1.2869,-1.2869]`（面板流向轴中心流坐标，非区间端点），建议在说明中明确「区间」与「中心」两种口径，避免后续核对歧义。
4. **P4｜（可选）** 为 `validate_fc_top_k` 与 `_fc_panel_geometry` 的异常路径补 `verify_viz.py` 断言：当前 `[2f]` 组未覆盖「非整数 k」与「H<=0 直接调用几何函数」两条路径（本次为外部测试发现）。

## 八、复现命令

```powershell
# 编译 / 静态检查
python -m compileall -q n3d_viz
node --check n3d_viz/assets/viewer.js
node --check n3d_viz/assets/viewer_fc.js
node --check n3d_viz/assets/viewer_smoke.js

# 模块自带验证（485/0/1，退出码 0）
python n3d_viz/verify_viz.py --report .lizhu_env/r22/verify_report.md

# 独立单元/接口测试（178 项，结果写 result_r22.json）
python -X utf8 .lizhu_env/r22/t_fc.py

# 定向复验（真缺陷确认，结果写 result_r22_round2.json）
python -X utf8 .lizhu_env/r22/round2b.py

# 渲染器逻辑冒烟（无 FC 13/13；有 FC 20/20）
python .lizhu_env/r22/extract_payload.py checkpoints/n3d_viz/viz_model.html .lizhu_env/r22/payload_no_fc.json
node n3d_viz/assets/viewer_smoke.js n3d_viz/assets/viewer.js .lizhu_env/r22/payload_no_fc.json

# 真实浏览器 E2E（40 项，需先 npm install playwright）
node .lizhu_env/r22_e2e/e2e_fc.mjs `
  .lizhu_env/r22/e2e/no_fc/viz_model.html `
  .lizhu_env/r22/e2e/fc_k3/viz_full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.html `
  .lizhu_env/r22/e2e/fc_k1/viz_full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.html
```

## 九、测试产物索引

- 模块自带验证报告：`.lizhu_env/r22/verify_report.md`（485/0/1）
- 独立测试结构化结果：`.lizhu_env/r22/result_r22.json`（178 项）；`.lizhu_env/r22/result_r22_round2.json`（19 项）
- 渲染器冒烟：`.lizhu_env/r22/payload_no-FC.json` / `payload_with-FC.json`
- E2E 结果：`.lizhu_env/r22/e2e/e2e_result.json`（40 项）
- 测试脚本：`.lizhu_env/r22/t_fc.py`、`.lizhu_env/r22/round2b.py`、`.lizhu_env/r22/extract_payload.py`、`.lizhu_env/r22_e2e/e2e_fc.mjs`
- E2E 用产物：`.lizhu_env/r22/e2e/{no_fc,fc_k3,fc_k1}/`
