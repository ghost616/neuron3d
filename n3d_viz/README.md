# n3d_viz —— 任意 N3D 拓扑产物的三维可视化工具（对几何零假设）

把已训练 checkpoint 中的**神经元位置**与**神经元级连接**渲染成三件套：
**自包含交互式三维 HTML** + **点云 PLY** + **线框 OBJ**。

- **对几何零假设**：图形**完全由 `neuron_pos` 决定**，不假设球 / 立方体 / 圆柱，也不假设晶格（FCC）或分层规整性。随机点云、任意曲面、非晶格、非均匀分层等任意 N3D 拓扑产物都走同一套渲染逻辑，模块里**不存在任何「形状类型」概念或分支**。
- **零新依赖**：只用 `torch` / `numpy` 与 Python 标准库（`tkinter` 属标准库）。没有 matplotlib / plotly / pyvista / tkinterdnd2，`requirements.txt` 无任何改动。
- **自包含**：只读取 checkpoint 文件的 `state_dict`，**不 import** `n3d_sphere` / `n3d_proto` / `n3d_shape` 的任何代码；这些模块的源码与产物零改动。
- **单实现**：GUI 与 CLI 共用 `core` 层同一套逻辑，模块内不存在第二份绘图实现。
- **产物不撞名**：文件名由 checkpoint 名派生（`viz_<ckpt名>.html/.ply/.obj`），同名已存在时明确提示，不静默覆盖。

### 边界（必须符合 N3D 拓扑 schema）

「几何零假设」指的是**不假设几何长什么样**，而不是放弃 schema 校验：产物仍须含
`core.REQUIRED_KEYS` 全部 12 个拓扑键且形状符合契约（`neuron_pos [N,3]`、
`edge_src/edge_dst [E]`、`level_node_reach [K,2]` 等）。不符合者明确报错并退出**码 3**，
不静默降级、不臆造图形。因此一期 `n3d_proto` 产物（只有突触级 `edge_index`）仍然退出码 3。

### 「层」的口径（不是几何切片假设）

**层 = 同时计算的神经元分组**，由 `level_node_reach` 给出的 `topo_index` 半开区间定义
（与 `n3d_sphere` 的 `level_node_reach` 同语义）。层参考平面取该层神经元沿流向轴位置的
**均值**（`layer_z = mean(zs)`）——因为「层」本身就由该分组定义，该平面必然包含这一组神经元。
该口径与几何是否规整无关：非均匀分层、每层只有 1 个神经元、K=1 单层都同样成立。
**不做最小二乘拟合平面，也不做「层结构退化」特判。**

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

# 非「二期默认产物」的用法：任意符合 N3D 拓扑 schema 的产物同一套逻辑渲染
# 下例是 K=15 的非均匀分层异构几何产物（走「K > 9 按均匀色相扩展」的层色板）
python -m n3d_viz -c checkpoints/n3d_shape/full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt

# 两端全连接包裹（config.fc_dim != 0）：额外展示两片 H 单元面板 + 边界块 + 抽样连线
# 下例实测 fc_dim=-1（宽度跟随 N）-> H=825、抽样 3,510 条（详见 §2.5）
python -m n3d_viz -c checkpoints/n3d_shape/full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.pt
python -m n3d_viz -c <fc产物>.pt --fc-top-k 5     # 抽样口径 k（范围 1..8，默认 3）
```

| 参数 | 说明 |
|---|---|
| `--checkpoint`, `-c` | 符合 N3D 拓扑 schema 的训练产物 `.pt` 路径；**省略时启动 GUI** |
| `--out-dir`, `-d` | 输出目录，不存在则创建（默认 `checkpoints/n3d_viz`） |
| `--out`, `-o` | 显式 HTML 路径；PLY / OBJ 与其同目录同名 |
| `--threshold`, `-t` | HTML 初始边权重阈值（默认 0.30） |
| `--no-plan-planes` | 默认不显示层参考平面（写入负载 `meta.showPlanes`，渲染器据此初始化复选框与绘制状态） |
| `--ply-ascii` | PLY 用 ascii 写出（默认 `binary_little_endian`） |
| `--with-ply-edges` | PLY 中附加 `edge` 元素（顶点索引对 + 权重），已经 CLI 透传至 `write_ply` |
| `--fc-top-k`, `-k` | **两端全连接包裹的抽样口径 k**（默认 `3`，合法范围 `1..8`）；仅 `fc_dim != 0` 的产物生效。非法值一律报错退出非 0，**不做静默截断** |
| `--quiet`, `-q` | 只输出最终摘要 |

退出码：`0` 成功 / `2` 参数错误或无法启动 GUI / `3` checkpoint 相关错误（路径不存在、文件损坏、非二期产物缺拓扑键、`fc_dim != 0` 但缺 FC 键）。

### 1.2 图形界面（GUI）

```bash
python -m n3d_viz          # 不带参数即弹出窗口
python n3d_viz/gui.py      # 等价入口
```

窗口包含：

- `.pt` 文件输入框 + **浏览…**（`filedialog.askopenfilename`）
- 输出文件夹输入框 + **浏览…**（`filedialog.askdirectory`）
- 边权重阈值滑块 + **默认显示层平面** 勾选框
- **全连接层抽样 k** 输入（`ttk.Spinbox`，`from_=1, to=8`，默认 `3`）；仅 `fc_dim != 0` 的产物生效
- **开始生成** 按钮、状态提示文字、可滚动日志区
- **打开 HTML**（默认浏览器）与 **打开输出文件夹** 按钮

输入框默认值遵循「**二期默认产物存在则用它，否则留空**」（`gui.default_checkpoint_text()`）：
`checkpoints/n3d_sphere/model.pt` 存在时预填该路径，不存在时留空由用户自行选择，
不臆造路径、也不绑定任何具体模块产物——任意符合 N3D 拓扑 schema 的 `.pt` 都可选。

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

### 层配色随 K 自动扩展（K > 9 不再循环复用）

**修复前实测的缺陷**：层色表只有 **9 个颜色**且用 `k % 9` 循环取用，于是 `K > 9` 时
第 9 层与第 0 层同色、分层着色失去可分辨性。实测来源
`checkpoints/n3d_shape/full_shapecylinder_a2_*.pt`（seed=42，非均匀分层，K=15）：
产物层色**去重数只有 9**（15 个层里 6 个与前面的层重复）。

**修复口径**（`core.layer_palette_hex` / `core.layer_palette_rgb`，HTML 与 PLY **同源**）：

| K | 色板来源 | 效果 |
|---|---|---|
| `K <= 9` | 既有 `LEVEL_PALETTE_BASE` 的**前 K 个** | **逐字节不变**（既有产物零回归） |
| `K > 9` | 按**均匀色相**生成 K 个颜色（标准库 `colorsys` 做 HSV→RGB） | 去重数恒等于 K |

K=15（cylinder λ=2，seed=42）修复后实测层色 15 个两两不同：

```
#f25c5c #f2985c #f2d45c #d4f25c #98f25c #5cf25c #5cf298 #5cf2d4
#5cd4f2 #5c98f2 #5c5cf2 #985cf2 #d45cf2 #f25cd4 #f25c98
```

`verify_viz.py` 对此有三组断言：`[2a]` 对 `K ∈ [1, 64]` 逐个检查（去重数 == K，且
`K <= 9` 与既有 9 色逐字节相同），`[2a]` 另用 `K = 1536` 覆盖**撞色回退分支**（见下），
`[2c]` 在真实异构产物上检查（K=15 产物去重数 == 15）。

#### 撞色回退分支的覆盖口径（K ≤ 64 区间不可达）

`core._hue_palette_rgb` 里除了纯色相扩展，还有一段**确定性撞色回退**：候选色若已被占用，
就沿「色相微移 (`_PALETTE_HUE_STEP`) + 明度微降」的固定序列取下一个候选，上限
`_PALETTE_SEARCH_LIMIT = 4096` 次，超出则抛 `ValueError`。

实测：在断言覆盖的 `K ∈ [10, 64]` 区间内，纯色相扩展的**撞色数为 0** —— 即
`candidate not in used` 恒为真、`step` 永远停在 0，**该分支不可达**。这属于「防了但没人守」
的盲区，故新增断言
`[2a] _hue_palette_rgb 撞色回退分支（必撞色构造）`：用 `K = 1536`（色相间隔 1/1536，
小于 8 位量化步长）**主动构造必然撞色**，三重把守 + 实测值：

| 把守点 | 实测（K=1536） |
|---|---|
| 前提成立：纯色相扩展撞色数 > 0 | 撞色 **636** 个 |
| 回退真的被执行：`_hsv_to_rgb_bytes` 调用次数 > K | 候选调用 **2,172** 次（回退 **636** 次，无回退时应恰好 == 1536） |
| 回退结果正确：最终去重数 == K（含公开入口 `layer_palette_hex(K)`） | **1536 == 1536** |

**承重证明**：第二项用调用计数直接证明分支被进入（若撞色消失则断言立即失败，杜绝「空转绿灯」）；
第一项若前提不成立也直接报错；第三项在回退写错（例如误用 `candidate in used` 分支、或跳过
未占用检查）时必然产出重复色而失败。

### 非规整几何实测（几何零假设的证据）

`verify_viz.py` 的 `[2b]` 组**就地构造**几何完全不规整但符合 schema 的 state_dict
（不依赖任何既有产物，因此永久有效），逐类断言顶点数、`l` 行数、层数、层色去重数、
坐标逐位一致、payload 长度自洽、产物名派生（8 项/样本，共 5 类）。实测
（`SYNTH_SEED = 20250925`，合成 checkpoint 与产物跑完即删）：

| 合成样本 | 破除的几何假设 | N | E | K | 顶点数 | `l` 行数 | 层色去重数 | 坐标 max\|diff\| |
|---|---|---|---|---|---|---|---|---|
| `random_cloud` 随机均匀点云 + 随机边 | 不假设晶格 / FCC | 120 | 200 | 6 | 120 ✓ | 200 ✓ | 6 ✓ | 0.000e+00 |
| `helix` 螺旋线点云 | 不假设凸包 / 中心对称 | 96 | 150 | 8 | 96 ✓ | 150 ✓ | 8 ✓ | 0.000e+00 |
| `one_per_layer` 每层仅 1 个神经元 | 不假设层内并行度 | 24 | 40 | 24 | 24 ✓ | 40 ✓ | 24 ✓ | 0.000e+00 |
| `single_layer` K=1 单层 | 不假设多层 | 40 | 60 | 1 | 40 ✓ | 60 ✓ | 1 ✓ | 0.000e+00 |
| `k33` K=33 | 不假设 K ≤ 9 | 64 | 90 | 33 | 64 ✓ | 90 ✓ | 33 ✓ | 0.000e+00 |

`[2c]` 组另用 `checkpoints/n3d_shape/` 的 **5 个真实非球几何产物**作泛化证据
（存在则跑、缺失则明确 SKIP 并计入报告），**只断言泛化不变量，不断言任何形状标签**：

| 真实产物（seed=42） | N | E | K | 层色去重数 | 坐标 max\|diff\| |
|---|---|---|---|---|---|
| `full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt` | 256 | 736 | 9 | 9 ✓ | 0.000e+00 |
| `full_shapecube_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt` | 256 | 713 | 9 | 9 ✓ | 0.000e+00 |
| `full_shapecylinder_a0.5_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt` | 256 | 679 | 5 | 5 ✓ | 0.000e+00 |
| `full_shapecylinder_a1_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt` | 256 | 705 | 9 | 9 ✓ | 0.000e+00 |
| `full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt` | 256 | 717 | **15** | **15 ✓**（修复前 9） | 0.000e+00 |

上述 5 个产物的三件套已用本模块重新渲染到 `checkpoints/n3d_viz/`（覆盖此前同名文件），
产物名由 checkpoint 名派生、六套（含二期默认 `viz_model.*`）两两不撞名。实测字节数：

| 产物（`checkpoints/n3d_viz/`） | HTML | PLY | OBJ |
|---|---|---|---|
| `viz_model.*`（二期默认，零回归锚点） | 88,521 | 4,120 | 15,695 |
| `viz_full_shapesphere_..._s42.*` | 88,634 | 4,177 | 15,752 |
| `viz_full_shapecube_..._s42.*` | 87,755 | 4,175 | 15,888 |
| `viz_full_shapecylinder_a0.5_..._s42.*` | 85,507 | 4,184 | 15,341 |
| `viz_full_shapecylinder_a1_..._s42.*` | 87,173 | 4,182 | 15,561 |
| `viz_full_shapecylinder_a2_..._s42.*` | 88,429 | 4,182 | 15,613 |

渲染器侧同样验证了色相扩展色板的端到端可用性：把 K=15 产物内联的 `viewer.js` 与数据用
Node + DOM 桩真实执行一次，**通过 12 / 失败 0**（图例色块数 == K+3 == **18**）——
与二期产物的 12/12（K+3 == 12）一致。

### 零回归锚点（两层：产物完整性 + 代码回归）

`[2d]` 组按**两层**断言二期默认产物 `viz_model.{html,ply,obj}` 与几何无关化**改动前逐位相同**
（K=9 走既有色表，输出逐字节不变）：

| 层 | 断言 | 守什么 |
|---|---|---|
| 层 1：产物完整性 | 磁盘 `checkpoints/n3d_viz/` 下的三件套存在、SHA256 与字节数 == 锚点常量（6 项） | 交付件未被改动 |
| 层 2：代码回归（**承重**） | 用**共享渲染入口** `core.render_default(ANCHOR_RERENDER_CKPT, …)` 重渲到 `_verify/_anchor_rerender/`，三件套的 SHA256 与字节数同样 == **同一组锚点常量**（6 项），比对后清理并断言无残留（2 项） | `K <= 9` 路径的**代码**行为未被改变 |

锚点常量（HTML / PLY / OBJ 的 SHA256 与字节数）：

```
viz_model.html 15A80EBBF2FD586BFB3C4C41E25F79F0E3E8F0C19D3B60AE22BE3F296EDA518C  88,521 字节
viz_model.ply  9A097D16306160F95C9E15826CB11ABED57C6809F2904660B700573889398801   4,120 字节
viz_model.obj  1F594ECF466E28F751C174A85A8A4E5459A304973F6266B3E11C567C25A09FED  15,695 字节
```

**为什么必须有层 2（反桩拒绝证明，实测留档）**：层 1 只比对磁盘上已有的文件，对代码侧回归
完全无感。把 `core.LEVEL_PALETTE_BASE` 前两项对调（`core.py` 一行改动）后实测：

| 断言组 | 注入缺陷后 | 说明 |
|---|---|---|
| `[2a]` 全部（含 `K <= 9 == 既有 9 色前 K 个`） | **5/5 PASS** | 该条是与同一个常量自比，对「常量本身被改」不敏感 |
| `[2d][磁盘]`（层 1） | **6/6 PASS** | 磁盘产物没动 |
| `[2d][重渲] html`（层 2） | **FAIL**（期望 `15A80EBB…`，实测 `4E147DA5…`） | 承重 |
| `[2d][重渲] ply`（层 2） | **FAIL**（期望 `9A097D16…`，实测 `B7449AB4…`） | 承重 |
| `[2d][重渲] obj`（层 2） | PASS | OBJ 不含颜色，不受影响 |
| `verify_viz.py` 退出码 | **1** | 非 0 |
| `core.py` SHA256 | 注入前 `2770BFCA…` → 注入后 `21B5CCBC…` → 恢复后 `2770BFCA…`（**逐字节相同**） | 恢复无损，恢复后全绿 |

> 结论：硬约束「`K <= 9` 逐字节不变」**有承重断言把守**——去掉该守卫（或改动 K≤9 路径的
> 任何可见行为），`[2d][重渲]` 会直接判失败并使退出码非 0。

### 参数集一致性：锚点语义永不脱耦

锚点的语义是「**CLI 默认形式**的产物」，因此「CLI 默认值」与「锚点重渲参数」必须是同一份定义。
现在二者都收敛到 `core.DEFAULT_WRITE_OPTIONS`（唯一事实来源），并经同一个入口
`core.render_default` 使用：

| 使用方 | 说明 |
|---|---|
| `python -m n3d_viz` **默认路径** | 参数集等于 `DEFAULT_WRITE_OPTIONS` 时，`__main__` 调用 `core.render_default(..., data=data)` |
| `verify_viz.py` `[2d][重渲]` | 同样调用 `core.render_default(ANCHOR_RERENDER_CKPT, out_dir=…)`，**不再手写参数集** |
| `verify_viz.py` `[2e]` | 三条**承重**断言把守三者一致：argparse 默认值映射 == `DEFAULT_WRITE_OPTIONS` == `render_default` 实际实参；CLI 选项表面 == 冻结清单 |

**为什么必须把耦合显式化**：若两边各自手写一份，日后任一默认值变化会导致两种坏结局之一 ——
锚点断言**无故 FAIL**（被误判为回归），或为了让断言变绿而两边一起改、**锚点悄悄漂移成
「另一套默认形式」的产物**。

**拒绝证明（三种注入，实测留档）**：`__main__.py` 源文件 SHA256 在三次注入前均为
`2E015E4EEDB72CF34ABF4732A6B1D3EB80563D904F2DFBF513484CAA8EEB569C`（7,352 字节），
每次恢复后**逐字节相同**，全量自检回到 173/0/0：

| 注入 | 命中条目 | 结果 |
|---|---|---|
| `--no-plan-planes` 的 argparse `default` 改为 `True` | `[2e]` 序号 138 | **FAIL**：`{…,'include_planes': False} != {…,'include_planes': True}`；`verify_viz.py` 退出码 **1**；此时 `[2a]` 5/5、`[2d]` 14/14 仍全 PASS（正说明该类改动只能由 `[2e]` 拦住） |
| `--threshold` 的默认值改为 `0.50` | `[2e]` 序号 138 | **FAIL**：`{'threshold': 0.5, …} != {'threshold': 0.3, …}`；退出码 **1**；`[2d]` 14/14 仍全 PASS |
| 新增一个 CLI 开关 `--dummy-new-flag` | `[2e]` 序号 140 | **FAIL**：选项表面 16 != 15；退出码 **1**（覆盖「新增布尔开关」这一风险场景） |

### 字节级复现口径：必须用**相对路径** `--checkpoint`

HTML 内嵌的 `meta.checkpoint` 记录的是**调用时给出的路径字符串**，因此字节级复核只在
「同一调用形式」下成立。交付件与锚点都是用**相对路径**产出的（如
`checkpoints\n3d_shape\...`）；改用绝对路径重渲会让 HTML **仅因该字段多 14 个字节**
（PLY / OBJ 无路径字段，仍逐字节相同）。实测对照：

| checkpoint | 相对路径 HTML | 绝对路径 HTML | 差 | PLY / OBJ |
|---|---|---|---|---|
| `checkpoints/n3d_sphere/model.pt` | 88,521 | 88,535 | +14 | 逐字节相同 |
| `full_shapesphere_..._s42.pt` | 88,634 | 88,648 | +14 | 逐字节相同 |
| `full_shapecube_..._s42.pt` | 87,755 | 87,769 | +14 | 逐字节相同 |
| `full_shapecylinder_a0.5_..._s42.pt` | 85,507 | 85,521 | +14 | 逐字节相同 |
| `full_shapecylinder_a1_..._s42.pt` | 87,173 | 87,187 | +14 | 逐字节相同 |
| `full_shapecylinder_a2_..._s42.pt` | 88,429 | 88,443 | +14 | 逐字节相同 |

因此 `[2d]` 的层 2 重渲固定使用相对路径常量 `ANCHOR_RERENDER_CKPT`
（见 `verify_viz.py`），不使用命令行传入的 `--checkpoint`。

### 2.5 两端全连接层（`config.fc_dim != 0`）的三维展示

`n3d_shape` 第三轮引入的 `fc_dim` 把 N3D 核心**夹在两个全连接层之间**。本模块据此
额外展示「全连接输入层 / 投影到 S_in / 从 S_out 收集 / 线性输出」这套结构。

#### 触发判定（三态，**只看 `fc_dim != 0`**）

| 产物状态 | 判定 | 行为 |
|---|---|---|
| `config` 无 `fc_dim` 键 | 无 FC | 走既有展示，**不报错**（二期与三期未启用产物实测都是这一形态） |
| `config.fc_dim == 0` | 无 FC | 走既有展示，**不报错** |
| `config.fc_dim != 0` 且 FC 键齐全 | **有 FC** | 展示 FC 层（本节主体） |
| `config.fc_dim != 0` 但缺 FC 键 | **产物损坏/不完整** | **报错并退出非 0**，不静默降级为无 FC 展示 |

**重要**：判定**只用 `fc_dim != 0`**，不依赖「FC 键是否存在」来启用；否则「产物损坏」
会被误判成「无 FC」。`fc_dim = -1` 是**有意义的取值**（有效宽度 `H` 跟随 `N`），
不是「关闭」。`H` 由 FC 张量的**实际形状**推出（`proj_weight` 的列数 /
`fc_out_weight` 的行数 / `fc_in_weight` 的行数 / `config.fc_width` 四者必须一致），
因为 `fc_dim = -1` 时 `config` 里不存在直接等于 `H` 的字段。

FC 键完整性由 `core.FC_REQUIRED_KEYS` 定义：`proj_weight`、`fc_out_weight`、
`fc_in_weight`、`fc_out_bias`。

#### 几何与不重叠判据（硬断言）

```
[输入边界块 784] ──▶ [H 单元面板·输入侧] ──(抽样连线)──▶ S_in 神经元
                                                       │ N3D 核心（既有展示不变）
[输出边界块 10] ◀── [H 单元面板·输出侧] ◀──(抽样连线)── S_out 神经元
```

- **H 单元面板**：两片平面，**垂直于流向轴**（`config.flow_axis`，默认 `z`）；分别置于
  神经元云流向轴跨度 `[lo, hi]` 的**两端外侧**；面板内按 `ceil(sqrt(H))` 列做网格排布
  （`H=825` → `29×29`）；面板在平面内的跨度取云在对应轴上的跨度 ×
  `FC_PANEL_SPAN_RATIO`（`1.0`），使面板与云**同尺度**。
- **间隙** = 云跨度 × `FC_PANEL_GAP_RATIO`（`0.15`）；面板自身在流向轴上的厚度 =
  单元中心间距 × `FC_PANEL_THICKNESS_RATIO`（`0.20`），夹在
  `[FC_PANEL_MIN_THICKNESS, 云跨度 × FC_PANEL_MAX_THICKNESS_RATIO(0.10)]` 之间。
- **不重叠判据（硬断言，构造期执行）**：把面板厚度也算进去，要求
  `面板输入侧区间 ∩ 神经元云区间 = ∅` 且 `神经元云区间 ∩ 面板输出侧区间 = ∅`，
  即 `in_hi < cloud_lo` 且 `cloud_hi < out_lo`。几何参数一旦被改坏会**立即抛错**，
  而不是画出一张重叠的图。厚度下界的作用是让「不相交」不退化成「不接触」。
- **边界块**：784 输入 / 10 输出各一块（块中心 + 三轴尺寸），置于面板**外侧**，
  以**聚合箭头**与面板块相连。
- **面板单元着色**：按该单元的**权重范数**映射（输入侧用 `fc_in_weight` 的行范数、
  输出侧用 `fc_out_weight` 的行范数），所用**配色函数复用既有实现**——
  Python 侧唯一实现在 `export_geometry.weight_color_rgb`，`assets/viewer_fc.js` 的
  `normToRgb` 是同一条公式的 JS 镜像（`r = 0.20+0.75t`、`g = 0.60-0.45t`、
  `b = 0.95-0.85t`，`t = (|w| - lo)/(hi - lo)`；`hi == lo` 时按 `t = 0`）。
  层色仍由 `core.layer_palette_hex` / `layer_palette_rgb` 提供，模块内**不存在第二份色表**。

#### 抽样口径（**抽样显示（每神经元 top-k），非全部连接**）

> **必须显式声明**：图形中的 FC 连线是**抽样显示（每神经元 top-k），非全部连接**。
> 该声明同时写进 **HTML 负载 `meta.fcDeclaration`**、**OBJ 伴随注释**、**PLY 头部注释**
> 与本文档；`verify_viz.py` 的 `[2f]` 组对此有专门断言（`meta 声明存在且含「非全部连接」`、
> `OBJ 伴随说明含「非全部连接」`、`PLY 头部注释含抽样口径与参数量`）。

| 侧别 | 抽取方式 | 默认 `k=3` 实测条数 |
|---|---|---|
| 输入侧 | 对每个 `S_in` 神经元，取 `proj_weight` 该**行**内 `\|w\|` 最大的 top-k | 582 × 3 = **1,746** |
| 输出侧 | 对每个 `S_out` 神经元，取 `fc_out_weight` 该**列**内 `\|w\|` 最大的 top-k | 588 × 3 = **1,764** |
| 合计 | —— | **3,510** |

`k` 由 CLI `--fc-top-k` 与 GUI「全连接层抽样 k」控制，范围 `[1, 8]`，越界报错退出非 0。

**为什么用 top-k 而不是全局阈值**（开工前实测，来源
`checkpoints/n3d_shape/full_shapesphere_N825_..._fc-1_s42_fc_align.pt`，seed=42）：

| 口径 | 保留条数 | 问题 |
|---|---|---|
| `\|w\| >= 0.30` | **1,308**（proj 942 + fc_out 366） | 阈值极敏感，且**静默丢掉整个神经元** |
| `\|w\| >= 0.20` | **12,688**（proj 9,376 + fc_out 3,312） | 阈值稍松即暴涨约 10 倍 |
| top-k（k=3） | 3,510 | **保证每个 S_in / S_out 神经元都至少有一条连线** |

`k=8` 时抽样 (582+588)×8 = **9,360** 条，实测仍可渲染（HTML 仍 < 2MB）。

#### 产物差异（有 FC vs 无 FC）

| 产物 | 无 FC（`fc_dim` 缺失或为 0） | 有 FC（`fc_dim != 0`） |
|---|---|---|
| HTML | 既有形态，**逐字节不变** | 追加内联 FC 叠加渲染器 + 负载含 `"fc"` 段与 FC meta 字段 |
| PLY | 既有形态，**逐字节不变** | `vertex` 元素后追加 `fc_node`（`2×H + 2` 点，带 `uchar kind` 分类标记：0=输入面板 / 1=输出面板 / 2=输入边界块 / 3=输出边界块）与 `fc_edge`（抽样连线）两个元素 |
| OBJ | 既有形态，**逐字节不变** | 追加 `2×H + 2` 个 `v` 行（面板单元 + 边界块中心）+ 抽样 `l` 行，并用 `g` / `o` 分组区分「核心边」（`n3d_viz_core_edges`）与「FC 抽样边」（`n3d_viz_fc_sampled_edges`）；附抽样口径说明 |

**零回归是硬约束**：无 FC 产物的 HTML / PLY / OBJ 必须**逐字节不变**。
由于 `assets/viewer.js` 与 `assets/viewer.html` 都被**逐字内联**进每一份 HTML，
改动它们的任何一个字节都会破坏该约束——因此 FC 渲染被完整隔离在
**独立的** `assets/viewer_fc.js` 中，`core.py` 只在 `data.fc` 非 None 时才把它
内联成**追加的一段脚本块**（追加在 `viewer.js` **之后**：`viewer.js` 是共享相机的
写入方，必须先执行；见 §2.6）。`viewer.js` 在本轮仅**新增 1 行**（`window.__n3d_cam = cam;`）。

该叠加渲染器与基础渲染器**共用同一个相机对象**（见 §2.6「相机单一事实来源」），
画在一块 `pointer-events: none` 的透明叠加 canvas 上，因此旋转 / 缩放 / 平移 / 悬停
由基础渲染器处理、两层始终同步，**3D 交互与既有元素完全一致**。
（注意：该句在「相机单一事实来源」轮之前是**不成立**的 —— 那时叠加层自建相机、
旋转与平移都不同步；这正是 §2.6 修复的缺陷。）

#### 有 FC 产物的实测值（来源与 seed 可追溯）

来源 `checkpoints/n3d_shape/full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.pt`
（**seed=42**，`config.fc_dim = -1`，`config.hidden_dim = 2048` 是预设默认值、**与实测 `H` 不等**）：

| 指标 | 实测值 |
|---|---|
| `N` / `E` / `K` | 825 / 2,588 / 15 |
| `\|S_in\|` / `\|S_out\|` | 582 / 588 |
| `fc_dim` / 有效宽度 `H`（=`fc_width`） | -1 / **825** |
| `fc_in_weight` 形状 / 参数量 | `[825, 784]` / 646,800 |
| `proj_weight` 形状 / 参数量 | `[582, 825]` / **480,150** |
| `fc_out_weight` 形状 / 参数量 | `[825, 588]` / **485,100** |
| 面板点数（构造量） | 2×825 = 1,650（+2 个边界块中心 = PLY 中 1,652） |
| 抽样连线（构造量，k=3） | 3,510 |
| 云流向轴（z）区间 | `[-0.9899, 0.9899]`（跨度 1.9799） |
| 输入面板流向轴**区间** | `[-1.2938, -1.2801]`（中心 -1.2869，厚度 0.01365；与云区间**不相交**） |
| 输出面板流向轴**区间** | `[1.2801, 1.2938]`（中心 1.2869；与云区间**不相交**） |
| 面板网格 | 29 × 29（`ceil(sqrt(825)) = 29`），单元中心间距 0.06827，面板横向跨度 ±0.9558 |
| 输入 / 输出边界块中心 z | -1.5511 / 1.5511（块尺寸 1.1879 × 1.1879 × 0.2772） |
| 三件套字节数（渲染实测） | HTML 591,709 / PLY 81,681 / OBJ 154,372 |
| 测试准确率 | 0.9853 |

**口径区分（重要）**：

1. **「区间」与「中心」是两个不同的量**，引用时必须写清。面板在流向轴上的位置有两种口径：
   `panels[].flow`（**中心**坐标，单点）与 `panels[].flow_interval`（**区间**，即
   `[中心 − 厚度/2, 中心 + 厚度/2]`）。不重叠判据用的是**区间**；
   本文档表格中凡写「区间」的均是 `flow_interval`。实测中心为 `∓1.2869`、区间为
   `[-1.2938, -1.2801]` / `[1.2801, 1.2938]`，两者不可混引。
2. **「参数量」与「抽样条数」是两个不同的量**：上表中「参数量」是
   `proj_weight` / `fc_out_weight` 的**全部**元素数（480,150 + 485,100 = 965,250），
   「抽样连线」是**渲染出来的**条数（3,510）——产物 meta、CLI 日志、OBJ 注释与本节
   都分别标注，**不得混用**。

#### 真实浏览器视觉验证与四个「只有真跑浏览器才会暴露」的缺陷（实测留档）

`checkpoints/n3d_viz/_verify/viewer_fc_screenshot.png`（547,213 字节）是在 Chromium
（`bun x playwright screenshot`，一次性图形验证工具，未引入任何依赖）中以 1440×900 打开
FC 产物得到的：可见**两片 29×29 面板**（色彩 = 单元权重范数）、输入 784 / 输出 10
两个边界块（线框方盒 + 标签）、以及从面板到 S_in / S_out 的抽样连线；右侧是独立的
FC 统计面板，底部图例含 FC 条目，左上工具栏有「全连接层（抽样）」开关。

**这张图不是装饰 —— 它暴露了 4 个 Node/DOM 桩测不出来的缺陷**（桩只验证「函数被调用」，
不验证「画到了可见区域」）：

| # | 缺陷 | 症状 | 根因与修复 |
|---|---|---|---|
| 1 | 叠加层 `z-index` 不够 | 改动 FC 绘制颜色后**截图字节数完全不变** → FC 一个像素都看不见 | `#view` 是 `position:absolute; inset:0` 且 z-index 为 auto；叠加层 `z-index:5` 仍排在它**之后**（同层叠上下文内定位元素并不必然压过未设 z-index 的定位元素）。改为 `z-index:20` |
| 2 | `draw()` 里的**正反馈环** | 面板被缩到几乎不可见（实测叠加层 `cam.dist` 由正确的 6.7565 涨到 12.2015 = 滚轮 clamp 上限 `radius*40`） | 原实现每帧「用上一帧的 dist 反推用户缩放倍率」：第 1 帧把 dist 归一到 `radiusWithFc*3.6`，第 2 帧又把它当「已缩放的距离」再乘一次 `radiusWithFc/radius`，逐帧放大。**当时**改为自带 `userZoom` 倍率（默认 1），滚轮/重置按钮通过捕获阶段事件同步 —— 该机制已在 §2.6 连同「自建相机」一并**删除**（现改为每帧从共享相机 `window.__n3d_cam` 直读，`userZoom` / `ZOOM_FACTOR` 在现行代码中**不存在**）。本行只是缺陷 1–4 的历史留档，现行口径以 §2.6 为准 |
| 3 | 面板相对云太小 | 面板只占云横向尺寸的约 40%，在 825 神经元 + 2,588 边的视图里读不出「两端包裹」 | 新增 `FC_PANEL_SPAN_RATIO = 1.0`：单元间距改为「面板目标跨度 / 网格边长」，使面板与云同尺度 |
| 4 | 抽样连线密度压过面板 | alpha 0.95 时输出侧面板完全糊成一片青蓝雾，网格结构不可辨 | 连线降为 `rgba(150,240,255,0.16)`、线宽 0.6px、虚线间隔拉大；面板单元改为**不透明**填充 + 亮边框并按 0.88 收缩留缝（29×29 等距实心会糊成一块色板） |

第 1、2 条是**功能性缺陷**（FC 结构在真实浏览器里根本看不见），由「截图字节数异常不变」
与「在页面内注入诊断盒读回叠加层的 `cam.dist`」两条独立手段定位；修复后全量自检与
渲染器冒烟重跑均通过。

#### 独立测试（离朱 R22）暴露的 3 类契约缺陷与修复

离朱以 731 项独立断言复核后，报告 **3 类真实缺陷**（均属「文档/规范声明了明确报错，
实际是未包装异常或静默截断」，**无功能性错误、无产物错误**）：

| 优先级 | 缺陷 | 实测症状 | 修复 |
|---|---|---|---|
| P1 | `core.validate_fc_top_k` 对**非整数浮点静默截断** | 原实现 `value = int(k)`，于是 `2.5→2`、`1.5→1`、`8.7→8`、`3.0→3`、`True→1` 全部被**接受**，与「非整数一律报错、不静默截断」相悖（CLI 不受影响，argparse `type=int` 已在入口拦下；受影响的是 Python API 调用方） | 改为**严格要求 `int` 且非 `bool`**：`bool` / `float` / `str` / 容器一律抛可读 `ValueError`（含 `3.0` 这类「值非法但类型是 float」的输入，不静默强转） |
| P2 | 同函数 `validate_fc_top_k(None)` 抛**裸 `TypeError`** | docstring 只声明 `ValueError`，按契约只捕获 `ValueError` 的调用方（CLI / GUI）会**漏接** | `None` 与其它非法类型统一抛 `ValueError` |
| P3 | `core._fc_panel_geometry(H<=0)` **逃逸未包装异常** | 在 `_place_units_in_panel` 的 `cell <= 0` 哨兵之前先算 `max(cols, rows)` / `math.sqrt(H)`，于是 `H=0` 抛裸 `ZeroDivisionError`、`H=-1` 抛裸 `ValueError: math domain error`。**主链路不受影响**（`extract_fc` 经 `_fc_hidden_width` 已用 `H<=0` 兜住），仅直接调用该几何函数时违约 | 把 `H` 为正整数、流向轴合法等校验**提到任何算术之前**，与既有的「云跨度为 0 / 面板相交」统一为 `CheckpointSchemaError` |

**回归防线**：`verify_viz.py` 的 `[2f]` 组新增 5 条承重断言
（`validate_fc_top_k` 拒绝 13 类非法输入且异常类型集合恒为 `{"ValueError"}`、
FC 几何入口对 `H=0` / `H=-1` / `H=True` / 流向轴非法 / 云跨度为 0 全部给可读
`CheckpointSchemaError`、且 `H=1` 与共线点云这类**合法边界仍不得误报**）。
修复后实测：`verify_viz.py` **通过 490 / 失败 0 / 跳过 1**，退出码 0
（**相机单一事实来源轮之后为 517 / 0 / 1**，见 §2.6 / §2.7）；
无 FC 零回归、FC 三件套**逐字节不变**（HTML 591,709 / PLY 81,681 / OBJ 154,372）。

### 2.6 相机单一事实来源（`window.__n3d_cam`）

**缺陷（用户报告）**：旋转神经元时输入/输出全连接层**不跟随**。

**根因（已复现确认）**：`assets/viewer.js` 的相机是 IIFE 私有变量 `var cam = {...}`，
**未挂到全局**；而 `assets/viewer_fc.js` 自建了一份等价相机，只同步了画布尺寸、滚轮与
重置，**没有任何 mousedown/mousemove 监听** → 其 `yaw/pitch` 恒为初值。
结果：**左键旋转、右键平移、重置均不同步**（仅滚轮缩放同步）。
真实浏览器实测（Playwright + Chromium，1000×700）：FC 产物左键拖拽旋转后，
叠加层非透明像素 bbox 由 `[356,680,188,481]` 到 `[356,680,188,481]` **完全不变**
（像素数 59,548 → 58,651 的微小变化来自 FC 统计面板重绘），而基础画布持续重绘
—— 缺陷形态确认。

**修复（方案甲：相机单一事实来源）**

| 文件 | 改动 |
|---|---|
| `assets/viewer.js` | **仅新增 1 行**：`window.__n3d_cam = cam;`（附行内注释说明用途与只读约定）。把相机对象**本身**挂到全局 —— 必须是**同一对象引用，不得拷贝**。`git diff --numstat` 实测 `1 0` |
| `assets/viewer_fc.js` | 删除自建相机的视图状态演化（自有 `yaw/pitch/dist/panX/panY`）以及为打补丁而存在的 `userZoom` / wheel 监听 / reset 监听；改为**每帧**从 `window.__n3d_cam` 直读（`sharedCam()`），`fcViewMatrix(camera)` / `fcProject(m, camera, …)` 只接收参数、**不写入**共享对象 |

**只读约定（必须遵守）**：`window.__n3d_cam` 的**写入方只有 `assets/viewer.js`**；
`assets/viewer_fc.js` 只读，**不得**写其中任何字段。这样四种交互天然同步，
且**不存在正反馈隐患**（旧实现「每帧用上一帧 `dist` 反推缩放倍率」曾使
`cam.dist` 由正确的 6.7565 涨到 12.2015 = 滚轮 clamp 上限）。

**取不到相机时的处置（不静默画错）**：`viewer_fc.js` 若读不到 `window.__n3d_cam`
（例如被单独加载、或内联脚本顺序被改动），**拒绝绘制**并给出可读告警
（控制台 `console.warn` + 页面内 `#fc-cam-warning` 面板）。告警按
`CAM_WARN_AFTER_FRAMES = 30` **去抖**，且**相机一旦就绪即自动撤销** —— 因为两层是
同一页面内先后执行的两段内联脚本，若顺序被改动，首帧读不到相机属正常时序而非缺陷
（曾因「立刻告警」留下永久误报，由 E2E 抓到并修复）。

**内联脚本顺序**：`core.build_html` 把 FC 块追加在 **`viewer.js` 之后**
（`viewer.js` 是共享相机的写入方，必须先执行）。实测产物块序：
数据块 → `viewer.js` → FC 块。

**块序不变式已是可执行断言（2026-09-28）**：该顺序不再只靠注释约定 ——
`core.build_html` 在返回 HTML 之前**显式断言**「`viewer.js` 的
`window.__n3d_cam = cam;` 出现在 `viewer_fc.js` 的 `var DATA = window.N3D_DATA;` 之前」，
违反即抛 `ValueError`（构建期失败，而不是产出一份两层不同步的 HTML）。
相应地，**上面那个「相机不可用」告警分支在当前产物里结构上不可达**：

| 场景 | 告警分支是否可达 |
|---|---|
| 正常产物（`core.build_html` 产出） | **不可达** —— 块序断言保证 FC 块排在 `viewer.js` 之后，首帧就能读到相机 |
| 手工注入 / 手改拼接顺序把 FC 块移到 `viewer.js` 之前 | **可达** —— `core.build_html` 直接抛 `ValueError`（构建期即被拦下，产物根本不会生成） |
| 把 `assets/viewer_fc.js` 单独加载到页面（不经 `build_html`） | **可达** —— 此时没有任何断言把关，只能靠页面内告警 + E2E 的「无告警面板」断言 |

因此该告警分支是**纵深防御的第二道**（第一道是块序断言）：断言覆盖「本模块自己拼 HTML」，
告警覆盖「脚本被单独加载 / 页面被手改」这类绕过构建路径的场景。

### 2.7 锚点重基线记录（2026-09-27「相机单一事实来源」轮）

零回归口径按用户批准放宽为：**「PLY/OBJ 逐字节不变 + HTML 锚点重新基线（登记旧→新与原因）」**。
原因是 `viewer.js` 新增那一行会使**每一份**产物的 HTML 内联渲染器源码变大，
所有 HTML 锚点必然变化；而几何与写出路径一行未动，**PLY / OBJ 必须逐字节不变**。

**PLY / OBJ：逐字节不变**（4 组 × 2 文件 = **8 项**，用当前代码重渲后逐项比对，实测 **0 失败**）

| 产物组 | PLY 字节 / SHA256 前 16 位 | OBJ 字节 / SHA256 前 16 位 |
|---|---|---|
| 二期 `viz_model` | 4,120 / `9A097D16306160F9` | 15,695 / `1F594ECF466E28F7` |
| 三期 sphere N256（K=9） | 4,177 / `28034930AAC972C7` | 15,752 / `5893E8825BE53C75` |
| 三期 cylinder λ=2（K=15） | 4,182 / `C4ED6E86461E7BCA` | 15,613 / `F650ADE9662AAD9D` |
| FC 产物（H=825） | 81,681 / `25C1008C7C412F0A` | 154,372 / `4E0C2A2F125F7C4F` |

**HTML：旧值 → 新值 + 原因**（登记在 `verify_viz.py` 的 `ANCHOR_REBASE_LOG`）

| 产物 HTML | 旧字节 | 新字节 | 旧 SHA256 → 新 SHA256（前 16 位） | 原因 |
|---|---|---|---|---|
| `viz_model.html` | 88,521 | **88,741** | `15A80EBBF2FD586B` → `A5FEE937B8023B98` | `viewer.js` 新增 1 行 `window.__n3d_cam = cam;` |
| `viz_full_shapesphere_N256_…_s42.html` | 88,634 | **88,854** | `F9535DAD22FA1C01` → `63B9CD4D4CEDB249` | 同上 |
| `viz_full_shapecylinder_a2_…_s42.html` | 88,429 | **88,649** | `FA5D5426C2064D79` → `9606F91E52AC4053` | 同上 |
| `viz_full_shapesphere_N825_…_fc-1_s42_fc_align.html` | 591,709 | **594,926** | `34CDDB922807787D` → `DDFE260E210A271F` | 同上 + `viewer_fc.js` 改读共享相机 + 脚本块顺序调整（FC 块移到 `viewer.js` 之后） |

**锚点组（`ANCHOR_GROUPS`）—— 维护约定：必须同时覆盖 K≤9 与 K>9 两类**

| 组 | checkpoint（相对路径） | 覆盖的层色路径 | HTML / PLY / OBJ 字节 |
|---|---|---|---|
| 二期 | `checkpoints/n3d_sphere/model.pt` | **K=9 → K≤9（取 9 色前 K 个）** | 88,741 / 4,120 / 15,695 |
| 三期 sphere | `…full_shapesphere_N256_…_s42.pt` | **K=9 → K≤9** | 88,854 / 4,177 / 15,752 |
| 三期 cylinder λ=2 | `…full_shapecylinder_a2_N256_…_s42.pt` | **K=15 → K>9（均匀色相扩展）** | 88,649 / 4,182 / 15,613 |

`verify_viz.py` 的 `[2d]` 组按这三组分别做**两层**校验（层 1：磁盘产物 == 常量；
层 2：用当前代码经 `core.render_default` 重渲 == 同一常量），并新增三条承重断言：

1. **锚点组同时覆盖 K≤9 与 K>9** —— 防止日后只剩 K≤9 锚点、让 K>9 色相扩展分支
   （历史上曾退化为只有 9 种颜色）**失去承重覆盖**；
2. **PLY/OBJ 与改动前快照逐项相同** —— 用当前代码重渲后比对（4 组 × 2 文件 = 8 项），
   任何字节变化一律视为**回归**而非重基线；
3. **HTML 锚点变更全部已登记** —— 必须给出旧值 + 原因，且新旧 SHA256 必须不同
   （防止「锚点悄悄漂移」或登记陈旧）。

**未重渲的组（如实说明）**：本次只重渲上述 3 组 + FC 产物，共 4 组。
其余 3 组（`viz_full_shapecube_…`、`viz_full_shapecylinder_a1_…`、
`viz_full_shapecylinder_a0.5_…`）**保持不动**，因此它们仍是**旧渲染器**
（无 `window.__n3d_cam`）的产物；它们的 HTML **不在任何锚点组内、与代码常量无关**，
需要时可用当前 CLI 随时重渲取得一致版本（实测同一 checkpoint 重渲的 PLY / OBJ 逐字节相同）。

---

## 3. 交互能力（HTML 视图）

| 操作 | 效果 |
|---|---|
| 左键拖拽 | 旋转（yaw 绕屏幕竖直轴，pitch 俯仰带限幅） |
| 滚轮 | 缩放（距离按 `exp(Δy·0.0012)` 变化，并夹在 `0.35R ~ 40R`） |
| 右键拖拽 | 平移（视图矩阵平移列叠加） |
| 悬停神经元 | 弹出详情：id、所在层、z 坐标、入度 / 出度、是否 S_in / S_out（有 FC 时 S_in / S_out 额外注明「连接输入/输出全连接层」） |
| 图层开关 | 神经元 / 连接 / 层平面分别显示隐藏；**有 FC 时多一个「全连接层（抽样）」开关**（由叠加渲染器动态注入） |
| 阈值滑块 | 只显示 `\|edge_weight\| ≥` 阈值的边，实时刷新保留边数（不影响 FC 抽样连线） |
| 重置视角 | 恢复初始 yaw / pitch / 距离 / 平移 |

有 FC 时页面额外呈现（详见 §2.5）：两片 H 单元面板（小方块，色 = 权重范数）、
两个边界块（半透明线框方盒 + 标签）、聚合箭头，以及**每神经元 top-k 的抽样连线**
（细虚线，虚线即「抽样显示，非全部连接」的视觉提示）；右侧另有**独立的 FC 统计面板**
（`#fc-stats`）显示 `fc_dim` / `H` / `top-k` / 面板点数 / 两个矩阵形状与参数量 / 抽样条数 / 声明文本。

配色约定：神经元按所在**层**着色（层色板由 `core.layer_palette_hex` 给出，与 PLY 顶点色**同源**：`K <= 9` 用既有 9 色，`K > 9` 按均匀色相扩展为 K 个两两不同的颜色）；**S_in** 以品红 `#ff5ec7` 同心小圆叠加（半径 0.55r），**S_out** 以金黄 `#ffd84d` 同心叠加（仅 S_out 时 0.55r，兼属两者时 0.26r 作为内圈），因此即便被高亮也能看出其所属层色；边按 `|w|` 归一化映射粗细（0.6~3.2 px）与颜色（弱=青蓝 → 强=橙红）；每层绘制一张参考平面并标注 `L1..L{K}`。

### 渲染原理（手写，无第三方库）

`n3d_viz/assets/viewer.js` 用 Canvas 2D 实现三维渲染：

1. **视图变换**：标准轨道相机。世界点 `p` → 相机坐标 `p_cam = R·(p − center) + (0, 0, −dist)`，其中 `R = RotX(pitch)·RotY(yaw)`，`center` 取点云质心。相机位于目标 +Z 方向 `dist` 处朝 −Z 观察，故目标点满足 `z_cam = −dist < 0`。
   > 实现要点：矩阵按**行主序**（`index = row*4+col`）存放。若把 `A[r][k]` 误读成 `a[k*4+r]`，得到的 `(A·B)ᵀ` 会把整个点云翻到相机背后，投影全部被剔除、画面只剩背景。
2. **透视投影**：`x_screen = W/2 + f·x_cam/(−z_cam)`，`y_screen = H/2 − f·y_cam/(−z_cam)`，焦距 `f = max(0.9H, 300)`；`z_cam > −0.001` 的点被剔除（位于相机之后）。
3. **深度排序（画家算法）**：把层平面、边（深度取两端点均值）、神经元（点深度）合并成一个列表，按 `depth = −z_cam` 升序（远 → 近）绘制，后画的覆盖先画的，从而在二维画布上得到正确遮挡关系。

### 渲染器逻辑冒烟（零第三方测试依赖）

`n3d_viz/assets/viewer_smoke.js` 用最小 DOM / Canvas 桩**真实执行** HTML 里内联的 `viewer.js`，断言：
圆形总数 == N + S_in + S_out、绘制线段数 ≥ 阈值内边数、投影 bbox 有限且落在画布附近、投影质心接近画布中心、
阈值滑块联动（阈值 0 时保留全部边）、**S_in 高亮圈数 == n_s_in**、**S_out 高亮圈数 == n_s_out**、悬停命中并填充详情、
图例色块数 == K+3（有 FC 时再 +2）。

**第三个参数给出 FC 叠加渲染器源码时**（`node viewer_smoke.js <viewer.js> <data.json> <viewer_fc.js>`），
额外断言 7 条：叠加画布已创建且尺寸与基础画布同步、面板单元点 == 2×H、抽样连线数 == `meta.fcSampleEdges`
== `(|S_in|+|S_out|)×k`、**面板流向轴区间与神经元云区间不相交**、`meta` 声明含「非全部连接」、
FC 统计面板独立存在且含声明与矩阵形状、叠加层实际产生绘制调用（方块 + 采样线）与标签。
由 `verify_viz.py` 自动调用（需要 `node`，缺失时该条标记 SKIP，不影响其它断言）。

**桩必须比实现更严格（本轮实测教训）**：早期桩的 `document.getElementById` 会「取不到就顺手创建一个」，
而真实 DOM 返回 `null`——于是渲染器里 `if (!document.getElementById("fc-stats"))` 这类**存在性判断永远为假**，
被守卫的代码块被静默跳过，而其他断言仍然全绿（典型的假绿灯，实测 FC 统计面板因此从未被创建）。
现在桩严格实现该语义（已存在才返回），并补齐 `document.body` 与 `document.createTextNode`。

### E2E「两层几何一致性」断言（补上此前漏检的盲区）

**为什么需要**：上一轮 `verify_viz.py` 490 项（相机轮 517 项）与 E2E 40 项**都没有**断言
「基础画布与 FC 叠加画布是否用同一套相机」，因此「FC 图层不跟随旋转」这个缺陷
**全部自检都通过**（渲染器冒烟只验证函数被调用、不验证两层是否同源）。本组断言专门补这个盲区。

脚本：**`n3d_viz/tests/e2e_two_layer_cam.mjs`**（2026-09-28 从 `.lizhu_env/r22_e2e/e2e_two_layer_cam.mjs`
**迁入版本控制** —— `.gitignore:48` 忽略了整个 `.lizhu_env/`，该目录 0 个文件被跟踪，
放在那里等于「唯一的守门测试不在版本控制内」）。Playwright + Chromium headless，
**运行依赖仍是既有那一份**：`playwright` 来自离朱 r22 轮已安装在
**`.lizhu_env/r22_e2e/node_modules`** 的安装（≈17.7 MB、183 个文件），**不是本次新增的运行依赖**，
`requirements.txt` 实测 0 变更行。

**Node 的 ESM 解析要求（接入时必须知道）**：ESM 的**裸导入不做「向上逐级」解析** —— 实测把脚本放在
`n3d_viz/tests/` 后，`node n3d_viz/tests/e2e_two_layer_cam.mjs …` 直接报
`ERR_MODULE_NOT_FOUND: Cannot find package 'playwright'`；`NODE_PATH`、`--preserve-symlinks` 也都无效
（`NODE_PATH` 只对 CJS 生效）。本仓库所在卷**不支持**目录联接（实测 `mklink /J`：
`Local NTFS volumes are required to complete the operation`），因此接入方式为：
把既有的 `.lizhu_env/r22_e2e/node_modules` **复制**到 `n3d_viz/tests/node_modules`
（`node_modules/` 在 `.gitignore` 中，不会入库；`n3d_viz/tests/` 下**不出现在 git 跟踪列表**里），
脚本**一个字符都不用改**（导入语句仍是原样的裸导入）。

```bash
# 0) 一次性接入（仅新克隆/新环境需要；依赖仍是既有 .lizhu_env 里的那一份）
cp -r .lizhu_env/r22_e2e/node_modules n3d_viz/tests/node_modules     # Windows: Copy-Item -Recurse -Force

# 1) 产出两件待测 HTML（默认写入 checkpoints/n3d_viz/，k 取默认 3）
python -m n3d_viz --checkpoint checkpoints/n3d_sphere/model.pt --out-dir checkpoints/n3d_viz --quiet
python -m n3d_viz --checkpoint checkpoints/n3d_shape/full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.pt --out-dir checkpoints/n3d_viz --quiet

# 2) 跑两层几何一致性 E2E（在仓库根执行；实测 通过 26 / 失败 0，退出码 0）
node n3d_viz/tests/e2e_two_layer_cam.mjs \
    checkpoints/n3d_viz/viz_model.html \
    checkpoints/n3d_viz/viz_full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.html
```

无 FC 时 `viewer.js` 也暴露共享相机，故**两件产物都必须传**（第 1–4 项断言专门守无 FC 回归）。
需要 `node` 与上一步的 `n3d_viz/tests/node_modules`；两者缺一时该 E2E **不参与**自动断言
（`verify_viz.py` 的 `[9]` 组只断言「脚本在版本控制内、断言数未被削减、未新增运行依赖」，
**不**自动执行浏览器 E2E）。

| # | 断言 | 口径 |
|---|---|---|
| 1–4 | **无 FC 产物回归** | 不创建 `#view-fc`、无 `#fc-stats`、无相机告警面板；且 `window.__n3d_cam` 仍为 object（`viewer.js` 无 FC 时也暴露相机） |
| 5–7 | 有 FC 产物前置 | 叠加层存在；`window.__n3d_cam` 为 object（两层级联同一引用）；无「相机不可用」告警 |
| 8 | 旋转前基准 | 用**共享相机在页面内复算**某个已知世界点（输入面板首个单元中心）的屏幕位置，叠加层在该处**确有像素** |
| 9–10 | **左键拖拽旋转** | 基础层与**叠加层都发生明显变化**（变化像素数阈值 500 / 200） |
| 11 | 旋转的决定性证据 | 共享相机下该世界点的预测屏幕位置**确实移动**（> 20px）；**旋转后叠加层出现在新预测位置**（跟随旋转） |
| 12 | 旋转的区分性 | 旧位置不再有像素、新位置在旋转前也有像素 —— 两者不得同时为真（否则该网格无法区分位置） |
| 13 | 旋转的相机证据 | 共享相机 `yaw`/`pitch` 确实变化 |
| 14–17 | **右键拖拽平移** | 两层都发生明显变化；两层「**变化像素质心**」的位移方向一致（单位向量点积 **> 0.5**，实测 0.971）；位移幅度比落在同一量级（0.3 ~ 3.0，实测 0.733） |
| 18–21 | **滚轮缩放** | 两层都发生明显变化；共享相机 `dist` 确实变化（实测 3.7404 → 1.6148，两层因此同步）；叠加层非透明像素数随缩放改变（实测 195,685 → 299,909） |
| 22 | 缩放的口径澄清 | 基础画布**恒为满画布**不透明（实测 700,000 == 视口面积），因此**不能**用「非透明像素数」做两层对比指标 —— 这是上一版断言的错误来源，已修正 |
| 23–24 | **点击重置** | 基础层与叠加层都**回到初始取景**（与初始缓冲**逐字节相同**） |
| 25 | 交互不崩 | 四种交互全程**无 JS 报错** |

**实测（真实 Chromium）**：**通过 26 / 失败 0**，退出码 0（迁移后以 `n3d_viz/tests/e2e_two_layer_cam.mjs`
在仓库根重跑复核，脚本断言逻辑与断言数均未改动）。
**方向一致性的容差标定**：平移用「变化像素质心」单位向量点积 > 0.5（实测 0.971，余量充足）；
旋转**不**用该口径 —— 两层渲染的是不同世界特征（神经元云 vs 端部面板），旋转时它们的
质心位移方向本就不同（实测点积可达 −0.78），改用上面的「预测位置命中」决定性口径。

**拒绝证明（两种注入，实测留档）**：

| 注入 | 结果 |
|---|---|
| 注释掉 `viewer.js` 的 `window.__n3d_cam = cam;`（模拟「忘记暴露」） | E2E **14 项 FAIL / 12 项 PASS**（共享相机 `typeof=undefined`、叠加层**0 像素**、四类交互断言全失败、告警面板出现）、退码 1，**且能正常打印汇总行** |
| 把 `viewer_fc.js` 改回「自建相机副本（yaw/pitch 恒定、不跟随旋转）」的旧形态（**精确复现原缺陷**） | E2E **2 项 FAIL**：`旋转：叠加层也发生了明显变化` 实测**变化像素 0**；`旋转后：旧位置已不再是…` 实测旧位置仍有像素且新位置旋转前也有像素、退码 1 |
| 恢复后 | `viewer.js` / `viewer_fc.js` SHA256 **逐字节回到注入前**，`node --check` 退出码 0，E2E 回到 26/0 |

**块序不变式的拒绝证明（2026-09-28 收口轮，实测留档）**：把 `core.build_html` 里那一次替换
从 `js + fc_block`（现行）改为 `fc_block + js`（2026-09-27 修正之前的顺序），
`core.py` SHA256 `35359CDA…` → `0327C9CD…`：

| 观测点 | 注入后实测 |
|---|---|
| `core.build_html` | 抛 `ValueError`：`块序不变式被破坏：FC 叠加渲染器（viewer_fc.js）必须排在 viewer.js **之后**（实测探针位置 viewer.js@571362 > FC@547944）` —— **产物根本生成不出来** |
| `[2g]` 产物侧探针断言 | FAIL：`viewer.js 探针 @27838 未排在 FC 探针 @4420 之前（FC 叠加层会读不到共享相机）` |
| 全量 `verify_viz.py` | 大批 FAIL（每个 FC 产物在 `[2c]` 的 8 项泛化不变量上都报同一条 `ValueError`，`[2d]` 的 PLY/OBJ 重渲亦 FAIL） |
| 恢复后 | `core.py` SHA256 **逐字节回到 `35359CDA…`**，`python -m compileall -q n3d_viz` 退出码 0 |

> 该实验只在**一次命令内**改一行、随即还原（见本节「验收命令」），不留任何中间产物。

> **脚本健壮性（离朱 R24 的 F1，已修）**：上表第一种注入下，脚本原先会在格式化
> 「预测屏幕位置」时抛 `TypeError: Cannot read properties of null (reading 'toFixed')`
> **崩溃且不打印汇总行** —— 因为点被相机剔除时 `projectWorldPoint` 返回
> `{visible:false}`（**不含** x/y 字段）。现统一用 `fmt()`（`Number.isFinite` 守卫）
> 格式化，并统一布尔化断言条件；修复后该场景**正常打印 12/14 汇总行**、退码 1。
> 更强的判别由**反证 A/B** 提供：离朱用修复前源码重建旧产物后同一条拖拽，
> 叠加层变化 **0 像素**，当前实现 **264,115 像素**。

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
├── core.py              # checkpoint 加载 / 拓扑键校验 / 拓扑抽取 / 产物命名派生 / 层配色 / HTML 数据负载
│                        # + 两端全连接包裹（fc_dim != 0）：三态判定 / FC 抽取 / 面板几何与不重叠硬断言 / top-k 抽样
├── export_geometry.py   # 零依赖 PLY（二进制/ascii）与 OBJ 线框写出器 + 回读解析器（层色取自 core）
│                        # + FC 附加点 / FC 抽样边 / 独立 group 名 / FC 元素回读解析器
├── render_html.py       # 单文件 HTML 生成 + 自包含断言
├── gui.py               # tkinter 界面（后台线程 + queue + after 轮询；全连接层抽样 k 控件）
├── __main__.py          # CLI 入口；不带参数启动 GUI
├── verify_viz.py        # 全部硬断言验证脚本（真实执行，末尾汇总退出码）
├── assets/
│   ├── viewer.html      # HTML 模板（含两个占位标记）
│   ├── viewer.js        # 三维交互渲染器（构建时内联进产物）
│   ├── viewer_fc.js     # 两端全连接包裹的**叠加渲染器**（仅在 fc_dim != 0 时内联）
│   └── viewer_smoke.js  # 渲染器逻辑冒烟脚本（Node + DOM 桩）
├── tests/
│   └── e2e_two_layer_cam.mjs  # 两层几何一致性 E2E（Playwright；26 项断言；**在版本控制内**）
└── README.md
```

构建时 `render_html` 读取 `assets/viewer.html` 与 `assets/viewer.js`，把 `/*__N3D_DATA_JSON__*/` 替换为 JSON 负载、`/*__N3D_VIEWER_JS__*/` 替换为渲染器源码，因此产物不含任何外部引用。

**为什么 FC 渲染器是独立的第三个资源文件**：`viewer.html` 与 `viewer.js` 都被**逐字内联**进每一份 HTML，
改动它们任何一个字节都会破坏「无 FC 产物逐字节零回归」的硬约束。因此 FC 渲染被完整隔离在
`assets/viewer_fc.js` 中，只在 `data.fc` 非 None 时作为**追加的一段脚本块**内联进产物
（`core.build_html` 把它追加在 **`viewer.js` 之后**，见 §2.6 与 core.py 的块序断言）。无 FC 产物因此连一个字节都不变。

---

## 5. 验收标准与断言口径

```bash
python n3d_viz/verify_viz.py --report checkpoints/n3d_viz/_verify/verify_report.md
# 实测：通过 173 / 失败 0 / 跳过 0，退出码 0
```

断言项数构成：几何无关化前的 **77 项**（其中二期基准组含写死的 `EXPECTED_N=256` /
`EXPECTED_E=736` / `EXPECTED_LAYER_COUNTS=[13,24,37,35,39,34,37,24,13]`，作为原验收口径不变）
\+ 几何无关化轮 **81 项** + 修复轮 **12 项** + 收尾轮 **3 项** = 既有 **173 项**；
本轮在全连接层支持下 `[2c]` 由 32 → 261 项（**+229**）、新增 `[2f]` **88 项**
（其中 5 项为离朱独立测试修复轮补入的契约回归防线），
故全连接层支持轮汇总为 **485 项**，再经离朱独立测试修复轮 **+5 项**（[2f] 契约回归防线）→ **490 项**，
再经**相机单一事实来源轮 +27 项**：`[2d]` 由单一锚点的 14 项改为**多锚点组**结构
（3 组 × 「磁盘 6 项 + 重渲 6 项」+ 清理 2 项 = 38 项）+ 新增 3 条承重断言
（锚点组覆盖 K≤9 与 K>9 / PLY·OBJ 逐字节不变 / HTML 重基线登记齐备）→ **517 项**，
再经**收口轮 +5 项**：新增 `[2g]` 3 项（块序不变式的产物侧探针顺序 / 注入拒绝证明 /
**活断言证明**）+ `[9]` 2 项（E2E 在版本控制内且 26 项断言未削减 / 迁入未新增运行依赖）
→ **现行 522 项**（实测通过 **522** / 失败 **0** / 跳过 **1**，退出码 **0**；
唯一 SKIP 是 `[2c]` 的「非 N3D 拓扑产物已明确跳过（逐个计入报告）」说明行）。
新增项全部**从 checkpoint 读实际 N / E / K / 层规模，不写死任何值**；既有构成：`[2a]` 5 项
（层配色可扩展性 + 撞色回退分支）+ `[2b]` 42 项（5 类非规整几何合成样本 × 8 项 +
2 项临时产物清理）+ `[2d]` 41 项（3 锚点组 × 12 项 + 清理 2 项 + 3 条结构/登记承重断言）
+ `[2e]` 3 项（CLI 默认形式 / `render_default` 实际参数 / CLI 选项表面 三者的一致性）。

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
| `n3d_viz` 不 import 业务模块 | 源码正则扫描 `^\s*(from\|import) n3d_(sphere\|proto\|shape)` | 0 命中 |
| 无第三方依赖 | 扫描全部顶层 import 与 `requirements.txt` | 仅 `torch` / `numpy` / 标准库 |
| GUI 冒烟 | `gui` 可导入，`withdraw()` 状态下构造并销毁 Tk 窗口（不进 mainloop） | 构造成功，顶层子控件 7 个 |
| 开关类参数产物差异 | 每个布尔开关开/关两态产物必须不同 | `include_planes` True/False 产物不同；`--no-plan-planes` 产物 `showPlanes == false`；`--with-ply-edges` 产物 PLY 含 736 条 `element edge` |
| 渲染器逻辑 | Node + DOM 桩执行内联 `viewer.js` | 通过 12 / 失败 0（断言清单序号 141） |
| 负例临时产物清理 | 负例跑完 `_bad_index/` 无文件残留、残留体积 == 0 字节 | 残留 0 个文件 / 0 字节（清理前曾残留 4 个共 ~69MB 的 `bad_*.pt`） |
| `[2a]` 层色去重数 | `K ∈ [1,64]` 逐个检查去重数 == K | 64 个 K 全部通过 |
| `[2a]` K ≤ 9 色板 | == 既有 9 色前 K 个（逐字节不变） | `K=0..9` 全部通过 |
| `[2a]` HTML/PLY 层色同源 | hex→RGB 与 RGB 色板逐一相等 | `K ∈ {1,9,10,15,33,64}` 全部通过 |
| `[2a]` `hex_to_rgb` 输入契约 | 非法文本抛 `ValueError`、分量恒 ∈ [0,255] | 正向 6 项、反向 15 项（含 `#-12345` / `#4e8c f` / `##4e8cff`）、`K=1..64` 分量全部通过 |
| `[2a]` 撞色回退分支 | `K=1536` 主动构造必然撞色：前提成立 + 回退真的执行 + 去重数 == K | 撞色 636 个；候选调用 2,172 次（回退 636 次）；去重数 1536 == 1536 |
| `[2b]` 非规整几何合成组 | 5 类样本 × 8 项（顶点数 / `l` 行数 / 层数 / HTML 层色去重数 / PLY 层色去重数 / 坐标逐位一致 / payload 长度自洽 / 产物名派生） | 40 项全部通过 + 2 项临时产物清理 |
| `[2c]` 真实异构几何组 | 5 个产物 × 6 项泛化不变量（**不断言形状标签**） | 30 项全部通过 + 产物名两两不同 + K=15 去重数 == 15 |
| `[2d]` 零回归锚点（**多锚点组 × 两层**） | 3 个锚点组（K=9 / K=9 / **K=15**）各做「层 1 磁盘产物完整性 6 项 + 层 2 **用共享入口 `core.render_default` 重渲**并与同一组锚点常量比对 6 项」+ 清理 2 项；另加 3 条承重断言：锚点组**同时覆盖 K≤9 与 K>9**、**PLY/OBJ 与改动前快照逐项相同**（4 组 × 2 文件，用当前代码重渲后比对）、**HTML 锚点变更全部已登记**（旧值 + 原因） | **41 项全部通过**；实测锚点组 3 个（K≤9 2 组 / K>9 1 组）、PLY/OBJ 8 项重渲后逐项相同、3 条 HTML 重基线登记齐备；注入 `LEVEL_PALETTE_BASE` 对调缺陷后层 2 的 html/ply 两条 FAIL、退出码 1（见 §2「零回归锚点」）；**相机轮重基线后 3 组 HTML 全部换新值、PLY/OBJ 不变**（见 §2.7） |
| `[2e]` 参数集一致性 | CLI 默认形式 == `core.DEFAULT_WRITE_OPTIONS` == `render_default` 实际参数；CLI 选项表面 == 冻结清单 | 3 项全部通过（`{'threshold':0.3,'ply_binary':True,'with_ply_edges':False,'include_planes':True,'fc_top_k':3}`；17 个 option string）；三种注入（`--no-plan-planes` 默认改 True / `--threshold` 默认改 0.5 / 新增一个 CLI 开关）均使对应条目 FAIL、退出码 1，恢复后源文件 SHA256 逐字节相同（见 §2「参数集一致性」） |
| `[2f]` 触发判定三态 | 无 `fc_dim` 键 / `fc_dim == 0` → 无 FC 段、不报错；`fc_dim != 0` 且键齐全 → 有 FC 段 | 三态全部通过；两态无 FC 产物的 HTML / PLY / OBJ **两两一致**（归一化掉 ckpt 文件名后逐字节比对） |
| `[2f]` 面板点数 == 2×H | PLY `fc_node` 中的面板单元 / 内联负载 `panels.units` 之和两条独立口径 | 1,650 == 2×825 |
| `[2f]` 抽样条数 | == `\|S_in\|×k + \|S_out\|×k`（负载 / OBJ group / PLY `fc_edge` 三条独立口径） | 582×3 + 588×3 = **3,510** |
| `[2f]` 抽样覆盖性 | 每个 S_in / S_out 神经元都至少有一条连线 | 582 / 582 与 588 / 588 全部覆盖（top-k 口径的保证） |
| `[2f]` 抽样权重正确性 | 独立回读 `proj_weight` / `fc_out_weight`，按 `(side, unit, neuron)` 直接取矩阵元素比对 | max\|diff\| ≤ 1e-6 |
| `[2f]` 面板与云不相交 | 面板流向轴区间 ∩ 云流向轴区间 == ∅（内存结构 + **落盘 PLY 坐标**两条口径） | 云 `[-0.9899, 0.9899]`；输入面板 **区间** `[-1.2938, -1.2801]`（中心 -1.2869）、输出 `[1.2801, 1.2938]`（中心 1.2869） |
| `[2f]` 间隙与网格 | 间隙 == 云跨度 × 0.15；列数 == `ceil(sqrt(H))` | 误差 < 1e-6；29 == `ceil(sqrt(825))` |
| `[2f]` 抽样声明 | `meta` / OBJ / PLY 三处都必须含「非全部连接」并写出两侧参数量 | 三处全部通过（参数量 480,150 + 485,100 分别标注） |
| `[2f]` PLY / OBJ FC 元素 | PLY 含 `fc_node`（2×H+2）+ `fc_edge`；OBJ 两个 group 名正确且行数分派正确 | `fc_node` 1,652 / `fc_edge` 3,510；核心边 2,588 + FC 边 3,510，`v` 行 2,477 == N + 2H + 2 |
| `[2f]` `fc_dim != 0` 缺 FC 键 | 5 个负例（缺 `proj_weight` / `fc_out_weight` / `fc_in_weight` / `fc_out_bias` / 两个同时缺）→ 退出码非 0、报错含缺失键名、无 traceback、**不产生任何产物** | 20 项全部通过（**不静默降级为无 FC 展示**） |
| `[2f]` `--fc-top-k` 边界 | `1` / `8` 合法；`0` / `9` / `-1` / `abc` / `1.5` 非法 → 退出码非 0 且无 traceback | 全部通过；k=1 → 1,170 条、k=8 → 9,360 条，且 k=1 与 k=8 产物**不同**（开关不得是空操作） |
| `[2f]` 临时产物清理 | 负例与三态临时 checkpoint 用完即删 | `_fc/` 下无 `.pt` 残留 |
| `[2c]` 预分拣 | `checkpoints/n3d_shape/*.pt` 中的非 N3D 拓扑产物（MLP 基线）**明确 SKIP 并计入报告**，不产生一堆 `KeyError` | 拓扑产物 9 个 + 非拓扑 18 个 == 总数 27 个 |

`[2c]` 真实异构几何组本轮由 32 项**扩展为 261 项**：`checkpoints/n3d_shape/` 新增了
MLP 基线与 10 个 `fc_align` 产物后，旧口径（「目录里所有 `.pt` 都是 N3D 拓扑产物、OBJ `l` 行数 == E」）
会成片误报。现在改为**预分拣**（`_non_topology_reason`）+ 拓扑产物跑完整泛化不变量，
其中「OBJ 核心边行数 == E」按 group 计数、「OBJ 总行数 == E + FC 抽样数」，
FC 产物另加「面板点数 == 2×H + 2」不变量 —— 因此 **FC 路径同样被真实产物覆盖**。

### 一键复现（口径要点）

```bash
# 1) 全量硬断言（含 [2d] 用当前代码重渲并与锚点常量比对、[2g] 块序不变式、[9] E2E 版本控制登记）
python -m compileall -q n3d_viz
python n3d_viz/verify_viz.py --report checkpoints/n3d_viz/_verify/verify_report.md   # 期望 522/0/1，退出码 0

# 2) JS 语法 + 渲染器逻辑冒烟
node --check n3d_viz/assets/viewer.js
node --check n3d_viz/assets/viewer_fc.js
node --check n3d_viz/assets/viewer_smoke.js

# 3) 两层几何一致性 E2E（26 项断言，在真实 Chromium 中执行）
#    需要先按「E2E 两层几何一致性」小节把既有 node_modules 复制到 n3d_viz/tests/
node --check n3d_viz/tests/e2e_two_layer_cam.mjs
node n3d_viz/tests/e2e_two_layer_cam.mjs \
    checkpoints/n3d_viz/viz_model.html \
    checkpoints/n3d_viz/viz_full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.html
# 实测：通过 26 / 失败 0，退出码 0

# 3) 重新生成某产物（字节级复核时 --checkpoint 必须是**相对仓库根**的路径，理由见 §2）
python -m n3d_viz -c checkpoints/n3d_sphere/model.pt --out-dir checkpoints/n3d_viz/_verify/_repro

# 4) 有 FC 产物的三件套（含面板 / 边界块 / 抽样连线）
python -m n3d_viz -c checkpoints/n3d_shape/full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.pt \
    --out-dir checkpoints/n3d_viz --fc-top-k 3
```

三条口径要点：

1. **必须用相对路径** `--checkpoint`：HTML 内嵌 `meta.checkpoint` 记录调用时的路径字符串，
   绝对路径会让 HTML 多 14 字节（PLY / OBJ 不受影响）——实测对照表见 §2「字节级复现口径」。
2. **`verify_viz.py` 的验证产出一律写入 `--out-dir`（默认 `checkpoints/n3d_viz/_verify/`）**，
   不覆盖 `checkpoints/n3d_viz/` 下的正式交付件；`[2b]`/`[2d]`/`[2f]` 的临时 checkpoint 与产物
   比对完即删（各有「无文件残留 / 残留体积 0 字节」类断言把守）。
3. `[2d]` 的层 2 重渲固定使用常量 `ANCHOR_RERENDER_CKPT`（相对路径），
   **不跟随**命令行传入的 `--checkpoint`，因此换个 ckpt 跑验证也仍会守护 K ≤ 9 路径。

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

**6.1 二期默认产物（`checkpoints/n3d_sphere/model.pt`，零回归锚点）**

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
| HTML / PLY / OBJ 大小 | 88,521 / 4,120 / 15,695 字节（HTML 为相机轮前旧值；现行 88,741 字节，见 §2.7） |
| 产物 SHA256 | HTML `15A80EBBF2FD586B…`（**相机单一事实来源轮前的旧值**；现行锚点为 `A5FEE937B8023B98…`，见 §2.7）/ PLY `9A097D16306160F9…` / OBJ `1F594ECF466E28F7…`（PLY·OBJ 在几何无关化、全连接层支持、相机轮与本收口轮前后**逐位相同**） |
| 验证汇总 | 通过 **522** / 失败 0 / 跳过 1（退出码 0）。按 `verify_report.md` 的**断言清单序号**定位（序号由断言插入顺序决定，新增断言会使后续序号顺移，故**以断言名称为准**）：渲染器逻辑冒烟 = 名称「内联渲染器可执行且投影正确」（内含 13 条子断言，均通过）；`[2f]` 组共 **88** 项；`[2c]` 组共 **261** 项（含 1 条 SKIP 说明）；本收口轮新增 `[2g]` **3** 项 + `[9]` **2** 项 |

**6.2 非规整 / 异构几何产物（`checkpoints/n3d_shape/`，seed=42）**

- **来源 checkpoint**：`checkpoints/n3d_shape/` 下的 5 个产物（sphere / cube / cylinder λ=0.5 / λ=1 / λ=2）
- **seed = 42**（均取自各自产物的 `config.seed`）
- **配置**：`N=256, y_in=y_out=8x8, H=D=0.1, flow_axis=z, placement=fcc, input_scope/readout_scope=any_isolated`
- 产物写入 `checkpoints/n3d_viz/`（覆盖此前同名文件），实测字节数与层色去重数见 §2「非规整几何实测」。

| 指标 | 实测值 |
|---|---|
| N / E / K 范围 | N 均为 256；E ∈ {679, 705, 713, 717, 736}；K ∈ {5, 9, 9, 9, 15} |
| 层色去重数 == K | 5 / 9 / 9 / 9 / **15** 全部满足（K=15 修复前仅 9） |
| 坐标 max\|diff\| | 5 个产物全部 0.000e+00（< 1e-6 容差） |
| 产物名两两不同 | 15 个文件名（5 ckpt × 3 件套）去重后仍为 15 |

**6.3 合成非规整几何样本（`verify_viz.py [2b]`，`SYNTH_SEED = 20250925`）**

- 就地构造、跑完即删，不依赖任何既有产物；N / E / K 与实测结果见 §2「非规整几何实测」表。

**6.4 两端全连接包裹产物（`checkpoints/n3d_shape/…_fc-1_s42_fc_align.pt`，seed=42）**

- **来源 checkpoint**：`checkpoints/n3d_shape/full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.pt`
- **seed = 42**（取自该产物 `config.seed`；记录 `test_acc = 0.9853`）
- **`fc_dim = -1`**（有效宽度 `H` 跟随 `N`，即 `H = 825`）；CLI 渲染命令与参数见 §1.1 / §5「一键复现」
- 全部指标见 §2.5「有 FC 产物的实测值」表（`|S_in|=582` / `|S_out|=588`、
  `proj_weight [582,825]` = 480,150 条、`fc_out_weight [825,588]` = 485,100 条、
  抽样 3,510 条、面板点 1,650、HTML 591,709 / PLY 81,681 / OBJ 154,372 字节；
  **HTML 591,709 是相机轮前旧值，现行 594,926 字节**，见 §2.7）
- 真实浏览器渲染证据：`checkpoints/n3d_viz/_verify/viewer_fc_screenshot.png`（1440×900，
  547,213 字节，含两片面板 / 两个边界块 / 抽样连线 / FC 统计面板；见 §2.5 末节的缺陷留档）

**引用纪律**：

- **边集（E）、边权重、突触位置随 seed 变化**：任何边级数字（E=736、阈值保留边数、`|w|` 统计、层入边数）都必须连同其来源 checkpoint 与 `seed=42` 一起引用，换 seed 后这些数字会变。
- **神经元位置与分层结构由排布参数（`placement`、`H`、`D`、`flow_axis`、`N` 及几何构造方式）决定，与 seed 无关**：N=256、K=9、层规模 13/24/37/35/39/34/37/24/13、层 z 取值属于几何常量，可在不同 seed 的同类产物间复用；但**异构几何产物的 E / K 各不相同**（如 cylinder λ=2 实测 K=15），引用时必须标注具体来源 checkpoint。
- 引用本 README 的数字时，请注明"来源 `<checkpoint 相对路径>`，seed=42"。
- `S_in` / `S_out` 计数（193 / 187）依赖 `input_scope=any_isolated` 与 `readout_scope=any_isolated` 及随机采样，**随 seed 变化**。
- `[2b]` 合成样本的坐标与边由 `SYNTH_SEED = 20250925` 决定，报告中的样本规模（N/E/K）必须与该 seed 一并引用。
- **`fc_dim != 0` 产物的 FC 数字（`H`、`|S_in|`、`|S_out|`、`proj_weight` / `fc_out_weight` 参数量、抽样条数）随 N 与 seed 变化**：引用时同样必须标注来源 checkpoint 与 seed；其中「参数量」与「抽样条数」是两个不同量（前者是矩阵元素数、后者是渲染出来的连线数），**不得混用**（见 §2.5）。

---

## 7. 注意事项

- **张量必须含 N3D 拓扑键**：`REQUIRED_KEYS` 定义在 `n3d_viz/core.py`，缺任一键即抛 `CheckpointSchemaError` 并携带全部缺失键名。「几何零假设」不代表放弃 schema 校验——键名与形状契约仍然强制，不符合即退出码 3、不产生任何产物。
- **不读 `syn_dist`**：`core._syn_dist_bytes` 只记录其字节数，从不把该张量转成 Python 对象。
- **层色随 K 自动扩展**：`K <= 9` 走既有 9 色（输出逐字节不变），`K > 9` 按均匀色相扩展（去重数 == K）。层色的唯一实现是 `core.layer_palette_hex` / `core.layer_palette_rgb`，HTML 与 PLY 同源；**不得在 `viewer.js` 或 `export_geometry.py` 里再写第二份色表**。
- **渲染默认形式的唯一事实来源是 `core.DEFAULT_WRITE_OPTIONS`**：CLI 默认路径与 `[2d]` 锚点重渲都经 `core.render_default` 使用它；改动 CLI 默认值或新增 CLI 开关时，`[2e]` 会立即 FAIL，请同步确认锚点语义（**不得**在 `verify_viz.py` 里重新手写一份参数集）。
- **产物覆盖**：同名产物存在时 CLI 打印提示、GUI 在日志与状态栏提示，verify 脚本打印 `existed` 标记；不做静默覆盖。
- **`verify_viz.py` 默认把产物写到 `checkpoints/n3d_viz/_verify/`**（含 `verify_report.md` 报告）**，避免与正式产物互相污染（对齐"验证类命令一律写入 `_verify/`"的产物纪律）。`[2b]` 合成组的临时 checkpoint 与产物跑完立即删除，`_synthetic/` 不残留文件（有 2 条断言把守）。
- **零回归锚点**：`checkpoints/n3d_viz/viz_model.{html,ply,obj}` 的 SHA256 记录在 `verify_viz.py` 的 `PHASE2_ANCHOR_SHA256`；任何使 `K <= 9` 路径输出变化的改动都会被 `[2d]` 组直接判失败。
- **无 FC 产物必须逐字节零回归**：`config` 无 `fc_dim` 键或 `fc_dim == 0` 的产物，三件套必须与改动前**逐字节相同**。由于 `assets/viewer.html` 与 `assets/viewer.js` 都被**逐字内联**进每一份 HTML，**不得**为了 FC 功能改动这两个文件；FC 渲染一律放在独立资源 `assets/viewer_fc.js` 中、只在 `data.fc` 非 None 时追加内联（详见 §2.5 与 §4）。
- **`fc_dim != 0` 时不得静默降级**：缺任一 `core.FC_REQUIRED_KEYS` 键必须报错退出非 0 且不产生任何产物，**不得**当作「无 FC」继续画一张不完整的图。
- **抽样口径必须显式声明**：FC 连线是「抽样显示（每神经元 top-k），**非全部连接**」，该声明写进 HTML `meta.fcDeclaration`、PLY 头部注释、OBJ 伴随注释与文档三处，`[2f]` 组有专门断言把守。
- **平台**：Windows 上"打开输出文件夹"用 `os.startfile`；Linux 用 `xdg-open`；macOS 用 `open`。
- **无图形环境**：`--skip-gui` 可跳过 GUI 冒烟断言；CLI 入口不受影响。
- **块序不变式（可执行）**：`core.build_html` 保证 FC 块排在 `viewer.js` **之后**，并在返回前用探针串
  （`core._VIEWER_CAM_PROBE` / `core._FC_MAIN_PROBE`）**断言**该顺序，违反即抛 `ValueError`
  （有 FC 时才有该断言；无 FC 产物字节不变）。配套的「相机不可用」告警分支因此**在当前产物里不可达**，
  仅在「手改拼接顺序」或「把 `viewer_fc.js` 单独加载」时可达 —— 见 §2.6。
- **零回归锚点的承重前提**：`PHASE2_ANCHOR_*` / `ANCHOR_GROUPS` / `PLY_OBJ_BASELINE` 的恒定值都取自
  「上游 checkpoints 产物逐字节不变」这一前提（`PLY_OBJ_BASELINE` 采集于 `HEAD = 927d32f`）。
  上游若**破坏性重建 / 清空重训** `checkpoints/`，这些条目会**集体 FAIL 而原因不在渲染器** ——
  判读顺序固定为「先核对上游产物是否同一份，再怀疑渲染路径」。
- **两层一致性 E2E 在版本控制内**：`n3d_viz/tests/e2e_two_layer_cam.mjs`（26 项断言）是唯一能拦住
  「两层不同步」回归的测试，已由 `.lizhu_env/r22_e2e/` 迁入（`.gitignore:48` 忽略整个 `.lizhu_env/`）。
  接入与运行方式见 §3「E2E 两层几何一致性」；`[9]` 组把守「在版本控制内 + 断言数未削减 + 未新增运行依赖」。
