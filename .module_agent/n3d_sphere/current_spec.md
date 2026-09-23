# n3d_sphere 功能说明（N3D 二期架构变体）

## 项目定位

本模块是 N3D 的**二期架构变体**，用于验证"用几何结构注入方向性与层次性"能否突破一期架构的信息损耗瓶颈。

- **与一期关系**：`n3d_proto` 为一期原型（立方体空间 + 四步闭环），本模块为其**并列存档的变体**。
- **自包含**：本模块不 import `n3d_proto`，六个源文件（`__init__.py` / `utils.py` / `data.py` / `config.py` / `model.py` / `train.py`）由一期拷贝而来并独立演进。
- **硬契约**：`n3d_proto/` 下任何文件一律不得修改；一期默认行为与既有产物完整存档、逐位不变。
- **规格偏离声明**：本变体的几何定义（球体分布 + 半球切分 + 全局流向）偏离原提示语 §2.1/§2.2/§2.3 的"核心设计，必须严格遵循"要求，故作为二期变体独立成模块，一期原型并列保留。

## 背景结论（一期已证否的维度）

1. 容量维度无效：N / y_in,y_out / D / T 全部无效或负收益（D 0.15→0.25 掉 1.94pp，T 4→6 掉 0.68pp）。
2. 拓扑种子无效：seed 42/7/2024/123 的 12-epoch 结果为 97.84% / 97.55% / 97.90% / 97.68%，极差 0.35pp。
3. 瓶颈在架构：同等控制变量下普通 MLP（1,628,170 参数）达 98.64%，主模型（1,691,730 参数）仅 97.84%，差 +0.80pp。
4. 四步闭环的信息损耗点：2a 分组 softmax 使输出为邻居凸组合（动态范围压缩）；2d LayerNorm 抹掉激活幅度；2c ReLU 负侧梯度恒零；readout 只取末轮 s_out。

## 功能领域

### 拓扑配置
在 `Config` 中新增三个字段，默认值必须使既有行为逐位不变：
- `topology: str = "cube"`（取值 `cube` | `sphere`）
- `flow_axis: str = "z"`（取值 `x` | `y` | `z`）
- `space_radius: float = 0.0`（0 表示使用等体积球默认半径 `L·(3/4π)^(1/3)` ≈ 0.620350）

新增字段必须同步纳入 `to_dict()` 与 `describe()`；新增校验必须**条件化**（仅 sphere 且 `space_radius > 0` 时才校验半径 > H），否则默认值会导致三个预设构造失败。

### 球形有向几何采样
坐标采样按 `topology` 分支：

| 项 | 立方体（现状，逐字保留） | 球体（新增） |
| --- | --- | --- |
| 神经元分布 | `[0, L]^3` 均匀 | 以原点为球心、等体积球内按体积均匀采样，保留 `min_neuron_dist` 拒绝采样 |
| 输入突触 | 所属神经元周围 **H 半径球体**内按体积均匀 | 半径 H 球内按体积均匀，限制在全局流向的**负半球**（−axis） |
| 输出突触 | 所属神经元周围 **H 半径球体**内按体积均匀 | 半径 H 球内按体积均匀，限制在全局流向的**正半球**（+axis） |

> 更正：一期突触采样实际为"神经元周围 H 半径球体内按体积均匀"，**不是** H 立方体。

**半径采样口径（两种拓扑一致，全模块表述统一，勿误读为半径线性采样）**：方向在单位球面上均匀（正态归一化得到），半径 `r = H · u^(1/3)`，其中 `u ~ U(0,1)`、`u^(1/3)` 是 **u 的立方根**（代码 `H * u.pow(1.0 / 3.0)`）；sphere 分支的神经元球半径同理取 `R · u^(1/3)`。立方根使采样点在球体内**按体积均匀**（半径分布函数 F(r) = (r/H)^3）。该口径在 `model.py` 顶部 docstring、`_sample_neuron_positions` / `_sample_synapse_positions` docstring、`config.py` 与 `README.md` 第 2.1 节四处一致。

半球切分用「方向生成后翻转流向轴分量（`d_axis ← sign·|d_axis|`）」实现：该操作**测度保持**（目标半球内每个方向恰由原始/翻转两种来源各命中一次），等价于半球拒绝采样但**不消耗额外随机数、无重试上限**；翻转只发生在 sphere 分支，对 cube 随机流零影响。

### 关键几何性质（实测结论）
流向轴边间距**统一使用代码口径**，不做 Δz / δ 分解：

    gap = axis(output_syn_pos) − axis(input_syn_pos)

其中 `axis` 是由 `flow_axis` 决定的坐标分量（`flow_axis_index`，取值 x/y/z），即"输出突触坐标 − 输入突触坐标"在流向轴上的分量——与 `__init__` 预计算的 `axis_gap` 及 `get_topology_stats()` 的 `axis_gap_*` 完全一致。（早期文档写作 `gap = Δz + δ_out − δ_in` 且 `δ_out − δ_in ∈ (0, 2H]`，与代码不符且两式自相矛盾，已删除。）

`gap > 0` 为正向边（沿 −axis → +axis 上行），`gap < 0` 为逆向边。因输入突触取 −axis 半球、输出突触取 +axis 半球，gap 的分布相对立方体**明显右偏**：各向同性（cube）时正/逆向 ≈ 50%（实测 0.5009 / 0.4991）。

实测（N=256 / y=8×8 / H=0.1 / D=0.15 / 等体积球 / seed=42）：cube 逆向边占比 0.4991（各向同性），sphere/z 为 **0.385006**（正向 0.614994）。→ 得到的是**明显上行偏置而非严格 DAG**；`get_topology_stats()` 如实返回逆向边占比，不做理想化裁剪。

### 拓扑统计
`get_connection_stats()` **保持与一期完全相同的 5 个键**（`num_edges` / `sparsity` / `zero_ratio` / `avg_out_degree` / `tau`），二期新增指标统一放在新增的 `get_topology_stats()` 中：
- 逆向边占比 / 正向边占比 / 逆向边绝对数
- 流向轴边间距 gap 的 mean / min / max
- 流向轴上的神经元高度分布 mean / min / max / std
- 2a 覆盖：静态（有入边的输出突触占比，预计算 buffer）+ 动态（`forward` 逐轮累计的非零列占比，末轮 / 全程均值 / 累计轮数；只做布尔统计、不参与计算图）
- 连通性：弱连通分量数、最大分量占比（`_compute_weak_components` 用最小标签传播 + 指针跳跃，`__init__` 内预计算并 `register_buffer`）
- 几何指纹：topology / flow_axis / space_radius

**动态 2a 覆盖容器契约（修复审查缺陷 W1）**：动态覆盖**不得**再用无界 `List[float]` 累加（旧实现 12ep×469batch×T=4 ≈ 22512 元素且只增不减）。现为 **O(1) 定长容器**——`reset_round_output_coverage()` 建立三个 Python 标量 `_round_coverage_sum` / `_round_coverage_count` / `_round_coverage_last`，`_accumulate_round_coverage()` 逐轮累加，`round_output_coverage()` 返回快照。`train.py::reset_dynamic_coverage(model)` 在**每次训练开始时**调用（`run_smoke_test` 与 `_run_training_with_config` 各一处），保证多阶段训练（先 `--max-batches` 限批快筛、再全量）与 `model.to(device)` / `load_state_dict` 之间统计不互相污染。

因该容器**不随 `state_dict` 持久化**，末轮 / 全程均值 / 累计轮数会被显式写入 checkpoint：`topology_stats` 内三者齐备，且在顶层另设 `dynamic_coverage` 快照块（由 `train.py::dynamic_coverage_metadata(model)` 生成，两处逐位一致）。因此 `.pt` 加载后可脱离内存态直接 `torch.load` 复核历史动态覆盖，不再出现"加载后为 NaN、无法复核"的情况。

**gap 缓存**：`__init__` 预计算的 `axis_gap` 同时注册为**非持久化 buffer** `edge_axis_gap`（`persistent=False`），`get_topology_stats()` 直接复用，不再每次调用都 `index_select` 重算 O(E)。非持久化意味着它**不进 `state_dict`**，一/二期 state_dict 键集增量仍恰为 3 个（`edge_reverse_flag` / `neuron_component_id` / `connected_output_mask`），逐位等价校验不受影响。

### 命令行入口
新增 `--topology {cube,sphere}`、`--flow-axis {x,y,z}`、`--space-radius`，纳入"未提供哨兵"语义（空串 / `-1.0`）与 `build_smoke_config` 的 explicit 判定（否则 `--smoke-test --topology sphere` 会被静默丢弃）。`run_full_training` 兼容模式同步新增 `topology_override` / `flow_axis_override` / `space_radius_override`。

### 产物隔离
- `CHECKPOINT_DIR = checkpoints/n3d_sphere/`，与一期 `checkpoints/` 物理隔离；`--checkpoint` 的 help 与 `run_full_training` docstring 必须写实际路径（默认 `checkpoints/n3d_sphere/model.pt`；冒烟与限批写入 `checkpoints/n3d_sphere/_verify/`），仅改文案、不改 `resolve_checkpoint_path` 行为；
- 冒烟产物 `smoke_<topology>_<flow_axis>.pt`（cube+z 退化为 `smoke.pt`）；
- 限批产物指纹加入 topology / flow_axis 维度：`verify_<bpe>_N{N}_y{...}_H{...}_D{...}_T{T}_top{topology}_ax{flow_axis}_s{seed}[_tag].pt`；
- 正式全量的默认路径：一期等价配置（cube + N256/y8x8/H0.1/D0.15/T3/seed42）保持 `model.pt`，其它几何/容量配置自动使用 `full_<指纹>.pt`。

### 连通性统计与自检（含 D1 修复）
神经元层弱连通分量按"突触级边 → 神经元层无向投影"计算，算法为**最小标签传播**
（`scatter_reduce(reduce="amin")`，只可能单调下降、固定点唯一）+ 指针跳跃
`min(nxt, nxt[nxt])`，全部在 `__init__` 预计算并 `register_buffer`（`neuron_component_id`）。

历史缺陷 D1（离朱第 2 轮实测、已修复）：初版"互为指针 + 对称化指针跳跃"会把二点循环
维持成**非真值固定点**（N=32 仅一条边时报 32 个分量，真值 31），使 `weak_components` /
`largest_component_ratio` 错误并落盘进产物。该缺陷**只影响统计量**，
`neuron_component_id` 不参与 forward/loss/梯度。

为固化回归，新增自检入口（均不参与训练）：
- `connectivity_selfcheck()`：7 个真值已知的**合成图**（单边 / 链 / 两条链 / 星形 /
  K4,4+孤立点 / 三角形 / 完全图 K32）+ 真实拓扑上"向量化 vs 独立并查集"的分量划分比对；
- `_components_of_synthetic_graph()` / `_expected_components()` / `_component_stats_crosscheck()`。

**实测真值结论**：在 N=256 / y=8×8 / H=0.1 / D=0.15（含 D=0.25）的全部实测配置下，
两种拓扑的神经元层弱连通图**都是单连通图**（分量数 1、最大分量占比 1.000000）。
即连通性不是本架构的瓶颈，刻画几何方向性的有效指标是**逆向边占比**与 **2a 覆盖率**。
## 文件与职责

| 文件 | 职责 |
| --- | --- |
| `n3d_sphere/__init__.py` | 二期变体包声明 |
| `n3d_sphere/utils.py` | 通用工具层（可复现性、设备、日志、稀疏分组 softmax、拓扑构建、统计），由一期原样拷贝 |
| `n3d_sphere/data.py` | MNIST 数据层（IDX 惰性解析 + 归一化 + DataLoader），由一期原样拷贝 |
| `n3d_sphere/config.py` | 超参配置层，含 topology / flow_axis / space_radius 三字段与预设 |
| `n3d_sphere/model.py` | 三维神经元空间模型，含形态采样分支（cube/sphere 半球切分）与拓扑统计 |
| `n3d_sphere/train.py` | 训练入口，含新 CLI、产物指纹与独立产物目录 |
| `n3d_sphere/README.md` | 二期几何定义、统计表、对照结果与结论 |

## 验收标准

1. **等价性门槛 A**：`python n3d_proto/train.py --smoke-test` 仍 9/9 PASS、退出码 0、`loss = 2.419689` 逐位不变（证明一期未被触碰）。
2. **等价性门槛 B**：`python n3d_sphere/train.py --smoke-test` 同样 9/9 PASS、退出码 0、`loss = 2.419689` 逐位不变（证明拷贝忠实且 sphere 分支未扰动 cube 随机流）。
3. `python -m compileall n3d_sphere` 退出码 0。
4. 新拓扑统计完整报告：E、平均出度、密度、逆向边占比、流向轴高度分布、2a 覆盖、弱连通分量数、最大分量占比。
5. 全量对照（12 epoch / bs=128 / AdamW lr=2e-3 / wd=1e-4 / dropout=0.1 / cosine / grad_clip=1.0 / seed=42 / 全量 60000 / 无增强无标准化）中给出球形拓扑的 test_acc，与立方体 97.84%、MLP 98.64% 对照。
6. 每个实验点均有独立、可 `torch.load` 复核的产物。

### 实测验收结果（已固定为可复现证据）

| 判据 | 实测 | 证据 |
| --- | --- | --- |
| 门槛 A | PASS：退出码 0，9/9 PASS，`loss=2.419689` | `python n3d_proto/train.py --smoke-test` |
| 门槛 B | PASS：退出码 0，9/9 PASS，`loss=2.419689` | `python n3d_sphere/train.py --smoke-test` |
| 逐位等价（更强） | PASS：两侧 `loss=2.419689416885376` 逐位相等；18 个共有 state_dict 键逐位相等；`grad_norms` 完全相同；`connection_stats` 既有 5 键逐位相同；二期仅新增 3 个指标 buffer | `verify_n3d_sphere_phase2.py proto` |
| 跨模块几何一致 | PASS：一期 `n3d_proto` 与二期 cube 的 `neuron_pos`/`input_syn_pos`/`output_syn_pos`/`edge_index`/`edge_dist` 逐位相等 | `verify_n3d_sphere_phase2.py geom` |
| 编译 | PASS：`python -m compileall n3d_sphere` 退出码 0 | 终端 |
| 半球切分正确性 | PASS：sphere 输入突触流向轴偏移 max ≤ 0、输出 min ≥ 0（**x / y / z 三轴均验证**） | `geom` |
| 神经元球内 + 体积均匀 | PASS：max\|p\|=0.620265 ≤ R=0.620350；`(\|p\|/R)^3` 均值 0.4841（理论 0.5） | `geom` |
| 产物纪律 | PASS：一期三件产物 SHA256 与本轮开工前完全一致 | `artifact` |
| 连通性（分量数 / 最大分量占比） | PASS：全部实测拓扑（cube、sphere x/y/z、sphere D=0.25）**均为单连通图**（分量 1、最大分量占比 1.000000），与朴素并查集 / BFS / networkx 三路真值一致 | `verify_connectivity_regression.py`（7 合成图 + 5 真实拓扑，退出码 0） |
| 参数快筛判据（E ∈ [3000,90000] 且连通） | 5/7 通过；D=0.20 / D=0.25 两点仅因 **E 超上界** 不满足（连通性本身全部通过） | `probe` |
| 全量 test_acc（sphere/z，12ep 全量 seed=42） | **97.83%**（vs cube 97.84% → −0.01pp，落在 ±0.3pp 噪声带内；vs MLP 98.64% → −0.81pp） | `checkpoints/n3d_sphere/full_sphere_z_N256_y8x8_H0.1_D0.15_T4_topsphere_axz_s42.pt` |

### 已修复缺陷（离朱第 2 轮实测）

* **D1（阻塞，已修复）**：`_compute_weak_components` 初版"互为指针 + 对称化指针跳跃"会收敛到**非真值固定点**，把分量提前劈开（N=32 单边时报 32 个分量，真值 31），导致 `weak_components` / `largest_component_ratio` 错误并落盘进产物。修复为"无向投影 + 最小标签传播（amin）+ 指针跳跃"；新增 `connectivity_selfcheck()` 等自检入口与 `verify_connectivity_regression.py` 固化回归（7 合成图 + 5 真实拓扑全部与三路真值一致）。**该缺陷不影响 forward / loss / 梯度，cube 逐位不变与 `loss=2.419689` 始终成立**；含旧值的产物已同配置重跑替换。
* **D2（已修正）**：README 中 cube 的正向/逆向边占比曾写反，现为逆向 **0.4991** / 正向 0.5009。
* **D3（已修正）**：README 关于"连通性不可达"的结论建立在 D1 之上，已按真值重写为"两拓扑均单连通、连通性不是瓶颈"。
* **D4（已修复）**：正式全量产物名不再经 `config_fingerprint`（原会产出 `full_verify_0_...`），改为 `full_N{N}_y{..}_H{..}_D{..}_T{T}_top{..}_ax{..}_s{seed}.pt`。
## 动态覆盖统计与持久化

### 动态 2a 覆盖：O(1) 定长容器与显式持久化
`forward` 的每一轮 2a 后累加一次"输出突触非零列占比"，累计状态保存在 `ThreeDNeuronSpace` 的
**三个 Python 标量** 中（`_round_coverage_sum` / `_round_coverage_count` / `_round_coverage_last`），
容器大小与训练轮数无关（修复审查缺陷 W1：原为无界 `List[float]`，12ep×469batch×T=4 ≈ 22512 元素
且只增不减）。

**口径（易误读点，已明确限定）**：`out_nonzero_coverage_rounds` 覆盖
**训练与评估两条路径的全部前向**，即

```
rounds = (训练批数 + 评估批数) × epoch 数 × T
```

实测佐证（SMALL_CONFIG，T=2，2 epoch ×（3 训练 + 3 评估）batch）：训练后 rounds 依次为
6 → 12 → 18 → 24，最终 **rounds=24**；冒烟路径（1 训练 batch）为 2 = T。
复现脚本 `checkpoints/n3d_sphere/_verify/verify_r4_round_count_scope.py`。

**阶段隔离**：`reset_round_output_coverage()` 在 `__init__` 末尾调用一次，并由
`train.reset_dynamic_coverage()` 在**每次训练开始时**再调用（`run_smoke_test` /
`run_full_training` 各一处），保证冒烟与全量、限批与全量之间统计不互相污染。

**持久化**：该统计不随 `state_dict` 保存，因此 `train.dynamic_coverage_metadata()` 在
`torch.save` 之前取**显式快照**写入 checkpoint 的 `topology_stats` 与顶层 `dynamic_coverage`，
使 `.pt` 加载后可脱离内存态复核历史动态覆盖。

**接口契约（三键）**：`out_nonzero_coverage_last`（末轮，未跑前向为 NaN）、
`out_nonzero_coverage_mean`（全程均值，未跑前向为 NaN）、`out_nonzero_coverage_rounds`
（累计轮数，未跑前向为 0.0）。取快照一律用 `.get(key, 默认值)`：**调用方返回键不完整的
字典时返回 NaN / 0.0 而非抛 `KeyError`**；`getattr` 取不到 `round_output_coverage` 的模型
（如 `MLPBaseline`）返回空字典 `{}`，以区分"无该维度"与"有该维度但缺键"。
