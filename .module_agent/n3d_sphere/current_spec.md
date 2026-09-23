# n3d_sphere 功能说明（纯球形分层有向无环架构）

## 项目定位

本模块是 N3D 的**纯球形分层有向无环架构（DAG）**实现：神经元按 FCC（面心立方）规则堆积放置在球空间内，突触按全局流向轴切分到正/负半球，连接规则强制沿流向轴严格上行，因此图天然无环。

- **纯球体几何**：网络上不存在其它几何分支，也不存在几何类型开关（既无几何类型字段，也无边长字段）。
- **自包含**：不 import `n3d_proto` 的任何模块；一期完整存档、默认行为与既有产物逐位不变（`python n3d_proto/train.py --smoke-test` 仍 9/9 PASS、退出码 0、`loss=2.419689`；`git status --porcelain -- n3d_proto` 为空；一期三件产物 SHA256 未变）。
- **与一期产物的物理隔离**：本模块产物一律写入 `checkpoints/n3d_sphere/`，验证类运行写入其 `_verify/` 子目录，绝不触碰一期 `checkpoints/` 根下的 `n3d_model_*.pt`。
- **可执行不变量**：全部架构不变量（几何、FCC、DAG、去重、逐边数值、感受野、判据语义、产物指纹、**设备契约**、**文档登记数字**）都固化为脚本判据，由 `verify_all.py` 一键执行并汇总；文档中出现的每一个实测数字都登记在 `_verify/doc_numbers.json` 并由 `verify_all.py` 现跑比对。
- 上游规格偏离声明：本模块的几何定义（球体分布 + 半球切分 + 全局流向 + 严格上行 DAG）偏离原提示语 §2.1/§2.2/§2.3 的"核心设计，必须严格遵循"要求，故独立成模块。
## 背景结论（一期已证否的维度）

1. 容量维度无效（**以下为一期历史实验记录，其迭代轮数维度在本模块当前架构中已不存在**）：N / y_in,y_out / D / 迭代轮数 全部无效或负收益（D 0.15→0.25 掉 1.94pp，迭代轮数 4→6 掉 0.68pp）。
2. 拓扑种子无效：seed 42/7/2024/123 的 12-epoch 结果为 97.84% / 97.55% / 97.90% / 97.68%，极差 0.35pp。
3. 瓶颈在架构：同等控制变量下普通 MLP（1,628,170 参数）达 98.64%，主模型（1,691,730 参数）仅 97.84%，差 +0.80pp。
4. 四步闭环的信息损耗点：2a 分组 softmax 使输出为邻居凸组合（动态范围压缩）；2d LayerNorm 抹掉激活幅度；2c ReLU 负侧梯度恒零；readout 只取末轮 s_out。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| `n3d_sphere/__init__.py` | 包声明与模块定位：纯球形分层 DAG、自包含、一期不变、快速入口 |
| `n3d_sphere/utils.py` | 通用工具层（可复现性 `set_seed`、设备 `get_device`、日志、参数/梯度统计 `count_parameters`/`tensor_grad_norms`），自包含实现；四步闭环算子为历史遗留、当前架构不调用 |
| `n3d_sphere/data.py` | MNIST 数据层（IDX 惰性解析 + 归一化 + DataLoader，num_workers>0 时按 (seed, worker_id) 播种） |
| `n3d_sphere/config.py` | 超参配置层：球半径窗口公式（`min_space_radius`/`max_space_radius`）、FCC 晶格常数派生量、判据开关、三预设（SMALL/DEFAULT/HIGHACC） |
| `n3d_sphere/model.py` | 球形分层 DAG 模型：FCC 规则堆积、半球突触采样、神经元级连接（同对去重取最近突触对）、Kahn 拓扑序与 CSR 分组、**整层向量化的两阶段双副本前向**（层节点集合由 `topo_index[s:e]` 张量切片得到）、**设备契约守卫 `_assert_index_device`**、统计与自检接口；另含 `MLPBaseline` 对照基线 |
| `n3d_sphere/train.py` | 训练入口：CLI（flow-axis/space-radius/input-scope/readout-scope/placement）、产物指纹与隔离、15 条冒烟判据、checkpoint 元数据 |
| `n3d_sphere/README.md` | 模块文档：几何与尺度、连接规则、两阶段双副本前向、判据开关、CLI、冒烟判据、实测拓扑统计（标注 seed）、产物纪律、验证脚本、文档数字防线、已知边界与三轮修复记录 |
| `checkpoints/n3d_sphere/_verify/verify_sphere_dag.py` | 几何 / FCC / DAG / 去重 / 双副本 / 逐边数值 / 感受野 / 判据 / 产物 验证脚本（R1-R7b）；每次运行把 **111 条全精度指标**落盘 `sphere_dag_metrics.json` 供文档数字比对 |
| `checkpoints/n3d_sphere/_verify/verify_device_regression.py` | **设备契约回归取证脚本（D1-D7）**：层拓扑量的类型/注册状态、`vars(model)` 通用扫描、层切分等价性、`meta` 设备搬运实验与"普通 Python list 搬不动"机制复现、两条守卫负例、CUDA 实测（无 GPU 时 skip 并标注静态取证）、`state_dict` 往返逐位一致 |
| `checkpoints/n3d_sphere/_verify/verify_scope_and_fingerprint.py` | 判据开关语义与产物指纹/冒烟命名验证脚本（S1-S5） |
| `checkpoints/n3d_sphere/_verify/verify_config_contracts.py` | 半径公式 / 窗口校验 / 字段清理 / 零残留验证脚本（C1-C6） |
| `checkpoints/n3d_sphere/_verify/verify_topology_snapshot.py` | 固定实验点 × 多 seed 拓扑量快照脚本（落盘 `topology_snapshot.json`） |
| `checkpoints/n3d_sphere/_verify/run_smoke_matrix.py` | 以当前代码重跑 **11 种冒烟组合**：解析实际落盘路径并与独立预测的产物名比对、回读 config 与期望逐字段比对、断言 `smoke.pt` 未被非默认组合污染；刷新 `smoke_matrix_f9.json` / `log_smoke_matrix_f9.txt` |
| `checkpoints/n3d_sphere/_verify/verify_all.py` | 一键验证入口：依次执行 **9 条**验收命令、汇总退出码，并执行**文档数字防线**（`doc_numbers.json` 现跑比对） |
| `checkpoints/n3d_sphere/_verify/doc_numbers.json` | **文档数字登记表**：README / 本说明中出现的实测值（90 项 = 32 产物字段 + 54 全精度指标 + 4 文本计数），由 `verify_all.py` 现跑比对，不一致即判失败 |
| `checkpoints/n3d_sphere/_verify/sphere_dag_metrics.json` | `verify_sphere_dag.py` 落盘的全精度实测指标（R1-R7b，111 条） |
| `checkpoints/n3d_sphere/_verify/smoke_matrix_f9.json` / `log_smoke_matrix_f9.txt` | 11 种冒烟组合的取证记录（产物名 / 退出码 / PASS / FAIL / loss / 梯度范数 / config 核对 / SHA256）与完整日志 |
| `checkpoints/n3d_sphere/_verify/topology_snapshot.json` / `smoke_scope_matrix.json` | 拓扑量快照与四种 scope 组合的冒烟汇总记录 |
## 验收标准

### 实测验收结果（唯一记录处；全部命令退出码 0）

所有实测数字均**登记在 `checkpoints/n3d_sphere/_verify/doc_numbers.json`**（93 项）并由
`verify_all.py` 在**同一轮现跑**中逐项比对；下表中的数字若与登记表或现跑不一致即判失败。

| 判据 | 实测证据 |
| --- | --- |
| E1 编译 | `python -m compileall -q n3d_sphere` 退出码 0 |
| E2 冒烟 | `python n3d_sphere/train.py --smoke-test` 退出码 0、**15/15 PASS**；四种 scope 组合均 15/15 PASS 且 **loss 互不相同**（`any/any` = 2.326995849609375；`any/all` = 2.299027681350708；`all/any` = 2.2900023460388184；`all/all` = 2.283543109893799），各有独立产物可 `torch.load` 复核（汇总记录 `_verify/smoke_scope_matrix.json`）；默认组合产物 `_verify/smoke.pt` 的 loss = 2.326995849609375，梯度范数（全精度）W_in 0.30047276616096497 / edge_weight 0.06744416803121567 / neuron_bias 0.16902117431163788 / W_out 0.12903930246829987；E=181、S_in=13、S_out=17、层数 7 |
| E2' 多配置冒烟 | 以当前代码重跑 **11 种组合**（四种 scope × `--seed 7/0/2024` × `--n 32` × `--preset default` × `--flow-axis x` × `--arch mlp`）**全部退出码 0、FAIL=0**，且实际落盘产物名与脚本独立预测的名字**逐条一致**、各产物 config 与期望逐字段自洽、`smoke.pt` 在全矩阵前后 SHA256 与 config 均未被污染，**同一产物路径被多条组合写入时逐位一致**（本轮 `smoke.pt` 被 3 条组合写入、SHA 去重后 1 种；离朱第 11 轮 D1 的回归判据）。如实标注：`--seed 0` 按 CLI 约定表示"不覆盖"、`--preset default` 在冒烟路径下的基线即 `SMALL_CONFIG`，故这两条与默认组合等价（产物同为 `smoke.pt` 且逐位相同）。取证：`_verify/log_smoke_matrix_f9.txt` 与 `_verify/smoke_matrix_f9.json` |
| R1-R7b 几何/FCC/DAG/去重/双副本/逐边数值/感受野/判据/产物 | `verify_sphere_dag.py all` 退出码 0（10 项判据标签全 PASS），**在 4 组配置上执行**（SMALL / DEFAULT / DEFAULT+`flow_axis=x` / DEFAULT+`space_radius=0.9`），111 条全精度指标落盘 `_verify/sphere_dag_metrics.json`。**R5**（零化 `a_in`）：max\|Δa_up\| = 0.941795 / 0.866614 / 0.918773 / 0.866614，受影响下游非 `S_in` 神经元 = 51 / 197 / 204 / 197，`d(sum a_up)/d(a_in)` 非零 = 166/192、713/768、711/768、713/768；**R5b**（独立逐节点重建 vs 生产实现）：最大偏差 = 2.980e-08 / 5.960e-08 / **2.384e-07** / 5.960e-08（判据阈值 1e-5），M1 形态反例错配 = 180/181、892/903、902/913 且结果差异 = 7.411e-01 / 7.512e-01 / 1.389；**R5c**：最深层祖先覆盖 7/7、9/9、10/10、9/9 层，第一层扰动传到最深层最大变化 = 0.004414 / 0.003379 / **0.007765** / 0.003379；**R7b**：`readout_scope` 两取值 \|S_out\| any=45 / all=13，同一输入下 logits 最大差异 = 0.119461 |
| **D1-D7 设备契约（F16）** | `verify_device_regression.py` 退出码 0（D1-D5 与 D7 共 6 项 PASS，D6 无 GPU 时 SKIP）：D1 `level_edge_reach` / `level_node_reach` 均为已注册 int64 `[K,2]` 张量（在 `named_buffers()` 与 `state_dict()` 中、**不再是 Python list**）；D2 `vars(model)` 通用扫描无未注册张量与含张量容器；D3 层切分与独立分层逐位一致、层边区间无缝覆盖 `[0,E)`；D4 `.to('meta')` 后 10 项索引张量全部搬到 meta，而对照的普通 Python list 中张量仍停留在 cpu（复现 F16 缺陷机制）；D5 两条守卫负例均抛 RuntimeError（设备不一致 / 未注册）；D6 本机无 GPU → 标注"CUDA 路径为静态取证"并 skip（有 GPU 时自动追加真实 `.cuda()` 前向比对）；D7 `state_dict` 往返后前向逐位一致 |
| **文档数字防线（F20）** | `verify_all.py` 现跑比对 `doc_numbers.json`：**一致 93 项 / 不一致 0 项 / 跳过 0 项**（32 产物字段按 `torch.load` 逐位相等、54 全精度指标取自本轮 `sphere_dag_metrics.json`、7 文本计数取自本轮命令标准输出）；离朱第 11 轮**负向验证**已确认：篡改登记值后 `verify_all.py` 退出码 1 并报 FAIL |
| S1-S5 判据语义与产物指纹 | `verify_scope_and_fingerprint.py` 退出码 0；`all/any` 下 `d(sum a_up)/d(a_in)` 非零 480/512、`any/any` 下 475/512；`input_scope=all_isolated` 的两种组合双副本神经元数均为 0（与第 7 节表一致） |
| C1-C6 半径公式/窗口/字段/零残留 | `verify_config_contracts.py` 退出码 0（残留扫描覆盖 n3d_sphere 全部源文件 + README + 本功能说明 + 文件定义；真实命中 0 行、上下文豁免 4 行，逐行打印并计数） |
| 拓扑快照 | `verify_topology_snapshot.py` 退出码 0（60 条记录落盘 `topology_snapshot.json`，自检 `：PASS`） |
| **E5 一期未被触碰** | `python n3d_proto/train.py --smoke-test` **9/9 PASS、退出码 0、loss=2.419689**；`git status --porcelain -- n3d_proto` 输出为空；一期三件产物 SHA256 与本轮开工前完全一致 |

### 阶段 A 冒烟判据（15 条，新架构口径）

旧架构的 `tau > 0`、连接稀疏度（密度 `E/(N*y_out*N*y_in)`）、边级参数数 == E 等判据已随架构失效。现判据集合为 **15 条**（日志逐条打印 PASS/FAIL，文案全 ASCII 以规避 GBK 控制台编码缺陷）；`--arch mlp` 对照基线只有其中 6 条适用：

1. 前向输出形状 == `[B, output_dim]`（**真实断言**：比较实际 logits 形状，非恒真）
2. 反向无错误（全部可学习参数都有梯度；缺失梯度由 `tensor_grad_norms` 记为 -1.0 并逐个核验）
3. 参与 loss 的参数梯度范数 > 0（按 arch 解析输出层参数名：neuron3d 为 `W_out`、MLP 为 `fc2.*`；neuron3d 另断言 `W_out` 非零梯度列数落在 `[1, |S_out|]` —— **区间断言，不是等式**，因为 ReLU 死神经元会合法地贡献零列）
4. loss 非 NaN/Inf
5. `S_in` 非空（阶段 1 真正被输入层驱动）
6. `S_out` 非空（readout 真正有信号）
7. 连接数 == 去重后的神经元对数（"同一神经元对只算一条连接"；**不是**"最大入度 <= 1"）
8. 无环 DAG 且每条边严格上行（`z_A < z_B`）
9. 最近邻距 == 2H（FCC 契约，容差 1e-5）
10. 球空间半径落在 `[R_min, R_max]` 内
11. `E > 0` 且平均出度 > 0
12. 不存在 `[N*y_out, N*y_in]` 形状的权重张量（未 materialize dense 矩阵）
13. readout 严格口径：`h` 的非零列**都属于** `S_out`，且 `h` 逐位等于 `a_up * out_scope_mask`（**只断言子集方向与掩码结构，不断言 `S_out` 列全覆盖** —— 某 `S_out` 神经元 pre-activation 在整批上全负时其 ReLU 输出为 0，`h` 对应列合法为 0）
14. 阶段 2 递推顺序 == 流向轴升序（`topo_matches_axis_order == 1`）
15. CPU 单 batch 前向+反向耗时 < 120s

### 验证脚本清单（均在 `checkpoints/n3d_sphere/_verify/`）

- `verify_sphere_dag.py`：R1 几何（球内/半球/体积均匀）、R2 FCC（晶格常数、最近邻距=2H、seed 无关性）、R3 DAG（无环/严格上行/拓扑序）、R4 去重（神经元对数==E、代表连接为块内最近）、R5 双副本、**R5b 逐边数值正确性**（独立逐节点重建 + M1 形态反例）、**R5c 感受野覆盖**、R6 判据（含 readout 严格口径）、**R7b readout_scope 生效性**、R7 产物可 torch.load；R1-R7b 均在 4 组配置上执行；运行时落盘 `sphere_dag_metrics.json`（111 条全精度指标）。
- `verify_device_regression.py`：**D1-D7 设备契约回归取证**（类型/注册状态、`vars(model)` 通用扫描、层切分等价性、`meta` 搬运实验与对照、守卫负例、CUDA 实测或静态取证、`state_dict` 往返）。
- `verify_scope_and_fingerprint.py`：S1-S3 判据语义与双副本随 scope 的行为（含 `d(sum a_up)/d(a_in)` 非零计数、共享权重梯度、`all/any` 双副本为 0 时依赖仍在）、S4-S5 产物指纹维度与冒烟命名规则。
- `verify_config_contracts.py`：C1 半径公式、C2-C3 窗口校验与 FCC 容纳性、C4-C5 字段清理与取值域、C6 零残留断言（含豁免计数与逐行打印）。
- `verify_topology_snapshot.py`：固定实验点 × 10 seed 拓扑量快照（落盘 JSON）。
- `run_smoke_matrix.py`：11 种冒烟组合重跑与取证刷新（产物名预测比对 + config 校验 + 默认产物未被污染 + **同一路径多次写入逐位一致**）。
- `verify_all.py`：一键依次执行 **9 条命令**（含 `--arch mlp` 冒烟、D1-D7 设备契约与一期回归）并汇总退出码，随后执行**文档数字防线**。
- 取证文件：`doc_numbers.json`（文档数字登记表，93 项）、`sphere_dag_metrics.json`（R1-R7b 全精度指标）、`log_smoke_matrix_f9.txt` / `smoke_matrix_f9.json`（11 种冒烟组合的退出码、PASS/FAIL、实际产物名、config 核对与 SHA 一致性字段）、`log_verify_all_f21.txt`（一键验证完整输出）。
## 纯球形分层有向无环架构

### 球体几何与尺度

神经元是半径 `H` 的球（突触云分布半径）。球空间半径由公式唯一确定（最优堆积系数 `φ = 0.7405`）：

- `R_min = H · (N / φ)^(1/3)`（非重叠容纳下界）
- `R_max = (H + D) · (N / φ)^(1/3)`（保证每个神经元的 D 邻域完整落在球空间内的上界）
- `space_radius` 默认 `0.0` = 取 `R_min`；显式传入时必须落在 `[R_min, R_max]` 内，越界在 `Config` 构造期报错（上下界两侧均有断言）。

**如实口径**：`space_radius` 仅作**半径窗口校验与元数据**，它**不改变神经元放置与拓扑** —— 神经元位置由 FCC 晶格与 `R_max` 决定的搜索半径唯一确定，因此实际 `placement_radius` 允许略超 `R_min`（FCC 格点是离散的，最近 N 个点的最远距离通常大于连续体积下界）。`verify_sphere_dag.py` 的 R1/R2 在含 `space_radius=0.9` 的配置上也验证这一点。

参考值（公式实测）：N=256/H=0.10/D=0.15 → `R_min=0.701840`、`R_max=1.754601`（比值 2.5 = (H+D)/H）；N=64/H=0.15/D=0.25 → `R_min=0.663198`、`R_max=1.768527`。

### FCC 规则堆积放置

晶格常数 `a = 2√2·H`，基元 `{(0,0,0), (½,½,0), (½,0,½), (0,½,½)}`，故**最近邻距恰为 2H**（相邻神经元的 H 半径突触云恰好相切、不重叠）。取距球心最近的 N 个格点，再按 **(流向轴坐标, 壳层名次)** 的显式字典序做确定性排序，使 `topo_index` 恰为流向轴升序。

- 神经元坐标**与 seed 无关**（完全由 H/N 决定），模块内以断言守护"最近邻距 == 2H"（容差 1e-5）。
- 不存在随机放置分支，也不存在最小间距拒绝采样。

### 突触采样与半球切分

突触在**各自神经元的 H 半径球内按体积均匀**采样：方向在单位球面均匀（正态归一化），半径 `r = H · u^(1/3)`（`u ~ U(0,1)`，立方根保证按体积均匀，**非半径线性采样**）。

输入突触取流向轴**负半球**（-axis）、输出突触取**正半球**（+axis）：把方向的流向轴分量翻转为目标半边（`-|c|` / `+|c|`）。该操作**测度保持**（目标半球内每个方向恰由原始/翻转两种来源各命中一次），等价于半球拒绝采样但不消耗额外随机数、无重试上限。

### 神经元级连接规则

`A -> B` 存在 ⟺ `A ≠ B` 且 `z_A < z_B` 且 `∃ o ∈ out(A), j ∈ in(B): d(o,j) <= D`（`z` 为 `flow_axis` 选定的坐标分量）。

- **同一神经元对只算一条连接**：多对突触满足条件时只保留间距最近的一对作为代表连接（`representative_syn_out` / `representative_syn_input`）。
- 判据是"**存在**至少一对"，故必须对合法突触对取 **`amin`** 后与 D 比较（写成 `amax` 语义相反，会把本该连接的神经元对判为不连接）。
- 4D 块视图 `[N_A, y_out, N_B, y_in]` **不可**再 reshape 成 `[N*N, y_out*y_in]` 后按 `flat[A*N+B]` 取值（stride 使轴序相对 2D 视为交换，会产生块内错位）；正确做法是在 `[N_A, N_B, y_out, y_in]` 上分别对两个突触维取 `amin/argmin`。代码内有两处契约断言守护（代表间距必须 == pair_min；不得出现反向边）。
- 全部拓扑量在 `__init__` 预计算并 `register_buffer`，`forward` 不重算 —— 含阶段 2 逐层向量化所用的 `topo_index` / `edge_perm_in` / `neuron_in_edge_reach` / `edge_dst_in` / **`level_edge_reach` / `level_node_reach`**。

### 孤立突触与判据开关

某突触"孤立" ⟺ 其 D 邻域内不存在任何合法连接的对端突触（合法连接 = 对端属于其他神经元且高低关系满足上行约束）。

| 开关 | 取值 | 含义 |
| --- | --- | --- |
| `input_scope` | `any_isolated` / `all_isolated` | 神经元有 ≥1 个 / 全部 y_in 个输入突触孤立 → 进入 `S_in` |
| `readout_scope` | `any_isolated` / `all_isolated` | 神经元有 ≥1 个 / 全部 y_out 个输出突触孤立 → 进入 `S_out` |

实测（DEFAULT 规模 seed=42）：`any/any` → S_in=50、S_out=45；`any/all` → 50/13；`all/any` → 13/45；`all/all` → 13/13。`E` 与判据选择无关（恒为 903）。以上 8 个数字均登记在 `doc_numbers.json`（键前缀 `r6.`）并由 `verify_all.py` 现跑比对。

### 两阶段前向与双副本展开

- **阶段 1（输入层驱动）**：`a_in[B] = ReLU(x · W_in[:,B] + b_B)`，仅对 `B ∈ S_in` 计算，其余恒为 0（不进入计算图）；`W_in` 形状 `[input_dim, |S_in|]`。
- **阶段 2（单遍逐层递推 + 整层向量化）**：严格按 Kahn 拓扑序（= 流向轴升序）**逐层递推一遍**：
  `a_up[B] = ReLU( Σ_{A→B} w_{A→B} · (a_up[A] + a_in[A]) + b_B )`。
  处理 `B` 时其全部上游（层更小）已算完，故一次前向即完成全部层的传播；**感受野覆盖全部层**（DEFAULT 规模 9 层；`verify_sphere_dag.py` 的 R5c 用"祖先层覆盖 + 第一层扰动可传到最深层"两点取证，四组配置的实测最大变化为 0.004414 / 0.003379 / 0.007765 / 0.003379）。
  **实现形态（F11/F16）**：循环次数 = **层数**（DEFAULT 9 / SMALL 7），而不是神经元数 N；每层只做一次 `index_add`（该层全部入边的消息按目标神经元散加）与一次**非原地** `index_copy`（写回 `ReLU(pre + bias)`，原地赋值会破坏 autograd）。层节点集合由 `topo_index[s:e]` **张量切片**得到。
- **无重复轮数参数**：架构中**没有**迭代轮数（同步迭代的旧设计已移除），故不存在"被固定跳数截断"的问题。
- **双副本展开**：求和项 `(a_up[A] + a_in[A])` 使神经元的"上游版本"与"输入层版本"**都参与后续传播**，两种版本**共享同一套边权** `w_{A→B}`。`a_in` 项作为常量参与每一次聚合；若遗漏则输入层副本被覆盖而永不生效（实测梯度恒为 0）。R5 以"零化 a_in 是否改变 a_up"与 `d(sum a_up)/d(a_in)` 取证：四组配置 max\|Δa_up\| = 0.941795 / 0.866614 / 0.918773 / 0.866614，受影响下游非 `S_in` 神经元 = 51 / 197 / 204 / 197。
- **逐边数值正确性（R5b）**：脚本独立重建阶段 2（按拓扑序逐节点、用原始边表精确递推）与生产实现对比，最大偏差 = 2.980e-08 / 5.960e-08 / 2.384e-07 / 5.960e-08（阈值 1e-5）；同项含 M1 形态**敏感性反例**（错配 180/181、892/903、902/913，结果差异 7.411e-01 / 7.512e-01 / 1.389），确保判据不会退化为恒真。
- **读出（严格口径）**：`h[n] = a_up[n]`（**仅当 `n ∈ S_out`**），否则 `h[n] = 0`；`logits = h @ W_out.T (+ b)`。
  即只有 `S_out` 中的神经元向输出层贡献信号，非 `S_out` 神经元被整体屏蔽 —— 其 `W_out` 列不参与计算图、梯度恒为 0（该口径的直接推论）。`readout_scope` 由此**真正生效**：两取值给出不同 `S_out`（any=45 / all=13），实测同一输入下 logits 最大差异 0.119461（R7b），四种 scope 组合的冒烟 `loss` 互不相同（见"实测验收结果"）。

### 参数集合与稀疏约束

`W_in [input_dim, |S_in|]`、`edge_weight [E]`（每条神经元级连接一个独立标量）、`neuron_bias [N]`（正初值 0.1）、`W_out [output_dim, N]`、（可选）`W_out_bias [output_dim]`。

禁止 materialize dense `[N*y_out, N*y_in]` 权重矩阵：`count_dense_weight_tensors()` 扫描 `named_parameters()` / `named_buffers()`（豁免几何量 `syn_dist`）并断言恒为 0。

**梯度口径（严格读出的直接推论）**：参与 loss 的参数梯度范数必须 > 0（判据 2/3）。neuron3d 的 `W_out` **非 `S_out` 列**梯度恒为 0，故判据断言"非零梯度列数落在 `[1, |S_out|]`"这个**区间**而不是等式：上界来自"只有 `S_out` 列可能非零"，下界 `1` 允许某 `S_out` 神经元的 pre-activation 在整批上非正（ReLU 输出恒 0，其在 `h` 中本来就是零列）—— 这是 ReLU 的正常行为，不是屏蔽错误。MLP 对照基线按 `fc2.*` 参数名断言其输出层梯度 > 0。

### 拓扑统计接口

- `get_connection_stats()`：`num_edges`（神经元级 E）、`num_neurons`、`avg/max_out_degree`、`avg/max_in_degree`、`num_layers`、`num_in_scope`、`num_out_scope`、`isolated_input_syn`、`isolated_output_syn`。
- `get_topology_stats()`：`flow_axis` / `placement` / `space_radius` / `placement_radius` / `lattice_constant` / `nearest_neighbour_dist` / 神经元流向轴高度分布 / 判据编码 / `num_edges` / `edge_dist_mean|min|max` / `dual_copy_count`。
- `connectivity_selfcheck()`：`dag_acyclic` / `all_edges_uphill` / `topo_covers_all` / `topo_matches_axis_order` / `representative_edge_count`。

### 设备契约与静态取证（F16）

**契约**：被 `forward` 当作索引张量使用的一切拓扑量，必须是 `register_buffer` / `nn.Parameter`，
从而 (a) 跟随 `.to(device)` 搬运、(b) 进入 `state_dict()` 持久化。
违反其一即属设备回归：在 CPU 上完全静默（全部 CPU 判据仍全绿），只在 CUDA 上抛 RuntimeError。

- **实现**：`level_edge_reach` / `level_node_reach` 两张 `[K,2]` int64 张量已注册；层节点集合由
  `topo_index[s:e]` 张量切片得到。历史缺陷是它们曾以**普通 Python `list`** 保存层节点张量 ——
  `nn.Module.to(device)` **不搬运普通 list 中的张量**。
- **运行时守卫**：`stage2_recurrence` 每次前向调用 `_assert_index_device(a_up)`，逐个校验
  `topo_index` / `edge_perm_in` / `edge_dst_in` / `edge_src` / `edge_dst` / `neuron_bias` /
  `level_edge_reach` / `level_node_reach`（以及启用时的 `W_out_bias`）**既已注册又与激活同设备**，
  并额外校验层节点切片的设备；任一不符即抛 `RuntimeError`（消息含张力名与两侧设备）。
- **静态取证**：本机 CPU-only（`torch.cuda.is_available() == False`，`torch 2.14.0+cpu`），
  故 CUDA 路径由 `verify_device_regression.py` 以 `meta` 设备做**搬运实验**（注册 buffer 会搬走、
  对照的普通 Python list 中张量停留在 cpu）+ **两条守卫负例**（设备不一致 / 未注册均必须抛错）取证；
  若在具备 GPU 的机器上运行，该脚本自动追加真实 `.cuda()` 前向与 CPU 结果比对（D6）。
- **回归基线**：设备改造后 CPU 冒烟 `loss` 与向量化前**逐位不变**（`2.326995849609375`，15/15 PASS）。

### 实测拓扑规模（可由 `verify_topology_snapshot.py` 复现，引用须标注 seed）

DEFAULT 测点（N=256 / y=8×8 / H=0.10 / D=0.15 / flow_axis=z / 两个 any_isolated / R=R_min）跨 10 个 seed：

- **E = 900 ~ 924**（seed=42 为 903；seed=7 为 912；seed=2024 为 924）
- 平均出度 = 3.5156 ~ 3.6094（seed=42 为 3.5273）；最大出度除 seed=3（6）外均为 5；最大入度恒为 5
- 层数恒为 9；S_in = 40 ~ 52；S_out = 44 ~ 50；双副本神经元数 = 27 ~ 39
- **可学习参数 35094 ~ 44489**（随 `|S_in|` 变，属"按连接构建"的必然结果；区间取自 `_verify/topology_snapshot.json` 中**该默认测点的 10 个 seed** 记录 —— 全快照含 6 个测点、合并范围为 5588 ~ 45285，因测点规模不同故不混用）

SMALL 测点（N=64 / y=4×4 / H=0.15 / D=0.25 / seed=42）：E=181、平均出度 2.8281、最大出度 5、层数 7、S_in=13、S_out=17、双副本 7、可学习参数 11077。
## 命令行与配置入口

### 几何与判据 CLI

几何相关 CLI（已完全移除几何类型开关）：

- `--flow-axis {x,y,z}`：全局流向轴（缺省沿用预设 z）；输入突触取负半球、输出突触取正半球。
- `--space-radius R`：球空间半径；哨兵 `-1.0` = 未提供、`0` = 取 `R_min`；显式值越界在 `Config` 构造期报错。
- `--input-scope S` / `--readout-scope S`：`any_isolated` / `all_isolated`（空串 = 沿用预设）。
- `--placement P`：缺省沿用预设 `fcc`（当前唯一取值）。
- `--seed S`：覆盖随机种子。**约定：负数报错、`0` 表示"不覆盖"**（`validate_overrides` 与 `--seed` 帮助文本一致）。因此 `--seed 0` 在冒烟矩阵中与默认组合等价（产物同为 `smoke.pt` 且逐位相同），真正的额外 seed 覆盖需用 `--seed 7` / `--seed 2024` 这类正值；`run_smoke_matrix.py` 的记录中已逐条注明。
- `--preset {small,default,highacc}`：冒烟路径的基线由 `build_smoke_config` 决定，`--preset default` 时基线即 `SMALL_CONFIG`（`base = PRESETS[args.preset] if args.preset != "default" else SMALL_CONFIG`），故该写法在冒烟下亦等价于默认组合。
- 全部新参数纳入 `build_smoke_config` 的 explicit 判定（否则 `--smoke-test --input-scope all_isolated` 会被静默丢弃），并同步进 `apply_overrides` 与 `run_full_training` 的兼容模式。
- **值不等价才算覆盖**：`apply_overrides` 构造出的 `Config` 若与基线**逐字段相同**，则**返回基线对象本身**而非新对象（值等价短路），使同一语义配置产出同一字节（见下节）。

### 产物字节确定性与 SHA256 语义（离朱第 11 轮 D1）

**`torch.save` 的产物字节不只由数值决定，还取决于 pickle 的记忆化（memo），而 memo 依赖
对象身份（`id`）而非取值** —— 因此"SHA256 相等"只有在**同一语义配置走同一代码路径**时才
等价于"数值相同"。

- **反例实测（D1）**：`--smoke-test`（走 `SMALL_CONFIG` 单例）与
  `--smoke-test --input-scope any_isolated --readout-scope any_isolated`（scope 取值来自
  argparse 构造的**等值新字符串**）写出的 `smoke.pt`：loss / `config` / `grad_norms` /
  `model_state_dict` 全部张量**逐位相等**，但 `data.pkl` 为 4097 vs 4117 字节、共 1520
  字节不同（单例下两个 scope 字段指向同一个 `"any_isolated"` 字符串对象，第二次出现被写成
  memo 引用；显式传参下两处分别内联写出）。后果：矩阵记录 [0] 登记的 SHA 在矩阵结束后
  **已被同轮后续写入覆盖**，取证文件不自洽。
- **根因侧修复**：`train.py` 的 `apply_overrides` 在"显式覆盖值与基线**逐字段相同**"时
  **复用基线对象本身**，使"同一语义配置 ⇒ 同一字节"。修复后 `--smoke-test` / 显式传等值
  scope / `--space-radius 0.0` / `--preset default` / `--seed 0` 等路径写出的 `smoke.pt`
  SHA256 **全部为 `1A9D68D8…43D7`**；离朱第 12 轮复测另确认 `build_smoke_config(...) is
  SMALL_CONFIG` 为 True（**同一对象**，而非仅字段相等）。
- **日志出现的精确范围（离朱第 12 轮 D2 澄清）**：`apply_overrides` 的短路分支会打印
  `显式覆盖参数与基线取值逐字段一致（值等价）：复用基线配置对象…`。**只有"显式给出覆盖
  参数且其值与基线相同"的形式**才会打印该行，实测为 3 条形式（`--input-scope
  any_isolated`、两个 scope 都显式传等值、`--space-radius 0.0`）；而**裸跑**、
  `--preset default`、`--seed 0` 三种形式在 `build_smoke_config` 的
  `if not explicit: return SMALL_CONFIG` 处**提前返回**（按 CLI 约定它们本就不算"显式覆盖"），
  不会进入 `apply_overrides`，故不打印该行 —— 但**产物字节与前者完全一致**（都走
  `SMALL_CONFIG` 单例）。真正改变取值的覆盖仍打印 `冒烟测试配置已被显式覆盖…` 告警且不打印
  值等价行，语义边界正确（离朱第 12 轮已用 9 组非等值覆盖验证未被过度短路）。
- **判据侧修复**：`run_smoke_matrix.py` 新增"**同一产物路径被多条组合写入时必须逐位一致**"
  的断言，并在每条记录追加 `artifact_sha256_at_end` / `artifact_sha256_stable` /
  `artifact_shared_writers` 字段。实测：`smoke.pt` 被 **3 条组合**写入
  （`scope any/any seed42` / `seed 0` / `preset default`）、SHA 去重后 **1 种**；
  11/11 记录的 `artifact_sha256_stable == true`、`artifact_sha256_at_end == artifact_sha256`
  且与盘上文件实际 SHA 一致。
- **引用规范**：报告引用产物 SHA256 时必须同时给出**配置与调用路径**；跨路径判断"数值
  等价"应以 `torch.load` 后的张量比对（`torch.equal`）为准，而不是文件 SHA。

### 配置字段（已清理）

保留：`N` / `y_in` / `y_out` / `H` / `D` / `flow_axis` / `space_radius` / `placement` / `input_scope` / `readout_scope` / `seed` / `input_dim` / `output_dim` / `hidden_dim`（仅 mlp 基线）/ 训练超参。

已删除（旧架构字段）：几何类型字段与边长字段、温度初始化与残差系数、最小间距与采样重试上限、读出头 dropout、以及**迭代轮数**；同时移除全部与球体无关的几何派生量。`dropout` 相关训练增强一并移除（参数集合中不再有读出头 dropout）。

**当前字段清单**：`N` / `y_in` / `y_out` / `H` / `D` / `flow_axis` / `space_radius` / `placement` / `input_scope` / `readout_scope` / `input_dim` / `output_dim` / `hidden_dim`（仅 mlp 基线用）/ 训练超参。**架构中不存在任何迭代轮数参数**（阶段 2 为单遍逐层递推）。

`to_dict()` / `describe()` / `apply_overrides` 已同步（新增字段缺席会在 `Config(**base.to_dict())` 往返时被静默丢弃，故必须纳入）。

### 产物命名与元数据

- 冒烟产物：**完全默认组合**（neuron3d + SMALL_CONFIG 的 N/y/H/D/seed/**batch_size** + flow_axis=z + 两个 any_isolated + `space_radius=0`）退化为 `_verify/smoke.pt`；其它任何组合（换 arch / 流向轴 / scope / 覆盖 N、y、H、D、seed、batch_size、space_radius）为 `_verify/smoke[_ar{arch}]_{完整配置指纹}.pt`。
- **指纹格式**：`N{N}_y{y_in}x{y_out}_H{H}_D{D}_pl{placement}_ax{axis}_is{scope}_rs{scope}_bs{batch_size}[_R{space_radius}]_s{seed}` —— **指纹维度必须与"默认判定"维度严格对齐**（历史缺陷：默认判定含 `batch_size` / `space_radius` 而指纹不含，导致 `--batch-size 64` 与 `--space-radius 0.9` 落回同名文件、互相覆盖取证产物；`arch` 缺维度亦曾导致 `--arch mlp` 与主模型互覆）。`run_smoke_matrix.py` 会把实际落盘产物名与按此格式**独立预测**的名字逐条比对。
- 限批产物：`verify_<bpe>_N{N}_y{y_in}x{y_out}_H{H}_D{D}_pl{placement}_ax{axis}_is{scope}_rs{scope}_s{seed}[_tag].pt`。
- 全量产物默认路径：与 `DEFAULT_CONFIG` 完全同配置时用 `model.pt`，其它配置 `full_N{N}_y{..}_H{..}_D{..}_pl{..}_ax{..}_is{..}_rs{..}_s{seed}[_tag].pt`（**不**复用限批前缀，避免 `full_verify_0_...` 这类误导名）。
- 指纹维度含 `flow_axis` / `input_scope` / `readout_scope` / `placement` / `seed` / `H` / `D` / `N` / `y`（架构中无迭代轮数，故指纹不含该维度） —— `seed` 影响突触采样（不同 seed 即不同边集），必须进指纹。
- **同一路径的多次写入必须逐位一致**：默认组合专属的 `smoke.pt` 会被多条等值组合写入（裸跑 / 显式传等值 scope / `--seed 0` / `--preset default`），`run_smoke_matrix.py` 对此做强断言并记录 `artifact_sha256_at_end` 等字段（见上节 D1）。
- `model_state_dict` 中的拓扑 buffer：`topo_index` / `edge_offset` / `edge_perm` / `edge_perm_in` / `neuron_in_edge_reach` / `edge_dst_in` / **`level_edge_reach` / `level_node_reach`** / `in_scope_mask` / `out_scope_mask` / 度分布 —— 全部 `register_buffer`，故 `.to(device)` 与 `state_dict()` 都能搬运/持久化（F16 设备契约）。
- checkpoint 元数据新增/更新：`placement` / `flow_axis` / `space_radius` / `effective_space_radius` / `min_space_radius` / `max_space_radius` / `input_scope` / `readout_scope` / `topology_stats`（几何指纹与统计）/ `dag_selfcheck`（无环、严格上行、拓扑序覆盖）。