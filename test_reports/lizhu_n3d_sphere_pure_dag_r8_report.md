# n3d_sphere 纯球形分层 DAG 架构重写 —— 离朱独立测试报告

- **测试对象**：`n3d_sphere/`（config.py / model.py / train.py / data.py / utils.py）
- **测试依据**：绑定的《待测试功能说明》（纯球形分层 DAG 架构重写）
- **环境**：Windows + Python 3.12.10 + torch 2.14.0+cpu（CPU only，无 CUDA）；MNIST 本地 IDX 复用 `data/mnist`
- **测试时间**：2026-09-23
- **测试类型**：单元测试（独立构造判据）、编译/导入检查、接口(CLI)测试、E2E（冒烟 / 限批真实训练循环）、一期回归与产物纪律

---

## 一、测试概览

| 测试类型 | 用例数 | 通过 | 失败 | 结论 |
|---|---|---|---|---|
| 独立单元测试 | 32 | 29 | **3** | 3 项均指向**真实缺陷**（非测试误报） |
| 编译 / 导入检查 | 2 | 2 | 0 | 通过 |
| 接口 / CLI 测试 | 12 | 12 | 0 | 通过 |
| E2E（冒烟 + 限批训练） | 8（含于上） | 8 | 0 | 13/13 冒烟判据 PASS、退出码 0 |
| 一期回归与产物纪律 | 3 | 3 | 0 | 9/9 PASS、loss 精确、SHA256 全一致 |
| **合计** | **49** | **46** | **3** | 存在 1 项高危缺陷 |

产物与证据：
- 独立单元测试脚本：`.lizhu_env/lizhu_tests/lizhu_n3d_r8_sphere_tests.py`（输出 `r8_unit_run.txt`）
- CLI/E2E 测试脚本：`.lizhu_env/lizhu_tests/lizhu_n3d_r8_cli_tests.py`（输出 `r8_cli_run.txt`）
- 语义隔离探针：`_r8_probe6.py` / `_r8_probe7.py` / `_r8_probe10.py`
- 全部判据**未复用**说明中列出的现有脚本，均为本轮独立构造（双重循环重算 / 独立 Kahn+DFS / 有限差分 / 独立重写前向）

---

## 二、缺陷清单

### 【高危 M1】`model.stage2_propagate` 源激活槽位与边顺序错配 —— 前向计算读错上游神经元

**位置**：`n3d_sphere/model.py` 第 730-745 行（`stage2_propagate`）

**成因**：`_build_edge_groups` 已把边按 CSR 顺序（`edge_perm`）重排，但 `stage2_propagate` 仍用
**未重排**的 `self.edge_src` 构造源槽位：

```python
perm = self.edge_perm
dst_all = self.edge_dst.index_select(0, perm)      # [E] 已按 CSR 顺序
w_all   = self.edge_weight.index_select(0, perm)   # [E] 已按 CSR 顺序
slot_in_edges = torch.repeat_interleave(          # ✗ 按 edge_offset 分组计数
    torch.arange(self.N, device=...), total_counts)  #   但边已重排 -> 槽位错配
msg = a_up.index_select(0, slot_in_edges) * w_all + rel
```

`repeat_interleave(arange(N), counts)` 隐含假设“第 e 条边属于第 `slot[e]` 号神经元”，
该假设仅在边按 `edge_src` 原始顺序排列时成立。边一旦被 `edge_perm` 重排，
**边 `e` 的权重取到了错误源神经元的激活**：`a_up[slot_shipped[e]]` 而非 `a_up[edge_src[edge_perm[e]]]`。

**实测证据**（`_r8_probe6.py`，同一份权重、同一 `a_in`，只切换源槽位映射）：

| 配置 | E | 边错配比例 | forward vs 现行映射 | forward vs 规格映射 | 相对误差 |
|---|---|---|---|---|---|
| N=16, y=2x2, seed=9 | 25 | 部分 | 4.8e-07（逐位复现） | **5.53e-01** | 0.405 |
| N=64, y=4x4（SMALL） | 181 | **174/181** | 6.6e-07（逐位复现） | **5.22e-01** | 0.445 |
| N=256, y=8x8（DEFAULT） | 903 | — | 5.4e-07（逐位复现） | **4.79e-01** | 0.493 |

即：`forward` 与“现行（错配）映射”逐位一致（≈6e-7，float32 噪声级），
与“按规格边界对应源激活”的正确映射相差约 **0.5 个绝对 logit**，
约为输出量级（|logits|max≈0.97~1.44）的 **41%~55%**。

**影响**：
1. 前向传播并非规格所定义的“每条边读取其起点神经元激活”，而是把激活按错位槽位分发，计算结果系统性错误；
2. 反向传播中 `edge_weight` / `w.r.t. a_in` 的梯度同样沿错误边分配，训练信号被破坏；
3. 冒烟 13 条判据、`verify_scope_and_fingerprint.py` 的“d(sum a_up)/d(a_in) 非零元素=486/512>0”
   等**只能证明梯度路径存在、无法证明梯度对应正确边**，故全部漏检。

**建议修复**（一行改动）：
```python
slot_in_edges = self.edge_src.index_select(0, perm)   # 与 dst_all / w_all 同序
```
（或保持 `slot_in_edges` 不变，改为不重排 `dst_all`/`w_all`，两者必须同序。）

**回归判据建议**：新增“对每条边，其贡献必须等于 `edge_weight[k] * (a_up[src_k] + a_in[src_k])`”的逐边断言；
本报告 F4 / F4b 两项用例即为可复用的最小判据。

---

### 【中 M2】`topo_index` 不是流向轴升序，违反文档契约（`topo_matches_axis_order=0`）

**位置**：`n3d_sphere/model.py` 第 505-553 行（`_build_topo_order`）、第 874-887 行（`connectivity_selfcheck`）

**实测**（`_r8_probe7.py` / `_r8_probe10.py`）：

| 配置 | N | `topo_matches_axis_order` | topo 序中轴坐标是否单调 |
|---|---|---|---|
| SMALL_CONFIG | 64 | **0.0** | 否（27/64 位置偏离，最大偏差 0.4243） |
| N=108, flow_axis=x | 108 | **0.0** | 否（56/108） |
| DEFAULT_CONFIG | 256 | **0.0** | 否（167/256） |

根因（两处叠加）：
1. `_build_fcc_positions` 末段的“按流向轴坐标升序”排序**实际是空操作**：
   `key = positions[:, axis] * (N+1) + arange(N)` 本身已按索引单调，故
   `torch.argsort(key, stable=True)` 恒等于恒等置换；`neuron_pos` 仍保持“按壳层距离排序”
   （实测 `neuron_pos` 并非轴升序，`axis[0]=-0.4243`、`axis[27]=-0.6364`，节点 0 却排在节点 27 前）。
2. `_build_topo_order` 的 Kahn 实现用 `ready.sort()`（按**神经元编号**排序）而非按轴坐标排序，
   于是编号较小的“更高壳层”节点被提前输出，导致拓扑序中轴坐标非单调。

**影响评估**：
- `topo_index` 仍是**合法拓扑序**（独立 Kahn / DFS 三色验证：无环、覆盖全部 N、每条边 `rank[src]<rank[dst]`）；
- 边分组实际按**流向轴坐标**排列（`key = rank[edge_src]` 中 `rank` 恰为轴序），故源神经元始终先于目标神经元被处理，
  数值上未产生额外错误（M1 与 M2 相互独立）；
- 但违反文档承诺（`_build_topo_order` 返回值说明“与前向遍历顺序一致”、
  `connectivity_selfcheck["topo_matches_axis_order"]` 的语义），且 `config.num_layers`（=7）远小于 N，
  说明大量神经元共享同一轴坐标——即“轴升序”需明确 tie-break 规则。

**建议**：将 Kahn 的就绪集排序键改为 `(轴坐标, 编号)`，或直接把 `topo_index` 定义为
`argsort(轴坐标, stable=True)` 并保留独立无环断言（两者等价且更简单）；
同时修正 `connectivity_selfcheck` 的比对口径或文档表述。

---

### 【低 M3】冒烟产物名不含 `arch`，`--arch mlp` 与默认路径互相覆盖

**位置**：`n3d_sphere/train.py` 第 132-158 行 `smoke_checkpoint_path()`

实测：先跑 `--smoke-test --arch mlp` 再跑项目自带 `verify_sphere_dag.py all`，
R7 报 FAIL（读到的是 mlp 产物：`stage=smoke arch=mlp E=0 S_in=0`，缺 `topology_stats` 全部键）；
重跑一次 `python n3d_sphere/train.py --smoke-test`（neuron3d）后 R7 恢复 PASS、整体退出码 0。

说明：`smoke_checkpoint_path` 仅由 `flow_axis/input_scope/readout_scope` 决定，
不含 `arch` 维度，两种架构写同一文件、互相覆盖，会破坏留痕（限批/全量产物命名已含 `arch` 之外的指纹，问题仅限冒烟档）。

**建议**：冒烟产物名追加 `_ar{arch}`。

---

## 三、各测试类型详细结果

### 3.1 独立单元测试（32 项，29 通过）

| 编号 | 用例 | 结果 |
|---|---|---|
| A1 | 半径公式精确值（R_min=0.701840、R_max=1.754601，与公式逐位一致；N/H/D 非法参数抛错） | PASS |
| A2 | `space_radius` 恰好等于 R_min / R_max 接受；越界 ±1e-6 拒绝（两侧，消息含“越界”）；负值拒绝 | PASS |
| A3 | `to_dict()` / `describe()` 含全部新增字段且 `Config(**to_dict())` 往返无损；已删字段（topology/L/tau_init/alpha/min_neuron_dist/max_sample_tries/dropout/`equivalent_sphere_radius`）确实不存在 | PASS |
| A4 | 枚举取值域：`flow_axis`/`input_scope`/`readout_scope`/`placement` 非法值抛 `ValueError`，合法值接受 | PASS |
| B1 | 最近邻距 == 2H（≤1e-5）、晶格常数 == 2√2·H、放置半径 ≤ R_max（3 组配置） | PASS |
| B2 | 真 FCC 格点结构：以 a/2 为单位的坐标分量为整数、最近邻距 == a/√2 | PASS |
| B3 | 神经元坐标与 seed 无关（逐位相同）；突触坐标随 seed 变化 | PASS |
| B4 | 三个流向轴：输入突触轴偏移 ≤0、输出 ≥0、距所属神经元 ≤H；半径按体积均匀（E[(r/H)³]≈0.5、E[r/H]≈0.75，排除线性采样） | PASS |
| C1 | 独立双重循环（N=16,y=2）重算神经元级边集与代表连接，与 buffer 逐位一致；代表间距 == 该对 amin 且 ≤ D | PASS |
| C2 | **反例**：改用 amax 判据会得到不同 E，证明实现取的是 amin | PASS |
| C3 | 孤立突触定义独立重算（合法对端=其他神经元且 z 上行），与 `input/output_isolated_mask` 逐位一致 | PASS |
| D1 | 独立 Kahn + DFS 三色：无环、覆盖全部 N、每边 rank[src]<rank[dst] 且 z_src<z_dst；拓扑序为 0..N-1 排列 | **FAIL（契约 M2）** |
| D2 | CSR 分组：`edge_offset` 单调、总数 == E、`edge_perm` 分组与拓扑位置一致 | PASS |
| E1 | 同 seed 两次构造：14 个几何/拓扑 buffer 与全部参数逐位相同 | PASS |
| E2 | 异 seed：神经元坐标逐位相同、边集不同 | PASS |
| F1 | 零化 `a_in` 后 `a_up` 改变，且确有**不在 S_in** 的下游神经元被打到；非 S_in 的 `a_in` 恒为 0 | PASS |
| F2 | 独立数值有限差分核对 `∂(Σa_up)/∂a_in` 非零（双副本未被覆盖） | PASS |
| F3 | `all_isolated` 下输入梯度非零（仍可微）、readout 两掩码互斥（`readout_in_mask == ~out_scope_mask`） | PASS |
| F4 | 独立复现 stage-2 累加：现行映射逐位复现 forward，**规格映射相差 5.5e-1** | **FAIL（高危 M1）** |
| F4b | CSR 源槽位不变量：`repeat_interleave(arange(N),counts) != edge_src[edge_perm]`（174/181 边错配） | **FAIL（高危 M1）** |
| F5 | 参数形状 `W_in[input_dim,|S_in|]`/`edge_weight[E]`/`neuron_bias[N]`/`W_out[output_dim,N]`；`count_dense_weight_tensors()==0`；无 `[N*y_out,N*y_in]` 可学习权重 | PASS |
| G1 | D 极小（E=0）给出可读异常（含 D 与距离统计），非张量崩溃 | PASS |
| G2 | 前向非法输入：非 Tensor→TypeError；1D/3D/维度不符→ValueError | PASS |
| G3 | `stage2_propagate` 形状不一致 / rounds<1 → ValueError | PASS |
| G4 | 三个流向轴：连接沿轴严格上行、E>0、指纹一致 | PASS |
| G5 | `get_connection_stats` / `get_topology_stats` 键集与数值口径（含 dual_copy_count 独立重算） | PASS |
| G6 | CLI 无任何几何类型开关；`--flow-axis/--space-radius/--input-scope/--readout-scope/--placement` 齐备 | PASS |
| G7 | 指纹命名含 `pl/ax/is/rs/seed`、不含几何类型维度；全量名无 `verify_` 前缀；不同配置不同名；非法 tag 报错；限批一律重定位 `_verify/` | PASS |
| G8 | `--smoke-test --input-scope all_isolated` 等几何参数纳入 explicit 判定（不静默丢弃）；`apply_overrides` 往返保留；`run_full_training` 兼容参数齐备 | PASS |
| G9 | 12 组负值参数 → `ValueError`，`main()` 返回非 0；argparse choices 拒绝非法枚举 | PASS |
| G10 | MLP 基线提供同名接口（`count_dense_weight_tensors`/`get_connection_stats`/`get_topology_stats`） | PASS |
| G11 | 小规模前向+反向：全部可学习参数梯度范数 > 0、loss 有限 | PASS |

### 3.2 编译 / 导入检查（2 项，全通过）

- `python -m py_compile` 六个源码文件：退出码 0；
- 包导入 `n3d_sphere` 及 config/model/train/data/utils：成功；`Config` 字段集为 26 个数据字段，
  新增 5 个（`flow_axis/space_radius/placement/input_scope/readout_scope`）齐备，已删字段无一残留。

### 3.3 接口 / CLI 测试（12 项，全通过）

| 用例 | 结果 |
|---|---|
| CLI1 负值参数（`--n/--d/--y-in/--space-radius/--seed`）→ 退出码 2 + 可读错误 | PASS |
| CLI2 非法枚举（`--flow-axis w` / `--input-scope bogus` / `--placement bcc` / `--readout-scope nope`）→ 退出码 2 | PASS |
| CLI3 `--help` 退出码 0、无几何类型开关、新开关齐备 | PASS |
| CLI4 `--space-radius 99.0` → 退出码 2 且消息含“越界”；合法 0.5 亦以退出码 2 结束（见备注） | PASS（见备注 B1） |
| E2E1 `--smoke-test` 默认：退出码 0、**13 PASS / 0 FAIL**、结论“全部通过”、耗时 6.8s | PASS |
| E2E2 `--smoke-test --input-scope all_isolated`：日志/config/产物 `smoke_axz_isall_rsany.pt` 均生效 | PASS |
| E2E3 `--smoke-test --flow-axis x`：产物 `smoke_axx_isany_rsany.pt`，`topology_stats.flow_axis==0` | PASS |
| E2E4 限批真实训练（`--max-batches 2 --epochs 1 --preset small --tag lz_r8`）：退出码 0、产物落 `_verify/`、文件名含全部指纹、顶层与 config 元数据齐备、无已删字段、`batches_per_epoch==2`、DAG 自检通过 | PASS |
| E2E5 不同 `flow_axis` 的限批产物互不覆盖（各自内容正确） | PASS |
| ART1 全量产物名 `full_N64_..._s7_t1.pt` 无 `verify_` 前缀；限批名一律重定位 `_verify/` | PASS |
| ART2 全程未生成正式产物 `checkpoints/n3d_sphere/model.pt`；一期三个产物存在且未被触碰 | PASS |
| E2E6 `--smoke-test --arch mlp`：退出码 0 | PASS |

### 3.4 一期回归与产物纪律（3 项，全通过）

- `python n3d_proto/train.py --smoke-test`：**退出码 0，9/9 PASS**，`loss = 2.419689`（与要求逐位一致）；
- `git status --porcelain n3d_proto` **无任何输出** → 一期文件零改动；
- 三个一期产物 SHA256 与要求**逐字符一致**：
  - full `888556B0913C9F46419A674117FD13A99F2C71BA6692367A839DB873A58D8924` ✅
  - highacc `9F21AC34C91977FE60F623138F3DEE24428E44B473BD86DE60A8B577F11BE3F8` ✅
  - capacity `0F7CF500C256BFE41408E3DC68CED9316C21EE4EE94527790F861C76E6C35011` ✅

### 3.5 零残留与可复现性

- 立方体几何残留扫描（源码 + README）：`cube/cuboid/box/--topology/--geometry/--geom/--shape/geometry_type/`
  `max_sample_tries/equivalent_sphere_radius/tau_init/dropout` 全部 clean；
  命中的“立方”均为**面心立方（FCC）**与**立方根**术语（球体语境，不构成残留）；
  `min_neuron_dist`（config.py:26）与 `alpha`（model.py:48、README.md:162）仅出现在“不存在/已移除”的说明句中；
- 可复现性：同 seed 两次构造的 14 个几何/拓扑 buffer 与全部参数逐位相同（E1）。

### 3.6 现有验证脚本的复核（说明中要求“不得只依赖它们”）

| 脚本 | 结果 |
|---|---|
| `verify_config_contracts.py`（C1-C6） | 退出码 0，全部 PASS |
| `verify_scope_and_fingerprint.py`（S1-S5） | 退出码 0，全部 PASS |
| `verify_sphere_dag.py all`（R1-R7） | 首次退出码 1（R7 FAIL）；定位为**测试次序耦合**：前一步 `--arch mlp` 冒烟覆盖了 `smoke.pt`；重跑 neuron3d 冒烟后 **R1-R7 全 PASS、退出码 0** |

（这些脚本覆盖了几何/DAG/去重/判据/指纹，但**均未对前向累加做逐边数值核对**，故 M1 未被其捕获。）

---

## 四、失败用例归因与修复建议（汇总）

| 编号 | 缺陷 | 严重度 | 建议修复 | 复现/回归判据 |
|---|---|---|---|---|
| M1 | `stage2_propagate` 源槽位与 `edge_perm` 错配，前向读错上游激活（相对误差 41%~55%） | **高危** | `slot_in_edges = self.edge_src.index_select(0, perm)` | 单元用例 F4 / F4b；或逐边断言 `contrib_k == w_k*(a_up[src_k]+a_in[src_k])` |
| M2 | `topo_index` 非轴升序（`topo_matches_axis_order=0`），违反文档契约；`_build_fcc_positions` 的轴排序为空操作 | 中 | 就绪集按 `(轴坐标, 编号)` 排序，或直接用 `argsort(轴, stable=True)` 定义 `topo_index` | 单元用例 D1；`connectivity_selfcheck()["topo_matches_axis_order"] == 1` |
| M3 | 冒烟产物名不含 `arch`，`--arch mlp` 与默认互相覆盖，破坏 `verify_sphere_dag.py` R7 | 低 | 产物名追加 `_ar{arch}` | 先跑 `--arch mlp` 冒烟再跑 `verify_sphere_dag.py all` 应仍为全 PASS |

**备注**
- B1：`--smoke-test --space-radius 0.5` 以退出码 2 结束属**预期行为**——`SMALL_CONFIG`（N=64,H=0.15,D=0.25）
  的窗口为 [0.6739, 1.7971]，0.5 越界被构造期拒绝；该用例仅验证“越界必拒”，非缺陷。
- B2：`n3d_sphere/utils.py` 保留了 `segment_softmax` / `build_scatter_matrix` / `connection_density` 等
  一期工具函数（新架构已不使用）。说明中的“零残留”针对立方体几何与几何类型开关，故不视为违规，
  但可作为后续清理项记录。
- B3：本会话的文件写入工具不可用（沙箱内 hardlink 报 EISDIR），测试脚本改由 shell 落盘，
  脚本内容与判据未受影响。

---

## 五、环境说明

| 项 | 值 |
|---|---|
| Python | 3.12.10 (MSC v.1943 64bit) |
| torch | 2.14.0+cpu（`torch.cuda.is_available() == False`） |
| 数据 | 复用工程内 `data/mnist` IDX 文件（未联网下载） |
| 冒烟耗时 | 默认冒烟 6.8s / 限批 2 batch 训练 7.6s（远低于 120s 阈值） |
| 未执行项 | 无（GPU 相关用例不适用于本环境，已在 CPU 下全量执行） |

## 六、结论

- **构建期配置契约、FCC 几何、半球采样口径、连接去重与代表连接、判据开关、产物指纹与命名、
  CLI 错误处理、一期回归与产物纪律：全部验证通过**（冒烟 13/13 PASS、一期 9/9 PASS、三个 SHA256 一致、`n3d_proto` 零改动）。
- **但前向传播存在一处高危实现缺陷（M1）**：`stage2_propagate` 的源激活槽位与 CSR 边顺序错配，
  使每条边读到错误源神经元的激活，实测与规格定义的差异达 logit 量级的 41%~55%，
  且反向梯度沿错误边分配。该缺陷不触发任何异常、不影响冒烟 13 条判据与现有验证脚本，
  因此**必须修复并补充“逐边数值正确性”判据**，否则本架构的“几何/连接规则正确”无法转化为“网络计算正确”。
- 另有 1 项中severity 契约不一致（M2）与 1 项低severity 产物覆盖（M3）建议一并修复。
