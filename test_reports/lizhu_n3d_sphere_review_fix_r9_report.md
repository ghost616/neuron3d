# n3d_sphere 皋陶审查修复轮 —— 离朱独立测试报告

- **测试对象**：`n3d_sphere/`（config.py / model.py / train.py / data.py / utils.py / README.md）
- **测试依据**：绑定的《n3d_sphere 皋陶审查修复轮 —— 待测试功能说明》
- **环境**：Windows + Python 3.12.10 + torch 2.14.0+cpu（CPU only，无 CUDA）；MNIST 复用本地 `data/mnist`
- **测试类型**：独立单元测试、编译/导入检查、接口(CLI)测试、E2E（四种 scope 冒烟 / 限批真实训练 / arch 对照）、一期回归与产物纪律
- **测试脚本**：`.lizhu_env/lizhu_tests/lizhu_n3d_r9_fix_tests.py`（29 项）、`.lizhu_env/lizhu_tests/lizhu_n3d_r9_cli_tests.py`（8 项）

---

## 一、测试概览

| 测试类型 | 用例数 | 通过 | 失败 | 结论 |
|---|---|---|---|---|
| 独立单元测试 | 29 | 26 | **3** | 均为真实问题（1 中 + 2 低），非测试误报 |
| 编译 / 导入检查 | 1 | 1 | 0 | `compileall -q n3d_sphere` 退出码 0 |
| 接口(CLI) / E2E | 8 | 8 | 0 | 四 scope 冒烟 15/15、限批训练、产物纪律全通过 |
| 一期回归与产物纪律 | 3 | 3 | 0 | 9/9 PASS、loss 精确、SHA256 全一致、`n3d_proto` 零改动 |
| **合计** | **40** | **37** | **3** | 无高危缺陷；**上轮 2 项缺陷已确认修复** |

> 另在 CLI/E2E 阶段独立发现 1 项**新回归**（`--arch mlp` 冒烟失败），已单列于 M1。

**上轮（第 8 轮）缺陷的回归确认**

| 上轮缺陷 | 本轮结论 | 证据 |
|---|---|---|
| 【高危】`stage2_propagate` 源槽位与 `edge_perm` 错配（前向读错上游神经元，相对误差 41%~55%） | ✅ **已修复** | 现用 `slot_src = self.edge_src.index_select(0, perm_in)`；独立参考实现逐元素比对 N=16/64/256 最大偏差 3e-8~6e-8（容差 1e-5）；错误映射反例仍发散 87.5%（判据敏感） |
| 【中】`topo_index` 非流向轴升序（`topo_matches_axis_order=0`） | ✅ **已修复** | N=64/108/256 下 `topo_index == argsort(轴坐标, stable=True)`，`topo_matches_axis_order==1.0`，轴值沿拓扑序单调 |
| 【低】冒烟产物名不含 `arch` | ✅ **已修复** | `--arch mlp` 产物为 `smoke_armlp_N64_y4x4_..._s42.pt`，neuron3d 产物不受影响 |

---

## 二、问题清单

### 【中 M1】`--smoke-test --arch mlp` 回归：判据 3 硬编码参数名 `W_out`，MLP 对照基线必然失败（退出码 1）

**位置**：`n3d_sphere/train.py` 第 1003-1016 行（`run_smoke_test` 判据 3）

**成因**：判据 3 用 `grad_norms.get("W_out", -1.0)` 取输出层梯度范数。`MLPBaseline` 的输出层参数名是 `fc2.weight` / `fc2.bias`，不存在 `"W_out"`，故取到哨兵 `-1.0`，判据 3 判定 FAIL。

```
[PASS] [1] 前向输出形状 == [B, output_dim] —— logits.shape=(32, 10)
[PASS] [2] 反向无错误（全部可学习参数都有梯度） —— 收集到 4 个参数的梯度，无缺失
[FAIL] [3] 参与 loss 的参数梯度范数 > 0 —— W_out grad_norm=-1.000000e+00
[PASS] [4] loss 非 NaN/Inf —— loss=2.298935
```
实测：`python n3d_sphere/train.py --smoke-test --arch mlp` → **5 PASS / 1 FAIL / 退出码 1**（结论“存在失败项”）。
而 MLP 的 4 个参数梯度范数实际全部正常（`fc1.weight 6.22`、`fc1.bias 0.20`、`fc2.weight 4.09`、`fc2.bias 0.19`）。

**影响**：
- `--arch mlp` 对照基线的冒烟验收**无法通过**（退出码 1），CI/验收脚本若包含该分支会红；
- 该分支下产物仍会写出（`smoke_armlp_...pt`），但结论为失败，易误导；
- 与说明“CLI 不得再有 `--arch` 之外分叉”“arch=mlp 走完全相同训练循环”的设计意图冲突——判据本身按 neuron3d 的参数名硬编码。

**建议修复**：按 arch 取输出层参数名（`is_neuron3d` 时 `W_out`，否则 `fc2.weight`），或改为“全部可学习参数梯度范数 > 0 且至少一个非零”而不绑定参数名：
```python
key = "W_out" if is_neuron3d else "fc2.weight"
w_out_gnorm = float(grad_norms.get(key, -1.0))
```

---

### 【中 M2】默认冒烟不再写 `smoke.pt`，遗留**过期产物**，与 README/spec 明确承诺不符

**位置**：`n3d_sphere/train.py` 第 1153-1159 行（`run_smoke_test` 调用 `smoke_checkpoint_path`）

**成因**：`run_smoke_test` 始终传入 `smoke_fingerprint(config)`，因此默认组合（z + 两个 `any_isolated`）不再退化为 `smoke.pt`，而是写指纹名：
```
smoke_checkpoint_path(..., arch, smoke_fingerprint(config))
  -> _verify/smoke_N64_y4x4_H0.15_D0.25_plfcc_axz_isany_rsany_s42.pt
smoke.pt 仅在 fingerprint == "" 时可达到，而没有任何调用点传空串
```
而说明与文档仍承诺“默认组合仍为 `smoke.pt`”：

| 出处 | 承诺 |
|---|---|
| 本轮测试说明 §3 | “默认组合仍为 `smoke.pt`” |
| `README.md:277/281/292` | “默认产物 `smoke.pt`… loss = 2.326995849609375（可 `torch.load` 复核 `_verify/smoke.pt`）” |
| `current_spec.md:78/189` | “默认组合产物 `_verify/smoke.pt` 的 loss = 2.326995849609375…退化为 `_verify/smoke.pt`” |
| `verify_sphere_dag.py:498/588` | 读取 `_verify/smoke.pt` |

**实测**（跑完四次 `--smoke-test` 之后）：

| 产物 | 最后写入 | loss | config 含 `T` | 与文档一致 |
|---|---|---|---|---|
| `_verify/smoke.pt` | **2026/9/23 21:13:54（陈旧）** | **2.4805126190185547** | **是（T=2）** | ❌ 文档称 2.326995849609375、无 `T` |
| `_verify/smoke_N64_..._isany_rsany_s42.pt` | 本轮刷新 | 2.326995849609375 | 否 | ✅ |

即 `smoke.pt` 是**上一版代码**（仍含 `T` 字段）留下的历史产物，当前代码路径永远不会再写它，README 给出的复核命令
`python -c "torch.load('.../smoke.pt')..."` 会打印出与文档不符的数值。

**影响**：说明 §4（F1 核心）要求“实测数字必须有可 `torch.load` 复核的产物支撑”，该条对 `smoke.pt` 不成立；
`verify_sphere_dag.py` 的 R7 正读该陈旧文件（因其只做“键齐全 + DAG 自检”的结构检查，未校验 loss 数值，故仍 PASS，掩盖了不一致）。

**建议修复（二选一）**：
1. 默认组合不传 fingerprint（保留 `smoke.pt` 历史语义），其它组合才用指纹名；或
2. 文档/`verify_sphere_dag.py` 统一改读指纹名 `smoke_N64_y4x4_H0.15_D0.25_plfcc_axz_isany_rsany_s42.pt`，并**删除陈旧 `smoke.pt`**（否则永远是误导性残留）。
推荐方案 2（指纹名信息更完整）＋ 删除陈旧产物。

---

### 【低 M3】`current_spec.md` 仍按 `T` 维度描述产物命名，且把 `T` 列为“保留字段”

**实测**（规格与代码逐条对照）：

| current_spec.md 行 | 规格表述 | 代码实际 |
|---|---|---|
| 181 | 保留字段列表含 `` `T` `` | `Config` **已无** `T`（`to_dict()` 无该键） |
| 190 | 限批产物 `verify_<bpe>_N{N}_..._H{H}_D{D}_T{T}_pl{...}...` | `verify_5_N64_y4x4_H0.15_D0.25_plfcc_axz_isany_rsany_s42.pt`（**无** `_T{T}`） |
| 191 | 全量产物 `full_N{N}_..._D{D}_T{T}_pl{...}...` | `full_N64_y4x4_H0.15_D0.25_plfcc_axz_isany_rsany_s42.pt`（**无** `_T{T}`） |
| 192 | 指纹维度含 `... / H / D / T / N / y` | 指纹维度为 `N/y/H/D/placement/axis/两个 scope/seed`（**无** `T`） |

另 `current_spec.md:13` 保留“T 4→6 掉 0.68pp”的**一期历史实验记录**——该句描述历史实验，可保留，但需与“架构已无 T”明确区分。

**影响**：纯文档不一致，不影响运行；但会让后续读者误以为仍存在轮数维度，与“删除迭代轮数参数”的修复目标相悖。

---

### 【低 M4】零残留扫描：3 处遗留（旧实现名 / 已删字段名出现在非源码文件）

| 位置 | 命中词 | 性质 |
|---|---|---|
| `n3d_sphere/utils.py:228` | `W_conn_sparse` | 一期 `segment_softmax` 的 docstring 仍写“边级 logits（= -edge_dist / tau + W_conn_sparse）”；该函数已被标注为不再被新架构调用，但旧实现名残留 |
| `.module_agent/n3d_sphere/current_spec.md:183` | `tau_init` / `max_sample_tries` / `equivalent_sphere_radius` | “已删除：…”说明句（叙述性提及已删字段） |
| `.module_agent/n3d_sphere/module_definition.json`（2 条 description） | `tau_init` / `min_neuron_dist` / `max_sample_tries` / `dropout` | “已删除旧架构字段 …”说明句（叙述性提及） |

说明 §5 要求这些文件中不得出现“被删除的 CLI 开关与配置字段、以及旧架构的统计/缓存实现名”。上述命中均为**叙述性/遗留 docstring**，非可用代码或缓存实现。
注：项目自带 `verify_config_contracts.py` 的 C6 扫描**仅覆盖源码+README**（未含 `current_spec.md` / `module_definition.json`），故其仍 PASS；本报告按说明 §5 的**更宽范围**独立复扫得出上述结论。

---

### 【信息】M5：README 参数区间“取自 10 个 seed 记录”的口径需限定测点

README:322 称“参数区间 **35094 ~ 44489**（取自 `_verify/topology_snapshot.json` 的 10 个 seed 记录）”。
实测快照共 **60 条记录、6 个测点 × 10 seed**：

| 测点 | 参数区间 |
|---|---|
| **default_any_any_z（N=256,y=8x8,H=0.1,D=0.15）** | **35094 ~ 44489** ✅ |
| default_any_any_x | 35086 ~ 43719 |
| default_any_any_y | 37439 ~ 45285 |
| default_all_all_z | 13908 ~ 13932 |
| small_any_any_z | 9519 ~ 14217 |
| small_all_all_z | 5588 ~ 5600 |

区间数值**正确**，但“取自 10 个 seed 记录”的表述未限定测点，全快照实为 5588 ~ 45285。建议补一句“（默认测点 `default_any_any_z` 的 10 个 seed）”。

---

## 三、通过项详细结果

### 3.1 独立单元测试（29 项，26 通过）

| 编号 | 用例 | 结果 |
|---|---|---|
| 1a | `Config` / `ThreeDNeuronSpace` 均无 `T`/`t_steps`/`rounds`；`to_dict()`/`describe()` 无 `T`；旧字段与 `equivalent_sphere_radius` 不存在；`stage2_propagate` 已移除 | PASS |
| 1b | 半径公式（R_min=0.701840、R_max=1.754601）、边界恰取 R_min/R_max 接受、±1e-6 越界拒绝、枚举域校验、`to_dict` 往返一致 | PASS |
| 2a | **单遍递推 vs 独立参考实现**（拓扑序逐节点 + 按 `edge_dst` 掩码取入边）：N=16 E=25 偏差 3.0e-8；N=64 E=181 偏差 3.0e-8；N=256 E=903 偏差 6.0e-8（容差 1e-5） | PASS |
| 2b | **错误源映射反例**（按位置切片取源而非按边自身源）：与生产实现相对偏差 **87.5%** → 判据敏感 | PASS |
| 2c | **感受野不截断**：扰动第一层 S_in 神经元 `a_in` +1.0 → 最深 13 个神经元中 **12 个** `a_up` 改变；最深层祖先层覆盖 **9/9 层** | PASS |
| 2d | **双副本语义**：零化 `a_in` 后 `a_up` 改变、且有非 S_in 下游神经元被打到；`d(Σa_up)/d(a_in)` 非零；数值有限差分与 autograd 一致 | PASS |
| 2e | `stage2_recurrence` 形状/设备守卫 | PASS |
| 3a | **readout 严格口径**：`h` 非零列全属 `S_out`；把 `S_out` 之外置零后 `h` **逐位不变**；`h == a_up ⊙ S_out` | PASS |
| 3b | `W_out` **非零梯度列数 == |S_out|（17 == 17）**；`S_out` 之外列梯度恰为 0 | PASS |
| 3c | `readout_scope` 真正影响前向：`any` vs `all` 的 logits 相对差 **53.5%**（\|S_out\| 17 vs 7） | PASS |
| 3d | `input_scope` 亦影响 logits | PASS |
| 4a | **入边表自洽**：`edge_perm_in` 为排列；`neuron_in_edge_reach` 区间内目标恰为 `topo_index[pos]`（N=16/64/256 全量核对）；每边 `rank[src] < rank[dst]` | PASS |
| 4b | **`topo_index == argsort(轴坐标, stable=True)`**、轴值单调、`topo_matches_axis_order==1`（上轮 M2 已修复） | PASS |
| 5a | 最近邻距 == 2H、晶格常数 == 2√2·H、放置半径 ≤ R_max、D 极小可读异常 | PASS |
| 5b | 参数形状 `W_in[784,|S_in|]`/`edge_weight[E]`/`neuron_bias[N]`/`W_out[10,N]`；`count_dense_weight_tensors()==0`；无 `[N*y_out,N*y_in]` 权重 | PASS |
| 5c | **同 seed 可复现**：25 个 `state_dict` 键逐位相同 | PASS |
| 5d | 前向非法输入守卫（非 Tensor / 1D / 维度不符） | PASS |
| 6a | 默认产物 `smoke.pt` 数值复核 | **FAIL（M2）** |
| 6b | 四种 scope 产物存在、文件名含全部指纹维度、**四个 loss 互不相同**（2.326996/2.299028/2.290002/2.283543）、与 `smoke_scope_matrix.json` 逐条一致（loss 与 grad_norms） | PASS |
| 6c | 冒烟指纹命名：含 `N/y/H/D/pl/ax/is/rs/s`，**不含 `T`**；默认组合（fingerprint 为空）退化为 `smoke.pt`；非 neuron3d 含 `_ar{arch}`；不同配置不同名；限批/全量指纹均无 `_T` | PASS |
| 6d | **参数区间 35094 ~ 44489**：快照 `default_any_any_z` 10 条记录 min/max 恰为 35094/44489；README 与 spec 均含该区间且无旧值 37440 | PASS |
| 6e | 指纹名 `smoke_N64_..._isany_rsany_s42.pt` 的 loss == 2.326995849609375、四个梯度范数与 README 逐位一致、config 无 `T` | PASS |
| 6f | 规格产物命名模式与代码一致（无 `_T{T}`） | **FAIL（M3）** |
| 7a | 零残留扫描（9 个文件 × 30 个禁用词） | **FAIL（M4）** |
| 7b | `current_spec.md` 六个 heading 无重复 | PASS |
| 7c | `utils.py` 文件头有调用现状标注 | PASS |
| 8a | `--t-steps 2` / `--t 2` / `--t-steps 1` 均被拒绝（退出码 2） | PASS |
| 8b | `--help` 不含 `--t-steps`/`--t`，含 5 个几何开关 | PASS |
| 8c | `--n -1` / `--d -1` / `--space-radius 99` / 非法枚举 → 非 0 退出码 | PASS |

### 3.2 编译 / 导入检查（1 项，通过）

`python -m compileall -q n3d_sphere` 退出码 0。

### 3.3 接口(CLI) / E2E（8 项，全通过）

| 用例 | 结果 |
|---|---|
| E1 一期三件产物 SHA256 | 全一致（full `888556B0…8924`、highacc `9F21AC34…3F8`、capacity `0F7CF500…5011`） |
| E2 `compileall` | 退出码 0 |
| E3 **四种 scope 组合各跑一次冒烟** | 4/4 退出码 0、各 **15 PASS / 0 FAIL**、产物名各含全部指纹维度、**四个 loss 互不相同**、`smoke_scope_matrix.json` 与四个产物逐条一致；耗时各 6.7~6.9s |
| E4 限批真实训练（`--max-batches 2 --epochs 1 --preset small --tag lzr9`） | 退出码 0；产物 `verify_2_N64_y4x4_H0.15_D0.25_plfcc_axz_isany_rsany_s42_lzr9.pt` 落 `_verify/`；config 无 `T`；`batches_per_epoch==2`；`topo_matches_axis_order==1`；正式 `model.pt` 未被触碰 |
| E5 `--arch mlp` 冒烟 | 产物 `smoke_armlp_N64_y4x4_H0.15_D0.25_plfcc_axz_isany_rsany_s42.pt` 已按 arch 分名（上轮 M3 已修复）；但**判据 3 失败、退出码 1**（见 M1） |
| E6 项目自带验证脚本 | `verify_config_contracts.py`（C1-C6）、`verify_scope_and_fingerprint.py`（S1-S5）、`verify_sphere_dag.py`（R1-R7，含 R5b 逐边数值 / R5c 感受野 / R7b readout 生效性）**均退出码 0、无 FAIL** |
| E7 `git status --porcelain -- n3d_proto` | 输出为空（一期零改动） |
| E8 `--space-radius` 两侧越界（99.0 / 0.0001） | 均非 0 退出码 |

### 3.4 一期回归（硬约束）

| 项 | 结果 |
|---|---|
| `python n3d_proto/train.py --smoke-test` | 9/9 PASS、退出码 0、`loss = 2.419689` |
| 一期三件产物 SHA256 | 与要求逐字符一致 |
| `git status --porcelain -- n3d_proto` | 空 |
| `compileall -q n3d_sphere` | 退出码 0 |
| 同 seed `state_dict` 逐位复现 | 25/25 键相同 |

---

## 四、修复建议汇总

| 编号 | 问题 | 严重度 | 建议修复 |
|---|---|---|---|
| M1 | `--arch mlp` 冒烟判据 3 硬编码 `"W_out"` → 5 PASS/1 FAIL、退出码 1 | **中** | 按 arch 取输出层参数名（`fc2.weight`），或改为不绑定参数名的“全部参数梯度 > 0 且至少一非零” |
| M2 | 默认冒烟不再写 `smoke.pt`，遗留含 `T` 的陈旧产物；README/spec/verify_sphere_dag 承诺与实际不符 | **中** | 统一改读指纹名并删除陈旧 `smoke.pt`；或默认组合传空 fingerprint 恢复 `smoke.pt` 语义 |
| M3 | `current_spec.md:181/190/191/192` 仍按 `T` 描述字段与产物命名 | 低 | 删除保留字段列表中的 `T`，从命名模式与指纹维度说明中移除 `T`（保留 :13 的历史实验记录并标注为历史） |
| M4 | `utils.py:228`（`W_conn_sparse`）、`current_spec.md:183`、`module_definition.json` 2 处 description 含旧实现名/已删字段名 | 低 | 清理 `utils.py` docstring 中的旧参数名；叙述性提及可改为“旧架构字段（不再列出具体名）”或加入扫描白名单 |
| M5 | README 参数区间未限定测点（全快照 5588~45285） | 信息 | 补注“（默认测点 `default_any_any_z` 的 10 个 seed）” |

---

## 五、环境说明

| 项 | 值 |
|---|---|
| Python | 3.12.10 (MSC v.1943 64bit) |
| torch | 2.14.0+cpu（`cuda.is_available() == False`） |
| 数据 | 复用工程内 `data/mnist` IDX（未联网） |
| 耗时 | 单次冒烟 6.7~6.9s；限批 2 batch 训练 7.1s（远低于 120s 阈值） |
| 未执行项 | 无（GPU 相关不适用本环境） |

## 六、结论

- **本轮核心修复已验证有效**：阶段 2 改为按拓扑序单遍递推后，与独立参考实现的逐元素偏差仅 3e-8~6e-8（上轮错配缺陷导致的 ~0.5 logit 偏差消失）；readout 严格口径、入边表自洽、`topo_matches_axis_order==1`、`W_out` 非零梯度列数 == `|S_out|`、感受野贯通全部 9 层、`T` 参数彻底移除、CLI 拒绝 `--t-steps`/`--t`、四种 scope 冒烟 15/15 且 loss 互异、产物与 `smoke_scope_matrix.json` 一致、参数区间 35094~44489 有快照支撑、一期回归全绿 —— **均通过**。
- **仍存在 2 项中等问题**：`--arch mlp` 冒烟因判据 3 硬编码参数名而回归失败（M1，退出码 1）；默认冒烟的 `smoke.pt` 已不再被写入，留下含 `T` 的陈旧产物，与 README/spec 的可复核承诺不符（M2）。建议尽快修复并同步文档。
- 另有 2 项低severity 文档/残留问题（M3/M4）与 1 项口径说明（M5）建议一并处理。
