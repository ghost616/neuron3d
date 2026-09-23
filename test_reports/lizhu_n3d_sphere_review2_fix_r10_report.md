# n3d_sphere 第二轮修复（皋陶第二轮审查）—— 离朱独立测试报告

- **测试对象**：`n3d_sphere/`（model.py / train.py / config.py / README.md）+ `.module_agent/n3d_sphere/`（spec、module_definition）+ `_verify/` 验证脚本
- **测试依据**：绑定的《n3d_sphere 第二轮修复（皋陶第二轮审查）—— 待测试功能说明》
- **环境**：Windows + Python 3.12.10 + torch 2.14.0+cpu（CPU only）；MNIST 复用本地 `data/mnist`
- **测试脚本**：`.lizhu_env/lizhu_tests/lizhu_n3d_r10_review2_tests.py`（21 项，判据全部独立构造）

---

## 一、测试概览

| 项 | 用例数 | 通过 | 失败 | 结论 |
|---|---|---|---|---|
| 独立单元测试（F9/F11/F12/F10/F13/F14 + 硬约束） | 21 | **20** | **1** | 唯一失败为低severity 文档残留（D2） |
| 冒烟命令覆盖（规格要求的 5 条 + 4 种 scope + mlp） | 9 | 9 | 0 | 全部退出码 0、判据全 PASS |
| 项目自带 `verify_all.py`（8 条命令） | 8 | 8 | 0 | 全部退出码 0、无 FAIL |
| 跨套件独立核对（默认判定 / 指纹 / 参考递推） | 12+ | 全部 | 0 | 见 §3.4 |
| **合计** | **50** | **49** | **1** | F9/F10/F11/F12/F13/F14 均已修复 |

**本轮六项修复目标的独立验证结论**

| 修复项 | 结论 | 关键证据 |
|---|---|---|
| F9 判据 13 结构性误报 | ✅ **已修复** | 见 §3.1；seed=7 实测 `\|S_out\|=11` 而 `h` 非零列=10（1 个 S_out 列因 ReLU 合法为 0）——旧等式断言必 FAIL，新子集+结构断言 PASS |
| F9 判据 3 口径 | ✅ **已修复** | 按 arch 解析 `W_out` / `fc2.*`；`W_out` 非零梯度列数改用区间 `[1, \|S_out\|]`（实测 seed 42/7/2024/3 分别为 17/11/14/17，均落在区间内） |
| F11 阶段 2 向量化 | ✅ **已修复** | 无 `for pos in range(self.N)`、无 `torch.tensor([node])`；循环 = 层数（N=64→7 层、N=256→9 层）；与独立参考实现最大偏差 2.4e-7~6.0e-7（< 1e-5） |
| F12 冒烟产物命名与默认判定 | ✅ **已修复**（含 1 项残留，见 D1） | 默认组合写 `smoke.pt`；`--seed 7` / `--n 32` / `--flow-axis x` / `--arch mlp` 各写独立指纹名；非默认运行**不再覆盖** `smoke.pt`（逐字节比对未变） |
| F10 规格/文件定义清残 | ✅ **基本修复** | 无重复 heading；`2.494203` 清零；`R1-R7`→`R1-R7b`；保留字段列表无 `T`；命名描述与实现一致；`module_definition.json` 无裸 `R1-R7` |
| F13/F14 验证脚本与文案 | ✅ **基本修复**（含 1 项残留，见 D2） | R5b 反例真实复现 M1 形态（174/181 错配边、结果差异非零）；R5c 无“占位”死代码；C6 含 7 个禁用词与豁免计数；冒烟告警不再提 `--t` |

---

## 二、问题清单

### 【低 D1】`smoke_fingerprint` 不含 `batch_size` / `space_radius`，不同配置**同名互覆**，与 F12 命名承诺相悖

**位置**：`n3d_sphere/train.py` 第 134-161 行（`smoke_fingerprint`）

**成因**：F12 的“完全默认组合”判定（`is_default_smoke`，第 1202-1218 行）**包含** `batch_size` 与 `space_radius`，但产物名指纹（`smoke_fingerprint`）**不含**这两个维度，于是“非默认配置”落回了不含该维度的文件名 —— 不同配置产生同名产物。

**实测**（四种配置的产物名完全相同）：

| 配置 | batch_size | space_radius | 产物名 |
|---|---|---|---|
| 默认 | 32 | 0.0 | `smoke_N64_y4x4_H0.15_D0.25_plfcc_axz_isany_rsany_s42.pt` |
| `--batch-size 64` | **64** | 0.0 | **同名** |
| `--space-radius 0.9` | 32 | **0.9** | **同名** |
| 两者同时覆盖 | **64** | **0.9** | **同名** |

已实测确认：执行 `--smoke-test --batch-size 64` 会重写上述文件（mtime 由 22:37:40 更新），且其 loss/grad_norms 与前一次 `--space-radius 0.9` 运行不同 —— **即后一次运行覆盖了前一次的取证产物**。

**与说明的冲突**：
- 测试说明 §3：“其它**任何**组合写 `smoke[_ar{arch}]_{完整指纹}.pt`”；
- `current_spec.md:182`：“其它任何组合（换 arch / 流向轴 / scope / **容量 N、y、H、D、seed 被覆盖**）为 `_verify/smoke[_ar{arch}]_{完整配置指纹}.pt`，保证不同配置不互相覆盖”；同句又自述“指纹含 N/y/H/D/placement/axis/两个 scope/seed”（确实不含 `batch_size`/`space_radius`），故该“保证”对这两个维度不成立。

**建议修复**：把 `batch_size`（与 `space_radius`，非 0 时）纳入 `smoke_fingerprint`（如 `_bs{batch_size}` / `_R{space_radius:g}`），或把 `is_default_smoke` 的判定维度与指纹维度严格对齐（二者取交集）。

---

### 【低 D2】`verify_all.py` 文件头仍写“13 条判据”“R1~R7”“9 条命令”（实际 15 条 / R1-R7b / 8 条）

**位置**：`checkpoints/n3d_sphere/_verify/verify_all.py` 第 3、6、7 行

| 行 | 现状 | 实际 |
|---|---|---|
| L3 | “与 README 的 **9 条**一致” | `COMMANDS` 共 **8** 条 |
| L6 | “E2 冒烟（**13 条判据**）” | 现为 **15** 条判据（实测日志 `PASS=15 FAIL=0`） |
| L7 | “**R1~R7** 几何/FCC/DAG/去重/双副本/判据/产物” | 现为 **R1-R7b**（含 R5b/R5c/R7b） |
| L39 | 命令标签 `"R1-R7 sphere_dag"` | 同上，应为 `R1-R7b` |

说明 §4 要求全文不得出现与实现互斥的描述；此文件属验证脚本范畴（说明 §5 亦列出 `verify_all.py`），且与同目录脚本（`verify_sphere_dag.py` 已改 `R1-R7b`、`module_definition.json` 已改 `R1-R7b`）不一致。

**建议修复**：同步为“8 条命令 / 15 条判据 / R1-R7b”。

---

### 【信息 D3】`space_radius` 仅作窗口校验与元数据、不影响几何与产物内容

测试说明 §1 要求 `space_radius` 语义明确。实测：`--space-radius 0.9` 与默认（`0.0`→`R_min`）**产物名相同**，且 `stage2`/几何仅取决于 `H`/`D`/`N`/`placement`/`flow_axis`（`neuron_pos` 由 `R_max` 内取最近 N 个 FCC 格点，与 `space_radius` 无关）。经核对，README/spec 已注明其“仅作窗口校验与元数据”，故**不构成缺陷**，此处仅记录以供口径确认。

---

## 三、通过项详细结果

### 3.1 F9：判据 13 / 判据 3 口径（关键修复）

**判据 13 —— 独立构造“S_out 神经元整批 pre-activation 全负”的情形**（遍历 8 个 seed）：

| seed | \|S_out\| | h 非零列 | S_out 中为 0 的列 | 旧等式断言 `h非零==\|S_out\|` | 新断言（子集+结构+下界） |
|---|---|---|---|---|---|
| 42 | 17 | 17 | 0 | PASS | PASS |
| **7** | **11** | **10** | **1** | **FAIL（误报！）** | **PASS** |
| 2024 | 14 | 14 | 0 | PASS | PASS |
| 0 / 1 / 3 / 5 / 11 | 15/17/… | 等于 \|S_out\| | 0 | PASS | PASS |

即：**seed=7 精确复现了 F9 描述的结构性误报场景**（1 个 S_out 神经元因 ReLU 输出 0，该列合法为 0），新口径下判据正确 PASS。
另独立验证 `h` 与 `a_up * out_scope_mask` 的转置形式**逐位相等**（8/8 seed，含比特级 `torch.equal`）。

**判据 3**：确认代码按 arch 解析输出层参数名（neuron3d `W_out`，MLP `fc2.weight` / `fc2.bias`），并断言“至少一个梯度范数 > 0”；neuron3d 另断言 `1 <= W_out 非零梯度列数 <= |S_out|`（区间，非等式）。独立实测 4 个 seed 均落在区间内，且 `S_out` 之外列梯度**恰为 0**。

**回归命令（说明 §1 要求全部退出码 0 且判据全 PASS）**：

| 命令 | 退出码 | PASS | FAIL |
|---|---|---|---|
| `--smoke-test` | 0 | 15 | 0 |
| `--smoke-test --seed 7`（F9 修复前会 FAIL） | **0** | **15** | **0** |
| `--smoke-test --preset default` | 0 | 15 | 0 |
| `--smoke-test --n 32` | 0 | 15 | 0 |
| 四种 scope 组合 | 0 ×4 | 15 ×4 | 0 |
| `--smoke-test --arch mlp` | **0** | 6 | 0 |

### 3.2 F11：向量化

- **代码结构**：`stage2_recurrence` 体内无 `for pos in range(self.N)`、无 `torch.tensor([node])` + 逐神经元 `index_copy`；改为 `for lo, hi, nodes in self._iter_levels()`，层内一次 `index_add` + 一次 `index_copy`。
- **循环次数 == 层数**：SMALL（N=64）→ **7** 层；DEFAULT（N=256）→ **9** 层（均 < N）。
- **层语义契约**（独立核对）：每层边区间内目标全属该层；`in_degree>0` 的层内节点**恰为**该区间的目标集合（`in_degree==0` 的节点合法不出现，其激活为 `ReLU(b)`，由空累加分支处理）；层区间首尾相接、覆盖全部 E 条边恰好一次；无层内边（沿流向轴严格上行）。
- **数值不变**：与独立参考实现（按拓扑序逐节点、用 `edge_dst` 掩码从**原始边表**取该节点入边）逐元素比对 —— N=16（E=25）**3.0e-8**、N=64（E=181）**6.0e-7**、N=256（E=903）**2.4e-7~6.0e-7**，均远小于 1e-5。
- **语义保持**：零化 `a_in` 后 `a_up` 改变、确有非 `S_in` 下游神经元被打到；`d(Σa_up)/d(a_in)` 存在非零元素。

### 3.3 F12：产物命名与默认判定

**默认判定矩阵（12 组，独立复算 `is_default_smoke` 逻辑）**：仅“完全默认”与“显式 `placement=fcc`”判为默认（后者等价于默认值，行为正确）；`seed/N/H/D/flow_axis/两个 scope/batch_size/space_radius` 任一偏离均判为非默认 → 全部符合预期。

**同名互覆实测**：

| 检查 | 结果 |
|---|---|
| 默认组合 → `smoke.pt` | ✅ 且内容为 `loss=2.326995849609375`、`grad_norms` 与 README 逐位一致、config 无 `T` |
| `--seed 7` → `smoke_..._s7.pt` | ✅ |
| `--n 32` → `smoke_N32_..._s42.pt` | ✅ |
| `--flow-axis x` → `smoke_..._axx_....pt` | ✅ |
| `--arch mlp` → `smoke_armlp_....pt` | ✅ |
| 连续执行 `--seed 7` / `--n 32` / `--flow-axis x` 后 `smoke.pt` | ✅ mtime 与逐字节内容**均未变**（非默认组合不再覆盖） |
| `run_smoke_test` 内 `torch.save(` 出现次数 | ✅ **恰好 1 次**（无“同时刷新指纹名副本”的冗余写入；该表述已从注释中移除） |

### 3.4 F10：规格与文件定义清残

| 检查项 | 结果 |
|---|---|
| `current_spec.md` heading 唯一性 | ✅ 6 个 heading，**无重复** |
| “实测验收结果”块 | ✅ 唯一 heading 块（`### 实测验收结果（唯一记录…）`）；L132 仅为“若与…不符”的交叉引用，非第二块 |
| 旧数字 `2.494203` | ✅ 清零 |
| `13 条判据` / `13/13（判据计数）` | ✅ 无（L121 的 `13/13` 是范围内 `S_in/S_out` 比例，属合法数值，非判据条数） |
| `R1-R7` → `R1-R7b` | ✅ spec 中 `verify_sphere_dag.py` 行、`module_definition.json` description 均为 `R1-R7b` |
| 配置字段保留列表含 `T` | ✅ 已清除（保留列表不再含 `T`） |
| 产物命名模式 `_T{T}` | ✅ 已清除 |
| `module_definition.json` “13 条 / 裸 R1-R7” | ✅ 无（正则 `R1-R7(?!b)` 无命中） |
| 冒烟命名描述与实现一致 | ✅ spec:182 描述与代码一致（除 D1 指出的 `batch_size`/`space_radius` 维度缺口） |

### 3.5 F13：验证脚本修正

| 检查项 | 结果 |
|---|---|
| R5b 反例真实复现 M1 形态 | ✅ 独立复算历史错槽位形态：**174/181 条边错配**（非恒等映射），且该映射结果与正确递推偏差显著 > 1e-6 |
| R5b 使用 `edge_perm` | ✅ 脚本中显式引用 `edge_perm` |
| R5c 无 `depth = {...}  # 占位` 死代码 | ✅ 全文无“占位” |
| C6 禁用词包含 7 项 | ✅ `tau_init` / `min_neuron_dist` / `max_sample_tries` / `W_conn_sparse` / `equivalent_sphere_radius` / `readout_in_mask` / `stage2_propagate` 全部在列 |
| C6 豁免计数与打印 | ✅ 含豁免逻辑与计数输出；独立运行退出码 0 且 PASS |

### 3.6 F14：文案与参数开销

| 检查项 | 结果 |
|---|---|
| 冒烟覆盖告警不再提 `--t` | ✅ 告警文案为“--preset/--n/--y-in/--y-out/--h/--d/--seed/--lr/--flow-axis/--space-radius/--input-scope/--readout-scope/--placement 之一”，无 `--t` |
| README/spec 标注非 `S_out` 列梯度恒为 0 属已知取舍 | ✅ 已标注，并给出 DEFAULT 测点 seed=42 的 `\|S_out\|=45` 与 211 列结构性静默 |
| 未改 `state_dict` / `config` 契约 | ✅ 抽查产物 `state_dict` 键集与 `config` 字段集与上一轮一致（仅移除 `T`，为本轮既定变更） |

### 3.7 硬约束（回归）

| 项 | 结果 |
|---|---|
| `python n3d_proto/train.py --smoke-test` | ✅ **9/9 PASS、退出码 0、`loss = 2.419689`** |
| 一期三件产物 SHA256 | ✅ 逐字符一致（full `888556B0…8924`、highacc `9F21AC34…3F8`、capacity `0F7CF500…5011`） |
| `git status --porcelain -- n3d_proto` | ✅ 输出为空 |
| `python -m compileall -q n3d_sphere` | ✅ 退出码 0 |
| 同 seed 可逐位复现 | ✅ `state_dict` 全键逐位相同 |
| 项目自带 `verify_all.py`（8 条命令） | ✅ 退出码 0、无 FAIL（E2 冒烟 15 PASS、E2b mlp 6 PASS、E5 一期 9 PASS） |

---

## 四、修复建议汇总

| 编号 | 问题 | 严重度 | 建议修复 |
|---|---|---|---|
| D1 | `smoke_fingerprint` 不含 `batch_size`/`space_radius`，而默认判定包含它们 → 四种配置同名互覆（`--batch-size 64` 实测覆盖了 `--space-radius 0.9` 的产物），与 F12“任何组合写完整指纹名、不同配置不互相覆盖”相悖 | 低 | 指纹追加 `_bs{batch_size}`（及非 0 的 `_R{space_radius:g}`），或让默认判定维度与指纹维度严格对齐 |
| D2 | `verify_all.py` 文件头仍为“9 条命令 / 13 条判据 / R1~R7”（实为 8 / 15 / R1-R7b），与同目录其他脚本不一致 | 低 | 同步为“8 条命令 / 15 条判据 / R1-R7b” |
| D3 | `space_radius` 不影响几何与产物内容（仅窗口校验+元数据） | 信息 | 文档已注明，无需改动；如期望其影响几何需另开需求 |

---

## 五、环境说明

| 项 | 值 |
|---|---|
| Python / torch | 3.12.10 / 2.14.0+cpu（`cuda.is_available() == False`） |
| 数据 | 工程内 `data/mnist` IDX（未联网） |
| 耗时 | 单次冒烟约 6~7s；`verify_all.py` 全套 8 条命令合计 < 3 分钟 |
| 未执行项 | 无（GPU 相关不适用本环境） |

## 六、结论

- **本轮六项修复目标（F9 判据口径、F11 向量化、F12 产物命名与默认判定、F10 规格清残、F13 验证脚本、F14 文案与开销）全部验证有效**：
  判据 13 的结构性误报已消除（seed=7 从 FAIL 转为 PASS，且 `h` 与掩码积逐位相等）；判据 3 按 arch 解析参数名并改用区间断言，`--arch mlp` 冒烟由“5 PASS/1 FAIL 退出码 1”恢复为 **6 PASS/退出码 0**；阶段 2 循环次数由 N 降为层数（7/9），与独立参考实现偏差 2.4e-7~6.0e-7；默认组合正确写 `smoke.pt` 且不再被非默认组合覆盖；规格与文件定义无旧数字、无重复 heading、`R1-R7b` 已同步；R5b 反例真实复现 M1 形态（174/181 错配）。
- **硬约束全部满足**：一期 9/9 PASS + `loss=2.419689` + 三个 SHA256 一致 + `n3d_proto` 零改动 + `compileall` 退出码 0 + 同 seed 逐位复现。
- **仅剩 2 项低severity 残留**：D1（`batch_size`/`space_radius` 未进冒烟指纹，导致不同配置同名互覆）与 D2（`verify_all.py` 文件头仍写 13 条判据 / R1~R7 / 9 条命令）。建议一并清理。
