# 离朱测试报告 —— n3d_sphere「D ≤ H 约束」改造（G1~G5）

- 轮次：R14（离朱）
- 被测试功能说明：`session-abe38eb5-a63b-4ab1-8a50-4a5711fcb0c3` 绑定的测试说明（`D ≤ H` 硬校验 / 三预设 `D = H` / 连通性下限 / 拓扑数字刷新 / 一键验收 / 一期零改动）
- 执行时间：2026-09-24 23:0x ~ 23:5x（本地）
- 结论：**通过（无产品缺陷）**

---

## 0. 结论摘要

| 项目 | 结果 |
|---|---|
| 独立测试用例 | **30 / 30 通过**（单元 12、接口/CLI 3、编译 2、产物与文档一致性 8、文档防线反向取证 1、一期硬回归 4） |
| 一键验收 `verify_all.py` | **退出码 0**：10 条命令全部 `exit=0` 且 `FAIL=0`；汇总判据 57 条全 PASS |
| 文档数字防线 | **一致 174 / 不一致 0 / 跳过 0** |
| 库内既有验证脚本 | H1-H4、R1-R7b、D1-D7、S1-S5、C1-C6、snapshot 全部 `exit=0` 且无 FAIL |
| 一期（`n3d_proto`）硬回归 | 零改动（git 干净）、冒烟 9/9 PASS、`loss=2.419689`、三产物 SHA256 逐字符未变 |
| E2E | **不适用**（仓库内无任何前端/UI 文件；详见 §9） |
| 环境阻塞 | 无 |
| 未发现的问题 | 产品侧 0 个 BLOCKER/MAJOR/MINOR；仅 1 条 info 级观察（§11） |

**核心判据实测结论**：`D > H` 在 `Config` 构造期被拒（含 `H=0.10, D=0.1000001` 与 `D = H + 1 ulp`）；三预设 `D = H`；退化配置 `H=0.10, D=0.05/0.03/0.02` 在模型构造期被连通性下限拦下（真实 `E=146/17/1 < N=256`）；拓扑数字 `DEFAULT |S_in|=193 / |S_out|=187 / E=736 / K=9`、`SMALL 55 / 53 / 106 / 7`、跨 10 seed 参数区间 `146237 ~ 164263`、默认冒烟 `loss=2.1496479511260986` 与四个梯度范数**逐位一致**，全部复现。

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| OS / Shell | Windows（PowerShell 7） |
| Python | 3.12.10（`sys.executable`） |
| PyTorch | 2.14.0+cpu |
| 工作目录 | `E:\neuron3d` |
| 环境构建 | **无需**（依赖已就绪；未执行任何 `npm/pip install`，未使用 `.lizhu_env/` 做环境构建） |
| Playwright | 已安装（npm，1.63.0）——但无前端可用，见 §9 |

**被测试版本锁定**（整轮测试**前后逐字节一致**，见 §8.5）：

| 文件 | SHA256（前 16 位） | 字节 |
|---|---|---|
| `n3d_sphere/config.py` | `17195C5E820A0956` | 21897 |
| `n3d_sphere/model.py` | `6E14DADBDFE6C4CC` | 73235 |
| `n3d_sphere/train.py` | `39E14C985F91A557` | 76531 |
| `n3d_sphere/README.md` | `A01B90663BE9E654` | 63819 |
| `.module_agent/n3d_sphere/current_spec.md` | `9A55B9D5108CFCDB` | 39705 |
| `checkpoints/n3d_sphere/_verify/doc_numbers.json` | `04F0D3EE599CE370` | 48658 |

---

## 2. 测试概览

| # | 测试类型 | 脚本 / 命令 | 用例数 | 通过 | 失败 | 跳过 | 证据文件 |
|---|---|---|---|---|---|---|---|
| 1 | 单元 + 接口 + 编译 | `python .lizhu_env/lizhu_tests/r14_unit.py` | 17 | 17 | 0 | 0 | `_out.txt`（`r14_unit_out.txt`） |
| 2 | 冒烟产物 + 文档数字 + 一期硬回归 | `python .lizhu_env/lizhu_tests/r14_artifacts.py` | 8 | 8 | 0 | 0 | `r14_artifacts_out.txt` |
| 3 | 跨 seed 拓扑 vs 文档 + 数字防线反向取证 | `python .lizhu_env/lizhu_tests/r14_docdefense.py` | 5 | 5 | 0 | 0 | `r14_docdefense_out.txt` |
| 4 | 一键验收 | `python checkpoints/n3d_sphere/_verify/verify_all.py` | 10 命令 / 57 判据 / 174 数字 | 全部 | 0 | 0 | `r14_verify_all_out.txt` |
| 5 | E2E | 不适用（无 UI） | — | — | — | 1（不适用） | §9 |

测试脚本与全部终端输出留档于 `.lizhu_env/lizhu_tests/`：

```
r14_unit.py            r14_unit_out.txt
r14_artifacts.py       r14_artifacts_out.txt
r14_docdefense.py      r14_docdefense_out.txt
                       r14_verify_all_out.txt
                       r14_hashes_before.txt  r14_hashes_after.txt
```

---

## 3. 单元测试（12 项，全部 PASS）

### 3.1 `D ≤ H` 硬校验（G1，4 项）

| 用例 | 输入 | 期望 | 实测 |
|---|---|---|---|
| 负例批量 | `H=0.10/D=0.15`、`H=0.15/D=0.25`、`H=0.10/D=0.1000001`、`H=0.10/D=0.2`、`H=0.30/D=0.3001` | 构造失败 `ValueError`，消息含 `H=`、`D=`、`D/H=` 与"不得超过" | 5/5 拒绝；如 `H=0.1 D=0.15 -> D/H=1.500000`、`H=0.1 D=0.1000001 -> D/H=1.000001` |
| **epsilon 极限边界** | `D = nextafter(0.10, +inf) = 0.10000000000000002` | 必须拒绝 | 拒绝（"不得超过"），`D == H` 精确相等则接受（`E=96`） |
| 正例 | `D == H`（0.10/0.10、0.15/0.15）、`D < H`（0.15/0.10、0.10/0.05） | 不得被 `D ≤ H` 守卫误伤 | `D==H` 构造成功（`E=96/106`）；`D<H` 仅被**连通性下限**拦下，异常串中**不含** `不得超过接收/发送范围半径`（证明两条守卫不混淆） |
| 字段默认值 | `Config()` | `H == D == 0.1` | 一致（`dataclasses.fields` 亦为 `0.1/0.1`） |

原始异常消息（实测，`H=0.10, D=0.15`）：

```
Config.D 不得超过 Config.H：连接半径 D 不得超过接收/发送范围半径 H，当前 H=0.1, D=0.15（D/H=1.500）。
请令 D <= H（本模块三个预设 DEFAULT/HIGHACC/SMALL 均取 D = H）。
```

**旁路路径同样受约束**（补充覆盖 `train.apply_overrides` 的两条真实入口）：

```
dataclasses.replace(DEFAULT_CONFIG, D=0.15)      -> ValueError（D/H=1.500）
dataclasses.replace(DEFAULT_CONFIG, D=0.1000001) -> ValueError（D/H=1.000）
dataclasses.replace(DEFAULT_CONFIG, H=0.05, D=0.10) -> ValueError（D/H=2.000）
Config(**{**DEFAULT_CONFIG.to_dict(), "D": 0.15}) -> ValueError（D/H=1.500）
dataclasses.replace(SMALL_CONFIG, D=0.15)         -> 接受 H/D = 0.15 0.15
```

### 3.2 三个预设均为 `D = H`（G2，2 项）

| 预设 | H | D | `D == H` | `describe()` 含 `H=`/`D=` | `Config(**to_dict()) == cfg` | `R_min` / `R_max` | `R_max/R_min` |
|---|---|---|---|---|---|---|---|
| `SMALL_CONFIG` | 0.15 | 0.15 | ✔ | ✔ | ✔ | 0.663198 / 1.326395 | 2.000000 |
| `DEFAULT_CONFIG` | 0.10 | 0.10 | ✔ | ✔ | ✔ | 0.701840 / 1.403681 | 2.000000 |
| `HIGHACC_CONFIG` | 0.10 | 0.10 | ✔ | ✔ | ✔ | 0.701840 / 1.403681 | 2.000000 |

- 派生关系自校验 `R_max/R_min == (H+D)/H == 2.0`（容差 1e-9），与 `D = H` 的承诺一致（改造前 D=1.5H 时为 2.5）。
- FCC 实测放置半径 `DEFAULT 0.721110` / `SMALL 0.670820`，均落在 `[R_min, R_max]` 内（容纳性断言未被 `R_max` 减半破坏）。
- `HIGHACC_CONFIG` 的 `weight_decay=1e-4`、`readout_bias=True`、`lr_schedule="cosine"`、`grad_clip=1.0` 仍完好，未被本轮改动波及。

### 3.3 连通性下限校验（G3，4 项）

**指标键与三预设实测值**（`connectivity_floor` 8 键齐备）：

| 预设 | E | N | E/N | 层数 K | \|S_in\| | \|S_out\| | H | D |
|---|---|---|---|---|---|---|---|---|
| `SMALL_CONFIG` | 106 | 64 | 1.65625 | 7 | 55 | 53 | 0.15 | 0.15 |
| `DEFAULT_CONFIG` | 736 | 256 | 2.87500 | 9 | 193 | 187 | 0.10 | 0.10 |
| `HIGHACC_CONFIG` | 736 | 256 | 2.87500 | 9 | 193 | 187 | 0.10 | 0.10 |

四条判据（`E ≥ N` / `K ≥ 2` / `|S_in| ≥ 1` / `|S_out| ≥ 1`）全部满足。

**退化配置必须被拦下，且消息字段完备**：

| 配置 | 期望 | 实测 |
|---|---|---|
| `H=0.10, D=0.05` | 构造期 `ValueError` | `E=146 < N=256`，消息含 `[连通性下限校验失败]`、`E=`、`N=`、`E/N=`、`层数K=`、`|S_in|=`、`|S_out|=`、`H=`、`D=` 全部 token |
| `H=0.10, D=0.03` | 同上 | `E=17 < N=256` |
| `H=0.10, D=0.02` | 同上 | `E=1 < N=256` |

实测消息片段：

```
[连通性下限校验失败] 该配置生成的图已退化，拒绝构造模型：E=146 < N=256（平均出度 E/N=0.5703 < 1）。
实测指标：E=146, N=256, E/N=0.5703, 层数K=9, |S_in|=256, |S_out|=256, H=0.1, D=0.05
（input_scope=any_isolated, readout_scope=any_isolated, seed=42）。请增大 D（硬约束 D <= H）或调整 N/H/scope。
```

**"不得永久旁路"取证（测试说明第 3 条）**：临时把 `ThreeDNeuronSpace.check_connectivity_floor` 置为 `lambda self: {}` 以取出被拒配置的**真实拓扑**，`finally` 立即还原，并断言：

```
旁路实测真实拓扑：{0.05: 146, 0.03: 17, 0.02: 1}（全部 E < N=256，确为退化图）
还原断言：函数对象同一（is original）✔  __code__ 同一 ✔  D=0.05 再次抛错 ✔
```

**接线取证**：`__init__` 中的调用点为
`self.connectivity_floor: Dict[str, float] = self.check_connectivity_floor()`，
且位置**早于** `self.W_in = nn.Parameter(...)`（退化图不会先分配参数）；四条判据分支 `if e < n` / `if layers < 2` / `if s_in < 1` / `if s_out < 1` 源码齐备。全仓库扫描确认：除测试/验证脚本的**临时**旁路外，产品代码中不存在对 `check_connectivity_floor` 的永久改写。

### 3.4 拓扑数字（G4，2 项）

| 测点 | \|S_in\| | \|S_out\| | E | 层数 |
|---|---|---|---|---|
| `DEFAULT`（H=D=0.10, N=256, y=8×8, seed=42, any-any, z） | **193** | **187** | **736** | **9** |
| `SMALL`（H=D=0.15, N=64, y=4×4, seed=42） | **55** | **53** | **106** | **7** |

跨 10 seed（`42, 7, 2024, 123, 0, 1, 2, 3, 4, 5`）独立重建 `DEFAULT` 模型：**参数区间实测 = 146237 ~ 164263**（与文档登记完全一致），同期 `E ∈ [727, 750]`、层数恒为 9、全部记录 `D == H`。

---

## 4. 接口测试（CLI，3 项，全部 PASS）

| 用例 | 命令 | 期望 | 实测 |
|---|---|---|---|
| `--help` 文案 | `python n3d_sphere/train.py --help` | 记录 `D <= H` 硬约束与连通性下限判据 | `exit=0`；输出含 `**硬约束 D <= H**`、`连通性下限`、`E >= N`、`层数 >= 2`、`|S_in| >= 1`、`|S_out| >= 1` |
| 非法参数 | `--smoke-test --h 0.10 --d 0.15` | 非零退出 + 可读报错 | `exit=2`，输出含"不得超过"，**无 Traceback**（走可读错误路径） |
| 退化参数 | `--smoke-test --h 0.10 --d 0.05 --n 256 --y-in 8 --y-out 8` | 非零退出 + 下限报错 | `exit=2`，输出含 `连通性下限校验失败` |

---

## 5. 编译测试（2 项，全部 PASS）

| 用例 | 命令 | 结果 |
|---|---|---|
| 字节码编译 | `python -m compileall -q n3d_sphere` | `exit=0`（成功时无输出） |
| 包导入 | `python -c "import n3d_sphere, .config, .model, .data, .utils, .train"` | `IMPORT_OK`，六个模块全部可导入 |

`verify_all` 内亦含 E1 `compileall` 与 E2b MLP 冒烟（6/6 PASS）。

---

## 6. 产物 / 文档数字一致性（8 项，全部 PASS）

### 6.1 默认冒烟：`loss` 与四个梯度范数逐位一致

`python n3d_sphere/train.py --smoke-test` → `exit=0`、**15/15 PASS**、`FAIL=0`；`_verify/smoke.pt` 的 9 项登记值（`exact: true`）全部**逐位相等**：

| 字段 | 实测 = 登记 |
|---|---|
| `loss` | `2.1496479511260986` |
| `grad_norms.W_in` | `1.5796021223068237` |
| `grad_norms.edge_weight` | `0.2537662386894226` |
| `grad_norms.neuron_bias` | `0.18929001688957214` |
| `grad_norms.W_out` | `0.49458175897598267` |
| `connection_stats.num_edges / num_in_scope / num_out_scope / num_layers` | `106 / 55 / 53 / 7` |

冒烟产物内 `config.H == config.D == 0.15`；`smoke.pt` SHA256 = `4E11F12FEFABE9C8F978348792C8775336E9D121D544BCA0D76350AA440E6A71`，与 README 登记一致；该文件在本轮被反复重写（≥3 次）而 SHA 始终不变 → **同 seed 逐位可复现**。

### 6.2 四种 `scope` 组合：`loss` 互不相同（`readout_scope` 仍生效）

| `input_scope` / `readout_scope` | 产物 | loss | 登记值比对 |
|---|---|---|---|
| any / any | `smoke.pt` | `2.1496479511260986` | 9 项逐位一致 |
| any / all | `smoke_..._H0.15_D0.15_..._isany_rsall_...pt` | `2.236379623413086` | 7 项逐位一致 |
| all / any | `..._isall_rsany_...pt` | `2.2899417877197266` | 7 项逐位一致 |
| all / all | `..._isall_rsall_...pt` | `2.337554454803467` | 7 项逐位一致 |

四种组合 `loss` 两两不同（4 个不同值），且产物**指纹**均含 `H0.15_D0.15`（`D = H` 已进入命名）。另在 `DEFAULT` 规模独立实测四种判据组合的生效性：

| input / readout | \|S_in\| | \|S_out\| | 双副本 | E |
|---|---|---|---|---|
| any / any | 193 | 187 | 180 | 736 |
| any / all | 193 | 14 | 180 | 736 |
| all / any | 13 | 187 | 0 | 736 |
| all / all | 13 | 14 | 0 | 736 |

与 README 第 7 节"判据开关实测"表逐格一致（含 `all_isolated` 下双副本为 0 的如实标注）。

### 6.3 跨 10 seed 拓扑表与文档逐列一致（独立重建）

用**自建代码路径**重建 10 个 seed 的 `DEFAULT` 模型，与 README 第 7 节表格的 12 列 × 10 行**全部一致**，并由独立的 `topology_snapshot.json` 二次确认：

```
seed |   E |  avgOut | maxOut |  avgIn | maxIn | K | S_in | S_out | dual | isoIn | isoOut | params
  42 | 736 |  2.8750 |      4 | 2.8750 |     4 | 9 |  193 |   187 |  180 |   605 |    555 | 154864  一致
   7 | 736 |  2.8750 |      4 | 2.8750 |     4 | 9 |  192 |   196 |  178 |   573 |    602 | 154080  一致
2024 | 740 |  2.8906 |      5 | 2.8906 |     5 | 9 |  194 |   185 |  178 |   537 |    563 | 155652  一致
 123 | 750 |  2.9297 |      5 | 2.9297 |     5 | 9 |  192 |   191 |  177 |   580 |    542 | 154094  一致
   0 | 738 |  2.8828 |      4 | 2.8828 |     5 | 9 |  197 |   189 |  182 |   580 |    570 | 158002  一致
   1 | 727 |  2.8398 |      5 | 2.8398 |     5 | 9 |  205 |   190 |  192 |   602 |    586 | 164263  一致
   2 | 738 |  2.8828 |      5 | 2.8828 |     5 | 9 |  195 |   190 |  180 |   593 |    568 | 156434  一致
   3 | 735 |  2.8711 |      4 | 2.8711 |     5 | 9 |  195 |   192 |   179 |   602 |    597 | 156431  一致
   4 | 729 |  2.8477 |      5 | 2.8477 |     5 | 9 |  187 |   203 |  173 |   543 |    622 | 150153  一致
   5 | 733 |  2.8633 |      4 | 2.8633 |     4 | 9 |  182 |   197 |  167 |   562 |    576 | 146237  一致
汇总：E ∈ [727, 750] ✔   params ∈ [146237, 164263] ✔   |S_in| ∈ [182, 205] ✔   层数恒 9 ✔
```

`topology_snapshot.json`：60 条 = 6 测点 × 10 seed，全部 `D == H`、`DAG 无环`、`严格上行`、`dense 权重张量 = 0`。

### 6.4 派生指标与产物矩阵自洽

- `sphere_dag_metrics.json` 111 条：`default 736` / `small 106` / `axis_x 730` / `radius_0.9 736`，且 `unique_neuron_pairs == num_edges`；代表连接最大间距 `0.09997506 ≤ D=0.10`（`small 0.14943191 ≤ D=0.15`）→ 边确实受 `D` 约束。
- `r7b.S_out_any = 187` / `r7b.S_out_all = 14`。
- `smoke_matrix_f9.json` 11 条记录：全部指向 `D0.15` 产物、`exit=0`、`item_ok=true`，且**每条 `artifact_sha256` 与磁盘现文件逐一相符（11/11）**。
- `smoke_scope_matrix.json` 4 条记录：SHA 全部相符；`any/any` 记录即 `smoke.pt`。
- MLP 对照基线冒烟 `--arch mlp` → `exit=0`、6/6 PASS；`smoke_armlp` loss `2.2989346981048584` 与 README 登记一致。
- 冒烟**未污染**正式产物路径（`checkpoints/n3d_sphere/model.pt` 不存在）。
- **本轮未新增任何 `D0.25`（改造前口径）产物**：磁盘上 11 个 `*_D0.25_*` 文件的 mtime 全部早于本轮（见 §11 观察项）。

### 6.5 文档数字防线**非空转**（反向取证）

对 `verify_all.py::check_doc_numbers` 做正/反向探针（**不触碰仓库内文件**，仅在 `.lizhu_env/lizhu_tests/_r14_tmp/` 下构造变体）：

```
篡改 1 项（r1.axis_x.syn_in_max_dist：0.09996657818555832 -> 1.0999665781855583）
    -> 一致 0 项 / 不一致 1 项 / 返回 False  ✔（防线确实会拦住）
同一项未篡改                -> 一致 1 项 / 不一致 0 项 / 返回 True   ✔
全部 111 项 metric_checks   -> 一致 111 项 / 不一致 0 项 / 返回 True ✔
探针结束：DOC_NUMBERS_PATH 已还原为仓库内文件；仓库内 doc_numbers.json 内容未被改写
```

---

## 7. 一键验收（`verify_all.py`）

命令：`python checkpoints/n3d_sphere/_verify/verify_all.py` → **退出码 0**

```
  E1 compileall        exit=0 PASS=0   FAIL=0
  E2 smoke             exit=0 PASS=15  FAIL=0
  E2b smoke_mlp        exit=0 PASS=6   FAIL=0
  R1-R7b sphere_dag    exit=0 PASS=10  FAIL=0
  H1-H4 d_le_h         exit=0 PASS=4   FAIL=0
  D1-D7 device         exit=0 PASS=6   FAIL=0
  S1-S5 scope_fp       exit=0 PASS=2   FAIL=0
  C1-C6 config         exit=0 PASS=4   FAIL=0
  snapshot             exit=0 PASS=1   FAIL=0
  E5 proto_smoke       exit=0 PASS=9   FAIL=0
  doc_numbers: 一致 174 项 / 不一致 0 项 / 跳过 0 项
总体结论：全部命令退出码 0 且无 FAIL；文档数字 全部一致（一致 174 / 不一致 0 / 跳过 0）
```

与测试说明第 5 条逐项吻合：**10 条命令全部 `exit=0` 且无 FAIL**；**174 / 0 / 0**。
`doc_numbers.json` 结构复核：`artifact_checks 49 + metric_checks 111 + text_checks 14 = 174`，其中含 `legacy.*`（15 处）与 `d_le_h.*`（7 处）条目。

---

## 8. 一期零改动硬回归（4 项，全部 PASS）

| 用例 | 命令 / 方法 | 实测 |
|---|---|---|
| 源码零改动 | `git status --porcelain -- n3d_proto` | 输出为**空** |
| 一期冒烟 | `python n3d_proto/train.py --smoke-test` | `exit=0`、**9/9 PASS**、`FAIL=0`、输出 `loss=2.419689` |
| 三产物指纹 | SHA256 | `n3d_model_full.pt = 888556B0…8924`、`n3d_model_highacc.pt = 9F21AC34…E3F8`、`n3d_model_capacity.pt = 0F7CF500…5011`，与登记值**逐字符一致** |
| 产物隔离 | `checkpoints/n3d_sphere/model.pt` | 不存在（冒烟只写 `_verify/`） |

### 8.5 测试前后被测试版本未漂移

整轮测试前后对 6 个手写文件重新计算 SHA256：**完全一致**（config.py / model.py / train.py / README.md / current_spec.md / doc_numbers.json）。
两个由验收脚本自身重算的派生产物也在前后保持**同一 SHA256**（内容确定性复现）：

```
checkpoints/n3d_sphere/_verify/sphere_dag_metrics.json  DE9930FFBCF2F629A46CD7B5E2695555CE207199418F9B8DA8A65FF625257794
checkpoints/n3d_sphere/_verify/topology_snapshot.json   AC4E6FE06364DB44BA4C14E61209146CAD0289E0248306A9EF6A9393CA5F4304
checkpoints/n3d_sphere/_verify/smoke.pt                 4E11F12FEFABE9C8F978348792C8775336E9D121D544BCA0D76350AA440E6A71
```

---

## 9. E2E 测试适用性判断：**不适用**

- 检测结果：Playwright **已安装**（`source: npm`，`Version 1.63.0`）。
- 但适用性判定为**不适用**：本模块是纯 PyTorch 数值/CLI 模块，仓库内（排除 `.lizhu_env/`、`.git/`）**不存在任何** `*.html / *.js / *.ts / *.jsx / *.tsx / *.vue / *.css` 文件，也没有任何 HTTP 服务端或前端路由，**不存在可驱动的 UI**。
- 因此 §"E2E 核心用户旅程 / 异常路径 / 跨页面状态"三类场景在本模块无对应实体；与其对应的"端到端用户旅程"已由 CLI 级端到端命令覆盖：`train.py --smoke-test`（前向+反向+15 条判据+落盘产物）、`--arch mlp`、四种 `scope` 组合、`verify_all.py` 全链路、一期 `n3d_proto` 冒烟。
- 若后续本模块引入可视化前端，再启用 `npx playwright test`。

---

## 10. 失败用例分析

**产品侧失败用例：0 个。**

本轮共出现 4 次失败信号，**全部经定位为本测试脚本自身的缺陷**并在修正后重跑为 PASS（如实记录，便于复核者区分"产品缺陷"与"测试脚本缺陷"）：

| # | 现象 | 定位结论 | 处理 |
|---|---|---|---|
| 1 | `U-G3 下限判据源码齐备且构造期生效` 首次 FAIL：断言 `"self.connectivity_floor = self.check_connectivity_floor()" in init_body` 不成立 | **测试脚本缺陷**：源码使用带类型标注的写法 `self.connectivity_floor: Dict[str, float] = self.check_connectivity_floor()`，断言字符串过窄。产品行为正确（调用点确实存在且早于参数创建） | 改为正则匹配（允许类型标注），重跑 PASS |
| 2 | `G4 四种 scope 组合` 首次读到 `H0.15_D0.25`（旧口径）产物，loss `2.299027681350708` 等与文档不符 | **测试脚本缺陷**：按文件名子串取"第一个匹配"，命中了上轮遗留的 `D0.25` 历史产物；产品实际写的是 `H0.15_D0.15` | 改为按 `mtime ≥ 本轮开始` 选取新产物 + 断言文件名含 `H0.15_D0.15`，重跑 4/4 逐位一致 PASS |
| 3 | `D 文档数字防线` 首次崩溃 `ValueError: I/O operation on closed file` | **测试脚本缺陷**：脚本先 `io.TextIOWrapper(sys.stdout.buffer)`，再导入 `verify_all`（它也会包装 `sys.stdout`），旧包装器析构时关闭了底层 buffer | 改用 `sys.stdout.reconfigure(...)`，重跑 PASS |
| 4 | `D 文档数字防线` 第二次 `AssertionError: 期望恰好 1 处不一致，实测 15` | **测试脚本缺陷**：变体保留了全部 14 条 `text_checks`，而探针未提供命令输出，故 14 条按"缺少命令输出"计入不一致（15 = 1 篡改 + 14 缺输出），与防线能力无关 | 变体改为只保留单项（正/反向各 1 项）与全量 111 项 metric，重跑 PASS |

---

## 11. 观察项与建议（info，不阻塞）

1. **`_verify/` 目录残留 11 个改造前（`H0.15_D0.25`）冒烟产物**：它们是 `D > H` 旧口径的历史产物（mtime ≤ 2026-09-24 00:42，`legacy_dh_baseline.json` 已固化其关键数字）。本轮**未新增**此类产物（已断言），`doc_numbers.json` 也未引用它们，故不影响验收；但其与当前 `H0.15_D0.15` 产物同目录、命名高度相似，容易误读（本轮测试脚本首次即被误选）。**建议**：后续如无取证需要，可将其归档到 `_verify/legacy/` 子目录并在 README 产物清单注明，降低误读与误引用风险。
2. **`doc_numbers.json` 未登记跨 seed 参数区间（146237 / 164263）**：该数字只出现在 README/spec 的表格与文字中，防线（174 项）未覆盖。本轮已用独立重建的 10-seed 模型逐列核对无误；**建议**（可选）把该区间或其 10 个 seed 记录纳入登记表，使跨 seed 结论也进入自动防线。
3. **`n3d_sphere/_verify/` 内两个派生产物（`sphere_dag_metrics.json` / `topology_snapshot.json`）会被验收命令重写**：本轮验证其重写后 SHA 不变（确定性），属预期行为，仅提示并发执行验收脚本时注意读写竞争。

---

## 12. 复现命令清单（工程根目录 `E:\neuron3d` 下执行）

```powershell
$env:PYTHONIOENCODING="utf-8"

# ① 单元 + 接口 + 编译（17 项）
python .lizhu_env/lizhu_tests/r14_unit.py

# ② 冒烟产物 / 文档数字 / 一期硬回归（8 项）
python .lizhu_env/lizhu_tests/r14_artifacts.py

# ③ 跨 seed 拓扑 vs 文档 + 文档数字防线反向取证（5 项）
python .lizhu_env/lizhu_tests/r14_docdefense.py

# ④ 一键验收（10 条命令 + 174 项文档数字）
python checkpoints/n3d_sphere/_verify/verify_all.py

# ⑤ 一期独立复核
git status --porcelain -- n3d_proto
python n3d_proto/train.py --smoke-test
```

---

## 13. 环境问题说明

- **无环境阻塞**：全部测试类型（单元 / 接口 / 编译 / 产物与文档一致性 / 端到端 CLI 链路 / 硬回归）均在本机真实执行并产生终端输出留档，无跳过项（E2E 属"不适用"而非"因环境跳过"）。
- 未执行任何环境构建命令（无 `pip install` / `npm install` / 脚手架初始化），依赖与数据（`data/mnist`）已就绪。
- 说明：本轮**未修改任何产品源码或文档**；仅新增 3 个测试脚本与 1 个报告，以及 `verify_all` 自身重写的两个派生 JSON 指标文件（SHA 前后不变）。

