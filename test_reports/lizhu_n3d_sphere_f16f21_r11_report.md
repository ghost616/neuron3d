# n3d_sphere 第三轮修复（F16-F21）—— 离朱独立测试报告

- **测试对象**：`n3d_sphere/model.py`、`_verify/` 验证脚本（含新增 `verify_device_regression.py` / `run_smoke_matrix.py` / `doc_numbers.json`）、README 与 `current_spec.md`
- **测试依据**：绑定的《n3d_sphere 第三轮修复（F16-F21）待测试功能说明》
- **环境**：Windows + Python 3.12.10 + torch 2.14.0+cpu（CPU only）；MNIST 复用本地 `data/mnist`
- **测试脚本**：`.lizhu_env/lizhu_tests/lizhu_n3d_r11_review3_tests.py`（19 项，判据全部独立构造）

---

## 一、测试概览

| 项 | 用例数 | 通过 | 失败 | 结论 |
|---|---|---|---|---|
| F16 设备契约（D1-D6） | 6 | **6** | 0 | 全部通过 |
| 逐层向量化（循环次数 / 数值一致） | 2 | **2** | 0 | 全部通过 |
| F20 文档数字防线（结构 / 汇总行 / 负向验证） | 3 | **3** | 0 | 全部通过（含负向验证） |
| F18 冒烟矩阵取证 | 1 | **1** | 0 | 通过（含 1 项证据文件注记，见 D1） |
| F17/F19 文档数字一致性 | 5 | **5** | 0 | 全部通过 |
| 硬回归与 `_verify/` 纪律 | 2 | **2** | 0 | 全部通过 |
| **合计** | **19** | **19** | **0** | 仅 1 项低severity 注记（不影响验收） |

**参考命令（说明 §3）实测**：`compileall` / `--smoke-test` / `--smoke-test --arch mlp` / `verify_sphere_dag.py all` / `verify_device_regression.py` / `verify_scope_and_fingerprint.py` / `verify_config_contracts.py` / `verify_topology_snapshot.py` / `run_smoke_matrix.py` / `verify_all.py` / `n3d_proto/train.py --smoke-test` —— **全部退出码 0**；`verify_all.py` 末行输出文档数字校验结果（一致 93 / 不一致 0 / 跳过 0）。

---

## 二、F16 设备契约（本轮 error 项）—— 逐条独立验证

| 判据 | 结果 | 实测证据 |
|---|---|---|
| `level_edge_reach` / `level_node_reach` 是 `torch.Tensor`（int64、`[K,2]`、K=层数） | ✅ | 3 组配置下均为 `Tensor`（非 list），`dtype=torch.int64`，`shape=(7,2)`（SMALL）/`(9,2)`（DEFAULT）等，且 `K == unique(flow_axis).numel()` 精确相等 |
| 同时出现在 `named_buffers()` 与 `state_dict()` | ✅ | 两者均为 True（3 组配置） |
| `vars(model)` 无“未注册 tensor 属性”/“含 tensor 的 list/tuple/dict” | ✅ | 全量扫描 45 个属性（排除 `_parameters`/`_buffers`），**offending = NONE** |
| `.to('meta')` 后索引张量 device 均为 meta | ✅ | `topo_index`/`edge_perm_in`/`edge_dst_in`/`edge_src`/`edge_dst`/`neuron_bias`/`level_edge_reach`/`level_node_reach` **8/8 全为 meta**；`state_dict()` 28 项 |
| 守卫负例 1：`level_node_reach` 换 CPU 张量 | ✅ | `RuntimeError`，消息含 `level_node_reach`（“在 cpu，但本次前向的激活在 meta”） |
| 守卫负例 2：删除 `level_edge_reach` | ✅ | `RuntimeError`，消息含 `level_edge_reach`（“未注册为 buffer/parameter”） |
| `state_dict()` → `load_state_dict(strict=True)` → 前向逐位相同 | ✅ | strict 载入无 missing/unexpected；**`torch.equal(logits) == True`**（maxdiff 0.0） |
| **回归硬约束**：冒烟 loss 必须逐位不变 | ✅ | `--smoke-test` **15/15 PASS、退出码 0、loss = 2.326995849609375**（与向量化前逐位一致） |

**独立实现的关键点**：负例必须在 **模型已 `.to('meta')` 之后**再改动 buffer（若先改再移到 meta，`Module._apply` 会把被替换的 CPU 张量一并搬到 meta，负例失效）。我在该顺序下复现了两个 RuntimeError。

**诚实边界（信息项）**：meta 张量不支持 `.item()`（实测 `RuntimeError: Tensor.item() cannot be called on meta tensors`），故 `stage2_recurrence` **在 meta 上跑不到底**是预期行为；说明 §1 要求的“`.to('meta')` 后索引张量 device 均为 meta”不涉及 meta 上完整前向，本项不构成缺陷。

---

## 三、逐层向量化（F16 相关）

| 判据 | 结果 | 证据 |
|---|---|---|
| 循环次数 == 层数（非 N） | ✅ | SMALL **7** 层、DEFAULT **9** 层；`level_edge_reach.shape[0]` 与之一致；层节点并集恰为全部 N |
| 无逐神经元循环 | ✅ | `stage2_recurrence` 体内无 `for … in range(self.N)`、无 `torch.tensor([node])`；层节点来自 `topo_index[s:e]` **张量切片** |
| 与独立参考实现一致（< 1e-5） | ✅ | 4 组配置：SMALL **5.96e-08**、DEFAULT **5.96e-08**、flow_axis=x **1.19e-07**、space_radius=0.9 **5.96e-08**（与说明登记的 2.98e-08~2.384e-07 同量级） |

---

## 四、F20 文档数字防线

| 判据 | 结果 | 证据 |
|---|---|---|
| `doc_numbers.json` 结构 | ✅ | `artifact_checks=32` + `metric_checks=54` + `text_checks=7` = **93**；`docs` 2 项；每项含 `value` 与 `doc` 字段 |
| `verify_all.py` 同一轮现跑比对全部 93 项 | ✅ | 输出 **一致 93 / 不一致 0 / 跳过 0**，退出码 0，末行“无 FAIL” |
| **负向验证** | ✅ | 临时篡改 `metric_checks[0]`（`r3.small.num_edges`）后：`verify_all.py` **退出码 1 且报 FAIL**；从备份还原后重新校验 **退出码 0 / 不一致 0** |
| PASS/FAIL 计数口径 | ✅ | `R1-R7b sphere_dag` = **10**、`D1-D7 device` = **6**、`E2 smoke` = **15**、`E2b smoke_mlp` = **6**、`E5 proto_smoke` = **9**、`snapshot` = **1** —— 与说明逐一相符 |

---

## 五、F18 冒烟矩阵取证

| 判据 | 结果 | 证据 |
|---|---|---|
| `run_smoke_matrix.py` 退出码 0、11/11 通过 | ✅ | 退出码 0；矩阵含 11 条记录 |
| 每条记录含 7 个必需字段 | ✅ | `artifact`/`artifact_name_matches_prediction`/`expected_artifact`/`artifact_sha256`/`torch_load_ok`/`config_matches_expectation`/`item_ok` **11/11 齐全** |
| `item_ok` 全为 true | ✅ | 11/11 true；`torch_load_ok` 与 `artifact_name_matches_prediction`、`config_matches_expectation` 亦全 true |
| 8 条旧记录曾引用不存在的产物名 → 已修正 | ✅ | **11/11 记录的 `artifact` 均真实存在且可 `torch.load`**（旧记录命名为 `..._s42.pt`，现为 `..._bs32_s42.pt`，已全部落盘） |
| `smoke.pt` 矩阵前后 SHA256 一致 = `1A9D…43D7` | ✅ | 矩阵运行后 `smoke.pt` SHA256 = **1A9D68D8F16FF88D7D848189A45D1AB6868F447ACE79950E3A545F69BC7443D7**（与说明登记值逐字符一致） |
| `smoke.pt` config 为 SMALL 默认组合 | ✅ | `N=64/y=4/H=0.15/D=0.25/seed=42/bs=32/axz/isany/rsany`，`arch=neuron3d`，loss=2.326995849609375 |

**独立稳定性测试**：连续 3 次裸跑 `--smoke-test`，`smoke.pt` 的 SHA256 **三次完全相同**（`1A9D…`），且两次跑出的文件**逐字节相同**；连跑两次 `run_smoke_matrix.py`，11 条记录的 SHA **逐条完全一致** —— 取证脚本本身是可复现的。

---

## 六、F17 / F19 文档实测数字一致性

| 判据 | 结果 |
|---|---|
| `smoke.pt`：loss `2.326995849609375`；grad_norms `W_in 0.30047276616096497` / `edge_weight 0.06744416803121567` / `neuron_bias 0.16902117431163788` / `W_out 0.12903930246829987`；E=181、S_in=13、S_out=17、层数 7 | ✅ **逐位精确匹配** |
| 四种 scope 组合 loss（2.326995849609375 / 2.299027681350708 / 2.2900023460388184 / 2.283543109893799） | ✅ 四个产物逐一精确匹配（`any/any` 落到 `smoke.pt`，与实现一致），四值互不相同 |
| R5/R5b/R5c 在 4 组配置上的值 | ✅ 数值核对通过（见 §3；`sphere_dag_metrics.json` 为全精度，README 登记值取 6 位有效数字，`small` 的 R5b 全精度 `2.9802322387695312e-08` → 登记 `2.980e-08`） |
| README 的 `d(Σa_up)/d(a_in)` 非零计数 | ✅ 脚本现跑输出 **475/512（any/any）与 480/512（all/any）**，与 README 一致；**README 已无旧值 486/512**，脚本亦不再报 486 |
| README / spec 登记文档数字校验命令 | ✅ 两处均引用 `doc_numbers` / 93 项口径 |

---

## 七、硬回归与产物纪律

| 项 | 结果 |
|---|---|
| `git status --porcelain -- n3d_proto` | ✅ 输出为空 |
| `python n3d_proto/train.py --smoke-test` | ✅ **9/9 PASS、退出码 0、loss=2.419689** |
| 一期三件产物 SHA256 | ✅ 逐字符一致（full `888556B0…8924`、highacc `9F21AC34…3F8`、capacity `0F7CF500…5011`） |
| `python -m compileall -q n3d_sphere` | ✅ 退出码 0 |
| 验证类产物全部落 `_verify/` | ✅ `doc_numbers.json` 登记的产物路径均位于 `_verify/` 下 |
| 未覆盖正式产物 | ✅ `checkpoints/n3d_sphere/model.pt` **不存在**；一期三件产物未被触碰 |

---

## 八、问题清单（1 项低severity 注记）

### 【低 D1】同一语义配置因“显式传默认值”而产生**不同字节**的 `smoke.pt`，导致矩阵记录 [0] 的 SHA 与盘上文件不符

**现象**：`run_smoke_matrix.py` 的记录 [0]（args = `--input-scope any_isolated --readout-scope any_isolated`）登记的 `artifact_sha256 = 4E09D031…`，而矩阵结束后盘上的 `smoke.pt` 为 `1A9D68D8…`（规范登记值）。连续两次运行矩阵，记录 [0] 恒为 `4E09D031…`（可复现），故不是随机抖动。

**根因（已独立复现并定位）**：

| 调用 | 生效配置 | 写盘目标 | `smoke.pt` SHA |
|---|---|---|---|
| `--smoke-test` | `SMALL_CONFIG` **单例** | `smoke.pt` | `1A9D68D8…` |
| `--smoke-test --input-scope any_isolated --readout-scope any_isolated` | 新建 `Config`（**值与 SMALL 逐字段相同**） | `smoke.pt` | `4E09D031…` |

- `build_smoke_config` 对**显式**给出的 scope 参数走 `apply_overrides` 分支，`explicit=True` → 返回**新建的 `Config` 对象**（`to_dict()` 与 `SMALL_CONFIG` **完全相同**，无任何字段差异），而裸跑返回 `SMALL_CONFIG` 单例；
- 两者 `is_default_smoke` 均为 True → 都写 `smoke.pt`；
- 实测两份 checkpoint：**loss、config、grad_norms、`state_dict` 全部张量逐位相等**，但**原始字节不同**（文件大小相同，373643 字节）→ SHA256 不同。

**影响**：
- 规范要求的 `smoke.pt` 固定 SHA（`1A9D…`）成立且稳定（裸跑 3/3 一致）；但矩阵取证文件里记录 [0] 的 SHA 指向一份**已被同轮后续写入覆盖**的内容，即**证据文件并非完全自洽**：该条目声称的 SHA 在矩阵结束后已不存在于盘上；
- 任何以“SHA 相等”作为语义等价判据的检查，会把两次**语义完全相同**的运行判为不同（字节级不确定性）；
- 不影响任何数值验收（loss/config/梯度/输出全等），故定为低severity。

**建议修复（二选一）**：
1. `build_smoke_config` 在“显式参数值与基线完全一致”时仍返回基线对象（即 `apply_overrides` 后若 `to_dict()` 与 `base.to_dict()` 相同则回退返回 `base`），使同配置同路径产出同字节；
2. 或在 `run_smoke_matrix.py` 中改用**数值/配置指纹**（而非 SHA256）判定同一组合的等价性，并在记录中标注“同一文件被同轮后续写入覆盖”。

---

## 九、环境说明

| 项 | 值 |
|---|---|
| Python / torch | 3.12.10 / 2.14.0+cpu（`cuda.is_available() == False`；F16-D6 的 CUDA 分支按脚本设计为 skip） |
| 数据 | 工程内 `data/mnist` IDX（未联网） |
| 耗时 | 单次冒烟约 6~7s；`run_smoke_matrix.py`（11 组合）约 2~3 分钟；`verify_all.py`（8 命令）< 4 分钟 |

## 十、结论

- **本轮 error 项（F16 设备契约）已彻底修复**：`level_edge_reach` / `level_node_reach` 为已注册的 int64 `[K,2]` 张量、`vars(model)` 无未注册张量或含张量容器、`.to('meta')` 后 8/8 索引张量随迁、两个守卫负例均按名抛 `RuntimeError`、`state_dict` 严格往返后前向**逐位相同**、冒烟 loss 保持 `2.326995849609375`（15/15 PASS）。
- **F20 文档数字防线可用且有效**：93 项（32+54+7）登记齐备，`verify_all.py` 一轮现跑输出「一致 93 / 不一致 0 / 跳过 0」，**负向篡改可被检出并返回非 0 退出码**，还原后重新通过；PASS 计数口径与说明逐项相符（10/6/15/6/9/1）。
- **F18 矩阵取证已修正**：11/11 记录的产物**真实存在、可 `torch.load`**、7 个字段齐全、`item_ok` 全 true；`smoke.pt` SHA 稳定为规范值 `1A9D…`（裸跑 3/3 字节一致）。
- **F17/F19 文档数字**与产物/脚本现跑一致（含 README 的 475/480 已替换旧值 486）。
- **硬回归全绿**：一期 9/9 PASS + loss=2.419689 + 三 SHA256 一致 + `n3d_proto` 零改动 + `compileall` 退出码 0；正式产物 `n3d_sphere/model.pt` 不存在，验证产物全部落 `_verify/`。
- **唯一注记 D1（低）**：显式传“与默认相同的 scope 值”会使 `smoke.pt` 产生**语义相同但字节不同**的内容，进而使矩阵记录 [0] 的 SHA 指向已被覆盖的文件。建议按上述二选一修复，以消除字节级不确定性并让取证文件完全自洽。
