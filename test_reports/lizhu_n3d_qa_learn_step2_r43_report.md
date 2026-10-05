# n3d_qa_learn 步骤 2 第 4 轮（replay 输入/输出语义分离）复测报告

- 测试智能体：离朱（R43）
- 测试时间：2026-10-05（本地 UTC+8）
- 工作目录：`E:\neuron3d`；解释器：`.venv\Scripts\python.exe`（Python 3.12.10 / torch 2.14.1+cpu / numpy 2.5.3）
- 本轮唯一代码变更：`n3d_qa_learn/step2_run.py`（`replay` 输入/输出分离，新增 `--report-dir`）
- 测试脚本：`lizhu_r43_scripts/verify_step2_r43.py`（**10/10 通过**）；明细 `lizhu_r43_scripts/_tmp/assertions_r43.json`

## 一、结论

**通过。** `compileall` / `probe` / `drill` / `guard` / `replay` 全部退码 0；R1（含**全新空目录首次执行**）、R2、R3 断言全部成立；R40/R41/R42 回归项全部不回退；上游零改动。

| 验收判据 | 结果 |
| --- | --- |
| compileall 退码 0 | 通过 |
| probe 退码 0 | 通过 |
| drill 退码 0 | 通过 |
| guard 退码 0 | 通过 |
| **replay 退码 0（全新空目录首次执行）** | 通过（修复前为 FileNotFoundError 退码 1） |
| R1 / R2 / R3 断言 | 全部成立 |
| 回归项（gradient / E1 / E2 / W2 / W3 / W12） | 全部不回退 |
| 上游零改动 | 通过 |

## 二、本轮重点：R1 `replay` 输入/输出语义分离

1. **全新空目录首次执行退码 0**：执行前 `checkpoints/qa_learn/_verify/step2_lz5` **不存在**（已 `Test-Path = False` 确认），直接执行规格命令 → **退码 0**、`all_match=True`、`n_cases=n_match=24`。
2. **日志首行打印实际采用的基准报告路径**：`[replay] 对账基准报告 = checkpoints\qa_learn\step2\step2_report.json`（正则匹配成功，路径存在，且与正式报告路径同文件）。
3. **`--report-dir` 优先级**（A/B 对照实测）：
   - `--report-dir .../_tmp/A`（正式报告的原样副本）→ 基准 = `_tmp/A/step2_report.json`，`all_match=True`，退码 0；
   - `--report-dir .../_tmp/B`（把 `s42_index` 的 `recall_at_1` 人为 +0.5 的篡改副本）→ 基准 = `_tmp/B/step2_report.json`，`all_match=False`、0/1 匹配、**退码 1**（`AssertionError`）。
   → 对账基准随 `--report-dir` 改变，且篡改可被检出，证明 `--report-dir` 确实**优先**且**生效**。
4. **`report_metrics_sha256`**：`021b3e0133e5c6522c3abf98cae4911bb4add4c7afa49f9de37a43e4d11d7d91`，**非空**；等于独立复算 `sha256(canonical_json(report 去掉 created_utc 与 report_paths))`；且等于 `step2_report.json` 的 `report_paths.metrics_sha256`。
5. 源码复核：`cmd_replay` 的候选顺序为 `[--report-dir] → [--verify-dir] → [--out-dir] → [正式产物目录]`，`--verify-dir` 仅作输出；`R42-1` 条目已登记在 README 第 7 节。

## 三、R2 README 与报告编号/口径一致性

- 第 7 节表头含说明行「编号沿用**审查原文**的条目编号；测试规格曾按复测顺序重排，两者内容一一对应」。
- README 声明 `metrics_sha256 = 021b3e0133e5c652…`，与 `step2_report.json` 的 `report_paths.metrics_sha256` 前 16 位**一致**（等于规格给出的期望值 `021b3e0133e5c652`）。
- 第 5 节 7 件产物：**字节数 + SHA256 前 16 位**与磁盘**逐件一致**，且与报告 `artifacts` **完全相同**（7/7 三方一致）。
- `step2_run.py` 源码指纹**三方一致**：README `5effee8a3970046d` == 报告 `source_manifest` == 现场重算磁盘 SHA256。

## 四、R3 产物 SHA 稳定性口径

- README 含「**产物 SHA 的稳定性口径**（如实说明）」说明，明确整包 SHA256 会因 `meta.created_utc` 变化、并点名三个内容成员 `key_table.pt` / `model_state_dict.pt` / `backbone_state.pt`。
- **独立验证（降级口径，规格允许）**：手头**没有**当前代码的第二轮 eval 产物（归档的 `_deprecated/{smoke,smoke2,_smoke3}` 属更早代码版本或限批 smoke 轮次，与之比对无法检验该口径），故按规格降级项执行——
  - 用**产物自带的写出器**从「产物自身 meta + 严格重建的模型」重新落盘：`model_state_dict.pt` 与 `backbone_state.pt` **逐字节相同**；`meta.json` 亦相同（本次重落用同一 meta，故整体逐字节一致），并给出实测 `created_utc` 值；
  - 键表产物同样重落：`meta.json` / `key_table.pt` **逐字节相同**，整包 SHA256 一致；键表 `meta.json` 中**确实存在** `created_utc` 字段（实测值已记录）。
- 该口径与 `README.md` 第七节对步骤 1 产物的如实口径同源。

## 五、回归项（R40/R41/R42 已通过项，逐条复核）—— 全部不回退

| 回归项 | 实测 |
| --- | --- |
| `drill.json` 梯度门禁 | `n_parameters_checked=7`、`zero_grad_parameters=[]`、`parameters_outside_graph=[]`、`nonzero=True` |
| 报告 MD「双模式一致性」节 | 不含 `None`；s42/s43/s44_candidates_same_model 三行 top-1 一致率 1.0、一致数 666/666、logits 逐位相同 False、最大偏差 3.874e-07/2.980e-07/3.576e-07，与 JSON 及 README 4.7 三处一致 |
| 路由候选空间构造期不变量 | `len(router.answer_keys)=230 == model.n_answers`、`router.irrelevant_index=230 == model.answer_index()`、`output_dim=231`、`answer_table` 形状 `[231,192]` |
| W2 | ⑤ ≤ ③ 在 3 seed × 5 任务 **15/15 成立**且 **15/15 全不相等**；①–⑤ 全部 ∈ [0,1] |
| W12 | `honest_notes` 含「路由退化」；`step2_reached_total` **15/15 全 0**；README 第 6 节有登记 |
| W3 日志 | `step2_lz5/*.log` 与 `_verify/step2/eval_full_v6.log`、`eval_full_v7.log`：首字节均非 `FF FE`/`FE FF`、UTF-8 可解码、U+FFFD 计数 0 |
| probe 事实 | 2665 行 / D=192 / 留出 68 → C=230 / 特征最大偏差 5.066e-07 |
| 产物一致性 | 报告 `artifacts` 的 7 件与磁盘 SHA256 + 字节数**逐件一致**（0 处不符） |

## 六、已知未达标项（按纪律如实登记，未当作通过项、未要求修复）

④ 无匹配类精确率/召回恒 0；`step2_reached_total` 全 0（步骤 2 通路不可观测）；N3D `q` 自检索 Recall@1（index 0.0000~0.0060、pointer 0.0000~0.0015）未超参照下限 0.986486，增益 −0.9805；⑤ 未超「全判未命中」平凡基线。训练后量本轮仅断言取值域 ∈ [0,1]、与报告逐位一致、以及 ⑤ ≤ ③ 关系，**未设门槛**。

## 七、发现（非阻断）

1. **运行窗口内产物被重新生成过**：首次清点（约 15:51–16:27）与末次复测（16:15–16:35）之间，`checkpoints/qa_learn/step2/` 的 7 件产物被重写过（键表 SHA256 由 `428aeeb5…` 变为 `16dcd28c…`）。这属于力牧按规格重跑 eval（`eval_full_v7`）的正常产物更替；末次复测时 README 第 5 节表、报告 `artifacts` 与磁盘**三方仍然完全一致**，`metrics_sha256 = 021b3e0133e5c652…` 亦与规格期望一致，故不影响验收。**提示**：若评测过程中再次重跑 eval，旧轮次的 `replay.json` / README 数字会立即失配，建议把「重跑 eval」与「复测」串行化。
2. **R3 的「另一轮产物」对照不可得**：同规格说明，`_deprecated` 下的归档产物来自更早代码版本或限批 smoke 轮次（键表含 `created_utc`、模型产物缺 `backbone_state.pt`），与之逐成员比对**不能**检验当前口径。已按规格降级口径执行并如实说明；若要真正验证「同参两轮逐成员一致」，需用当前代码连跑两次 eval。

## 八、边界与产物纪律

- 本轮**未重跑 eval**（规格不要求；已由力牧跑过 `eval_full_v7`）。
- 新增文件仅在 `lizhu_r43_scripts/`（脚本 + `_tmp/A|B|A_out|B_out` 对照材料）；四个入口的验证输出写 `checkpoints/qa_learn/_verify/step2_lz5/`（该目录按要求**从不存在开始**创建）。
- A/B 对照使用**副本**，未改动 `checkpoints/qa_learn/step2/` 的正式报告与产物。
- 工具说明：本工作区 `write` 工具创建文件报 `EISDIR`（硬链接失败），本轮一律改用 PowerShell here-string + `Set-Content -Encoding utf8`。

## 九、上游零改动核验（通过）

| 项 | 结果 |
| --- | --- |
| `git status --porcelain -- n3d_shape n3d_sphere n3d_proto` | 空（无跟踪改动） |
| 上述三模块源码 mtime | 最新为 2026-10-03，早于步骤 2 任何运行 |
| `git status --porcelain -- n3d_qa` | `M README.md`、`M __init__.py`、`?? adapters.py / build_qa.py / probe_zh.py / verify_qa.py / zh_features.py / tools/`（mtime 14:02–14:03，属步骤 1 在制改动，非步骤 2 引入） |
