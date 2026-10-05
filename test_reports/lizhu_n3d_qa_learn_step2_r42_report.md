# n3d_qa_learn 步骤 2 审查问题修复轮（第 3 轮）测试报告

- 测试智能体：离朱（R42）
- 测试时间：2026-10-05（本地 UTC+8）
- 工作目录：`E:\neuron3d`；解释器：`.venv\Scripts\python.exe`（Python 3.12.10 / torch 2.14.1+cpu / numpy 2.5.3）
- 被测对象：`n3d_qa_learn/step2.py`、`step2_run.py`（五入口）、`backends.py`、`train.py`、`README.md`、`README_step2.md`
- 测试脚本：`lizhu_r42_scripts/verify_step2_r42.py`（E1/E2/W2–W13 + I1/I2，**13/13 通过**）；明细 `lizhu_r42_scripts/_tmp/assertions_r42.json`

## 一、结论

**通过。** 验收判据全部成立：`compileall` / `probe` / `drill` / `guard` / `replay` 退码全 0；`drill.json` 的 `gradient.nonzero=True` 且两类空列表成立；E1 / E2 / W2–W13（含 I1/I2）独立复算断言 **13/13** 成立；上游零改动。

| 验收判据 | 结果 |
| --- | --- |
| `compileall -q n3d_qa_learn` 退码 0 | 通过 |
| `probe` 退码 0 | 通过 |
| `drill` 退码 0 | 通过 |
| `guard` 退码 0 | 通过（`all_tampered_rejected=True`） |
| `replay` 退码 0 | 通过（`all_match=True`，24/24） |
| `drill.json`：`nonzero=True`、`zero_grad_parameters=[]`、`parameters_outside_graph=[]` | 通过（7 个参数全非零且全在图上） |
| E1 / E2 / W2–W13 独立复算 | 通过（13/13） |
| 上游零改动 | 通过 |

## 二、逐条复测

### E1 报告 MD 双模式一致性不再出现 `None` —— 通过
- 「双模式一致性」节**不含子串 `None`**；含 `s42/s43/s44_candidates_same_model` 三行。
- 三处逐位一致（MD ↔ JSON ↔ README 4.7）：

| 用例 | top-1 一致率 | 一致数 / 总数 | logits 逐位相同 | logits 最大绝对偏差 |
| --- | --- | --- | --- | --- |
| `s42_candidates_same_model` | 1.0（README 1.0000） | 666 / 666 | False | 3.8743019104003906e-07（README 3.87e-07） |
| `s43_candidates_same_model` | 1.0 | 666 / 666 | False | 2.980232238769531e-07（README 2.98e-07） |
| `s44_candidates_same_model` | 1.0 | 666 / 666 | False | 3.5762786865234375e-07（README 3.58e-07） |

MD 与 JSON 为逐位一致（容差 1e-12 / 整数字面量一致）；README 数值列按 2 位有效数字打印（相差 < 1e-6 相对），方向与量级一致。
（本轮我最初的比较脚本把 `666 / 666` 与 `666/666` 的空格差异、以及 README 的两位有效数字截断误判为不一致，已修正比较口径后复测通过——**非被测对象缺陷**。）

### E2 路由候选宽度（构造期不变量，独立复算）—— 通过
独立构造模型与路由器后实测：`len(router.answer_keys) = 230` == `model.n_answers = 230`；`router.irrelevant_index = 230` == `model.answer_index() = 230`；`model.output_dim = 231`；`answer_table` 形状 **[231, 192]**（**不是** 299×192）。统一答案表全量 **298** 类未交给路由器（对照组 `[299,192]` 仅为反例）。`cmd_eval` 已含这三条构造期硬断言（源码可核）。审查条目 E2 的归因经本轮独立复算**同样不成立**，与 README 第 7 节一致。

### W2 ③ 与 ⑤ 口径分离 —— 通过
- ⑤ `e2e_answer_accuracy` ≤ ③ `routing_decision_accuracy` 在 **3 seed × 5 任务 = 15 项上全部成立**（0 项违反）；
- 两者**全部不相等**（15/15 均严格小于，例如 seed 42：judge ③ 0.465649 vs ⑤ 0.408397；choice ③ 0.480045 vs ⑤ 0.224916；blank ③ 0.648846 vs ⑤ 0.069259；solve ③ 0.628205 vs ⑤ 0.019231；triviaqa ③ 0.888889 vs ⑤ 0.000000）——⑤ 不再退化为 ③；
- ①–⑤ 全部 ∈ [0, 1]（0 项越界）。按纪律未设任何门槛。

### W3 日志编码 —— 通过
`_verify/step2_lz3/{probe,drill,guard,replay}.log` 与 `_verify/step2/eval_full_v6.log` 五份：首字节均为 `5b`（`[`），**非** `FF FE` / `FE FF`；UTF-8 解码无异常且 **U+FFFD 替换字符计数 = 0**。新增 `--log-file` 由 Python 以 UTF-8（无 BOM）自写。

### W4 `replay` 的 `report_metrics_sha256` 非空且可复算 —— 通过
`replay.json`：`all_match=true`、`n_cases=n_match=24`；`report_metrics_sha256 = b29b0bc95de7f1c84a8ff509ec259983a6cd57ed6fc61ffa31355c35dd393797` **非空**；与**独立复算**值（`sha256(canonical_json(report 去掉 created_utc 与 report_paths))`）**逐位一致**；且与 `step2_report.json` 的 `report_paths.metrics_sha256` **相等**。

### W12 报告与 README 登记「路由退化 / 步骤 2 不可观测」—— 通过
- `step2_report.json` 的 `honest_notes`（共 7 条）含「**路由退化**」条目与含「**没有被走到**」的等价表述（第 6 条），并含 ⑤/③ 口径分离条目（第 7 条）；
- 分项字段 `step2_reached_total` 在 **3 seed × 5 任务 = 15 项上全为 0**；
- `README_step2.md` 第 6 节第 1/2/4 条有对应登记（含「路由退化」与「步骤 2 的端到端贡献为 0 且不可观测」）。

### W5–W10 / I1 / I2 静态与结构 —— 全部通过
| 条目 | 实测 |
| --- | --- |
| W5 README_step2.md 围栏配对 | 围栏行 **4** 行（偶数），无孤立围栏行 |
| W6 `backends.py` 的 `field` 标识符 | 出现次数 **0** |
| W7 `train.py` | 无 `if True:`；`history.append` 中 `"epoch": int(...)`、`"batches": int(...)`；含 `REBUILD_LOGIT_SCALE_INIT`（=20.0）且 `rebuild_model` **不再**用 `meta["model"]["logit_scale"]` 当构造初值（改用该常量） |
| W8 `README.md` 产物一致性表述 | 不再出现「同参数重复运行产物逐字节一致」的断言式表述；含「逐字节一致…整包因 `meta.created_utc` 不同而不同」的如实口径 |
| W9/W10 `step2.py` | `__all__` **只在 L1861 赋值一次**（0 处二次赋值）；`import math` 在模块顶部 L56，**函数体内 0 处** |
| I1 骨架文件 | `n3d_qa_learn/_write_probe_limu.txt`、`_writetest.txt` **均不存在**（前者在 `_verify/step2/legacy_scaffold/`） |
| I2 smoke 目录归档 | `_verify/step2/` 下**不存在** `smoke/`、`smoke2/`、`_smoke3/`；三者均在 `_verify/step2/_deprecated/`，且该目录含 `README_DEPRECATED.md` |

### 既有判据回归（一并复测）
- `drill`：7 个可学习参数、`zero_grad_parameters=[]`、`parameters_outside_graph=[]`、`nonzero=True`、末 epoch loss 5.0409（证明骨干入图，非均匀 softmax 的 5.44）；
- `probe`：2665 行 / D=192 / 特征最大偏差 5.066e-07 超容差 0 / 留出 68 → C=230；
- `guard`：合法产物加载成功、篡改键表与篡改口径指纹均被拒、`all_tampered_rejected=True`；
- 两份正式报告 `checkpoints/qa_learn/step2/step2_report.json` 与 `_verify/step2/step2_report.json` **SHA256 完全相同**（58664 字节），`report_paths` 已回填。

## 三、已知未达标项（按纪律如实登记，未当作通过项、未要求修复）

| 项 | 实测 | 状态 |
| --- | --- | --- |
| ④ 无匹配类精确率/召回 | 三 seed × 五任务**恒 0.0000** | 未达标，如实登记 |
| `step2_reached_total` | 15 项**全 0**（步骤 2 通路不可观测） | 未达标，如实登记（W12） |
| N3D `q` 自检索 Recall@1 vs 参照下限 | index 0.0000~0.0060、pointer 0.0000~0.0015；下限 0.986486；增益 **−0.9805** | 未达标，如实登记 |
| ⑤ vs「全判未命中」平凡基线 | 如 judge ⑤ 0.4046 < 0.5344 | 未达标，如实登记 |

训练后量（①–⑤、Recall@1/@5、双模式一致率）本轮仅断言取值域 ∈ [0,1]、与报告逐位一致、以及 W2 的内部关系（⑤ ≤ ③），**未断言任何门槛**。

## 四、发现（非阻断）

1. **规格的 `replay --verify-dir` 指向目录不含报告（复现第 2 轮同一问题）**：规格给的是 `--verify-dir checkpoints/qa_learn/_verify/step2_lz3`，但该目录为空；`cmd_replay` 既写报告又**从该目录读取** `step2_report.json` 作对账基准，故**首次执行必然 `FileNotFoundError` 退码 1**（本轮现场复现）。规格同时声明「既有正式报告在 `checkpoints/qa_learn/step2/` 与 `_verify/step2/`（两份一致）」，两者互相矛盾。本轮把已有报告复制进 `step2_lz3` 后重跑，退码 0、24/24。**建议**：或把「报告输入目录」与「对账输出目录」拆成两个参数（`--report-dir` / `--verify-dir`），或在 `cmd_replay` 中于默认/指定目录缺失时回退到正式报告目录。
2. **README 第 7 节的 W 编号与测试规格的 W 编号不是同一套映射**（README 把「未用 field 导入」标为 W7、把 `if True:` 标为 W8、`REBUILD_LOGIT_SCALE_INIT` 标为 W9 等，规格则是 W6/W7/W5+W9/W10）。本轮按**测试规格**的条目定义判定，两者内容实质一致，仅编号错位，建议统一以免后续轮次对不上。
3. **W7 的 `if True:` 检查口径说明**：`train.py` 中确无 `if True:` 行；但若审查原意是「恒真分支」，本轮按字面检查，未做语义级等价检查（如 `if 1 == 1`）。

## 五、边界与产物纪律

- 本轮**未重跑 eval**（规格不要求），故 4.5/4.8 的训练后量仍为既有正式报告的数字；E1/W4/W12 均基于**既有正式报告**独立复算完成（未依赖重跑）。
- 新增文件仅在 `lizhu_r42_scripts/`；四个入口的验证输出写独立目录 `checkpoints/qa_learn/_verify/step2_lz3/`；**未修改**正式产物 `checkpoints/qa_learn/step2/`（仅为让 replay 对账向 `step2_lz3` 复制了一份报告与 MD 副本）。
- 工具说明：本工作区 `write` 工具创建文件报 `EISDIR`（硬链接失败），本轮一律改用 PowerShell here-string + `Set-Content -Encoding utf8` 写文件。

## 六、上游零改动核验（通过）

| 项 | 结果 |
| --- | --- |
| `git status --porcelain -- n3d_shape n3d_sphere n3d_proto` | 空（无跟踪改动） |
| 上述三模块源码 mtime | 最新为 2026-10-03，早于步骤 2 任何运行 |
| `git status --porcelain -- n3d_qa` | `M README.md`、`M __init__.py`、`?? adapters.py / build_qa.py / probe_zh.py / verify_qa.py / zh_features.py / tools/`（mtime 14:02–14:03，属步骤 1 在制改动，非步骤 2 引入） |
| 报告消费的 n3d_qa 冻结产物 | 与 `product_files` 记录一致（缺失 0） |
