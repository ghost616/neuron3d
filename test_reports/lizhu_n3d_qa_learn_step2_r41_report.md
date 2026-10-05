# n3d_qa_learn 步骤 2（第 2 轮：R40 三条缺陷修复后复测）测试报告

- 测试智能体：离朱（R41）
- 测试时间：2026-10-05（本地 UTC+8）
- 工作目录：`E:\neuron3d`；解释器：`.venv\Scripts\python.exe`（Python 3.12.10，torch 2.14.1+cpu，numpy 2.5.3）
- 被测对象：`n3d_qa_learn/step2.py`、`n3d_qa_learn/step2_run.py`（五入口）、`n3d_qa_learn/README_step2.md`
- 本轮测试脚本：
  - `lizhu_r41_scripts/verify_step2_r41.py`（独立复算 9 项断言，**9/9 通过**）
  - `lizhu_r41_scripts/recompute_metrics_from_artifacts.py`（**自建代码路径**复算训练后量）
  - 明细：`lizhu_r41_scripts/_tmp/assertions_r41.json`

## 一、结论

**通过（9 条验收判据全部成立；9/9 独立复算断言成立）。** R40 登记的三条缺陷已逐条修复并经现场复测确认；另发现 2 项**非阻断**问题（规格文字笔误、replay 的 `--verify-dir` 复合语义），建议澄清但与验收判据无关。

| 验收判据 | 结果 |
| --- | --- |
| `compileall -q n3d_qa_learn` 退码 0 | **通过**（exit 0） |
| `probe` 退码 0 | **通过**（exit 0） |
| `drill` 退码 0 | **通过**（exit 0） |
| `guard` 退码 0 | **通过**（exit 0，`all_tampered_rejected=True`） |
| `replay` 退码 0 | **通过**（exit 0，`all_match=True`，24/24） |
| `drill.json` 的 `gradient.nonzero == True` 且 `zero_grad_parameters`、`parameters_outside_graph` 均为空 | **通过**（`n_parameters_checked=7`，两类均为 `[]`） |
| 断言 1–9 独立复算全部成立 | **通过**（9/9） |
| 上游零改动（`n3d_qa`/`n3d_shape`/`n3d_sphere`/`n3d_proto`） | **通过** |

## 二、R40 三条缺陷的复测（逐条）

### 缺陷 1：drill 硬门禁必然失败 —— **已修复**

- 修复点：`step2.build_model()` 新增 `head_input_mode` 形参，默认 `DEFAULT_HEAD_INPUT_MODE = "concat"`（`step2.py` L132/L795），**不吃上游默认值**；`_gradient_report` 把「参与前向但恰为 0」与「根本不参与前向」**分开报**（`step2_run.py` L289–318）。
- 现场复测：`drill` 退码 **0**；`n_parameters_checked=7`、`zero_grad_parameters=[]`、`parameters_outside_graph=[]`、`nonzero=True`；末 epoch loss 5.0409（不再是均匀 softmax 的 5.4424，证明骨干确实入图）。
- 独立复核：`concat` 下骨干梯度非零、`raw` 下为零（本轮 A6/M2 口径复算一致）。

### 缺陷 2：产物与代码架构不兼容、无法复算 —— **已修复**

- 修复点：产物 zip 改为 `meta.json` + `model_state_dict.pt`（头/缓冲）+ **`backbone_state.pt`**（N3D 骨干）；`meta` 新增 `head_config`（`head_input_mode`/`normalize_query`/`answer_table_mode`/`logit_scale_init`/`learn_logit_scale`/`mix_logit_init`）与 `source_manifest`（源码 SHA256）；`rebuild_model_from_meta()` 严格重建，缺骨干**直接抛错**（`step2_run.py` L884–911）。
- 现场复测（独立脚本、真实产物）：报告列出的 **6 件模型产物全部严格重建并 `load_state_dict(strict=True)` 成功**，无 size mismatch / missing key；产物成员均含 `backbone_state.pt`；**删除该成员后 6/6 全部被拒**（`ValueError: 产物缺少骨干权重…`）；`backbone_state.pt` 的键集合与形状与全新构造的 `model.adapter.model.state_dict()`**逐键一致**（每件 20 键）；`N3DQA.state_dict()` 仅 **5 键**（`answer_table`/`logit_scale`/`mix_logit`/`q_head.weight`/`q_head.bias`），**不含任何骨干键**（缺陷根因复现确认）。
- 训练后量复算：`replay --full-metrics` **24/24 全部匹配、偏差 0.0**；另用**自建代码路径**（不调用 `cmd_replay`）从产物独立复算：6 组自检索 `Recall@1/@5` 与 rank 直方图**偏差全部 0.0**、seed 42 五任务分项**最大偏差 0.0**。

### 缺陷 3：README 与实现不一致 —— **已修复**

- 现场对账：README 声明的 `metrics_sha256 = 60ac19fd93b12ca7` 与按同口径（剔除 `created_utc` 的规范化 JSON SHA256）现场重算**逐位一致**；README 第 5 节 7 件产物的「字节数 + SHA256 前 16 位」与磁盘**逐件一致**，且与报告 `artifacts` 列表**完全相同**；README 声明的源码指纹 `heads.py = e8354ffb7e03411d`、`train.py = 2f1f911872eff95e`、`step2.py = 43ed403aeea3ed89` 与报告 `source_manifest` 及**现场重算源码 SHA256 三者一致**；`source_manifest` 覆盖 20 个 `.py`。

## 三、九项独立复算断言（全部 PASS）

| # | 断言 | 结果 | 关键实测 |
| --- | --- | --- | --- |
| 1 | 划分可复核 | PASS | `default_rng(20261005).permutation(2665)`：库 1999 / 查询 666 / 交集 0 / 并集 2665；复现查询集合 == `positives>=1` 的 666 行；行索引表 `index` 字段与行序一致 |
| 2 | 特征实现与产物等价 | PASS | 全量 2665 行最大绝对偏差 **5.066394805908203e-07**、超容差 0 行；**非自洽对照**（`buckets_per_order=32`）偏差升到 **0.894**，证明该比对确由产物驱动 |
| 3 | 行向量口径 | PASS | `dim=192=64*3`；`spec_hash` 与冻结口径一致；重复编码逐位一致；不消耗 numpy/torch 全局 RNG；空文本全零无 NaN |
| 4 | 冻结键表守卫 | PASS | 现场重算 `key_table_sha256=121f27a2…4f21`、`line_ids_sha256`、`feature_spec_hash` 与 meta 一致；**自行构造 6 件篡改产物**（改张量 / 改口径指纹 / 改 meta 键表指纹 / 改维数 / 删 `key_table.pt` / 删 `meta.json`）**全部抛 ValueError**；`guard.json` 的 `all_tampered_rejected=True` |
| 5 | 开集协议 | PASS | 独立复算 judge 2→1、choice 4→1、blank 153→31、solve 78→16、triviaqa 100→20；并集 **68**、`C=230`；清单 SHA256 `2944e155…9605` 与报告、probe **双侧一致**；留出类均在 `answer_table.jsonl` 且不在候选空间 |
| 6 | 产物可复算（本轮重点） | PASS | 见缺陷 2 复测：6/6 严格重建、缺骨干 6/6 被拒、骨干键集合逐键一致、`state_dict()` 不含骨干 |
| 7 | replay 对账 | PASS | `all_match=True`、24/24（6 自检索 + 3 双模式 + 15 分项）、最大偏差 0.0；独立复核其中 2 项（某 seed 自检索 Recall@1、某任务 `①`）偏差 **0.0** |
| 8 | 双模式结构性取证 | PASS | 同一 trained index 模型（由产物重建）以两种候选来源对 **666 行**打分：top-1 一致率 **1.0**（666/666）、logits 最大绝对偏差 **3.8743019104003906e-07 < 1e-5**、`logits_bitwise_equal=False`、键表 `requires_grad=False`；与报告逐位一致 |
| 9 | 跨 seed 聚合可复算 | PASS | 由 `per_seed` 独立重算 5 任务 × 7 字段的均值/极差/最小/最大/n，与 `cross_seed` **最大绝对偏差 0.0**（容差 1e-12） |

**补充（纪律遵守）**：`①`–`⑤` 仅断言 ∈ [0,1] 与与报告逐位一致，**未断言任何门槛**；`②` 逐任务同值且 `step2_recall_at_1_is_task_independent=True`；重新复算的 `gain = 0.006006… − 0.986486… = −0.9804804804804805` 与报告一致。

## 四、已知未达标项（如实登记，未当作通过项、未要求修复为通过）

| 项 | 实测 | 状态 |
| --- | --- | --- |
| `④` 无匹配类精确率/召回 | 三 seed × 五任务**恒为 0.0000** | 未达标，如实登记 |
| N3D `q` 自检索 Recall@1 vs 参照下限 | index 0.0000~0.0060、pointer 0.0000~0.0015；参照下限 **0.986486**；增益 **−0.9805** | 未达标，如实登记 |
| `⑤` vs 平凡基线 | 如 judge `⑤ = 0.4656` < 「全判未命中」0.5344 | 未达标，如实登记 |

README 第 6 节（12 条）已包含上述内容，且新增两条有价值的如实登记：缺陷 2 的根因（`N3DQA.adapter` 非 `nn.Module` → `state_dict()` 不含骨干）与「该缺陷同样适用于步骤 1 的 `train.build_artifact_bytes`」。**独立复核成立**：`n3d_qa_learn/train.py` L550 仍为 `_state_to_bytes(model.state_dict())`，确实只存头/缓冲。

## 五、非阻断问题（建议澄清 / 修复）

1. **规格文字与产物清单不一致（文档级）**：测试规格写「逐个 `torch.load` **7 件**模型产物」，而报告与 README 一致声明的是 **6 件模型产物 + 1 件冻结键表 = 7 件产物**（3 seed × 2 模式 = 6）。本轮按「报告中列出的全部模型产物」判定并要求 6/6 严格重建通过；若确需 7 件模型产物（例如补 1 个配置），需补跑 eval。
2. **`replay` 的 `--verify-dir` 是「输入+输出」复合语义（可用性）**：`cmd_replay` 既把报告写到 `--verify-dir`，又**从该目录读取** `step2_report.json` 作为对账基准；因此按规格给出的「`--verify-dir checkpoints/qa_learn/_verify/step2_lz2`」在全新目录上**首次执行必然 `FileNotFoundError` 退码 1**（本轮现场复现）。本轮将该已有报告置于该目录后重跑，退码 0、24/24。建议：或把报告输入与对账输出拆成两个参数，或在报文里明确「该目录须已存在报告」。

## 六、复算边界（如实说明）

- 本轮**未重跑 eval**（长跑约 25 分钟；规格允许）。因此报告中 4.5/4.8 的训练后量是**上一轮生成**的数字，本轮**未重新生成**；但这些数字已可通过 `replay --full-metrics`（24/24）与**本轮自建路径的独立复算**（6 组自检索 + 5 任务分项，偏差 0.0）从落盘产物逐位复现——即「训练后量可复算」这一实质性要求已成立。
- `replay` 的 24 项对账、以及 `probe` 的入口自断言，均由被测模块自身代码产生；本报告不把它们当作独立证据，独立证据是第三节 9 项与 `recompute_metrics_from_artifacts.py`。
- 唯一的非逐位相等处仍是 `probe` 特征重算与产物 `feature` 词袋块的 **5.066e-07**（float32 舍入），已在 `probe.json` 以 `max_abs_deviation`/`atol` 显式记录。

## 七、上游零改动核验（通过）

| 项 | 结果 |
| --- | --- |
| `git status --porcelain -- n3d_shape n3d_sphere n3d_proto` | 空（无跟踪改动） |
| 上述三模块源码 mtime | 最新为 2026-10-03，**晚于步骤 2 任何运行** |
| `git status --porcelain -- n3d_qa` | `M README.md`、`M __init__.py`、`?? adapters.py / build_qa.py / probe_zh.py / verify_qa.py / zh_features.py / tools/`（mtime 14:02–14:03，属**步骤 1** 在制改动，非步骤 2 引入） |
| 报告消费的 15 个 `n3d_qa` 冻结产物 | 现场逐个重算 SHA256 与字节数，与 `product_files` **全部一致**（缺失 0） |

## 八、环境与产物纪律

- 未发生环境阻断；全部命令一次通过（唯一一次 `replay` 失败为第五节第 2 条的输入目录问题，非环境问题）。
- 本轮新增文件仅在 `lizhu_r41_scripts/`；`probe`/`drill`/`guard`/`replay` 的验证输出写独立目录 `checkpoints/qa_learn/_verify/step2_lz2/`，**未覆盖** `checkpoints/qa_learn/step2/` 的正式产物与 `_verify/step2/` 的既有报告（为让 replay 可对账，向 `step2_lz2` 复制了一份 `step2_report.json` 副本，正式报告未被修改）。
- 工具说明：本工作区 `write` 工具创建文件报 `EISDIR`（硬链接失败），本轮一律改用 PowerShell here-string + `Set-Content -Encoding utf8` 写文件。
