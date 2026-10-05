# n3d_qa_learn 步骤 2：文本数据集匹配 + 分项验收（README_step2）

> 本文档记录**步骤 2（文本数据集匹配）**的口径、命令与**现场实测**验收结果。
> 全部数字由第 3 节命令现场运行产出（主日志 `checkpoints/qa_learn/_verify/step2/eval_full_v6.log`，
> **UTF-8 无 BOM**），未使用任何硬编码常量。报告指标指纹
> `metrics_sha256 = 021b3e0133e5c652…`（现场重算一致）。

## 1. 定位与业务逻辑

两级业务路由的**第二级**（步骤 1 未命中「不相关」时才进入）：

```
问题文本 --(D 维确定性特征)--> N3D 后端 + q 头 --> q ∈ R^D
                                                    |
                         候选键 K（两种来源，按全局开关二选一）
                                                    |
                                  logits = q @ K^T --> softmax --> top-1 行原文
```

* **index（索引式）**：候选键表由**库行文本的确定性特征**构造、**冻结不参与梯度**，
  固化进本模块自己的产物 zip（`meta.json` + `key_table.pt`）。
* **pointer（指针式）**：候选键表**由输入提供**（`K ∈ R^[B, L, D]`），产物内不存固定候选表。
* 两模式**共用同一条 q 通路**（后端特征 → q 头）。

## 2. 冻结口径

| 项 | 取值 | 来源 |
| --- | --- | --- |
| 连接参数 `D` | **192** | `buckets_per_order=64` × `n_gram_orders=(1,2,3)`；与 `n3d_qa` 产物 `doclines_rows.jsonl` 的 `feature` 词袋块同 salt / 同归一化 |
| 特征口径指纹 | `8f2523e41484adf39f6f1f455ef13f473a8dd027499af31219b675eade259fae` | `n3d_qa.zh_features` 的 `spec_hash()` |
| 文本行（只读） | `checkpoints/qa_learn/dataset/doclines_rows.jsonl`（2665 行） | `n3d_qa` 冻结产物 |
| 库 / 查询划分 | 非查询库 **1999** / 查询 **666** / 交集 **0** / 并集 **2665**（seed 20261005） | 产物 `doclines.npz` 的 `meta.split` |
| 检索池 | **全量 2665 行**（含查询行自身） | 产物 `label_rule`：`label=1 iff the candidate library row IS the query row` |
| 统一答案表 | **298** 类 | `checkpoints/qa_learn/dataset/answer_table.jsonl` |
| **模型候选空间** | **C = 230**（保留类表，不含留出类） | `select_held_out_per_task` 的保留类；**路由一律用这张表**，不是 298 |
| 开集协议 | 按任务分层留出 **68** 类 | 冻结，含 SHA256 |
| 任务 | judge / choice / blank / solve / triviaqa | 产物 `manifest.json` 的 `tasks` |
| 后端 | `n3d_shape`（N=64, y_in=y_out=4, H=D=0.15） | `n3d_qa_learn.backends.recommended_config` |
| q 头输入口径 | **`head_input_mode="concat"`（显式固定）** | `step2.DEFAULT_HEAD_INPUT_MODE`；**不吃上游默认值**（上游默认 `raw` 会让 N3D 骨干脱离计算图） |

**拓扑 seed 前提（跨 seed 报告必读）**：三个后端的 `Config` 由
`backends.recommended_config` 统一给出，其中**后端自身的 seed 恒为 42**（该函数对
`n3d_shape` / `n3d_sphere` / `n3d_proto` 三个分支都显式写死 `seed=42`）。N3D 的神经元位置
与突触几何由该 seed 决定，因此**跨 seed 报告中的所有运行共用同一套拓扑**；变化的只有
训练侧的数据顺序与参数初始化。**即：本报告的「极差」是训练随机性的极差，不含拓扑随机性。**

**源码溯源**：报告与模型产物 meta 均落 `source_manifest`（本模块 + `n3d_qa` 共 20 个 `.py`
的 SHA256）。本文档数字对应的关键源码：
`heads.py = e8354ffb7e03411d…`、`train.py = ba1c309d93f7d2a2…`、
`backends.py = 205ff3579338e5ec…`、`step2.py = 9e9d43289b3c91b1…`、
`step2_run.py = 5effee8a3970046d…`。

## 3. 命令

```bash
python -m compileall -q n3d_qa_learn

# 数据与口径取证（不训练）
python -m n3d_qa_learn.step2_run probe --product-dir checkpoints/qa_learn/dataset \
    --out-dir checkpoints/qa_learn/_verify/step2

# 单条端到端演练（硬门禁：梯度非零后才放全量）
python -m n3d_qa_learn.step2_run drill --product-dir checkpoints/qa_learn/dataset

# 产物守卫拒绝证明
python -m n3d_qa_learn.step2_run guard

# 全量分项验收（跨 seed x 双模式；正式产物与报告写 checkpoints/qa_learn/step2/）
python -m n3d_qa_learn.step2_run eval --product-dir checkpoints/qa_learn/dataset \
    --seeds 42,43,44 --modes index,pointer --epochs 12 \
    --log-file checkpoints/qa_learn/_verify/step2/eval_full_v6.log

# 从落盘产物复算并与报告逐项对账（不训练）
python -m n3d_qa_learn.step2_run replay --product-dir checkpoints/qa_learn/dataset --full-metrics
```

所有子命令都支持 `--log-file <path>`：由 **Python 自己**以 `encoding="utf-8"`（无 BOM）落日志，
**不要用 PowerShell 的 `Tee-Object`**（PowerShell 5.1 默认写 UTF-16LE，中文会乱码）。

> **`replay` 的输入 / 输出已分离（修复离朱 R42 的非阻断缺陷）**：`--report-dir` 指定**对账基准**
> `step2_report.json` 所在目录，`--verify-dir` 只作**输出**目录（写 `replay.json` 与日志）。
> 未指定 `--report-dir` 时按 `[--report-dir] -> [--verify-dir] -> [--out-dir] -> [正式产物目录]`
> 顺序自动回退，故**在全新空目录首次执行也能成功**（现场实测：`--verify-dir` 指向空目录 →
> 退码 0、`all_match=True`）。此前 `--verify-dir` 兼作输入语义，首次执行必然 `FileNotFoundError` 退码 1。

### 4.1 数据与口径取证（`probe`，退码 0）

| 取证项 | 实测值 |
| --- | --- |
| 文本行数 | 2665 |
| 库 / 查询划分 | 1999 / 666；交集 **0**；并集 **2665**；复现结果 == 产物行索引表（`positives>=1`）：**True** |
| 特征重算（vs 产物 `feature` 词袋块） | 检查 **2665** 行，最大绝对偏差 **5.066e-07**，超容差（atol 1e-6）**0** 行 |
| 统一答案表 | 298 类 |
| 按任务分层留出 | judge 2→1、choice 4→1、blank 153→31、solve 78→16、triviaqa 100→20；并集 **68** 类 → 模型候选空间 **C=230** |
| 合并样本 | 记录 12727；`train_known=4576 / train_unknown=3908 / test_known=2245 / test_unknown=1998` |
| 消费的产物文件 | 15 个，缺失 0（逐个 SHA256 记入报告的 `product_files`） |

### 4.2 单条端到端演练（`drill`，退码 0）

| 项 | 实测值 |
| --- | --- |
| 限批训练 | 1 epoch / 2 batch / 256 样本 |
| 梯度取证 | 检查 **7** 个可学习参数；**参与前向但零梯度 = 0 个**；**不在计算图上 = 0 个**（`nonzero=True`） |
| 极小自检索（库 32 行自检索） | Recall@1 = **0.0312** |
| 参照下限（同集合） | Recall@1 = 1.0000 |
| 双模式一致率 | 1.0000 |
| top-1 行回填 | 成功返回行 id + 行原文 |

> 梯度取证**分开报**「参与前向但恰为 0」与「根本不参与前向」两类，避免把
> 「口径把某条支路排除出计算图」误诊为「梯度恰好为 0」。

### 4.3 产物守卫拒绝证明（`guard`，退码 0）

| 用例 | 期望 | 实测 |
| --- | --- | --- |
| 合法产物加载 | 成功 | **成功**（[2665, 192]） |
| 篡改键表张量内容 | 拒绝 | **拒绝**：`键表指纹校验失败` |
| 篡改 `meta.feature_spec_hash` | 拒绝 | **拒绝**：`特征口径不一致` |

`all_tampered_rejected=True`。

### 4.4 产物复算对账（`replay`，退码 0）

| 项 | 实测值 |
| --- | --- |
| 用例数 / 匹配数 | **24 / 24**（`all_match=True`） |
| 覆盖 | 6 组「模型产物严格加载 + 自检索复算」、3 组「双模式一致性复算」、15 组「按任务分项指标复算」 |
| 最大绝对偏差 | **0.000e+00**（全部分项） |
| `report_metrics_sha256` | `021b3e0133e5c6522c3abf98cae4911bb4add4c7afa49f9de37a43e4d11d7d91…`（**非空且与正式报告一致**；此前因报告先写验证目录、`report_paths` 后填而恒为空串，已修复） |

> 该入口是**训练后量可复算**的硬证据。首轮实现曾因
> `N3DQA.adapter` 是普通对象、`model.state_dict()` **不含 N3D 骨干权重**（实测仅 5 个键），
> 导致产物只能复算出随机骨干的结果（当时 24 项只有 5 项匹配）；现产物改存
> `model_state_dict.pt` + `backbone_state.pt` 两份，缺骨干的产物在加载时**直接报错拒绝**。
### 4.5 自检索留出法（步骤 2 主判据；666 查询 / 2665 检索池）

| seed | 模式 | Recall@1 | Recall@5 | rank 直方图（1..5） |
| --- | --- | --- | --- | --- |
| 42 | index | **0.0060** | **0.0150** | 4 / 2 / 7 / 4 / 2 |
| 43 | index | **0.0000** | **0.0210** | — |
| 44 | index | **0.0060** | **0.0285** | 4 / 2 / 7 / 4 / 2 |
| 42 | pointer | **0.0015** | 0.0075 | 1 / 0 / 0 / 3 / 2 |
| 43 | pointer | **0.0015** | 0.0090 | — |
| 44 | pointer | **0.0000** | 0.0060 | 0 / 1 / 1 / 2 / 0 |

### 4.6 参照下限（纯确定性特征检索，不经 N3D；不作门槛）

| 指标 | 实测值 |
| --- | --- |
| Recall@1 | **0.986486**（657 / 666） |
| Recall@5 | **1.000000** |

**N3D 的 `q` 相对参照下限的 Recall@1 增益 = −0.9805**（取最优 seed 的 index 结果 0.0060
与参照下限 0.986486 之差）。即：经 N3D 后端与 `q` 头之后，**自相似结构几乎完全丢失**，
两个模式都没有把「同一行」认出来。

### 4.7 双模式一致性（只报实测值，不设门槛）

报告 JSON / 报告 MD / 本文档**三处口径一致**（此前报告 MD 因读取不存在的扁平键而恒渲染 `None`，已修复）。

| 用例 | top-1 一致率 | 一致数 / 总数 | logits 逐位相同 | logits 最大绝对偏差 |
| --- | --- | --- | --- | --- |
| `s42_candidates_same_model` | **1.0000** | 666 / 666 | False | 3.87e-07 |
| `s43_candidates_same_model` | **1.0000** | 666 / 666 | False | 2.98e-07 |
| `s44_candidates_same_model` | **1.0000** | 666 / 666 | False | 3.58e-07 |
| `s42_candidates_cross_model` | 0.0105 | 7 / 666 | False | 8.71e-01 |
| `s43_candidates_cross_model` | 0.0000 | 0 / 666 | False | 8.88e-01 |
| `s44_candidates_cross_model` | 0.0045 | 3 / 666 | False | 8.25e-01 |
| 步骤 1 跨模式 `s42_cross_model`（n=4243） | 0.3483 | — | — | — |
| 步骤 1 跨模式 `s43_cross_model`（n=4243） | 0.1046 | — | — | — |
| 步骤 1 跨模式 `s44_cross_model`（n=4243） | 0.1176 | — | — | — |

> 「同一模型双模式」是**结构性**检查：两模式在同一候选集上做的是同一次 `q @ K^T`，
> 一致率 1.0000 属预期；`logits_bitwise_equal = False` 与 ~3e-07 的偏差源于 2D `matmul`
> 与 `bmm` 两条内核路径的浮点累加顺序差异，不影响 top-1。
### 4.8 分项 5 项指标（按任务 × 按 seed）

`①` 步骤 1 答案准确率　`②` 步骤 2 Recall@1（任务无关，见 4.9）　`③` 路由正确率（宽松：只看是否走到步骤 1 答案分支）
`④` 无匹配类精确率 / 召回　`⑤` **端到端最终答案正确率（严格：命中且答案文本等于金标展示文本）**　`step2_reached` 落到文本匹配分支的样本数

逐 seed（主模型 = `index`）：

| seed | 任务 | ① | ② | ③ | ④P | ④R | ⑤（严格） | step2_reached |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | judge | 0.8770 | 0.0060 | 0.4656 | 0.0000 | 0.0000 | 0.4084 | 0 |
| 42 | choice | 0.4685 | 0.0060 | 0.4800 | 0.0000 | 0.0000 | 0.2249 | 0 |
| 42 | blank | 0.1067 | 0.0060 | 0.6488 | 0.0000 | 0.0000 | 0.0693 | 0 |
| 42 | solve | 0.0306 | 0.0060 | 0.6282 | 0.0000 | 0.0000 | 0.0192 | 0 |
| 42 | triviaqa | 0.0000 | 0.0060 | 0.8889 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 43 | judge | 0.8852 | 0.0000 | 0.4656 | 0.0000 | 0.0000 | 0.4122 | 0 |
| 43 | choice | 0.4701 | 0.0000 | 0.4800 | 0.0000 | 0.0000 | 0.2257 | 0 |
| 43 | blank | 0.1142 | 0.0000 | 0.6488 | 0.0000 | 0.0000 | 0.0741 | 0 |
| 43 | solve | 0.0374 | 0.0000 | 0.6282 | 0.0000 | 0.0000 | 0.0235 | 0 |
| 43 | triviaqa | 0.0000 | 0.0000 | 0.8889 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 44 | judge | 0.8443 | 0.0060 | 0.4656 | 0.0000 | 0.0000 | 0.3931 | 0 |
| 44 | choice | 0.4724 | 0.0060 | 0.4800 | 0.0000 | 0.0000 | 0.2268 | 0 |
| 44 | blank | 0.0843 | 0.0060 | 0.6488 | 0.0000 | 0.0000 | 0.0547 | 0 |
| 44 | solve | 0.0374 | 0.0060 | 0.6282 | 0.0000 | 0.0000 | 0.0235 | 0 |
| 44 | triviaqa | 0.1250 | 0.0060 | 0.8889 | 0.0000 | 0.0000 | 0.1111 | 0 |

跨 seed（seed 42/43/44，**均值 ± 极差**；极差 = max − min）：

| 任务 | ① | ② | ③ | ④P | ④R | ⑤（严格） | step2_reached |
| --- | --- | --- | --- | --- | --- | --- | --- |
| judge | 0.8689 ± 0.0410 | 0.0040 ± 0.0060 | 0.4656 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | **0.4046 ± 0.0191** | 0 ± 0 |
| choice | 0.4703 ± 0.0039 | 0.0040 ± 0.0060 | 0.4800 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | **0.2258 ± 0.0019** | 0 ± 0 |
| blank | 0.1017 ± 0.0300 | 0.0040 ± 0.0060 | 0.6488 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | **0.0660 ± 0.0194** | 0 ± 0 |
| solve | 0.0351 ± 0.0068 | 0.0040 ± 0.0060 | 0.6282 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | **0.0221 ± 0.0043** | 0 ± 0 |
| triviaqa | 0.0417 ± 0.1250 | 0.0040 ± 0.0060 | 0.8889 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | **0.0370 ± 0.1111** | 0 ± 0 |

**③ 与 ⑤ 的关系**：③ 是宽松口径（`source == "qa"` 即算命中，不校验答案对错），
⑤ 是严格口径（还要 `res.answer == 金标展示文本`），故恒有 **⑤ ≤ ③**；本轮两者不相等，
说明「命中但答错」确实存在（例如 judge：③ 0.4656 vs ⑤ 0.4046）。

### 4.9 口径说明与测试集规模

`②` 由**文本行自检索留出法**给出，**与任务无关**（文本行库不随任务变化），故分项表里
每个任务填入同一全局值（index 主模型逐 seed 的 Recall@1）；逐任务字段
`step2_recall_at_1_is_task_independent=True` 显式标记。

| 任务 | n_test_known | n_test_unknown |
| --- | --- | --- |
| judge | 122 | 140 |
| choice | 1287 | 1394 |
| blank | 534 | 289 |
| solve | 294 | 174 |
| triviaqa | 8 | 1 |
## 5. 产物清单（`checkpoints/qa_learn/step2/`）

| 文件 | 字节 | SHA256（前 16 位） |
| --- | --- | --- |
| `qa_step2_textrows_n3d_shape_index_D192_L2665_s0.pt.zip`（**冻结索引式键表**） | 306577 | `16dcd28c248d3246` |
| `qa_step2_model_n3d_shape_index_D192_C230_s42.pt.zip` | 299294 | `f6878cd22ae03068` |
| `qa_step2_model_n3d_shape_pointer_D192_C230_s42.pt.zip` | 231236 | `858a266e27a02dd2` |
| `qa_step2_model_n3d_shape_index_D192_C230_s43.pt.zip` | 299330 | `24a1b998d0f148f9` |
| `qa_step2_model_n3d_shape_pointer_D192_C230_s43.pt.zip` | 231249 | `7eee183afe2c11a8` |
| `qa_step2_model_n3d_shape_index_D192_C230_s44.pt.zip` | 299402 | `9140cb19124146d6` |
| `qa_step2_model_n3d_shape_pointer_D192_C230_s44.pt.zip` | 231181 | `3c04f17bd2b0d379` |

> **产物 SHA 的稳定性口径（如实说明）**：键表与模型产物的 `meta.json` 含 `created_utc`
> （挂钟时间），因此**同参重跑的整包 SHA256 会变**（上表为 `eval_full_v7` 一轮的值）；
> 逐成员拆开后 `key_table.pt` / `model_state_dict.pt` / `backbone_state.pt` 在同参重跑间
> **逐字节一致**（这与 `README.md` 第七节对步骤 1 产物的如实口径同源）。若需要整包稳定，
> 应把 `created_utc` 移出产物 `meta`，改由报告承载。

产物成员：键表 zip = `meta.json` + `key_table.pt`；模型 zip = `meta.json` +
`model_state_dict.pt`（头与缓冲）+ **`backbone_state.pt`（N3D 骨干）**。
模型 meta 含 `head_config`（`head_input_mode` / `normalize_query` / `answer_table_mode` /
`logit_scale_init` / `learn_logit_scale` / `mix_logit_init`）与 `source_manifest`（源码 SHA256）。

报告：正式目录与验证目录各一份内容一致的 `step2_report.json` / `step2_report.md`
（均带 `report_paths`）。验证目录另有 `probe.json`、`drill.json`、`guard.json`、`replay.json`
与 UTF-8 日志 `probe.log` / `drill.log` / `guard.log` / `replay.log` / `eval_full_v6.log`，
以及两件**刻意篡改**的产物 `tampered_keytable.pt.zip` / `tampered_spechash.pt.zip`。
历史轮次已归档到 `_verify/step2/_deprecated/`（`smoke/`、`smoke2/`、`_smoke3/` 与废弃说明），
脚手架文件归档到 `_verify/step2/legacy_scaffold/`。

## 6. 如实登记（未达标项 / 不可复现项 / 边界）

1. **未达标（如实报告）：`④` 无匹配类的精确率与召回恒为 0.0000。** 三个 seed、五个任务上
   模型**从未输出「不相关」类**，因此 `③` 路由正确率退化为「已知题命中率」。
   这是**开集能力未落地**的直接证据，不是通过项：`index` 模型含未知样本的末 epoch 训练准确率
   仅 0.1993~0.2003。**未达标，如实上报。**
2. **未达标（如实报告）：步骤 2 的文本行匹配在任何被测指标里都没有被走到。**
   分项表的 `step2_reached_total` **全部为 0**（跨 seed 均值 0）。根因同第 1 条：步骤 1 从不
   拒绝，路由永远不会落到第二级。**这意味着本报告的分项指标只覆盖了「步骤 1 + 路由」链路，
   步骤 2 的端到端贡献为 0 且不可观测**；步骤 2 本身的能力只能由第 4.5/4.6/4.7 节的自检索
   留出法、参照下限与双模式一致性来刻画。**未达标，如实上报。**
3. **未达标（如实报告）：N3D 的 `q` 在步骤 2 未带来增益，且差距很大。** `index` 模式自检索
   Recall@1 = 0.0000~0.0060、`pointer` 0.0000~0.0015，纯确定性特征检索为 0.986486，
   增益 = **−0.9805**。结论：`q` 是为「答案分类」训练的，**并未保留行级自相似结构**；
   要让 `q` 在步骤 2 稳定带来增益，需要为文本行匹配单独训练 `q`（或联合训练），**本轮未做**。
4. **未达标（如实报告）：`⑤` 端到端正确率未超「全判未命中」平凡基线。** 例如 `judge`
   `⑤ = 0.4046`，而该任务全判未命中在 122 已知 + 140 不相关上即 140/262 = 0.5344。
5. **「库与查询分离」与「命中自身行」的联合口径**：产物 `label_rule` 规定
   `label=1 iff the candidate library row IS the query row`，故候选池必须**包含查询行自身**；
   本实现取检索池 = 冻结行表**全量 2665**，并把冻结的 1999/666 互斥性作为划分审计项单列
   （`intersection=0`，已独立复核）。这与「库与查询严格分离」的字面读法**不可同时成立**，
   此处按产物口径实现并显式登记。
6. **「不相关」样本非数据集原生**：`n3d_qa` 冻结产物只导出已入表答案的正样本
   （移除题数：judge 0、choice 15、blank 3060、solve 21864），故本轮用**按任务分层的留出类**
   开集协议构造「不相关」，清单冻结并落 SHA256，**不随运行重抽**。
7. **triviaqa 任务样本量过小，其分项不可作结论**：仅 27 题（train 16/2、test 8/1）。
   `①`/`⑤` 的极差 0.1250/0.1111 来自 8 条测试样本，属**统计噪声**。
8. **跨 seed 极差不含拓扑随机性**：所有运行共用后端 `seed=42` 的同一套拓扑（见第 2 节前提）。
9. **不可复现项**：无已知不可复现项。`replay` 对 24 个用例的复算偏差全为 0；唯一非逐位相等处
   为 `probe` 的特征重算与产物 `feature` 词袋块的偏差 **5.066e-07**（float32 舍入），已记录。
10. **上游零改动**：本模块只读 `n3d_qa` 冻结产物与 `n3d_shape`/`n3d_sphere`/`n3d_proto` 的
    `config`/`model`（经 `backends` 代理层），未写入任何上游目录。
11. **运行期上游漂移（如实记录）**：本轮执行期间上游被改写多次（13:06 指针尺度更名；
    13:44 新增 `head_input_mode`/`answer_table_mode`；14:29 默认改 `raw`、q 头恢复 192→192）。
    对策是显式固定 `head_input_mode="concat"` 并把 `head_config` 与 `source_manifest` 写进
    产物与报告。**若上游继续变更，需重跑 `probe`/`drill`/`eval`/`guard`/`replay` 后方可引用本文档。**
12. **早期数轮结果已废弃（如实记录）**：13:06 版（index R@1 0.0000~0.0015）、13:44 版
    （0.8754~0.8934、增益 −0.0931）以及 smoke/smoke2/smoke3 三轮限批结果，均因上游重构或口径
    变更而**不再代表当前代码**，已归档到 `_verify/step2/_deprecated/`，不作为验收依据。
13. **模块级缺陷（本轮修复，如实记录）**：`N3DQA.adapter` 是普通 Python 对象、
    不是 `nn.Module`，因此 `N3DQA.state_dict()` **不包含 N3D 骨干权重**（实测只有 5 个键）。
    只保存 `state_dict()` 会得到「加载后骨干随机初始化」的假产物。本模块已改为
    `model_state_dict.pt` + `backbone_state.pt` 双份存储，缺骨干的产物加载时直接报错拒绝。
    **该缺陷同样存在于步骤 1 的 `train.build_artifact_bytes`**（仍是只存 `state_dict()`），
    建议步骤 1 同步修复。

## 7. 本轮审查问题修复对照（逐条）

> 编号沿用**审查原文**的条目编号；测试规格曾按复测顺序重排，两者内容一一对应
> （离朱 R42 已核对实质一致，仅编号错位）。


| 审查条目 | 处置 | 现场证据 |
| --- | --- | --- |
| E1 双模式一致性在 MD 中恒为 `None` | 渲染层改为按 seed 遍历 `s*_candidates_same_model` / `s*_candidates_cross_model` 并输出逐用例表；`cmd_eval` 同时写扁平汇总键 | 4.7 节表；JSON/MD/README 三处一致 |
| E2 路由候选宽度错配 | **现场复核不成立**：`len(router.answer_keys)` = 模型 `n_answers` = **230**、`router.irrelevant_index` = `answer_index()` = **230**、`answer_table` 形状 **[231, 192]**，末位可达；按建议把三条不变量写成 `cmd_eval` **构造期硬断言** | 断言已生效（eval 退码 0）；`step2_reached_total=0` 的真实根因是「不相关」类没学出来，已在第 6 节第 1/2 条如实登记 |
| W1 `TextRowMatcher.logits` 重复前向 / q 被丢弃 | **现场复核不成立**：现行 `logits` 只调一次 `q_of` 并在 index/pointer 两分支复用同一个 `q` 张量，不存在丢弃或二次骨干前向 | `step2.py` `TextRowMatcher.logits` 实现 |
| W2 ⑤ 口径与实现不一致 | 已**分离口径**：③ 宽松（是否走到答案分支）、⑤ 严格（命中且答案文本等于金标展示文本），恒有 ⑤ ≤ ③ | 4.8 节：judge ③ 0.4656 vs ⑤ 0.4046 |
| W3 日志为 UTF-16LE | 新增 `--log-file`，由 Python 以 UTF-8（无 BOM）自写；不再用 `Tee-Object` | 各 `.log` 首字节非 `FF FE` |
| W4 `report_metrics_sha256` 恒为空 | 报告先写正式目录拿路径 → 回填 `report_paths` → 重写正式报告并写验证目录副本；指标指纹改为排除 `created_utc` **与** `report_paths` | 4.4 节：`b29b0bc95de7f1c8…` 非空且可复算 |
| W5 README_step2.md 孤立围栏 | 已删除；围栏配对校验通过 | 围栏计数为偶数 |
| W6 README.md「产物逐字节一致」失实 | 改为如实口径「同参数权重与答案表逐字节一致，整包因 `meta.created_utc` 不同而不同」并附两组实测 SHA256 | `README.md` 第七节 |
| W7 `backends.py` 未用 `field` 导入 | 已删除（全文件 `field` 出现次数 1→0） | `compileall` 退码 0 |
| W8 `train.py` `if True:` | 已去掉并回退 66 行缩进 | `compileall` 退码 0 |
| W9 `rebuild_model` 用训练后 `logit_scale` 当初值 | 新增固定常量 `REBUILD_LOGIT_SCALE_INIT = 20.0` 并注明真实值由 `load_state_dict` 恢复 | `train.py` |
| W10 `history` 字段类型不一致 | `epoch` / `batches` 统一为 `int` | `train.py` |
| W11 模块根目录残留脚手架文件 | `_write_probe_limu.txt` 移入 `_verify/step2/legacy_scaffold/` | 目录可核 |
| W12 未登记「路由/步骤 2 通路不可观测」 | 报告 `honest_notes` 已加两条（路由退化 + ⑤/③ 口径分离），本文档第 6 节第 1/2/4 条同步 | 报告 JSON `honest_notes` |
| W13 smoke/ smoke2/ 口径不一致 | 与其后的 `_smoke3/` 一并移入 `_verify/step2/_deprecated/` 并写 `README_DEPRECATED.md` | 目录可核 |
| I1 `__all__` 二次赋值 + 函数体内 `import math` | 已合并进首个 `__all__`；`import math` 提到模块顶部 | `step2.py` |
| R42-1 `replay` 首次在空目录执行必然退码 1 | **已修**：新增 `--report-dir`（对账基准目录），`--verify-dir` 只作输出；未指定时按 `[--report-dir]→[--verify-dir]→[--out-dir]→[正式产物目录]` 回退 | 现场：`--verify-dir` 指向全新空目录 → **退码 0**、`all_match=True` |
| R42-2 README 与测试规格的 W 编号错位 | 已在第 7 节表头注明「编号沿用审查原文」 | 本条 |
| R42-3 `if True:` 仅按字面检查 | 如实说明：本项目该项为**字面级**检查（`train.py` 已无该字面行），未做恒真分支的语义等价证明 | 本条 |
| I2 README.md 精确性 | ① drill 行改为「不写死 2/2，判据只看零梯度集合为空」；② CLI 列表补充 `step2_run` 五入口交叉引用 | `README.md` |