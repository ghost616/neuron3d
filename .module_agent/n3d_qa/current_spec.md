N3D QA 数据集处理模块（通用）。当前内置数据集：TriviaQA（OpenDataLab/TriviaQA 镜像）——证据段落二分类，产出 4 个 base + 4 个 rich 的 npz 供 n3d_shape 以 --dataset npz 消费。后续 QA 数据集按 adapter 挂入。
## 模块总览与对外能力

N3D QA 数据集处理模块（通用）：任意问答数据集 -> N3D 数组格式（`X[M,D] float32` / `y[M] int64` / `meta` 的 npz）。

* 参考实现：**TriviaQA**（证据段落二分类）——归档流式扫描、split 表、答案匹配、base/rich 特征口径、8 个正式产物（`checkpoints/triviaqa/`）与 E0–E8 验证。**该路径与其产物字节冻结**，本轮增量扩展一行未改。
* 增量扩展（2026-10-05）：**中文文本特征口径**（字符级 1/2/3-gram 哈希词袋）+ **QA 匹配产物**（Math1 四题型 + TriviaQA top-100 + 跨任务统一答案表）+ **文本行产物**（`data/doc` 非空行、库/查询留出集），落 `checkpoints/qa_learn/dataset/`；并新增中文判别力探针（硬门禁）与 F0–F9 产物验证。
* 对外入口：
  * `n3d_qa/build_dataset.py`（既有）：TriviaQA npz 构建（CLI `--split/--no-bag/--features/--hash-dim/--max-questions/--out-dir/--archive`）。
  * `n3d_qa/verify_dataset.py`（既有）：E0–E8 产物验证。
  * `n3d_qa/build_qa.py`（新）：中文 QA/文本行产物构建（CLI `--tasks/--doclines/--neg-per-question/--out-dir/--ngram-orders/--buckets-per-order/--triviaqa-splits/--triviaqa-topn/--doc-query-ratio/--doc-neg-per-query/--max-records/--seed/--drill/--dry-run`）。
  * `n3d_qa/probe_zh.py`（新）：中文判别力探针（CLI `--sets/--sweep/--min-auc/--max-empty-frac/--no-gate`），退码 3 表示门禁失败。
  * `n3d_qa/verify_qa.py`（新）：F0–F9 验证（CLI `--dir/--checks/--run-verify-dataset`）。
* 本模块不导入、不修改 n3d_proto / n3d_sphere / n3d_shape / n3d_viz / framework 的任何代码（仅只读导入 `n3d_shape.data.load_npz_arrays` 判契约）；零新依赖（torch / numpy / 标准库）。
## 文本特征口径（英文 Token 词袋 + 中文 字符 n-gram）

文本特征口径是本模块的「数据集无关」层，两种语言口径**并存且互不影响**：

* **英文口径（既有，冻结）**：`build_dataset.TOKEN_RE = [a-z0-9]+`，整体小写后分词 + blake2b 落桶 + L2 归一化；列定义 `bow_hash{N}` + 6 列覆盖度/长度/数字答案标记（base D=70、nobag D=6、rich D=74/10）。
* **中文口径（新增）**：`n3d_qa/zh_features.py`。为什么需要它：既有 `[a-z0-9]+` **丢弃全部非 ASCII 字符**，中文文本词袋恒全零（阻塞项）。
  * 规范化：`NFKC -> 去掉全部 Unicode 空白 -> casefold()`；分词：字符级，对每个 `n in n_gram_orders` 产出全部连续 n-gram。
  * 落桶：`f"{order}:{ngram}"` 经 `blake2b(ZH_HASH_SALT + unit, digest_size=8)` 取大端整数 `% buckets_per_order`，块偏移 `order_index * buckets_per_order`；块做 L2 归一化（范数 0 保持全零）。
  * 盐：`ZH_HASH_SALT = b"n3d-qa-zh-charbow-v1\x00"`。
  * 列定义（9 条，`feature_columns()` 现场生成）：`zh_char{1,2,3}gram_hash{N}` 三块 + `q_to_c_coverage` / `c_to_q_coverage` / `jaccard` / `log1p_candidate_chars` / `log1p_query_chars` / `length_ratio`。
  * 维度与样本量匹配（历史纠正 #12）：QA 侧 `buckets_per_order=100` → `bag_dim=300`、`D=306`；文本行侧 `64` → `192`、`D=198`。
  * 口径参数（阶数、每阶桶数、盐、归一化方式、分词规则、落桶规则、块归一化）全部写入产物 meta（`features.spec`）并折叠为单一 `spec_hash`（QA 侧实测 `275feb01683b8352…`、文本行侧 `8f2523e41484adf3…`）。
  * 确定性：纯哈希 + 显式 seed 驱动，**不消耗全局 RNG**；同一文本重复调用逐位一致（F1 断言）。
* 共享的额外特征（两侧同源定义，均落 `EXTRA_COLUMN_NAMES`）：覆盖度、Jaccard、候选/查询字符数 log1p、长度比。
## 数据集适配层（Math1 / data/doc 文本行 / 统一答案表）

`n3d_qa/adapters.py`：数据集输入解析与「与数据集强相关」的口径，**不混入通用特征层**。

* **Math1 adapter**：八文件 `data/kupasai/math1/.../data/{离散数学,高等数学}/{subject}_{qtype}.jsonl`，字段键集合实测 `['answer','choices','explanation','id','qtype','question','sampling_results','subject']`（八文件一致）；共 37,639 条（88/405/579/895 + 696/5,124/22,689/7,163）。
  * 问题文本 = `question` + 全部 `choices`（换行连接）；答案文本 = 候选答案表层串 + 该题 `explanation`。
  * 答案归一化规则 `ANSWER_NORM_RULE = "NFKC -> 去掉全部 Unicode 空白 -> casefold()"`，实现 `normalize_answer_key`；**键空间单一**（`answer_key_variants` 只返回该 canonical key），保证选择/建表/覆盖/标签判定同源。
* **文本行 adapter**：`data/doc/{a,b,c,d}.md` 按非空行（`strip()` 后非空）切分，实测 51/2,403/167/44 = **2,665 行**；`row_id = blake2b("line\x1f<file>\x1f<line_no>", digest_size=8)`。
* **跨任务统一答案表**：`build_answer_table` 把各任务选中的答案键合并去重，每类记录 `count`（含该键的源题目数）与 `sources`（`任务 -> 计数`）；产出 `AnswerTable(classes, index_by_key, alias_index, n_questions_seen, n_questions_in_table, n_multi_value_questions)`，并落 `answer_table.jsonl` 可回读。
* **负样本采样**：`class_sampling_weights` + `sample_wrong_answers`（类级、按 `count` 加权、禁用提问自身的正确类、可选 `restrict` 限制在任务家族内），保证负样本数**精确等于请求数**；随机性全部来自显式 seed 的 `np.random.default_rng`。
* **库/查询划分**：`split_library_query(rows, query_ratio, seed)` —— 固定 seed 置换，返回升序、互斥、并集覆盖全部行的 `(library, query)`。
* **事件/答案判定（既有，TriviaQA 侧）**：`doc_contains_answer` / `compile_answer_patterns`（大小写不敏感 + 词边界 + 短数字保护）等，本轮未改。
## QA 与文本行产物构建

`n3d_qa/build_qa.py`：新增产物的编排、向量化与落盘。

* **任务与产物**：`judge`（判断题 2 类）/ `choice`（选择题 a–d）/ `blank`（填空题，答案频次 ≥ 5）/ `solve`（解答题，答案频次 ≥ 5）/ `triviaqa`（归档 split 的答案频次 top-100）/ `all`（五任务行向拼接）；另有文本行产物 `doclines`。
* **标签规则**：`label=1` ⇔ 候选答案类属于该题正确答案类（别名解析后比对，`validate_task_product` 现场复算校验）；`label=0` = 同任务家族内按类频次加权抽出的错误类（提问自身的正确类被禁用 → 负样本数精确等于请求数）。文本行侧 `label=1` ⇔ 候选行 row_id == 查询行 row_id。
* **采样参数**：`MATH1_NEG_PER_QUESTION = {judge:2, choice:3, blank:3, solve:3}`、`TRIVIAQA_NEG_PER_QUESTION=3`、`QA_NEG_SEED=20261005`（各任务 `seed+i` 保持独立）、`DOC_SPLIT_SEED=20261005`、`DOCLINES_QUERY_RATIO=0.25`、`DOCLINES_NEG_PER_QUERY=1`。
* **选择常量**：`MIN_CLASS_SAMPLES=5`、`MIN_REPEAT=5`、`CHOICE_LETTERS=('a','b','c','d')`、`TRIVIAQA_TOPN=100`、`TRIVIAQA_SPLITS_DEFAULT=('wiki','web')`。
* **落盘**：npz 复用既有 `build_dataset.save_npz_deterministic`（自写 zip、固定时间戳）；文本侧 JSONL 由 `write_jsonl_deterministic`（LF、UTF-8、不转义非 ASCII、先写 `.tmp` 再 `os.replace`）；`manifest.json` 记录逐文件 `bytes`/`sha256` 与两侧 `spec_hash`。
* **确定性硬约束（实测踩坑后写死）**：产物 meta **不得含**任何挂钟或缓存状态字段——`build_seconds`（原写在 meta，导致两次构建 npz 字节不同）与 TriviaQA 的 `archive_passes`（冷缓存 1 / 暖缓存 0）均已移出产物，只由 CLI 报告；由验证项 F10 遍历 meta 树守卫。
* **口径自洽断言（构建期不变量，非训练后量）**：统一答案表的键集合必须**等于**各任务选中键的并集（否则报 `BuildQaError`）；`questions_paired == selection.kept_samples`；`kept+removed+unkeyed == total`；`X`/`y` 形状与 meta 计数一致；每行 bag 块非全零。
## 产物验证与判别力探针

新增两层验证/探针，与既有 E0–E8 **并存**（`verify_dataset.py` 一行未改，仅被只读导入 `_cv_scores` / `_fit_logistic`）。

**中文判别力探针 `n3d_qa/probe_zh.py`（硬门禁，先跑）**

* 评估集合：`lexical`（文本行词面匹配，真实 `data/doc` 2,665 行按产物同款 split）、`doclines`（实际产物行本身）、`math1judge`（Math1 判断题 QA 匹配行，**只报读数、不设门禁**）。
* 指标：E7 同口径 5 折 CV（折划分与逻辑回归**直接从 `verify_dataset` 导入**），报 bag-only / extra-only / full 的 acc 与 AUC + 主类基线。
* 门禁：`empty_bag_fraction ≤ 0.01` 且 bag-only AUC ≥ 0.75；失败退码 3（`--no-gate` 只报不失败）。
* 实测（全零行 0/全行）：`lexical` 与 `doclines` bag-only AUC **0.8309**、acc 0.7598（full 1.0000）；`math1judge` bag-only AUC 0.1799、acc 0.5744（主类 0.6667）——**近随机，如实登记**，根因：Math1 答案为计算得出的短串、与题面无词面重叠，且候选文本中 `explanation`（实测 113 字符）占主导稀释信号、正确答案不出现在题面。

**产物验证 `n3d_qa/verify_qa.py`（F0–F10）**

| 项 | 内容 |
| --- | --- |
| F0 | 特征口径契约：meta 口径与常量逐字比对、`spec_hash` 重算一致、列定义无缝铺满 `[0,D-1]` |
| F1 | 向量化确定性：同文本两次调用逐位一致（含全空文本、全角/大小写用例） |
| F2 | 产物契约：dtype/shape/无 NaN-Inf/每行 bag 非全零/bag L2 范数 = 1/计数自洽 |
| F3 | 逐字节幂等：同参连跑两遍并逐文件 SHA256 与正式产物比对 |
| F4 | 文本侧 JSONL 可回读：行数/标签与 npz 逐行一致、题面答案非空、`row_id` 唯一 |
| F5 | 行级重算：由 pairs.jsonl 文本重新向量化逐位等于 npz 行 |
| F6 | 统一答案表一致：键唯一、`count == sum(sources)`、与 meta 计数一致 |
| F7 | 剔除登记可回读：`kept+removed+unkeyed == total`、键算术闭合、`coverage_loss == 1 - coverage_rate` |
| F8 | 文本行产物：行表/行索引覆盖全行、库∩查询 = 0、每行特征宽度一致 |
| F9 | 零回归：`checkpoints/triviaqa/` 快照逐位不变 + 既有 E 检查子进程退码 0 |
| F10 | meta 可复现：meta 树内不得出现挂钟/缓存状态字段（`build_seconds` / `archive_passes` 等） |

辅助工具：`n3d_qa/tools/snapshot_triviaqa_sha256.py`（改动前后 SHA256 快照与零回归比对）、`n3d_qa/tools/recon_math1_doc.py`（Math1 字段键集合/答案分布/归一化前后类数、`data/doc` 非空行与字符构成现场复核）。
