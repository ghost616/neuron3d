# n3d_qa：通用 QA 数据集处理模块（QA 数据集 → N3D 数组格式）

> 本模块为**通用 QA 数据集处理模块**，当前内置 **TriviaQA**（证据段落二分类）作为**参考实现**；
> 后续 QA 数据集按 **adapter** 挂入（输入解析 / 文档定位 / 答案合并 / split 表），共用特征层与
> npz 落盘层。本文以下所有口径（归档、split、列定义、产物名）**都是 TriviaQA 参考实现的口径**。

## 1. 模块职责

本模块负责把**问答数据集**转成 N3D 可直接训练的 numpy 数组；当前内置实现取 TriviaQA 的
**evidence 文档 + 问答对**：产物是 npz，键为 **`X[M, D] float32`** 与 **`y[M] int64`**
（另附 `meta`，见 §5），供 `n3d_shape` 以 **`--dataset npz --dataset-path <产物>`** 直接训练
（完整命令见 §7）。
本模块**自包含**：不导入、不修改 `n3d_proto` / `n3d_sphere` / `n3d_shape` / `n3d_viz` / `framework`
的任何代码；不联网；**不整包解压**（只做 `tarfile "r|gz"` 顺序流式 + 按需 `extractfile`）。
[!] 产物路径 `checkpoints/triviaqa/`、产物文件名前缀 `n3d_triviaqa_verified_*` 与产物 meta 里的
`module` 字段**冻结为 TriviaQA 参考实现口径**，不随模块目录名变化（改名不得改动既有产物字节）。

## 2. 数据源与 split 口径

### 2.1 归档（现场实测）

| 项 | 值 |
| --- | --- |
| 路径 | `data/triviaqa/OpenDataLab___TriviaQA/raw/triviaqa-rc.tar.gz` |
| SHA256 | `ef94fac6db0541e5bb5b27020d067a8b13b1c1ffc52717e836832e02aaed87b9`（构建前校验，不符即退码 2） |
| 大小 / 成员数 / 解压总字节 | 2,665,779,500 B / 487,254 / 7,341,073,957 B |

### 2.2 QA JSON 成员表（流式枚举实测；序号为归档内 1-based 成员序号）

| 成员序号 | 成员名 | 字节数 | 对应 split | 用途 |
| --- | --- | --- | --- | --- |
| 487246 | `qa/wikipedia-dev.json` | 17,318,460 | `wiki-dev` | 全量 dev（剔除 verified 后构建） |
| 487248 | `qa/verified-wikipedia-dev.json` | 797,321 | `wiki` | verified 子集（缺省正式产出） |
| 487251 | `qa/verified-web-dev.json` | 1,096,903 | `web` | verified 子集（缺省正式产出） |
| 487253 | `qa/web-dev.json` | 47,655,088 | `web-dev` | 全量 dev（剔除 verified 后构建） |

归档内另含 `qa/web-test-without-answers.json`(#487247)、`qa/wikipedia-test-without-answers.json`(#487249)、
`qa/wikipedia-train.json`(#487250)、`qa/web-train.json`(#487252)，本模块**不使用**。

### 2.3 split 口径（题数与候选文档均现场实测）

| split | 来源 | 题数 | 剔除规则 | 候选文档 |
| --- | --- | --- | --- | --- |
| `wiki` | `verified-wikipedia-dev.json` | 318 | — | verified 的 `EntityPages ∪ SearchResults` |
| `web` | `verified-web-dev.json` | 407 | — | 同上 |
| `wiki-dev` | `wikipedia-dev.json` | 7,993 → **7,675** | 按 `QuestionId` 剔除 `wiki` 的 318 题 | **9,658** |
| `web-dev` | `web-dev.json` | 9,951 → **9,544** | 按 `QuestionId` 剔除 `web` 的 407 题 | **59,150** |

`--split all` **只**展开 `wiki` + `web`（保持"缺省产出 = 已有正式产物"语义）；两个 dev split 需显式指定。

### 2.4 文档映射规则与访问口径

| QA 字段 | 归档成员名 | 实测形态 |
| --- | --- | --- |
| `EntityPages[].Filename` | `evidence/wikipedia/<basename>` | 维基侧为**扁平目录**，故取 basename |
| `SearchResults[].Filename` | `evidence/web/<相对路径>` | web 侧为**数字子目录**，如 `158/158_2486.txt` |

归档内 `evidence/wikipedia` 74,021 个成员、`evidence/web` 412,972 个成员（流式探针实测）。
访问一律用 `tarfile.open(path, "r|gz")` **顺序流式**遍历 + `extractfile` 按需读取，**从不 `extractall`**。

## 3. 任务定义

样本 = **(Question, Evidence 文档) 对**，二分类：

* **正样本 `label=1`**：文档来自本题的 `EntityPages ∪ SearchResults`（provenance 判定）；
* **负样本 `label=0`**：固定种子 `20261003` 从**同一 split 内其他题**的文档池按 **1:1** 采样
  （与正样本数相等；同题的正负文档集合不相交）；
* **行序**：正负**严格交替**并各占 50%（与 `n3d_shape` 的"每 5 取 1"交错切分对齐，见 §7.5）；
* **答案匹配**：大小写不敏感 + **词边界**；答案字符串合并 `Answer.Value` / `Answer.NormalizedValue` /
  `Answer.Aliases` / `Answer.NormalizedAliases`；纯数字答案在 `len <= 3` 时额外禁止与相邻数字/
  小数点连写（避免 `3` 命中 `3.14`）。

## 4. 特征口径

`--features {base,rich}`（缺省 `base`）。列定义**由产物 meta 现场回读**：

| 列名 | 含义 | base | rich |
| --- | --- | --- | --- |
| `bow_hash64` | 哈希词袋块（`--hash-dim` 桶、L2 归一化、blake2b 确定性哈希，不消耗 RNG） | ✓ | ✓ |
| `q_to_d_coverage` | 问题 token 被文档覆盖的比例 | ✓ | ✓ |
| `d_to_q_coverage` | 文档 token 被问题覆盖的比例 | ✓ | ✓ |
| `jaccard` | 问题/文档 token 集合的 Jaccard | ✓ | ✓ |
| `log1p_doc_tokens` | 文档 token 数（含重数）的 log1p | ✓ | ✓ |
| `log1p_question_tokens` | 问题 token 数（含重数）的 log1p | ✓ | ✓ |
| `numeric_answer_flag` | 该题答案是否含纯数字项 | ✓ | ✓ |
| `qcov_idf` / `dcov_idf` | IDF 加权的问题/文档覆盖率 | — | ✓ |
| `tfidf_cos` | 问题/文档 TF-IDF 余弦（两侧同用 TF-IDF 范数，取值 ≤ 1） | — | ✓ |
| `ans_isnum_frac` | 答案中纯数字项的占比 | — | ✓ |

维度对照（`D = feature_dim(hash_dim, no_bag, features)`）：`_h64` → **70**（64+6）、`_nobag` → **6**、
`_h64_rich` → **74**（64+10）、`_nobag_rich` → **10**。

## 5. 产物（8 个 npz）

| 产物 | 路径 | `X` 形状 | D |
| --- | --- | --- | --- |
| wiki / h64 | `checkpoints/triviaqa/n3d_triviaqa_verified_wiki_dev_h64.npz` | (1280, 70) | 70 |
| wiki / nobag | `checkpoints/triviaqa/n3d_triviaqa_verified_wiki_dev_nobag.npz` | (1280, 6) | 6 |
| web / h64 | `checkpoints/triviaqa/n3d_triviaqa_verified_web_dev_h64.npz` | (820, 70) | 70 |
| web / nobag | `checkpoints/triviaqa/n3d_triviaqa_verified_web_dev_nobag.npz` | (820, 6) | 6 |
| wiki / h64_rich | `checkpoints/triviaqa/_verify/n3d_triviaqa_verified_wiki_dev_h64_rich.npz` | (1280, 74) | 74 |
| wiki / nobag_rich | `checkpoints/triviaqa/_verify/n3d_triviaqa_verified_wiki_dev_nobag_rich.npz` | (1280, 10) | 10 |
| web / h64_rich | `checkpoints/triviaqa/_verify/n3d_triviaqa_verified_web_dev_h64_rich.npz` | (820, 74) | 74 |
| web / nobag_rich | `checkpoints/triviaqa/_verify/n3d_triviaqa_verified_web_dev_nobag_rich.npz` | (820, 10) | 10 |

npz **键契约**：`X`（`float32 [M, D]`）、`y`（`int64 [M]`，取值 0/1）、`meta`（0 维 Unicode 数组，
内容为紧凑 JSON：split / 归档 SHA256 / 特征维度与列定义 / 计数与 provenance 等）。
`n3d_shape.data.load_npz_arrays` 只挑 `X` 与 `y`，**多余键不影响契约**。
[!] `meta.module` 现场回读为 **`"n3d_triviaqa"`**（TriviaQA 参考实现的产物归属标注，随产物字节冻结）；
模块目录改名 `n3d_triviaqa/ → n3d_qa/` **不改该字段**，否则 8 个产物的 SHA256 会全部变化。

## 6. 用法

```bash
python n3d_qa/build_dataset.py --split all                      # 缺省：wiki+web 的 base 产物（D=70）
python n3d_qa/build_dataset.py --split all --no-bag             # 无词袋版（D=6）
python n3d_qa/build_dataset.py --split all --features rich      # rich 特征（D=74，落 _verify/）
python n3d_qa/build_dataset.py --split all --features rich --no-bag   # rich + 无词袋（D=10）
python n3d_qa/build_dataset.py --split wiki-dev                 # 扩容 dev（compact，单 split 小时级）
python n3d_qa/build_dataset.py --split web-dev
```

产物验证（E0–E8，逐项真实执行；E0 恒定执行）：

```bash
python n3d_qa/verify_dataset.py --checks E2,E4,E8 --product-set all   # 实测：通过 15 / 失败 0 / 跳过 2，退码 0
python n3d_qa/verify_dataset.py --checks E7 --product-set base        # 实测：PASS，rich 十列 0.9430（wiki）/ 0.9646（web）
```

演练构建（`--max-questions` > 0）落 `checkpoints/triviaqa/_verify/`，**不覆盖**上述 8 个正式产物。

## 7. 与 N3D 的对接

### 7.1 产物即接口

npz 里的 `X[M, D] float32` / `y[M] int64` 就是与 `n3d_shape` 之间的**唯一接口**，无需任何适配层。

### 7.2 完整消费命令（**已实跑，退码 0**；模块路径改名后按下方命令复跑仍为退码 0）

```bash
python n3d_shape/train.py --dataset npz \
  --dataset-path checkpoints/triviaqa/n3d_triviaqa_verified_wiki_dev_h64.npz \
  --input-dim 70 --output-dim 2 --preset default --seed 42 --n 256 --shape sphere \
  --input-scope any_isolated --readout-scope any_isolated \
  --threads 0 --fc-dim -1 --geo-field none --epochs 20
```
（注：该命令**只用新模块的产物路径**，不含任何 `n3d_qa/` 模块路径；改名复跑实测：退出码 0、
`n3d_shape` 侧自报加载 `train=1024, test=256, D=70`，最终 `test_acc` 87.11%。）

实测（本机）：**退出码 0，脚本自报总耗时 9.9 s，最终 `test_acc` 87.11%**（该耗时是脚本内计时，
重跑会随机器负载略有差异，例如另一次实跑自报 8.7 s）；checkpoint 落
`checkpoints/n3d_shape/full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_dsnpz_d70x2_nosyn_s42.pt`。

### 7.3 维度对照（必须显式给 `--input-dim`）

`_h64` → **70**；`_nobag` → **6**；`_h64_rich` → **74**；`_nobag_rich` → **10**。
`--output-dim 2`：两套 split 都是**二分类**（正/负各半）。必须显式给维度，是因为 npz 在
`n3d_shape` 数据集注册表里是**占位 0**（`input_dim=0, num_classes=0` 表示"由文件现场解析"）。

### 7.4 产物命名与隔离

`n3d_shape` 侧按配置指纹命名，其中含 **`_dsnpz_d{D}x{C}`** 段（如上例 `_dsnpz_d70x2`），故本数据集的
checkpoint 与 MNIST 等既有产物**互不覆盖**；两者也分属不同目录（本模块 `checkpoints/triviaqa/`，
`n3d_shape` 侧 `checkpoints/n3d_shape/`）。

### 7.5 数据切分

`n3d_shape` 对 npz 用 **"每 5 个取 1" 交错切分**：wiki 1280 条 → 训练 **1024** / 测试 **256**
（web 820 条 → 训练 656 / 测试 164）。本模块行序已按正负严格交替排布，故该切分下两份类别分布一致。

### 7.6 空间参数不受影响

接入本数据集**只改 `--input-dim`**：`N=256`、`D=0.1`（H）、`E=736`、`|S_in|=193`、`|S_out|=187`、
层数 K=9 等空间/拓扑参数由 `n3d_shape` 侧决定，与本模块无关。

## 8. 历史留档与维护边界

模块目录由 `n3d_triviaqa/` 改名为 `n3d_qa/` 后，仓库内仍有若干**历史留档**按旧包名/旧路径书写，
其维护状态如下（**不在本模块的维护范围内**，改名时未同步）：

| 留档 | 现状 | 维护状态 |
| --- | --- | --- |
| `.lizhu_env/triviaqa/`（约 30 个历史测试脚本） | 仍以 `import n3d_triviaqa.build_dataset` / `sys.path` 指向 `n3d_triviaqa/`、并以 `python n3d_triviaqa/build_dataset.py` 起进程 | **按旧包名冻结，不再维护**（本轮不同步）。这些脚本已不可直接运行；如需复跑，请改指向 `n3d_qa/`，或直接使用下表的现行入口 |
| `lizhu_r30–r37_scripts/`（历史回归脚本） | 同上（旧包名 + 旧路径） | 同上，按历史留档冻结 |
| `test_reports/lizhu_n3d_triviaqa_r*.md` | 历史测试报告（记录当时的命令与读数） | 只读留档，**不得改动**（其中路径/读数反映当时状态） |
| `.module_agent/n3d_triviaqa/` | 改名前的模块元数据目录 | 历史元数据，只读 |
| `checkpoints/triviaqa/_verify/_diag/`、`checkpoints/triviaqa/_probe/` | **已随改名同步修复**（`sys.path` 指向 `n3d_qa/`；3 个 `_probe` 脚本改为按脚本位置相对定位 `n3d_qa/build_dataset.py`，不再硬编码绝对路径） | 可运行，按下表使用 |

现行可复现入口（均实测退码见括号）：

```bash
python checkpoints/triviaqa/_verify/_diag/recompute_learning_baselines.py        # 学习成功标准基线复算（退码 0）
python checkpoints/triviaqa/_verify/_diag/test_ac_equivalence.py [--samples N --fuzz N --timing]
                                                                                # AC 预筛等价性对照（0=无漏判 / 1=有漏判；实测有漏判故退码 1，但报告完整）
python checkpoints/triviaqa/_verify/_diag/analyze_feature_power.py              # 特征判别力分解（退码 0）
python checkpoints/triviaqa/_verify/_diag/analyze_feature_power_rich.py         # rich 对照分解（退码 0，自动在 _verify/ 找 rich 产物）
python checkpoints/triviaqa/_verify/_diag/test_d1_custom_archive.py             # D1 自定义归档不覆盖正式产物（退码 0）
python checkpoints/triviaqa/_verify/_diag/test_d3_record_order.py               # D3 乱序 records 行序一致（退码 0）
python checkpoints/triviaqa/_probe/probe_archive_members.py                     # 归档成员表探针（退码 0）
python checkpoints/triviaqa/_probe/_inspect.py                                  # 源码切片查看（退码 0）
```

注：`test_ac_equivalence.py` 会把折叠陷阱用例（`ſ`/`İ`/`ς` 等）原样打印出来，在 Windows GBK 控制台下
会 `UnicodeEncodeError`；脚本已在入口放宽 stdout/stderr 错误策略（`errors="replace"`，与
`verify_dataset.configure_console_encoding` 同口径），故默认控制台下也能跑完并给出结论行。


---

# 附录 A：中文文本特征口径与新增产物（增量扩展，2026-10-05）

> 本节记录**与 TriviaQA 参考实现并存**的中文口径。**第 1–8 节全部不变**：既有
> `n3d_qa/build_dataset.py`、`n3d_qa/verify_dataset.py` 与 `checkpoints/triviaqa/` 下 8 个正式 npz
> **零改动**（实测见 §A.9）。新增代码全部在新文件里：`zh_features.py` / `adapters.py` /
> `build_qa.py` / `probe_zh.py` / `verify_qa.py` / `tools/`。

## A.1 为什么需要新口径（阻塞项，现场复核）

既有英文口径 `build_dataset.TOKEN_RE = r"[a-z0-9]+"` **丢弃全部非 ASCII 字符**，故对中文文本：

| 文本 | 既有口径 token 数 | 词袋块取值 |
| --- | --- | --- |
| `一个容量为80的样本最大值为143` | 2（`80` / `143`） | 几乎全零 |
| `最高人民法院关于适用《中华人民共和国民法典》…` | 0 | 恒全零 |

即中文文本在既有口径下**特征向量全零**，第 1 步（中文特征）与第 2 步（中文判别力）都无法工作。
新增 `n3d_qa/zh_features.py` 提供**字符级 1/2/3-gram 哈希词袋**口径修掉该阻塞项，且**不改动**
既有英文口径的任何字节（不改 `TOKEN_RE`、不改 `feature_columns`、不改 meta 字段）。

## A.2 中文特征口径（口径参数全部写入 meta 并可回读）

规范化（`zh_features.NORMALIZATION_RULE`，实测字符串）：

```
NFKC -> drop all Unicode whitespace -> casefold()   （纯 ASCII 字节 < 0x80，UTF-8 落盘跨平台字节稳定）
```

分词（`zh_features.TOKENIZER_RULE`）：

```
character level: 取规范化后的字符序列（空白已被去除），对每个 n in n_gram_orders 产出全部连续 n-gram
```

落桶（复用既有 blake2b 口径，仅换盐命名空间）：

```
hash_unit  = f"{order}:{ngram}"                          # 例： "2:最高"、"3:人民法院"
hash       = int.from_bytes(blake2b(ZH_HASH_SALT + unit, digest_size=8).digest(), "big")
bucket     = hash % buckets_per_order
block 偏移 = order_index * buckets_per_order
块归一化   = L2（范数为 0 时保持全零，不产生 NaN）——与既有 hash 块同口径
```

`ZH_HASH_SALT = b"n3d-qa-zh-charbow-v1\x00"`（hex 实测
`6e33642d71612d7a682d63686172626f772d763100`）。**不消耗任何全局 RNG**；口径折叠为单一
`spec_hash`（本节所用口径实测 `275feb01683b83526898d7eb0a1af99974420c3eb93640e118db675775145ddd`，
文本行侧 64 桶口径为 `8f2523e41484adf39f6f1f455ef13f473a8dd027499af31219b675eade259fae`）。

列定义（9 条，`feature_columns()` 现场生成、写入 meta，无缝铺满 `[0, D-1]`）：

| 列名 | 含义 |
| --- | --- |
| `zh_char1gram_hash{N}` | 字符 1-gram 哈希词袋块（N 桶、L2 归一化） |
| `zh_char2gram_hash{N}` | 字符 2-gram 哈希词袋块 |
| `zh_char3gram_hash{N}` | 字符 3-gram 哈希词袋块 |
| `q_to_c_coverage` | 查询侧字符 n-gram 被候选侧覆盖的比例 |
| `c_to_q_coverage` | 候选侧字符 n-gram 被查询侧覆盖的比例 |
| `jaccard` | 查询/候选字符 n-gram 集合的 Jaccard |
| `log1p_candidate_chars` | 候选侧规范化字符数的 log1p |
| `log1p_query_chars` | 查询侧规范化字符数的 log1p |
| `length_ratio` | 候选/查询字符数之比 |

**维度与样本量匹配（历史纠正 #12）**：词袋桶数按「每阶 10^1~10^2」取，且**分侧独立取**：

| 侧 | 样本量 | `buckets_per_order` | `bag_dim` | `D` |
| --- | --- | --- | --- | --- |
| QA 匹配（`judge/choice/blank/solve/triviaqa/all`） | 443 ~ 50,865 行 | **100** | **300** | **306** |
| 文本行（`doclines`） | 2,665 行 + 1,332 行 | **64** | **192** | **198** |

## A.3 新增数据源与 adapter（现场复核）

| 数据源 | 现场实测 | adapter |
| --- | --- | --- |
| Math1 八文件（`离散数学`/`高等数学` × `判断题/填空题/解答题/选择题`） | 88 / 405 / 579 / 895 + 696 / 5,124 / 22,689 / 7,163 = **37,639 条**；字段键集合实测为 `['answer','choices','explanation','id','qtype','question','sampling_results','subject']`（八文件一致） | `adapters.load_math1_task` |
| `data/doc/*.md` | `a.md` 51 / `b.md` 2403 / `c.md` 167 / `d.md` 44 = **2,665 非空行**（汉字占比 0.8695 / 0.8698 / 0.8944 / 0.9068） | `adapters.load_doc_lines` |

题面/答案文本口径：**问题文本 = `question` + 全部 `choices`（换行连接）**；
**答案文本 = 候选答案表层串 + 该题 `explanation`（换行连接）**。

答案字符串归一化规则（`adapters.ANSWER_NORM_RULE`，显式断言于 F0）：

```
NFKC -> 去掉全部 Unicode 空白 -> casefold()
```

现场实测「归一化前/后类数变化」（`n3d_qa/tools/recon_math1_doc.py`，全量）：

| 文件 | 原始答案类数 | 归一化后类数 | 变化 |
| --- | --- | --- | --- |
| 离散数学_判断题 | 2 | 2 | 0 |
| 离散数学_填空题 | 339 | 332 | −7 |
| 离散数学_解答题 | 564 | 558 | −6 |
| 离散数学_选择题 | 7 | 7 | 0 |
| 高等数学_判断题 | 2 | 2 | 0 |
| 高等数学_填空题 | 3,280 | 3,141 | −139 |
| 高等数学_解答题 | 20,953 | 20,810 | −143 |
| 高等数学_选择题 | 9 | 9 | 0 |

## A.4 统一答案表与各任务口径（所有数字由现场命令产出）

各任务的选择规则（原文写入 `meta.selection.answer_table_rule` 与 `answer_table.jsonl`）：

| 任务 | 规则 | 保留类数 | 剔除类数 | 题目总数 | 可答题目 | 覆盖率 |
| --- | --- | --- | --- | --- | --- | --- |
| `judge` | 全部答案键（两个布尔类） | 2 | 0 | 784 | 784 | **1.0000** |
| `choice` | 单字母 a–d（多字母如 `bc`、超出 d 的 `e`/`f` 登记为剔除） | 4 | 7 | 8,058 | 8,043 | **0.9981** |
| `blank` | 归一化频次 ≥ 5 | 153 | 3,239 | 5,529 | 2,469 | **0.4466** |
| `solve` | 归一化频次 ≥ 5 | 78 | 21,277 | 23,268 | 1,404 | **0.0603** |
| `triviaqa`（wiki+web, top-100） | 按含该答案键的题目数降序取前 100 | 100 | 10,939 | 725 | 27 | **0.0372** |

跨任务合并去重后的**统一答案表**（落 `answer_table.jsonl`，每类含来源任务与计数）：
**298 类**、别名 298、进入合并的题目 38,364、答案在表内 12,727、**不相关（正确答案不在表内）25,637**、
多值答案题目 306。

**剔除披露（覆盖率损失，逐条可回读）**：`blank` 剔除 3,060 题（−55.34%）、`solve` 剔除 21,864 题
（−93.97%，另有 8 题答案值归一化后为空键，单独记为 `unkeyed`）、`choice` 剔除 15 题（−0.19%）、
`triviaqa` 剔除 698 题（−96.28%）。所有剔除键与其题数登记在 `meta.removals`（前 50 条明细 +
计数），并由 F7 闭合算术校验。

### A.4.1 关于 TriviaQA 覆盖率仅 3.7% 的实测说明（**两个不同原因，必须区分**）

* **先说被排除的假因**：首版实现里 TriviaQA 侧的答案键用了「仅去空白」的变体键（`likeaprayer`），
  与统一答案表的 canonical 键（`"""likeaprayer"""`）**不同空间**，导致「表里明明有答案却判为不相关」。
  该键空间不一致已修复（现统一为 `normalize_answer_key`），修复前后 `triviaqa` 命中题目数
  实测 27 题（修复后）——即**修复并没有改变覆盖率**，因为真正的限制在数据本身。
* **真因（数据特性）**：TriviaQA 的 verified 子集在答案键层面极度稀疏——725 题共 **11,039 个不同
  答案键**，最高频键也只出现在 3 题中，故「答案频次 top-100」天然只能覆盖 27 题。作为对照，
  非 verified 全量 dev（17,944 题）有 **96,922 个不同键**、top-100 只覆盖 86 题（0.48%）。
  两个 split 家族都远达不到 1e3 量级样本，所以 top-100 口径下的 TriviaQA 产物**规模很小但不为空**。
* 需要更大 TriviaQA 子集时可显式调 `--triviaqa-topn`（及 `--triviaqa-splits`）；本节所有数字固定
  登记为 `topn=100 / splits=wiki,web` 口径。

## A.5 新增产物清单

产物目录：`checkpoints/qa_learn/dataset/`（验证类产物落 `_verify/`，快照落 `_snapshot/`）。

| 产物 | 形状 / 规模 | 说明 |
| --- | --- | --- |
| `n3d_qa_judge.npz` | `X (2352, 306)` / `y (2352,)` | 判断题 QA 匹配 |
| `n3d_qa_choice.npz` | `X (32172, 306)` | 选择题 QA 匹配 |
| `n3d_qa_blank.npz` | `X (10274, 306)` | 填空题 QA 匹配 |
| `n3d_qa_solve.npz` | `X (5624, 306)` | 解答题 QA 匹配 |
| `n3d_qa_triviaqa.npz` | `X (443, 306)` | TriviaQA top-100 QA 匹配 |
| `n3d_qa_all.npz` | `X (50865, 306)` | 上述五任务行向拼接 |
| `n3d_qa_<task>_pairs.jsonl` | 与 npz 同行数 | 问答对文本侧（`row_id` / `question_text` / `answer_text` / `label` / 候选索引与键） |
| `answer_table.jsonl` | 298 行 | 统一答案表（`index` / `key` / `display` / `count` / `sources` / `raw_variants`） |
| `doclines.npz` | `X (1332, 198)` / `y (1332,)` | 文本行库/查询留出匹配 |
| `doclines_rows.jsonl` | 2,665 行 | 行表：`row_id` / `file` / `line_no` / `text` / `text_norm` / `char_len` / `feature[198]` |
| `doclines_row_index.jsonl` | 2,665 行 | 行索引：`positives` / `negatives` / `rows_for_this_query` |
| `manifest.json` | — | 逐文件 `bytes` + `sha256` + 两侧 spec_hash + 答案表摘要 |

标签规则：QA 侧 `label=1` ⇔ 候选答案类属于该题正确答案类（别名解析后比对，F5 现场校验）；
`label=0` = 从**同一任务家族**按类频次加权抽出的错误类（提问自身的正确类被禁用，故负样本数精确等于请求数）。
文本行侧 `label=1` ⇔ 候选行就是查询行本身（按 **row_id** 比对，不是按文本比对：相同文本出现在另一行
仍是合法负样本）。库/查询划分：seed `20261005`、查询比例 `0.25` → 库 1,999 / 查询 666、
**交集 0**、并集 2,665。

## A.6 中文判别力探针（硬门禁，先跑）

```bash
python n3d_qa/probe_zh.py
```

实测读数（E7 同口径 5 折 CV，`_cv_scores` / `_fit_logistic` **直接从 `verify_dataset` 导入**）：

| 集合 | 行数 | 主类基线 | bag-only acc | bag-only AUC | full acc | full AUC | 是否门禁 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lexical`（文本行词面匹配，2,665 行真实中文） | 1,332 | 0.5000 | **0.7598** | **0.8309** | 1.0000 | 1.0000 | ✅ 门禁 |
| `doclines`（实际产物行本身） | 1,332 | 0.5000 | **0.7598** | **0.8309** | 1.0000 | 1.0000 | ✅ 门禁 |
| `math1judge`（Math1 判断题 QA 匹配） | 2,352 | 0.6667 | 0.5744 | 0.1799 | 0.5770 | 0.1780 | ⚠️ 只报读数 |

* 全零行：三个集合都是 **0 / 全部行**（`empty_bag_rows=0`，中文特征**非全零**）；bag 块每行 L2 范数实测 = 1。
* 门禁判据：`empty_bag_fraction ≤ 0.01` 且 bag-only AUC ≥ **0.75** → 实测 0.8309 **通过**（退码 0）。
* **如实登记的负读数**：Math1 判断题的 QA 匹配集合为 **近随机**（bag-only AUC 0.18，逐列单特征
  AUC ≤ 0.51）。根因经对照实验定位：①Math1 的答案是**计算得出**的短串（正确/错误、数字），与题面
  无词面重叠；②候选文本 `答案 + explanation` 中 explanation（实测 113 字符）占绝对主导，正确答案的
  2 字符差异被 L2 归一化稀释；③正确答案本身不出现在题面里（用例：题面 `…可以分成{blank}组．`，
  答案 `10`，`10` 不在题面）。同口径在文本行词面匹配上 AUC 0.8309，说明**不是特征实现缺陷**
  （合成可分数据的 CV AUC 实测 0.97+ 亦佐证 CV 实现无缺陷）。
* 桶数扫描（`--sweep`，report-only）实测：32 桶 0.3063 / 64 桶 0.2302 / 100 桶 0.1799 /
  128 桶 0.1531 / 256 桶 0.1158（Math1 判断题集合），单调下降，与「该集合无词面信号」一致。

## A.7 构建与验证命令

```bash
# 全量构建（5 个 QA 任务 + all + 文本行；实测退码 0，用时 392.2 s）
python n3d_qa/build_qa.py --tasks judge,choice,blank,solve,triviaqa --doclines \
    --out-dir checkpoints/qa_learn/dataset

# 单条端到端演练（先演练再放全量；不写正式产物）
python n3d_qa/build_qa.py --drill --tasks judge --max-records 6 --dry-run

# 中文判别力探针（硬门禁；另加 --sweep 看桶数曲线）
python n3d_qa/probe_zh.py

# 新增产物验证（F0–F9；实测全绿退码 0）
python n3d_qa/verify_qa.py --checks F0,F1,F2,F4,F5,F6,F7,F8,F9 --run-verify-dataset
python n3d_qa/verify_qa.py --checks F3          # 逐字节幂等（连跑两遍 + 与正式产物比对）

# 既有 TriviaQA 验证（必须仍通过）
python n3d_qa/verify_dataset.py --checks E2,E4,E8 --product-set all

# 改动前后快照 / 零回归比对
python n3d_qa/tools/snapshot_triviaqa_sha256.py
python n3d_qa/tools/snapshot_triviaqa_sha256.py --check

# Math1 与 data/doc 现场复核（字段键集合、答案分布、归一化前后类数、非空行数）
python n3d_qa/tools/recon_math1_doc.py
```

## A.8 验证项清单（`verify_qa.py`，F0–F9）

| 项 | 内容 | 实测 |
| --- | --- | --- |
| F0 | 特征口径契约：meta 口径与 `zh_features` 常量逐字比对、`spec_hash` 重算一致、列定义无缝铺满 | PASS（6 个产物） |
| F1 | 向量化确定性：同一文本两次调用逐位一致（含全空文本、全角/大小写用例） | PASS |
| F2 | 产物契约：`float32[M,D]` / `int64[M]` / 无 NaN-Inf / **每行 bag 非全零** / bag L2 范数 = 1 | PASS（91,930 行） |
| F3 | 逐字节幂等：同参连跑两遍 + 与正式产物逐文件 SHA256 比对 | PASS |
| F4 | 文本侧 JSONL 可回读：与 npz 行数/标签逐行一致、题面答案文本非空、`row_id` 唯一 | PASS |
| F5 | 行级重算：从 pairs.jsonl 文本重新向量化 **逐位等于** npz 行 | PASS（91,930 行，`max_dev=0`） |
| F6 | 统一答案表一致：键唯一、`count == sum(sources)`、与 meta 计数一致 | PASS（298 类） |
| F7 | 剔除登记可回读：保留+剔除+空键 == 总数、键算术闭合、`coverage_loss == 1 - coverage_rate` | PASS |
| F8 | 文本行产物：行表/行索引覆盖全行、**库与查询交集 = 0**、每行特征宽度一致 | PASS |
| F9 | 零回归：`checkpoints/triviaqa/` 189 个 npz 快照逐位不变 + 既有 E 检查退码 0 | PASS |
| F10 | meta 可复现：meta 树内不得出现挂钟/缓存状态字段（`build_seconds` / `archive_passes` 等） | PASS（6 个产物 0 命中） |

F3 实测输出（两遍完整构建 + 正式产物**三方逐字节一致**；下为 SHA256 前 12 位）：

```
answer_table.jsonl           0294e66cc2c4   doclines.npz                 909aa0262e85
doclines_row_index.jsonl     6e369fc400b6   doclines_rows.jsonl          e1b98a67e621
manifest.json                4a260cb0454b   n3d_qa_all.npz               e2520f821309
n3d_qa_all_pairs.jsonl       18428b30db6e   n3d_qa_blank.npz             16a8bcce86c5
n3d_qa_blank_pairs.jsonl     d8e5e1f6ee61   n3d_qa_choice.npz            a852baf1bae8
n3d_qa_choice_pairs.jsonl    3372ce71af43   n3d_qa_judge.npz             c12ede3754dc
n3d_qa_judge_pairs.jsonl     93f6d9eb316a   n3d_qa_solve.npz             f38a9426bc64
n3d_qa_solve_pairs.jsonl     a49c25cee5c4   n3d_qa_triviaqa.npz          5fef16f3d566
n3d_qa_triviaqa_pairs.jsonl  a9c004f8cb65
```

`manifest.json` 也逐字节一致（其内容只有逐文件 `bytes`/`sha256` 与两侧 `spec_hash`，不含时间戳）。

## A.9 零回归证据（硬门槛）

* 改动前快照：`checkpoints/qa_learn/_snapshot/triviaqa_npz_sha256.json`（189 条 `(路径, 字节数, SHA256)`）。
* 8 个正式产物的 SHA256（改动前后**逐位不变**）：

```
c7b142e89b9b5a212698c9c7bc9311aa973c01f509a70545b732d634f262b78e  checkpoints/triviaqa/n3d_triviaqa_verified_wiki_dev_h64.npz
a544eb8bd99421a73d8ecf308ddab81b14545e1b309c51125a584831362f4802  checkpoints/triviaqa/n3d_triviaqa_verified_wiki_dev_nobag.npz
bfdf4ca91c6dc9863af1202bc3f5cf0c9a9f9e8ed26be5ac2c98da60bc94efa0  checkpoints/triviaqa/n3d_triviaqa_verified_web_dev_h64.npz
15de4699084c7b9e063636253d80d750128e280255e227d8a7a21b0151cbc7c2  checkpoints/triviaqa/n3d_triviaqa_verified_web_dev_nobag.npz
1ea65eae226c6e4707f2f1be651dc8f98e3175c45bc58c46ca76580a438875b5  checkpoints/triviaqa/_verify/n3d_triviaqa_verified_wiki_dev_h64_rich.npz
057ae8d85c3a9f33a3f3695a56ca79b6c8ecb625b9f8c36a727acc783085e141  checkpoints/triviaqa/_verify/n3d_triviaqa_verified_wiki_dev_nobag_rich.npz
02555fe15359d1536af6bcb50e22e61d4e7abc5a8e659a6c6cbfa5609d3204a7  checkpoints/triviaqa/_verify/n3d_triviaqa_verified_web_dev_h64_rich.npz
8cadecbd7185f9ffa6ea26d6b8f785fbbf78532c32afbddb11fdb6f14798ea62  checkpoints/triviaqa/_verify/n3d_triviaqa_verified_web_dev_nobag_rich.npz
```

* 既有验证器复核：`python n3d_qa/verify_dataset.py --checks E2,E4,E8 --product-set all` →
  **通过 15 / 失败 0 / 跳过 2，退码 0**（由 F9 子进程自动执行并断言退码）。
* 本模块**未改动** `n3d_qa/build_dataset.py`、`n3d_qa/verify_dataset.py` 的任何一行；
  `verify_dataset.py` 仅被 `probe_zh.py` / `verify_qa.py` **只读导入**（`_cv_scores` / `_fit_logistic`）。

## A.10 不可复现项与边界（如实登记）

| 项 | 说明 |
| --- | --- |
| 耗时类读数 | `build_qa.py` 报告的 `seconds`（冷缓存 392.2 s / 暖缓存 244.9~320.0 s）与 `verify_qa.py` 的 440.4 s（F0–F10）/ 565.9 s（F3 连跑两遍）随机器负载浮动，**不作为验收判据**（故**不写入产物**，由 F10 守卫）；验收判据是 SHA256 与各项断言 |
| `--sweep` 与 `math1judge` 读数 | 属**只读报告**，不进门禁；其数值已固定登记在 §A.6，若口径变更需重跑登记 |
| TriviaQA 侧规模 | top-100 口径下仅 27 题可答（443 行），是该子集答案极度稀疏的数据特性；如需更大规模需显式调 `--triviaqa-topn` / `--triviaqa-splits`，届时须重新登记 |
| `solve` 覆盖率 6.03% | 属「答案频次 ≥ 5」硬规则的**直接后果**，已按纪律量化披露（剔除 21,864 题，另有 8 题答案值归一化后为空键、单独记为 `unkeyed`），未做任何粉饰 |
| 中文语义 | 本口径是**词面级**（字符 n-gram），不声称任何语义等价性；Math1 判断题匹配集的近随机读数即为该边界的具体证据 |
| 历史踩坑留档 | 首版曾把 `build_seconds`（挂钟）与 TriviaQA 的 `archive_passes`（冷缓存 1 / 暖缓存 0）写入产物 meta，导致 `X`/`y` 逐位相同但 npz 字节不同；两者已移出产物，并新增 F10 遍历 meta 树守卫该回归 |