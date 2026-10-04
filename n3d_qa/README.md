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
