# n3d_triviaqa：TriviaQA → N3D 数组格式（证据段落二分类）

## 1. 模块职责

把 TriviaQA 的 **evidence 文档 + 问答对** 转成「证据段落二分类」数据集
（`X[M, D] float32` / `y[M] int64`），产出 npz 供 `n3d_shape` 以 `--dataset npz` 直接训练。

| 文件 | 职责 |
| --- | --- |
| `n3d_triviaqa/__init__.py` | 包声明（对外能力索引，不含逻辑） |
| `n3d_triviaqa/build_dataset.py` | 归档流式访问 + QA 解析 + 答案判定 + 特征构造 + 1:1 负采样 + 确定性 npz 落盘 + CLI |
| `n3d_triviaqa/verify_dataset.py` | 产物验证 E0–E6（口径回读 / 幂等 / 契约 / 均衡与答案复核 / 无泄漏 / 端到端训练 / 零回归） |
| `checkpoints/triviaqa/_verify/_diag/` | 对照与回归脚本（判别力分解、D1/D3 回归、E5 对照），非模块源码 |

**本模块自包含**：不导入、不修改 `n3d_proto` / `n3d_sphere` / `n3d_shape` / `n3d_viz` /
`framework` 的任何源码（`verify_dataset.py` 只**只读**导入 `n3d_shape.data.load_npz_arrays`
判定产物契约，见 E2）。不联网、不整包解压、不写入 `data/mnist`。

## 2. 数据源与归档口径（全部为**现场实测**，写死进代码常量）

| 项 | 值 |
| --- | --- |
| 归档 | `data/triviaqa/OpenDataLab___TriviaQA/raw/triviaqa-rc.tar.gz` |
| SHA256 | `ef94fac6db0541e5bb5b27020d067a8b13b1c1ffc52717e836832e02aaed87b9`（构建前校验，不符即退码 2） |
| 压缩字节数 | 2 665 779 500 |
| 成员数 | 487 254 |
| 解压后总字节数 | 7 341 073 957（≈6.84 GiB） |
| 成员分布 | `evidence/web` 413 173 个 / `evidence/wikipedia` 74 070 个 / `qa/*.json` 8 个 / `README` 1 个 |

| split | QA 成员 | 实测题数 | 文档映射 |
| --- | --- | --- | --- |
| `wiki` | `qa/verified-wikipedia-dev.json` | 318 | `EntityPages[].Filename` → `evidence/wikipedia/<basename>`（扁平目录） |
| `web` | `qa/verified-web-dev.json` | 407 | `SearchResults[].Filename`（形如 `158/158_2486.txt`）→ `evidence/web/<该相对路径>`；`EntityPages` 同样按维基规则映射 |

**访问口径（不整包解压）**：只用 `tarfile.open(path, "r|gz")` 顺序流式扫描 + `extractfile`
按需抽取目标成员，**从不调用 `extractall`**、从不把归档解压到磁盘。
**[!] 遍数的实测事实**：`qa/*.json` 位于归档**末尾**（成员序号 487 248 / 487 251，总数 487 254），
gzip 流不可回退，故"先知道要抽哪些 evidence"不可能在同一遍内完成：

* **冷构建 = 2 遍**（第 1 遍只抽 QA JSON 落 `_cache/<sha16>_<split>_qa.json`，第 2 遍只抽目标 evidence）；
* **缓存命中 = 1 遍**。

实际遍数**只写运行日志**、不写进产物 meta —— 否则产物字节会随缓存状态而变（见 §5 确定性说明与实测）。

## 3. 任务定义与行序口径

* 样本 = `(Question, Evidence 文档)` 对；
* **正样本 `label=1`**：文档来自该题的 `EntityPages ∪ SearchResults`（**provenance 判定**，不要求文档里一定出现答案）；
* **负样本 `label=0`**：由固定 seed `20261003` 从**同 split 内其他题的文档**中无放回采样，1:1 均衡；
  候选池 = 纳入题目的正样本文档并集（升序），排除该题自身文档；
* 有 evidence 引用的题目才产生样本（`web` split 有 48 题两类引用皆空 -> 不产生样本）。

**答案判定**：合并 `Answer.Value` / `Answer.NormalizedValue` / `Answer.Aliases` /
`Answer.NormalizedAliases`（**`Answer.HumanAnswers` 按口径不参与**）；大小写不敏感 + 词边界
（首/末为非词字符时相应省略该侧边界，否则 `C++` 这类答案永不命中）；**短数字答案额外保护**：
`strip` 后匹配 `^[0-9]+$` 且长度 `<= 3` 的答案改用 `(?<![\w.])<ans>(?![\w.])`，
在词边界之外额外禁止与小数点连写（防 `3` 命中 `3.14`）。

**行序口径**（与 `n3d_shape` 数据层的「每 5 个样本取 1 个做测试集」交错切分对齐）：
轮次 `i` 从 0 到 `max_k-1`、每轮按 `QuestionId` 升序遍历题目，逐题依次发出"第 `i` 个正样本 +
第 `i` 个负样本" -> 全表**严格正负交替**（偶数为正、奇数为负），故任何"每 k 个取 1 个"的交错切分
都得到**恰好均衡**的两份（`M = 1280` 时测试集 256 条 = 128 正 + 128 负）。
`build_split` 入口另按 `QuestionId` 排序（使对外 API 的乱序输入不会静默改变行序）。

## 4. 特征口径（**已修订**）

| 口径 | 开关 | D | 产物名 | 状态 |
| --- | --- | --- | --- | --- |
| 缺省（降维） | 无（`--hash-dim` 缺省 **64**） | **70** = 64 + 6 | `..._dev_h64.npz` | **正式产物** |
| 无词袋 | `--no-bag` | **6** | `..._dev_nobag.npz` | **正式产物** |
| 原计划口径（失败对照） | `--hash-dim 1024` | 1030 = 1024 + 6 | `..._dev_h1024.npz`（落 `_verify/`） | 已实测不达标，仅留档 |

列定义（由 `feature_columns(hash_dim, no_bag)` **单一生成**，写入 meta 后回读逐字比对）：

| 口径 | 列 | 名称 | 定义 |
| --- | --- | --- | --- |
| h64 | 0–63 | `bow_hash64` | 把 `Question + "\n" + Evidence` 拼接、整体小写后按 `[a-z0-9]+` 分词；每 token 经 `blake2b(salt+token, digest_size=8)` 取大端整数 `mod 64` 落桶计数；该块整体 **L2 归一化**（零范数行保持全 0）。**不消耗全局 RNG** |
| h64 | 64–69 / nobag: 0–5 | `q_to_d_coverage` | `|T(q) ∩ T(d)| / |T(q)|`（token 集合口径） |
| | | `d_to_q_coverage` | `|T(q) ∩ T(d)| / |T(d)|` |
| | | `jaccard` | `|T(q) ∩ T(d)| / |T(q) ∪ T(d)|` |
| | | `log1p_doc_tokens` | `log1p(文档 token 数)`（**含重数**） |
| | | `log1p_question_tokens` | `log1p(问题 token 数)`（含重数） |
| | | `numeric_answer_flag` | 该题答案集合中存在纯数字答案（`^[0-9]+$`）→ 1.0，否则 0.0（逐题常量） |

哈希盐 `HASH_SALT = b"n3d-triviaqa-bow-v1\x00"`；`--no-bag` 时 meta 的 `features` 段**不写**哈希字段
（避免"写了却没产出该块"的口径歧义）。`--no-bag` 与 `--hash-dim` **互斥**（同时给出退码 2）。

## 5. 产物

| 文件 | 形状 | 内容 | SHA256（实测） |
| --- | --- | --- | --- |
| `checkpoints/triviaqa/n3d_triviaqa_verified_wiki_dev_h64.npz` | `X`(1280, 70) float32 / `y`(1280,) int64 | wiki / D=70 | `c7b142e89b9b5a212698c9c7bc9311aa973c01f509a70545b732d634f262b78e` |
| `checkpoints/triviaqa/n3d_triviaqa_verified_web_dev_h64.npz` | `X`(820, 70) / `y`(820,) | web / D=70 | `bfdf4ca91c6dc9863af1202bc3f5cf0c9a9f9e8ed26be5ac2c98da60bc94efa0` |
| `checkpoints/triviaqa/n3d_triviaqa_verified_wiki_dev_nobag.npz` | `X`(1280, 6) / `y`(1280,) | wiki / D=6 | `a544eb8bd99421a73d8ecf308ddab81b14545e1b309c51125a584831362f4802` |
| `checkpoints/triviaqa/n3d_triviaqa_verified_web_dev_nobag.npz` | `X`(820, 6) / `y`(820,) | web / D=6 | `15de4699084c7b9e063636253d80d750128e280255e227d8a7a21b0151cbc7c2` |
| `checkpoints/triviaqa/_verify/legacy_d1030/*.npz` | `X`(1280/820, 1030) | 原计划口径**失败对照** | 见 §7.3 |

npz 键：`X` / `y` / `meta`（`meta` 为 0 维 Unicode 数组，内容是紧凑 JSON 字符串；
`n3d_shape.data.load_npz_arrays` 只挑 `X`/`y`，多余键不影响其契约）。

meta 字段：源 split 与 QA 成员、归档 SHA256/字节数/成员数/解压后字节数/**流式访问口径**、
映射规则、标签规则（负样本种子/比例/候选池/行序）、答案匹配规则、特征口径
（`feature_dim` / `hash_dim` / `no_bag` / 哈希算法与盐 / 落桶规则 / 分词正则 / **逐列定义**）、
计数（正负/文档池/答案命中/`bag_columns`/文档类别分布/特征极值）、题目清单、文档清单、
**逐样本 provenance**（`[题目索引, 文档索引, label, 答案命中]`）、`notes`（易误读字段的定义）。

**确定性落盘**：`np.savez` 会把当前时间写进 zip 时间戳，故本模块自写 zip
（成员时间戳固定 `(1980,1,1,0,0,0)`、`create_system=0`、`external_attr` 尝试置 0 —— CPython 3.12
的 `zipfile` 在 `external_attr == 0` 时会改写为 `0o600 << 16`，实测 `0x01800000`，与 `np.savez`
原生口径一致，不影响确定性），meta 不含时间戳/环境指纹。实测：**冷构建（2 遍）与暖构建（1 遍）
产出的 npz 逐字节相同**，与缓存状态、扫描遍数无关。

**产物目录纪律**：出现下列任一情形时构建缺省落 `checkpoints/triviaqa/_verify/`，文件名分别带
`_qK` / `_hN`（非缺省维度）/ `_arch<8 位 hex>`（非计划归档）后缀，绝不与正式产物同名同目录：
`--max-questions > 0`、`--hash-dim` 非 64、`--archive` 不是写死的计划归档。
（`--no-bag` 自 2026-10-03 口径修订起属**正式产物**，落正式目录。）

## 6. 命令行用法

```bash
# 缺省构建（D=70，两个 split）
python n3d_triviaqa/build_dataset.py --split all

# 无词袋版（D=6，两个 split）
python n3d_triviaqa/build_dataset.py --split all --no-bag

# 演练构建（前 4 题；自动落 _verify/ 且文件名带 _h64_q4 / _nobag_q4）
python n3d_triviaqa/build_dataset.py --split all --max-questions 4

# 失败对照口径（落 _verify/，文件名带 _h1024）
python n3d_triviaqa/build_dataset.py --split all --hash-dim 1024

# 验证（E0 恒定逐产物执行）
python n3d_triviaqa/verify_dataset.py --checks E1,E2,E3,E4,E6
python n3d_triviaqa/verify_dataset.py --checks E5          # 逐个产物跑端到端
python n3d_triviaqa/verify_dataset.py --checks E5 --e5-path <npz>   # 只跑指定产物
python n3d_triviaqa/verify_dataset.py                      # 全部
```

其他参数：`--archive`、`--out-dir`、`--negative-seed`、`--no-cache`、`--refresh-cache`、
`--allow-missing-evidence`（缺省严格：evidence 未命中即退码 2）。退出码：`0` 成功；
`2` 归档校验失败 / 映射未命中 / 契约断言失败 / `--no-bag` 与 `--hash-dim` 同时给出。

**[!] `--max-questions` 的下限**：候选池 = 纳入题目的文档并集，K 太小会不够给某题凑 1:1 负样本
（实测 wiki 取前 2 题即明确报错退码 2，**不静默降级**）。演练请用 `--max-questions 4` 起步。

## 7. 实测结果（真实执行）

### 7.0 先单条端到端演练，再放全量

```bash
python n3d_triviaqa/build_dataset.py --split all --max-questions 4            # D=70：wiki X=(16,70) / web X=(8,70)
python n3d_triviaqa/build_dataset.py --split all --max-questions 4 --no-bag   # D=6 ：wiki X=(16,6) / web X=(8,6)
python n3d_shape/train.py --preset default --dataset npz \
  --dataset-path checkpoints/triviaqa/_verify/n3d_triviaqa_verified_wiki_dev_h64_q4.npz \
  --input-dim 70 --output-dim 2 --epochs 1 --seed 42 --n 256 --shape sphere \
  --input-scope any_isolated --readout-scope any_isolated --threads 0 --fc-dim -1 --geo-field none \
  --max-batches 2 --checkpoint checkpoints/n3d_shape/_verify/triviaqa_drill_d70.pt
# 实测（退码 0）：数据集加载完成（npz）：train=13，test=3，D=70；[epoch 1/1] loss=0.7117 test_acc=66.67%
# D=6 同口径（--input-dim 6）：train=13，test=3，D=6；[epoch 1/1] loss=0.7175 test_acc=100.00%
```

演练产物落 `_verify/`（文件名带 `_h64_q4` / `_nobag_q4`），`--max-batches` 触发的 checkpoint
落 `checkpoints/n3d_shape/_verify/` —— 演练**不覆盖任何正式产物**。

### 7.1 全量构建（真实终端输出）

```
[n3d_triviaqa] 归档校验通过：SHA256=ef94fac6db0541e5bb5b27020d067a8b13b1c1ffc52717e836832e02aaed87b9（2665779500 字节，487254 个成员）
[n3d_triviaqa] 特征口径：hash_dim=64 -> D = 70（缺省口径）
[n3d_triviaqa]   evidence 命中 1021/1021（wikipedia 660 / web 361），未命中 0
[n3d_triviaqa] [OK] wiki: X=(1280, 70)（D=70）float32 / y=(1280,) int64 正 640 负 640 （文档池 621，正样本含答案 633/640，负样本含答案 31/640）
[n3d_triviaqa]      产物 ...\checkpoints\triviaqa\n3d_triviaqa_verified_wiki_dev_h64.npz（SHA256=c7b142e89b9b5a21...）
[n3d_triviaqa] [OK] web: X=(820, 70)（D=70）float32 / y=(820,) int64 正 410 负 410 （文档池 410，正样本含答案 409/410，负样本含答案 7/410）
[n3d_triviaqa] 特征口径：--no-bag（无词袋块） -> D = 6
[n3d_triviaqa] [OK] wiki: X=(1280, 6)（D=6）float32 / y=(1280,) int64 正 640 负 640
[n3d_triviaqa] [OK] web: X=(820, 6)（D=6）float32 / y=(820,) int64 正 410 负 410
```

实测：维基侧 evidence 映射 **640 命中 / 0 未命中**；两 split 合并目标 evidence **1021/1021 全部命中**。

### 7.2 验证 E0–E6（两版各跑，真实执行）

| 项 | 实测 | 判定 |
| --- | --- | --- |
| E0（逐产物 ×4） | 4 个产物 meta 逐项 OK：归档 SHA256/大小/成员数/解压字节数、`no_bag` 与实际口径一致、`feature_dim = hash_dim + extra_dim`（70 / 6）、缺省口径 D=70、`hash_bucket_rule` 的 mod 与实际维度一致、`counts.bag_columns = hash_dim`、列名与 `feature_columns(hash_dim, no_bag)` 逐字一致且无缝覆盖 `[0, D-1]`、`X` 列数 = meta.feature_dim、负样本种子/比例/短数字阈值、provenance 计数与索引范围、文档路径前缀 | **PASS** |
| E1（逐口径连跑两次） | `[h64]` 2 次构建（--split all）122.5 s / 149.7 s：wiki `A=B=正式产物=c7b142e89b9b5a21…`、web `A=B=正式产物=bfdf4ca91c6dc986…`；`[nobag]` 147.9 s / 155.7 s：wiki `a544eb8bd99421a7…`、web `15de4699084c7b9e…` —— **每个口径三次 SHA256 完全相同** | **PASS** |
| E2（修 warning 1） | 4 个产物均**委派 `n3d_shape.data.load_npz_arrays` 读取成功**；期望维度**取自产物 meta 的 `features.feature_dim`**（70 / 6 各自断言，不再硬编码 1030），`X.shape=(1280,70)/(1280,6)/(820,70)/(820,6)`、dtype float32/int64、无 NaN/Inf、标签 ⊂ {0,1} | **PASS**（对 D=70/D=6 无假失败） |
| E3（修 warning 2） | 归档实测 SHA256 = `ef94fac6…`（**本次 E3 的 QA 缓存键来源**，与写死常量一致）；4 个产物正负 640/640 与 410/410（正占比 0.5000）；抽样 20 正 + 20 负/产物从归档重算命中**与 meta 一致 40/40**（4 个产物共 160/160）；抽样正样本含答案 20/20，负样本 1/20 | **PASS** |
| E4 | 4 个产物：同题正负文档交集 **0**、重复 `(QuestionId,文档)` 对 **0**、负样本属本题证据 **0**；**口径修订专项**：同 split 的 h64 与 nobag **样本口径逐条一致**（行序/QuestionId/文档/label）且 **6 列附加特征逐位一致**（(1280,6)/(820,6)）；跨 split 信息项：QuestionId 交集 13、文档交集 10、共有对 2、标签冲突 0 | **PASS** |
| E5（逐产物 ×4） | wiki/h64 **87.11%**、wiki/nobag **90.62%**、web/h64 **83.54%**、web/nobag **90.85%**，全部退码 0，判据 `> 0.75` -> **4/4 达标** | **PASS** |
| E6 | `git status --porcelain -- n3d_proto n3d_sphere n3d_shape n3d_viz` **无改动**；四目录最新 mtime 均早于本模块（n3d_shape 为 `README.md` 2026-10-03 00:41:42） | **PASS** |

汇总（`--checks E1,E2,E3,E4,E6`）：**通过 9 项、失败 0 项、跳过 0 项**（含 4 个 E0）；
`--checks E5`：**通过 5 项、失败 0 项**。E5 逐 epoch 曲线（末轮为准）：
`wiki/h64 [77.34 … 87.11]`、`wiki/nobag [89.84 … 90.62]`、`web/h64 [51.83 … 83.54]`、
`web/nobag [89.63 … 90.85]`。

### 7.3 口径修订的实测依据（D=1030 为何必须降维）

**(a) 端到端对照**（同一 R1 口径 20 epoch，`n3d_shape/train.py --dataset npz`）：

| 产物 | D | 末轮 test_acc | 判定 |
| --- | --- | --- | --- |
| 原计划口径 wiki（`_verify/legacy_d1030/`） | 1030 | **59.38%** | 未达 0.75（**旧口径 FAIL**） |
| 原计划口径 web（同上） | 1030 | 48.17% | 未达 0.75 |
| 修订后 wiki/h64 | 70 | **87.11%** | PASS |
| 修订后 wiki/nobag | 6 | **90.62%** | PASS |
| 修订后 web/h64 | 70 | **83.54%** | PASS |
| 修订后 web/nobag | 6 | **90.85%** | PASS |

**(b) 特征判别力分解**（`_verify/_diag/analyze_feature_power.py`：逐列 |Pearson r| +
5 折交叉验证逻辑回归，纯 numpy、确定性）：

| 产物 | D | 词袋块 max\|r\| | 词袋块 \|r\|>0.3 列数 | 词袋块 CV | 6 列附加 CV | 全部列 CV | 单列 q→d 覆盖率 CV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 原计划 wiki D=1030 | 1030 | 0.0774 | **0** | 0.3406 | 0.8852 | 0.7820 | 0.8570 |
| 原计划 web D=1030 | 1030 | 0.0861 | **0** | 0.2927 | 0.9329 | 0.7805 | 0.8793 |
| 修订 wiki h64 | 70 | 0.0534 | 0 | 0.4586 | 0.8852 | **0.8867** | 0.8570 |
| 修订 web h64 | 70 | 0.0647 | 0 | 0.4537 | 0.9329 | **0.9061** | 0.8793 |
| 修订 wiki nobag | 6 | — | — | — | 0.8852 | 0.8852 | 0.8570 |
| 修订 web nobag | 6 | — | — | — | 0.9329 | 0.9329 | 0.8793 |

**读数**：1024 维词袋块单独使用**比随机还差**（CV 0.3406 / 0.2927，逐列 |r| 最大仅 0.0774 / 0.0861、
且 |r|>0.3 的列数为 0），而 6 列附加特征与单列 `q→d` 覆盖率分别达 0.8852/0.9329 与 0.8570/0.8793；
把该块并入后"全部列 CV"反而从 0.8852 掉到 0.7820 —— 与端到端 59.38% 的现象同源。
降到 64 桶后"全部列 CV"回升到 0.8867 / 0.9061，端到端随之达标。
风后独立实测（numpy 逻辑回归 5 折 CV，D=1030 产物）给出 词袋 0.3445 / 0.2866、6 列附加
0.8844 / 0.9354、单列 qcov 0.8570 / 0.8793、词袋 max|r| 0.0774 / 0.0861 ——
与本表**逐项吻合**（其中 qcov 与 max|r| 完全一致），两条独立测量路径互相印证。

[!] 口径修订**只改特征列**：实测同 split 的 h64 与 nobag 产物**样本逐条一致**
（行序/QuestionId/文档/label）且 6 列附加特征**逐位一致**（E4 断言），标签与采样的
1:1 均衡、行序严格交替、负样本种子等口径一字未动。

### 7.4 对照与回归脚本

```bash
# 特征判别力分解（逐列 |r| + 5 折 CV；缺省跑 4 个正式产物 + D=1030 失败对照）
python checkpoints/triviaqa/_verify/_diag/analyze_feature_power.py
# D1 回归：非计划归档不得覆盖正式产物（走缺省路由）
python checkpoints/triviaqa/_verify/_diag/test_d1_custom_archive.py
# D3 回归：build_split 乱序输入与正序产出逐位一致
python checkpoints/triviaqa/_verify/_diag/test_d3_record_order.py
```

### 7.5 离朱两轮独立测试与 D1–D5 缺陷修复

| 编号 | 级别 | 现象 | 处置（已实测复核） |
| --- | --- | --- | --- |
| D1 | 中危 | `resolve_out_dir` 未校验 `--archive`：非缺省归档 + 缺省 `--out-dir` 时产物以**正式文件名**写入正式目录，静默覆盖正式产物 | **已修**：新增 `same_archive` / `archive_tag`，非计划归档落 `_verify/` 且文件名带 `_arch<8hex>`；回归脚本 `test_d1_custom_archive.py`（按离朱建议走**缺省路由**）**PASS** |
| D2 | 低危 | `external_attr=0` 的文档与运行时不符（CPython 3.12 改写为 `0o600 << 16`） | **已修**：docstring 如实说明，确定性结论不变 |
| D3 | 低危 | `build_split`（对外 API）信任调用方顺序，乱序会静默改变行序 | **已修**：入口按 `QuestionId` 排序 + 重复断言；`test_d3_record_order.py` **PASS** |
| D4 | 信息项 | `questions_without_evidence_ref` 易被误读 | **已修**：meta 新增 `notes` 段写明定义与"答案命中不参与特征列" |
| D5 | 信息项 | 同口径下 web（原 D=1030）末轮仅 48.17% | 已补入 §7.3；修订后 web/h64 83.54%、web/nobag 90.85% |

离朱独立断言合计 **829 条**（R26 330 + R27 499）**全部通过、0 失败**；两轮均确认 E1/E2/E3/E4/E6
与产物契约成立，且测试期间未修改本模块与四个既有模块的任何源码。

## 8. 已知性质与边界（如实披露）

1. **负样本"含答案"比例不为 0**：标签是 provenance 判定，别的题的文档可能恰好提到同一答案。
   实测负样本含答案比例 wiki `31/640`、web `7/410`（抽样复核 1/20），作为诊断指标写入 meta，
   **不做过滤**（过滤会改变任务定义并造出"负样本不含答案"的人造分布）。
2. **正样本"含答案"比例不为 1**：wiki `633/640`、web `409/410`。逐条复核分两类：**词边界规则
   按口径生效**（`Foot` 只出现在 `football`、`Pakistan` 只出现在 `Pakistanis`、`Convict` 只出现在
   `convictions`）；**远距离监督固有缺口**（`Henry Gondorf` vs 文档里的 `Henry Gondorff`、
   `IDRIS I` vs `King Idris`、`Dogs` 与 `HENRY THE SIXTH` 在对应文档中不出现）。
   下游训练**不依赖**该标志（只写 meta 作诊断；列 `numeric_answer_flag` 只由**问题侧**答案集合决定）。
3. **词袋块即使降到 64 桶仍弱**：单独使用 CV 仅 0.4586 / 0.4537（低于 6 列附加与单列 qcov）；
   它的价值在于与覆盖率类特征互补（"全部列 CV" 0.8867 / 0.9061 >= 仅附加列 0.8852 / 0.9329）。
   若追求端到端最高准确率，`--no-bag`（D=6）实测最优（wiki 90.62% / web 90.85%）。
4. **`same_archive` 不解析 symlink**：`abspath + normcase` 不解析软链接/junction，故用软链接路径
   指向计划归档会被判为"非计划归档"（落 `_verify/` + `_arch` 后缀）。该方向**失败安全**：只会多出
   一个副本，不会覆盖正式产物。
5. **负样本候选池 = 本 split 正样本文档的并集**（不含未被任何题目引用的 evidence 文档）；
   演练构建（`--max-questions K`）会同时缩小候选池，故其负样本与全量构建不同（`max_questions` 写入 meta）。

## 9. 硬约束

* 不修改 `n3d_proto` / `n3d_sphere` / `n3d_shape` / `n3d_viz` / `framework` 任何源码（E6 复核 `git status`）；
* 不联网；不整包解压（只用 `r|gz` 流式 + 按需 `extractfile`）；不写入 `data/mnist`；
  空间参数（N=256 / D=0.1 / E=736 / |S_in|=193 等）由 `n3d_shape` 侧决定，本模块不涉及；
* 代码内标识符（文件名/字段名/键名）全部现场枚举得出（归档成员名、QA JSON 键、npz 键、
  `n3d_shape` 数据层契约），未凭记忆手写。
