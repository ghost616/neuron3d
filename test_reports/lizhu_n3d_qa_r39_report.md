# 留档脚本旧模块路径断链修复 测试报告（离朱 R39 · n3d_triviaqa → n3d_qa）

## 0. 结论速览

| 说明书功能点 | 判据 | 结论 |
| --- | --- | --- |
| **1 8 个脚本无导入类错误** | 9 个脚本实跑（8 授权 + 1 越界）：**零 `ModuleNotFoundError: No module named 'build_dataset'`、零 `NameError`、零 `KeyError`、零 `UnicodeEncodeError`、零 Traceback**；唯一非零退出是 `test_ac_equivalence.py` 的**语义性退码 1** | **满足** |
| **2 recompute_learning_baselines.py** | 退码 0；读数**逐值命中**：单列 qcov **0.8570 / 0.8793**；base extra 6 cols **0.8859 / 0.9329**；rich extra 10 cols **0.9430 / 0.9646**；legacy 1030 对照 **0.7789 / 0.7732** | **满足** |
| **3 _probe 三脚本** | `_inspect.py` 打印 **`len 159733`** 且退码 0；`_cmp_anchor.py` / `_cmp_old2.py` 均**成功打开 `n3d_qa/build_dataset.py`** 并输出 `contains: False` + 最长匹配前缀长度（71） | **满足** |
| **4 test_ac_equivalence.py** | 无 ModuleNotFoundError；输出 **`total pairs=5367  AC misses=1240`** 与 `RESULT: MISSES FOUND - do not replace`；退码 **1**（语义性，已知 AC 漏判） | **满足** |
| **5 test_d1_custom_archive.py** | 退码 0，输出 `[D1] PASS`；测试前后 4 个正式产物「文件名→SHA256」快照**逐字一致**，并与独立指纹交叉核对 4/4 相符 | **满足** |
| **6 test_d3_record_order.py** | 退码 0，输出 `[D3] PASS`，正序/乱序 QuestionId 行序完全一致 | **满足** |
| **7 两个 analyze 脚本** | 均退码 0；`analyze_feature_power.py` 实分析 **6** 个产物（4 正式 + 2 legacy）；`analyze_feature_power_rich.py` **默认参数真正分析 wiki 与 web 两对，无任何 `[skip]`** | **满足** |
| **8 路径残留分类** | `checkpoints/**/*.py` 中 `n3d_triviaqa` 残留 **7 处，全部为产物名类**（`recompute` PATHS 6 处 + `diag_e5_reference.py:68` 1 处）；**`n3d_triviaqa/` 模块目录引用 0 处**；历史 json 属明列允许项 | **满足** |
| **9 8 个产物 SHA256 硬闸门** | 8/8 前缀命中（c7b142e8 / bfdf4ca9 / a544eb8b / 15de4699 / 1ea65eae / 02555fe1 / 057ae8d8 / 8cadecbd）；测试前后 **33 个产物零增/零删/零改**；正式目录顶层文件集合未变 | **满足** |
| **10 其他模块零改动** | `git status --porcelain -- n3d_proto n3d_sphere n3d_shape n3d_viz framework data/triviaqa` **空输出** | **满足** |

**零回滚 10 条：全部满足。** 断言合计 **100 条（100 通过 / 0 失败）** + **9 条退码门**（8 个预期值全中）。**本轮未重建、未删除、未覆盖任何产物。**

### ⚠️ 一个越界发现（不在本轮清单，请风后判断）
`checkpoints/triviaqa/_verify/_diag/diag_e5_reference.py` **退码 1**，但它**不属本轮授权修复的 8 处清单**：
```
FileNotFoundError: npz 文件不存在：E:\neuron3d\checkpoints\triviaqa\n3d_triviaqa_verified_wiki_dev.npz
```
根因是该脚本 `main()` 里拼的产物名缺 `_h64` 后缀（实际产物为 `n3d_triviaqa_verified_wiki_dev_h64.npz`）。这是**既有断链**（R30/R31 期该脚本本就按 1030 维旧产物名书写），**非本轮引入、本轮未授权修改**，故**不计入通过/失败**，仅作残留分类的旁证与后续待办上报。

---

## 1. 测试环境与执行清单

* 环境：Windows / PowerShell 5.1 / Python 3.12.10（Windows Store 版）/ `numpy 2.5.3`；盘 exFAT。
* 被测：`checkpoints/triviaqa/_verify/_diag/` 下 6 个脚本 + `checkpoints/triviaqa/_probe/` 下 4 个脚本；`n3d_qa/README.md` §8。
* **未重建任何 npz**；本套件全程只读产物。
* 说明：`.venv/` 为无 numpy 的空环境（`include-system-site-packages=false`），已按 R38 口径改测系统 `python`。

| 步骤 | 命令 | 退码 | 用时 |
| --- | --- | --- | --- |
| t2 | `python .../recompute_learning_baselines.py` | **0** | 5 s |
| t3a | `python .../_probe/_inspect.py` | **0** | 0 s |
| t3b | `python .../_probe/_cmp_anchor.py` | **0** | 0 s |
| t3c | `python .../_probe/_cmp_old2.py` | **0** | 0 s |
| t4 | `python .../test_ac_equivalence.py` | **1**（语义性） | 159 s |
| t5 | `python .../test_d1_custom_archive.py` | **0** | 69 s |
| t6 | `python .../test_d3_record_order.py` | **0** | 1 s |
| t7a | `python .../analyze_feature_power.py` | **0** | 106 s |
| t7b | `python .../analyze_feature_power_rich.py` | **0** | 4 s |
| t11a | `python .../_probe/probe_archive_members.py`（附带） | **0** | 58 s |
| t11b | `python .../diag_e5_reference.py --split wiki --epochs 2`（越界） | **1** | 5 s |

| 套件 | 覆盖功能点 | 结果 |
| --- | --- | --- |
| `lizhu_r39_scripts/verify_r39.py`（对数级断言 + 指纹闸门 + 残留分类） | 1–9 | **100 通过 / 0 失败** |
| `_pre_state.json` → `_post_state.json` 指纹复比 | 9 | **ZERO CHANGE**（33/33 一致，无增无删无改） |
| `git status --porcelain`（指定 6 路径） | 10 | **空输出** |

---

## 2. 关键实测明细

### 2.1 8 个脚本可运行性（功能点 1）

9 个实跑脚本**全部无导入类错误**：`checkpoints/*.py` 中 `sys.path` 均已指向 `n3d_qa/`（grep 实证：6 个 `_diag` 脚本 + `probe_archive_members.py` 均含 `sys.path.insert(0, os.path.join(..., "n3d_qa"))`，3 个 `_probe` 小脚本改为**按脚本位置相对定位** `n3d_qa/build_dataset.py`）。退出码分布与说明书预期**完全一致**：`{0,0,0,0,1,0,0,0,0}`。

### 2.2 recompute_learning_baselines.py（功能点 2）—— 逐值命中

```
[wiki] M=1280 (pos 640 / neg 640); base D=70; rich D=74
  single qcov        (col=  64) acc=0.8570 auc=0.9226     ← 期望 0.8570 ✓
  base extra 6 cols  (col=64..)      acc=0.8859 auc=0.9553 d=6   ← 期望 0.8859 ✓
  rich extra 10 cols (col=64..)      acc=0.9430 auc=0.9844 d=10  ← 期望 0.9430 ✓
  all 1030 dims (noisy bag, control) acc=0.7789 auc=0.8529       ← 期望 0.7789 ✓
[web] M=820 (pos 410 / neg 410); base D=70; rich D=74
  single qcov        (col=  64) acc=0.8793 auc=0.9459     ← 期望 0.8793 ✓
  base extra 6 cols  (col=64..)      acc=0.9329 auc=0.9741 d=6   ← 期望 0.9329 ✓
  rich extra 10 cols (col=64..)      acc=0.9646 auc=0.9903 d=10  ← 期望 0.9646 ✓
  all 1030 dims (noisy bag, control) acc=0.7732 auc=0.8470       ← 期望 0.7732 ✓
```
**8 个登记读数 8/8 命中**（容差 5e-5）；`legacy1030` 对照读数与历史登记值（0.7734 等）不完全一致，按说明书属**既有登记事实，不判失败**。此处读数与 E7（`verify_dataset --checks E7`）**同源**：wiki/web 单列 qcov 0.8570/0.8793、rich 十列 0.9430/0.9646 与 R38 实测逐字相同 —— 证明该脚本的 `sys.path` 修复后与正式验证链**读数一致**。

### 2.3 _probe 三脚本（功能点 3）

```
$ python .../_probe/_inspect.py
idx 10847
raw slice: "哈希词袋：把 (Question + '\\n' + Evidence) 拼接后"
backslash count in slice: 2
has_rich_columns: 13115
extra_columns def at: 11269
check_features def at: 9296
len 159733                      ← 说明书要求 len 159733 ✓（= 当前 build_dataset.py 实际字节数）

$ python .../_probe/_cmp_anchor.py
contains: False
longest matching prefix len: 71
anchor[n-40:n+40] = 'im: int = HASH_DIM, no_bag: bool = False\n) -> Tuple[Dict[str, Any], ...]:\n    "'
src   [n-40:n+40] = 'im: int = HASH_DIM, no_bag: bool = False, features: str = FEATURES_DEFAULT\n) -> '

$ python .../_probe/_cmp_old2.py
contains: False
longest matching prefix len: 71 of 1048
MINE: 'HASH_DIM, no_bag: bool = False\n) -> Tuple[Dict[str, Any], ..'
FILE: 'HASH_DIM, no_bag: bool = False, features: str = FEATURES_DEFAULT'
```
两侧切片对照精确指认了断链点：R30 前的旧源码切片在 `no_bag: bool = False` 后直接接 `)`，而现行源码已加入 `features: str = FEATURES_DEFAULT` 参数 —— 即 `contains: False` 是**正确结果**（锚文本为旧源码切片），**不是错误**。`_inspect.py` 的 `len 159733` 同时反证了相对定位路径确实打开的是 `n3d_qa/build_dataset.py`（该文件实测 159 733 B）。

### 2.4 test_ac_equivalence.py（功能点 4）

```
[1 corpus] pairs=450 misses=32
[2 manual] pairs=857 misses=490
[3 fold-traps] pairs=... misses=...
[4 fuzz] pairs=... misses=...
==============================================================================
total pairs=5367  AC misses=1240
RESULT: MISSES FOUND - do not replace
```
退码 **1**（语义为「有漏判=1」）。**`total pairs=5367  AC misses=1240` 与说明书逐字一致。** 折叠陷阱用例（`ſ`/`İ`/`ς` 等）确实被打印出来（日志含 `[MISS] fold/fold: ans=('ſ',) text='x ſ y'`）且**未抛 `UnicodeEncodeError`** —— 证明入口新增的 `_relax_console_encoding()` 生效，脚本能跑完并给出结论行（`configure_console_encoding` 同口径）。

### 2.5 test_d1_custom_archive.py（功能点 5）

```
[D1] 缺省路由 A -> E:\neuron3d\checkpoints\triviaqa\_verify / n3d_triviaqa_verified_wiki_dev_h64_archea08d860.npz
[D1] 缺省路由 B -> n3d_triviaqa_verified_wiki_dev_h64_archfbe86aa7.npz
[D1] 测试前正式产物：{'n3d_triviaqa_verified_web_dev_h64.npz': 'bfdf4ca91c6d…', '…web_dev_nobag.npz': '15de4699084c…',
                     '…wiki_dev_h64.npz': 'c7b142e89b9b…', '…wiki_dev_nobag.npz': 'a544eb8bd994…'}
[D1] 测试后正式产物：（与测试前逐字相同）
[D1] PASS：缺省路由下自定义归档产物落 _verify（…_archea08d860.npz / …_archfbe86aa7.npz），正式产物集合与 SHA256 全不变
```
关键点全部满足：**不给 `--out-dir`** 走缺省路由 → 落 `_verify/`；文件名带 `_arch<8hex>`；两个不同路径的同字节归档产出**不同文件名**（不互覆）；4 个正式产物 SHA256 快照**测试前后逐字一致**，并与我方**独立指纹 4/4 交叉相符**（`c7b142e8` / `bfdf4ca9` / `a544eb8b` / `15de4699`）。

### 2.6 test_d3_record_order.py（功能点 6）

```
[D3] 正序行序：['q1', 'q1', 'q2', 'q2', 'q3', 'q3', 'q4', 'q4', 'q1', 'q1']
[D3] 乱序行序：['q1', 'q1', 'q2', 'q2', 'q3', 'q3', 'q4', 'q4', 'q1', 'q1']
[D3] PASS：乱序输入与正序输入产出完全一致（行序口径不再依赖调用方顺序）
```

### 2.7 两个 analyze 脚本（功能点 7）

* `analyze_feature_power.py`：退码 0，实分析 **6** 个产物（4 个正式 base 产物 + 2 个 `legacy_d1030` 对照），每段均打印「词袋块 max|r| / 6 列附加 / 全部列 / 单列 q->d 覆盖率」四组 CV acc。`OUT_NAME_TEMPLATE.format(slug=split)` 修复生效 —— **无 `KeyError: 'slug'`**（旧的 `.format(split=…)` 会直接崩）。
* `analyze_feature_power_rich.py`：退码 0，**默认参数真正分析 wiki 与 web 两对**，**日志中零 `[skip]`**：
```
wiki: base=checkpoints\triviaqa\n3d_triviaqa_verified_wiki_dev_h64.npz
      rich=checkpoints\triviaqa\_verify\n3d_triviaqa_verified_wiki_dev_h64_rich.npz
  M=1280  D_base=70  D_rich=74  正/负=640/640
  5 折逻辑回归 CV：
    base 六列附加       d=  6  acc=0.8852  auc=0.9553
    rich 全部附加十列    d= 10  acc=0.9430  auc=0.9848
```
`_resolve()` 回退把 rich 产物在 `_verify/` 下找到（否则默认必走 SKIP 分支）。**注**：其逐列 `|r|`（如 `|r(q_to_d_coverage)| = 0.0083`）与「单列 qcov(base) acc=0.4781」使用的是**字面列号**而非 meta 列号 —— 按说明书属**既有口径缺陷（R30/R31 已对 verify_dataset 修过同类问题）**，本轮只保证可运行与退码 0，**不判失败**；仅在此登记。

### 2.8 路径残留分类（功能点 8）

`checkpoints/**/*.py` 全量扫描，`n3d_triviaqa` 命中 **7 处，全部为产物名类**：

| 文件 | 行 | 内容性质 |
| --- | --- | --- |
| `_diag/recompute_learning_baselines.py` | 30–35（6 处） | `PATHS` 里的产物文件名 `n3d_triviaqa_verified_*`（**允许**） |
| `_diag/diag_e5_reference.py` | 68 | f-string 模板 `n3d_triviaqa_verified_{args.split}_dev.npz`（**允许**，说明书明列） |
| `_verify/_lizhu_r26/e5_independent_results.json` | — | 历史 npz 路径（**允许**，说明书明列） |

**`n3d_triviaqa/` 形式的模块目录引用：0 处**（含 f-string、注释、docstring 全量匹配）。8 处授权修复逐一核验：6 个 `_diag` 脚本 + `probe_archive_members.py` 的 `sys.path` 均指向 `os.path.join(..., "n3d_qa")`；3 个 `_probe` 小脚本均已删除硬编码绝对路径，改为 `os.path.join(ROOT, "n3d_qa", "build_dataset.py")` 相对定位。

### 2.9 硬闸门与零改动（功能点 9/10）

| 产物 | 期望前缀 | 实测 |
| --- | --- | --- |
| `checkpoints/triviaqa/…wiki_dev_h64.npz` | c7b142e8 | **c7b142e89b9b…** |
| `checkpoints/triviaqa/…web_dev_h64.npz` | bfdf4ca9 | **bfdf4ca91c6d…** |
| `checkpoints/triviaqa/…wiki_dev_nobag.npz` | a544eb8b | **a544eb8bd994…** |
| `checkpoints/triviaqa/…web_dev_nobag.npz` | 15de4699 | **15de4699084c…** |
| `checkpoints/triviaqa/_verify/…wiki_dev_h64_rich.npz` | 1ea65eae | **1ea65eae226c…** |
| `checkpoints/triviaqa/_verify/…web_dev_h64_rich.npz` | 02555fe1 | **02555fe15359…** |
| `checkpoints/triviaqa/_verify/…wiki_dev_nobag_rich.npz` | 057ae8d8 | **057ae8d85c3a…** |
| `checkpoints/triviaqa/_verify/…web_dev_nobag_rich.npz` | 8cadecbd | **8cadecbd7185…** |

测试前/后对**全部 33 个** `n3d_triviaqa_verified_*.npz` 做 `(size, SHA256)` 双指纹复比：**33/33 一致，无增、无删、无改**；正式目录顶层文件集合亦未变（`test_d1` 运行只往 `_verify/` 写 `_arch*` 演练产物，未污染正式目录）。`git status --porcelain -- n3d_proto n3d_sphere n3d_shape n3d_viz framework data/triviaqa` **空输出**。

### 2.10 README §8（文档变更核对）

`n3d_qa/README.md` 新增 **§8「历史留档与维护边界」**（L176–205），内容与实现一致：留档表逐项标注冻结/只读状态（`.lizhu_env/triviaqa/`、`lizhu_r30–r37_scripts/`、`test_reports/lizhu_n3d_triviaqa_r*.md`、`.module_agent/n3d_triviaqa/` 冻结；`_diag`/`_probe` **已同步修复**），并给出 **8 条现行可复现入口**（§8 代码块实测恰为 8 条 `python …` 命令），与说明书「8 条现行可复现入口」一致；注中还记录了 `test_ac_equivalence.py` 的 GBK 控制台 `UnicodeEncodeError` 修复口径。

---

## 3. 失败用例分析

本轮**首次运行**有 2 条断言 + 1 处脚本抛错，**全部为验证脚本自身问题，非产品缺陷**：

| 序号 | 现象 | 根因 | 定性 |
| --- | --- | --- | --- |
| 1 | `_cmp_old2` 断言「输出最长匹配前缀长度 = None」FAIL | **验证脚本断言口径错**：我按 `_cmp_anchor.py` 的格式（`len: N`）去匹配 `_cmp_old2.py` 的二分实现（实际格式 `len: N of M`）。实测输出 `longest matching prefix len: 71 of 1048` 与 `MINE:`/`FILE:` 两行**完全符合说明书要求** | 非产品缺陷 |
| 2 | `_cmp_old2` 断言「输出源码切片对照」FAIL | 同 #1：该脚本的对照行标签是 `MINE:`/`FILE:`，不是 `src   [n-40:n+40]` | 非产品缺陷 |
| 3 | `verify_r39.py` 抛 `json.JSONDecodeError` | **验证脚本解析方式错**：D1 的产物快照是 Python `dict` repr（单引号），我误用 `json.loads`；已改 `ast.literal_eval` | 非产品缺陷 |

修正后 **100/100 断言通过**。**未发现任何产品侧缺陷**（越界发现 `diag_e5_reference.py` 见 §0 与 §4）。

---

## 4. 环境问题说明与越界发现

| 项 | 说明 | 影响 |
| --- | --- | --- |
| `diag_e5_reference.py` 退码 1 | `FileNotFoundError`：缺 `checkpoints/triviaqa/n3d_triviaqa_verified_wiki_dev.npz`（产物名缺 `_h64`）。该脚本**不在本轮授权修复的 8 处清单**，且其 `sys.path` 未做 `n3d_qa` 修复（仅 `n3d_shape` 可直接从仓库根导入，故**未出现导入错误**，只是产物路径断链） | **不计入通过/失败**。属**既有断链**（R30/R31 期即按 1030 维旧产物名书写），非本轮引入。**建议列入后续待办**（若需修复：把 `f"n3d_triviaqa_verified_{args.split}_dev.npz"` 改为按 `bd.out_name_for(split, 0, bd.HASH_DIM, ...)` 现场生成，或指向 `_verify/legacy_d1030/`） |
| `analyze_feature_power_rich.py` 读数口径 | 逐列 `\|Pearson r\|`（如需 `q_to_d_coverage` 实读 0.0083）与「单列 qcov(base)」用**字面列号**而非 meta 列号 | **既有缺陷，非本轮引入**（说明书已明列）。本轮只保证可运行 + 退码 0，**判为满足** |
| `recompute` 的 legacy1030 读数 | 0.7789 / 0.7732 与历史登记值（0.7734 等）不完全一致 | **既有登记事实**（R36 报告已记录 1030 维读数在仓库内不可复现），**不判失败** |
| `.venv/` 无 numpy | 空环境；已改测系统 `python`（numpy 2.5.3） | 无影响 |
| **无阻断性环境问题** | — | **无测试类型因环境被跳过**。E2E 无适用面（本轮变更全为 Python 脚本 + Markdown 文档，无 UI） |

---

## 5. 修复建议

1. **本轮 8 处授权修复全部验证通过，无需返工。** 3 处附带修复（`_relax_console_encoding` / `.format(slug=…)` / `_resolve()`）均实证有效且为「否则脚本不可用」的最小改动。
2. 后续待办（**非本轮范围**，供风后决定）：
   * `diag_e5_reference.py:68` 的产物名断链（缺 `_h64`）——建议改用 `bd.out_name_for(...)` 现场生成，避免再次随口径漂移而断链；
   * `analyze_feature_power_rich.py` 的字面列号 → meta 列号（与 R30/R31 对 `verify_dataset` 的同类修复对齐）。
3. 建议把 `lizhu_r39_scripts/verify_r39.py` 沉淀为「留档脚本可运行性」标准回归套件（9 脚本实跑 + 读数对账 + 残留分类 + 指纹闸门四段式），本类改动可一键复验。

---

## 6. 留档

| 类别 | 路径 |
| --- | --- |
| 测试脚本 | `lizhu_r39_scripts/run_r39.ps1`、`verify_r39.py` |
| 指纹留档 | `lizhu_r39_scripts/_pre_state.json`、`_post_state.json`（33 产物，ZERO CHANGE） |
| 运行日志 | `checkpoints/triviaqa/_verify/_lizhu_r39/`（`_exitcodes.txt` + 11 个 step 日志） |
