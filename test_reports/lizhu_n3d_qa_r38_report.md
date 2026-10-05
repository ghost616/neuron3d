# n3d_qa 测试报告（离朱 R38 · 容器改名 `n3d_triviaqa` → `n3d_qa` + 定位通用化）

## 0. 结论速览

| 说明书功能点 | 判据 | 结论 |
| --- | --- | --- |
| **1 包导入与导出面** | `import n3d_qa` 解析到 `n3d_qa/__init__.py`；`__all__ == ["build_dataset","verify_dataset"]`；包 docstring 首行含「通用 QA 数据集处理模块」；旧包名 `n3d_triviaqa` 已 `ModuleNotFoundError` | **满足** |
| **2 自引用改写正确性** | build/verify docstring 首行为 `n3d_qa.build_dataset` / `n3d_qa.verify_dataset`；`[n3d_triviaqa]` 零残留、`[n3d_qa]` 27 处；argparse description 以 `n3d_qa ` 开头；stderr 前缀 `[n3d_qa.verify]`（**运行时实测**）；报告标题 `n3d_qa 产物验证报告`；README 8 条命令全为 `python n3d_qa/*.py` | **满足** |
| **3 硬约束未被改动** | `OUT_NAME_TEMPLATE(_PLAIN)` / `out_name_for("wiki",0)` / `out_name_for("web-dev",0,features="rich")` / `HASH_SALT` / `DEFAULT_OUT_DIR` / `DEFAULT_ARCHIVE` / `ARCHIVE_SHA256` 全部逐字未变；`PROJECT_ROOT` 未因包目录改名而错位；`meta["module"] == "n3d_triviaqa"` 回读一致 | **满足** |
| **4 8 个正式产物 SHA256 硬闸门** | 8/8 前缀命中（c7b142e8 / bfdf4ca9 / a544eb8b / 15de4699 / 1ea65eae / 02555fe1 / 057ae8d8 / 8cadecbd）；测试前后 33 个 `n3d_triviaqa_verified_*.npz` 逐个 SHA256 复比**零变化** | **满足** |
| **5 CLI 可运行性** | 3 条命令退码均为 0；`E2,E4,E8 + all` 汇总**通过 15 / 失败 0 / 跳过 2**（与说明书逐字一致）；`E7 + base` PASS，rich 十列 0.9430（wiki）/ 0.9646（web），单列 qcov 0.8570 / 0.8793（逐字一致） | **满足** |
| **6 下游消费契约** | 训练命令退码 0，日志自报 `train=1024, test=256, D=70`，`test_acc 87.11%` | **满足** |
| **7 其他模块零改动** | `git status --porcelain -- n3d_proto n3d_sphere n3d_shape n3d_viz framework` **空**；`-- data/triviaqa` **空**；全仓除 `.module_agent/` 外无其他 tracked 文件改动 | **满足** |
| **8 失效路径检查** | `n3d_triviaqa/` 目录不存在（git 记 4 个 `D`）；`n3d_qa/*.py` 中零处旧模块目录路径串；README 仅 1 处（L110 改名说明条目，说明书明列允许） | **满足** |

**零回滚 8 条：全部满足。** 断言合计 **142 条（142 通过 / 0 失败）**，另 **4 条 CLI 退码门**（全 0）。**本轮未重建、未删除、未覆盖任何既有产物。**

---

## 1. 测试环境与执行清单

* 环境：Windows / PowerShell 5.1 / Python 3.12.10（Windows Store 版）/ `numpy 2.5.3`；盘 exFAT。
* 被测：`n3d_qa/__init__.py`（1 316 B）、`n3d_qa/build_dataset.py`（159 733 B）、`n3d_qa/verify_dataset.py`（65 403 B）、`n3d_qa/README.md`（174 行）。
* 本轮**未重建任何正式 npz**；演练产物落 `checkpoints/triviaqa/_verify/`。
* 环境说明：`.venv/` 为 `include-system-site-packages = false` 的空环境（无 numpy），**未使用**；统一用系统 `python`（numpy 来自用户级 site-packages）。曾出现 1 次解释器解析瞬时异常（同一命令重跑即恢复），未影响任何结论，重跑 3 次均稳定复现同一结果。

| 套件 / 命令 | 覆盖功能点 | 结果 |
| --- | --- | --- |
| `lizhu_r38_scripts/regress_r38.py`（静态/导入/文件级/产物闸门） | 1、2、3、4、8 | **58 通过 / 0 失败** |
| `lizhu_r38_scripts/regress_r38b.py`（运行时 stderr 前缀 + 边界界定） | 2、8 | **10 通过 / 0 失败** |
| `lizhu_r38_scripts/lib/astcmp2.py`（归一化 AST 比对 力牧 备份） | 「代码逻辑零改动」 | **6 通过 / 0 失败** |
| `poststate_r38.py`（测试前后产物指纹复比） | 4 | **3 通过 / 0 失败** |
| `python -m compileall -q n3d_qa` | 编译测试 | **退码 0** |
| `build_dataset.py --help` / `verify_dataset.py --help` | 正向 CLI | **退码 0 / 0** |
| `build_dataset.py --features bogus` / `--no-bag --hash-dim 64` / `--archive <不存在>` | 反向 CLI | **退码 2 / 2 / 2**（均带 `[n3d_qa]` 前缀） |
| `build_dataset.py --split all --max-questions 4` | 5 | **退码 0**（73.8 s，流式扫描 1 遍） |
| `verify_dataset.py --checks E2,E4,E8 --product-set all` | 5 | **退码 0**，通过 15 / 失败 0 / 跳过 2 |
| `verify_dataset.py --checks E7 --product-set base` | 5 | **退码 0**，E7 PASS |
| `train.py --dataset npz …wiki_dev_h64.npz…` | 6 | **退码 0**，`train=1024, test=256, D=70`，`test_acc 87.11%` |
| `git status --porcelain`（3 组路径） | 7、8 | **全部符合预期** |

---

## 2. 关键实测明细

### 2.1 「代码逻辑零改动」的强证据（归一化 AST 比对）

以 力牧 的**迁入时备份**（`.module_agent/n3d_qa/backups/<hash>/<最早 .bak>`，由 `mapping.json` 定位）为基线，去除 docstring 后比对可执行 AST：

| 文件 | 备份 | 结论 |
| --- | --- | --- |
| `n3d_qa/build_dataset.py` | `1e1825e2…/1791128214323.bak`（159 513 B） | 归一化（`[n3d_triviaqa]`→`[n3d_qa]` 等 4 条规则）后 AST **完全一致、零残差** |
| `n3d_qa/verify_dataset.py` | `92cdabc2…/1791128215932.bak`（65 262 B） | 归一化后 AST **完全一致、零残差** |
| `n3d_qa/__init__.py` | `11017f66…/1791128214282.bak`（976 B） | 去掉 docstring 后 AST **逐字相同**（连字符串常量都未变，仅 docstring 重写） |

未归一化时的**全部**差异（逐对列举、无遗漏）恰好是说明书明列的 4 类改写，**无第 5 类**：

* `build_dataset.py`：**29 对** = 28 处 `'[n3d_triviaqa] …'` → `'[n3d_qa] …'` 日志前缀 + 1 处 `read_npz_meta` docstring 交叉引用（→ `:mod:`n3d_qa.verify_dataset``）。**其余常量/分支/参数/算法一字未动。**
* `verify_dataset.py`：**4 对** = `description` 首字 `n3d_triviaqa ` → `n3d_qa `、`'[n3d_triviaqa.verify] '` → `'[n3d_qa.verify] '`、报告标题 `'n3d_triviaqa 产物验证报告'` → `'n3d_qa 产物验证报告'`、模块 docstring 首行。
* `__init__.py`：docstring 全文重写，**可执行 AST 逐字相同**（`__all__` 取值未变，仅注释/说明变化）。

> 即：**无任何常量、分支、参数、算法逻辑被改动**——这正是「8 个产物 SHA256 逐字节不变」的机理保证。

### 2.2 8 个正式产物 SHA256 硬闸门（只读复核）

| 产物 | 期望前缀 | 实测 | 与测试前指纹 |
| --- | --- | --- | --- |
| `checkpoints/triviaqa/…wiki_dev_h64.npz` | c7b142e8 | **c7b142e89b9b…** | 未变 |
| `checkpoints/triviaqa/…web_dev_h64.npz` | bfdf4ca9 | **bfdf4ca91c6d…** | 未变 |
| `checkpoints/triviaqa/…wiki_dev_nobag.npz` | a544eb8b | **a544eb8bd994…** | 未变 |
| `checkpoints/triviaqa/…web_dev_nobag.npz` | 15de4699 | **15de4699084c…** | 未变 |
| `checkpoints/triviaqa/_verify/…wiki_dev_h64_rich.npz` | 1ea65eae | **1ea65eae226c…** | 未变 |
| `checkpoints/triviaqa/_verify/…web_dev_h64_rich.npz` | 02555fe1 | **02555fe15359…** | 未变 |
| `checkpoints/triviaqa/_verify/…wiki_dev_nobag_rich.npz` | 057ae8d8 | **057ae8d85c3a…** | 未变 |
| `checkpoints/triviaqa/_verify/…web_dev_nobag_rich.npz` | 8cadecbd | **8cadecbd7185…** | 未变 |

测试前/后对 `checkpoints/triviaqa/` + `_verify/` 下**全部 33 个** `n3d_triviaqa_verified_*.npz` 做 `(size, SHA256)` 双指纹复比：**33/33 完全一致，零删除、零新增、零覆盖**。演练构建 4 题产物 `…_h64_q4.npz` 亦与既有文件逐字节相同（`084dff36…` / `ed22485f…`），即正式产物的可复现性再次现场确认。

### 2.3 CLI 实测（说明书 5）

**`python n3d_qa/build_dataset.py --split all --max-questions 4`** → 退码 0（73.8 s，第 2 遍流式扫描 1 遍命中）

```
[n3d_qa] [OK] wiki: X=(16, 70)（D=70）float32 / y=(16,) int64 正 8 负 8
[n3d_qa]      产物 …\checkpoints\triviaqa\_verify\n3d_triviaqa_verified_wiki_dev_h64_q4.npz
[n3d_qa] [OK] web:  X=(8, 70)（D=70）float32 / y=(8,) int64 正 4 负 4
[n3d_qa]      产物 …\checkpoints\triviaqa\_verify\n3d_triviaqa_verified_web_dev_h64_q4.npz
[n3d_qa] 特征口径：features=base / hash_dim=64 -> D = 70（缺省口径）
[n3d_qa] 归档校验通过：SHA256=ef94fac6…87b9（2665779500 字节，487254 个成员）
```

落点正确（`_verify/`）、**未覆盖** 8 个正式产物、日志前缀 `[n3d_qa]` 正确。

**`python n3d_qa/verify_dataset.py --checks E2,E4,E8 --product-set all`** → 退码 0

```
汇总：通过 15 项，失败 0 项，跳过 2 项；用时 3.9 s
结论：全部通过（跳过项为环境原因，不计失败）
```

严格匹配说明书「通过 15 / 失败 0 / 跳过 2」；跳过 2 项为 `wiki-dev` / `web-dev` 的 rich 口径产物尚未构建（预先存在的环境性跳过，非本轮缺陷）。报告标题、检查项标签均为 `n3d_qa`。

**`python n3d_qa/verify_dataset.py --checks E7 --product-set base`** → 退码 0，`[E7] PASS`，逐值匹配说明书：

| 读数 | wiki | web | 说明书要求 |
| --- | --- | --- | --- |
| CV rich 附加块（十列）acc | **0.9430** | **0.9646** | 0.9430 / 0.9646 ✓ |
| CV 单列 `qcov`(base) acc | **0.8570** | **0.8793** | 0.8570 / 0.8793 ✓ |
| 结论 | rich 0.9430 > base 0.8859 | rich 0.9646 > base 0.9329 | 均判 rich 优于 base ✓ |

### 2.4 下游消费契约（说明书 6）

```
python n3d_shape/train.py --dataset npz --dataset-path checkpoints/triviaqa/n3d_triviaqa_verified_wiki_dev_h64.npz \
  --input-dim 70 --output-dim 2 --preset default --seed 42 --n 256 --shape sphere \
  --input-scope any_isolated --readout-scope any_isolated --threads 0 --fc-dim -1 --geo-field none --epochs 20
```

退码 **0**（13 s）；`[N3D][INFO] 数据集加载完成（npz）：train=1024，test=256，D=70，标签取值 [0, 1]`；最终 `test_acc 87.11%`（与 README §7.2 留档一致）。产物路径与模块目录名解耦，改名后消费链**零影响**。

### 2.5 反向/边界覆盖

| 场景 | 期望 | 实测 |
| --- | --- | --- |
| `--features bogus` | argparse 拒绝 | 退码 **2**，usage 到 stderr |
| `--no-bag --hash-dim 64`（互斥） | 显式拒绝 | 退码 **2**，`[n3d_qa] [FAIL] --no-bag 与 --hash-dim 互斥（…当前同时给了 --hash-dim 64）` |
| `--archive <不存在>` | 归档校验失败 | 退码 **2**，`[n3d_qa] [FAIL] 归档校验失败：归档不存在：…` |
| `verify_dataset --checks E99` | 未知检查项 | **运行时实测** stderr 首行 `[n3d_qa.verify] 未知检查项 ['E99']；可选 ('E1',…,'E8')`，退码 1 |
| `--help`（两脚本） | 正常用法输出 | 退码 **0 / 0** |

### 2.6 边界界定（非缺陷项，逐条复核）

* **README L110**：`` 模块目录改名 `n3d_triviaqa/ → n3d_qa/` **不改该字段**… `` —— 说明书明列「说明性文字中的冻结条目」，**非模块路径引用**。已逐行核验：§6 用法命令块内 `n3d_triviaqa/` 出现 **0 次**，8 条命令全为 `python n3d_qa/*.py`。
* **`n3d_qa/*.py`**：旧模块目录路径串 **零处**（仅 `<string>` 级 `n3d_triviaqa_verified_` 产物名与 `meta["module"]` 取值，均为冻结口径）。
* **`lizhu_r*_scripts/`、`test_reports/lizhu_n3d_triviaqa_r*_report.md`、`.module_agent/n3d_triviaqa/`**：历史留档，按说明书**不在改动范围**，本轮未触碰。

---

## 3. E2E 测试：不适用（已核验）

`check_playwright` → 已安装（npm，1.63.0）。但本轮 4 个变更文件全部为 **Python 模块 / Markdown 文档**，**不含任何前台 UI 交互**（无 HTML/CSS/JS、无 DOM、无浏览器入口）；模块自身 README 亦声明「不联网」。故 **E2E 无适用面**，不编写 Playwright 脚本——CLI 与数据契约链（构建 → 校验 → 训练）已由 §2.3–§2.5 端到端覆盖。

---

## 4. 失败用例分析

| 序号 | 现象 | 根因 | 定性 |
| --- | --- | --- | --- |
| 1 | `regress_r38.py` 报 `verify_dataset.py stderr 前缀为 [n3d_qa.verify]` FAIL | **测试脚本自身断言口径错误**：脚本要求源码中出现带引号的 `"[n3d_qa.verify]"` 字面量，而源码是 f-string 内的 `[n3d_qa.verify] `。已改为源码级精确匹配 **+ 运行时实测**（`regress_r38b.py` 4/4 PASS） | 非产品缺陷 |
| 2 | `regress_r38.py` 报 README 存在 `n3d_triviaqa/` 引用 FAIL | **测试脚本正则过宽**：命中的是 L110 改名**说明条目**（说明书明列允许）。已收窄为「命令/路径引用」检测 | 非产品缺陷 |
| 3 | `regress_r38.py` 报 `n3d_qa/` 下存在旧目录路径串 FAIL | 同 #2（同一处 L110） | 非产品缺陷 |
| 4 | `lib/astcmp.py` 报两文件「可执行 AST 完全一致」FAIL | **比对方法过严**：未对说明书明列的日志前缀改写做归一化。已用 `lib/astcmp2.py` 归一化比对，**零残差**（见 §2.1） | 非产品缺陷 |

上述 4 项均为我方断言/方法口径问题，**已修正并全部转为 PASS**；修正后三套件 **142/142 通过**。**本轮未发现任何产品侧缺陷。**

---

## 5. 环境问题说明

| 项 | 说明 | 影响 |
| --- | --- | --- |
| `.venv/` 为空环境 | `include-system-site-packages = false`，`python -c "import numpy"` 报 `ModuleNotFoundError` | **无影响**：改测系统 `python`（numpy 2.5.3 来自用户级 site-packages）。`sklearn` 在全机均缺失，但 E7 已自实现 numpy 逻辑回归（`_fit_logistic`），**不依赖 sklearn** |
| 解释器解析瞬时异常 | 首轮探测中同一 `python -c "import numpy"` 出现 1 次 `ModuleNotFoundError`，重跑即恢复；随后 3 次重复均稳定通过 | **无影响**：全部测试在该状态下完成并复现 |
| PowerShell 5.1 无 `pwsh` | 首版 heavy 启动脚本用 `pwsh -File` 启动失败（`CommandNotFoundException`） | **无影响**：改用 `powershell.exe -NoProfile -File` 后全部执行成功 |
| 目录写入 EISDIR | 本会话文件写入工具对 `lizhu_r38_scripts/` 报 `EISDIR` | **无影响**：改用 PowerShell 写文件，脚本落地正常（已实测可执行） |
| **无阻断性环境问题** | — | **无测试类型因环境被跳过**（E2E 属无适用面，非环境跳过） |

---

## 6. 修复建议

1. **本轮无需修复**——8 条功能点逐一实证满足，未发现产品缺陷。
2. 可选（非本轮要求，供后续参考）：
   * 把 `regress_r38.py` / `regress_r38b.py` / `lib/astcmp2.py` / `poststate_r38.py` 沉淀为常规回归套件，作为「包改名类」改动的标准验收脚本（其中「归一化 AST 比对」与「产物前后指纹复比」两条对本类改动最有价值）。
   * `wiki-dev` / `web-dev` 的 rich 口径产物仍未构建（导致 E2/E4/E8 出现 2 项 SKIP）；如需全绿，可另行安排 compact 构建（属既有待办，与本轮改名无关）。
   * 若希望 CI 稳定复现，建议在测试入口显式固定解释器（避免 Windows Store 别名与 `.venv` 空环境混用）。

---

## 7. 留档

| 类别 | 路径 |
| --- | --- |
| 测试脚本 | `lizhu_r38_scripts/regress_r38.py`、`regress_r38b.py`、`poststate_r38.py`、`lib/astcmp.py`、`lib/astcmp2.py`、`run_r38_heavy.ps1` |
| 指纹留档 | `lizhu_r38_scripts/_pre_state.json`（33 个产物）、`_prod_sha.json` |
| 运行日志 | `checkpoints/triviaqa/_verify/_lizhu_r38/`（`_exitcodes.txt` + 4 个 step 日志） |
