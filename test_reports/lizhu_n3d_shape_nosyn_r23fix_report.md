# 离朱测试报告 —— n3d_shape 第 23 轮·修复轮 R23-fix（文档一致性 2 处修复 + 1 处瑕疵披露）

**测试时间**：2026-09-29（本副本）
**被测模块**：`n3d_shape`（本轮仅文档与临时探针脚本改动）
**结论**：**全部通过，0 缺陷**。上一轮的 4 项文档断言失败（2 个根因 D1/D2）**已全部修复并复验通过**；O1 瑕疵已按“如实留档、不美化取证文件”口径披露；代码与受保护产物**零变更**。

---

## 一、本轮被测版本与「只动文档」举证

| 文件 | 本轮 SHA256 | 与上一轮对比 |
| --- | --- | --- |
| `n3d_shape/model.py` | `dbede6396f1b49680cdd8e522d68a01b305573ba7b35e5f98e2807665d640394`（110,299 B） | **逐位相同**（spec 要求值） |
| `n3d_shape/train.py` | `d33a1fa8a8fb5c97c5afeafbfc06a342ce085d21c88bca53b0b3866c36d06885`（99,578 B） | **逐位相同**（spec 要求值） |
| `n3d_shape/README.md` | `26144b0058b69459c8eda81eb62ee370e930ffcd393e484264eb6cf61e01c426`（142,043 B） | 已改（本轮修复内容） |
| `.module_agent/n3d_shape/current_spec.md` | `46385651cdd2063074af4126b1ca0b7a4fd7024b05df42fb902cff676e8c98f9`（60,700 B） | 已改（本轮同步内容） |
| `checkpoints/n3d_shape/_verify/nosyn_probe/probe_snapshot.py` | `249ce682cd3ddca9badc9067b13b3f73b53a017594326f2eb4edda83300d8ee1`（10,637 B） | 仅 docstring 增补（**行为未变**，见 §三-P3） |
| `checkpoints/n3d_shape/_verify/nosyn_probe/baseline.json` | `8ba1a70ffa37380be355a9719519e5a3c92792c73f497bd865687e00049f9f91`（48,007 B，mtime `2026-09-29T03:52:54.1805858Z`） | **未被事后修改**（字节/哈希/mtime 三者与上一轮记录一致） |

- **`n3d_proto` / `n3d_viz` 源码零改动**：`n3d_proto/model.py`（`663A2E65…`）、`n3d_viz/core.py`（`CF94A72A…`）、`n3d_viz/__main__.py`（`A80173FE…`）哈希在测试前后一致；`git status` 中 `n3d_proto`/`n3d_viz` 仅有 `.module_agent/**` 记账类文件，**无源码变更**。
- 环境同上一轮：Python 3.13.15 / `torch 2.14.0+cpu`；浏览器 E2E 用系统 Google Chrome 150.0.7871.187（chromium 下载受网络限制，见 §五）。

---

## 二、测试概览

| 测试类型 | 断言/用例数 | PASS | FAIL | 说明 |
| --- | --- | --- | --- | --- |
| **R23-fix 专项验证**（D1/D2/O1 + 版本与只读约束） | **18** | **18** | 0 | `test_r23fix.py` |
| 文档一致性套件（上一轮 15 项，含 D1/D2 两项失败） | 15 | **15** | 0 | `test_docs_consistency.py`：**上轮 11/15 → 本轮 15/15** |
| 单元测试（重跑，确认代码未变） | 27 | 27 | 0 | `test_unit_nosyn.py` |
| 变异测试（重跑） | 8 | 8 | 0 | 8 个变异 0 逃逸 |
| 改动前基线逐位比对（重跑） | 6 | 6 | 0 | 5 个配置 |
| 产物往返与兼容性边界（重跑） | 6 | 6 | 0 | |
| `n3d_viz` 契约回归（重跑） | 5 | 5 | 0 | |
| **自动断言合计** | **85** | **85** | **0** | |
| 接口/CLI 回归 | 4 条命令 | 4/4 符合预期 | 0 | 冒烟 ×2、compileall、verify_shape |
| E2E（浏览器，重跑） | 2 | 2 | 0 | FC 产物 / fc0 产物 HTML |

---

## 三、逐项复核结果（对照 spec §二）

### 1. D1 修复（字节数与 SHA 自洽）— 6/6 PASS

| 断言 | 实测 |
| --- | --- |
| 现场实测 `_verify/smoke_nosyn.pt` | **193,551 B**，SHA256 `a1f4772f797487eadf37ce777ad367fbda02af22d5540bd41286c6c83eaeb6fc` |
| README §19.2 表格行（L1687）字节列 == 实测字节，且与同行 SHA 指向**同一文件** | `193,551` / `a1f4772f…` **双双相符**（脚本按行解析表格并与 `os.stat` + 哈希现场比对） |
| `192,755` 出现位置 | 仅见于 **L1708–1709 历史留档**（历史名 `smoke.pt`，`e9c83f61…`，已随改名删除）与 **L1710–1712 修订留档（误沿用说明）**；**任何表格行都不再与 `smoke_nosyn.pt` 配对**；历史记录**未被删除** |
| README §19.2「修订留档（离朱 R23 发现 D1）」 | 存在，且写明“与同一行的 SHA256 自相矛盾”“已按现场实测更正为 **193,551 B**” |
| current_spec「本轮冒烟产物」表（L625） | 字节列 == 实测 193,551，SHA 相符；表格行不与 `192,755` 配对；D1 修订留档存在（L628–630） |
| current_spec「验收」节（L214 等） | 已为 **193,551 B** + SHA；`192,755` 仅出现在“字节数更正”说明句中，且句中明确归属**历史名 `smoke.pt`** |

### 2. D2 修复（失效表述就地标注）— 3/3 PASS

| 断言 | 实测 |
| --- | --- |
| README §12.2「**需风后决策的事项**」句（L607–608）后就地标注 | L609–612 标注含「**第 19 轮修订**」「该**历史轮次**的决策记录」「一期零改动 + 二期同批同步改造」「见 §19.3」，**原句保留未删** |
| 标注**未改变原决策项状态陈述** | 明写「本决策项的状态**未变**」，并保留 `_build_edge_groups` 注解与 **D2 排序口径**“仍不在本轮同步改造范围内” |
| 全仓无未标注的「二期零改动」类失效表述 | README 中所有涉及 `n3d_sphere` 且含 `零改动/零回归/不变/干净` 的行，均带就地标注（同行或下一行）；`n3d_proto`（一期）单独声明零改动属**现行有效**，不计 |
| current_spec fc_dim 节旧名表述 | 节内 `防静默覆盖 smoke.pt` 等旧名表述（L442）由**节尾 L562–564「[第 19 轮注（名字类记录同步）]」**覆盖解释（注明旧名 `_verify/smoke.pt` 自第 19 轮起已改名为 `_verify/smoke_nosyn.pt`） |

### 3. O1 披露（baseline.json 的 persistent 字段不可用）— 4/4 PASS

| 断言 | 实测 |
| --- | --- |
| README §19.1（L1652–1660）披露要素齐备 | 含 `baseline.json` / `buffer_meta[*].persistent` / **不可用** / 根因“判据写反（应为 `not in`）” / **未参与任何结论** / 结论来自 `after` 模式对 `_non_persistent_buffers_set` 与 `state_dict()` 的**直接复算** / 其余字段（键集合、逐张量 SHA256、参数量、forward 哈希）**均有效** / **保持原样不修改**；且标注来源「离朱 R23 观察 O1」 |
| current_spec「产物持久性（nosyn 口径）」节（L604–609） | 同口径披露齐备 |
| 措辞未夸大 | 全文无「已修复基线 / 基线已修复 / baseline 已更正 / 已重写/重建基线」；披露段无「已修复」修饰 `baseline.json`/`buffer_meta` |
| 取证文件未被事后修改 | `baseline.json` **48,007 B / `8ba1a70f…` / mtime `03:52:54`** 与上一轮记录一致；**5 个配置组的 `persistent` 字段仍全部为 `false`（原始瑕疵状态保留，未被“修好”）** |
| 未削弱原技术结论 | §19.1 仍声明“8 个张量逐位相同”“`edge_dist` 等 `persistent` 仍全部为 True” |

### 4. 回归（确认本轮只动文档）

**(a) 版本与只读（P1–P3）**
- P1 `model.py`/`train.py` SHA256 与 spec 给定值**逐位相同** → 本轮未改代码。
- P2 `baseline.json` 未被事后修改（见 §三-3）。
- P3 `probe_snapshot.py` **行为未变**：`after` 模式**退码 0**，且**输出与上一轮留档逐行完全相同**（`Compare-Object` 与脚本内字符串比对双口径一致）；其模块 docstring 已含 O1 瑕疵说明（“判据写反/瑕疵”）。即：该文件为 docstring-only 改动，探针判据逻辑未动。

**(b) CLI 回归**

| 命令 | 期望 | 实测 |
| --- | --- | --- |
| `python n3d_shape/train.py --smoke-test` | 16/16、退码 0 | **16 PASS / 0 FAIL，退码 0**；落盘产物 **193,551 B / `a1f4772f…`**，与 README §19.2 表内数字**完全一致**；日志仍含 `[产物格式] nosyn：8 个突触类张量…persistent=False、不进入 state_dict；edge_dist 与全部索引拓扑量仍持久化（n3d_viz 契约）` |
| `python n3d_shape/train.py --smoke-test --fc-dim -1` | 16/16、退码 0 | **16 PASS / 0 FAIL，退码 0**；产物 `smoke_shapesphere_N64_…_fc-1_nosyn_s42.pt` |
| `python -m compileall -q n3d_shape` | 退码 0 | **退码 0** |
| `python n3d_shape/verify_shape.py --quick` | 78/78、退码 0 | **断言总数 78；通过 78；失败 0，退码 0**（含 S12b 与二期 27 张量 `torch.equal`） |

**(c) 重跑上一轮全部套件（代码未变，结果应一致）**：单元 **27/27**、变异 **8/8**、基线逐位比对 **6/6**、产物往返/兼容性 **6/6**、`n3d_viz` 契约 **5/5**、文档一致性 **15/15** —— 全部退码 0。

**(d) `n3d_viz` 契约**：由 full 产物删去 8 键另存临时产物 → `python -m n3d_viz -c … -d … -o …` **退码 0**、FC 分支识别（`[FC] fc_dim=-1 H=256 [13,256]×[256,14]`）、HTML/PLY/OBJ 三件套齐备。

**(e) 浏览器 E2E（重跑，Playwright + 系统 Chrome）**：FC 产物 HTML 与 fc0 产物 HTML 各 **1 passed**（加载无 JS 错误、内联 JS 渲染 `#stats`、canvas 布局、阈值交互 372→13→372 与 43→0、刷新状态复位；FC 段有无与 `fc_dim` 一致）。

**(f) 受保护产物零变更（测试前后逐位不变，共 12 项）**：
`checkpoints/n3d_shape/full_shapesphere_N256_…_fc-1_s42.pt`、`checkpoints/n3d_shape/_verify/verify_2_shapesphere_N256_…_fc-1_s42.pt`、`checkpoints/n3d_viz/` 既有三件套（平铺版）与 `checkpoints/n3d_viz/full_shapesphere_…/` 子目录版三件套、`baseline.json`、`n3d_proto/model.py`、`n3d_viz/core.py`、`n3d_viz/__main__.py` —— 全部 **UNCHANGED**（旧格式产物仅被读取）。

---

## 四、失败用例

**无**。本轮 85 项自动断言全部通过。上一轮的 2 个根因（D1 字节数自相矛盾、D2 失效表述缺标注）已修复，且 O1 瑕疵已按口径披露（不修改取证文件）。

---

## 五、环境说明与已知不可复现项（非缺陷）

- 浏览器下载受限同上一轮：`npx playwright install chromium` 退码 1（`Failed to download Chrome for Testing 153.0.8010.12 … Download failure, code=1`），改用系统 Google Chrome 完成 E2E，**未跳过**。
- 历史台账缺失导致的三条命令（本轮复跑结果与上一轮一致，属**声明过的不可复现项**）：
  - `verify_full_runs.py` 退码 1 → `[FAIL] 账本不存在：…full_runs_shape.json`
  - `verify_ladder.py` 退码 1 → `[FAIL] 台账不存在：…ladder_runs.json`
  - `verify_fc_alignment.py` 退码 1 → `[FAIL] 台账不存在：…fc_alignment_runs.json`
- 仓库根 `_tmp_*.py` / `_tmp_*_out.txt` / `_verify_tmp/` 属同期 `n3d_sphere` 任务遗留，按 spec **未删除**。

---

## 六、复现命令

```powershell
$py = "C:\Users\wb3094\AppData\Local\Programs\Python\Python313\python.exe"
& $py .lizhu_env\n3d_shape_r23\test_r23fix.py                 # 18/18 PASS，退码 0（D1/D2/O1 + 版本与只读）
& $py .lizhu_env\n3d_shape_r23\test_docs_consistency.py       # 15/15 PASS，退码 0（上轮 11/15）
& $py .lizhu_env\n3d_shape_r23\test_unit_nosyn.py             # 27/27
& $py .lizhu_env\n3d_shape_r23\test_mutation_asserts.py       # 8/8
& $py .lizhu_env\n3d_shape_r23\test_baseline_before_after.py  # 6/6
& $py .lizhu_env\n3d_shape_r23\test_artifacts.py              # 6/6
& $py .lizhu_env\n3d_shape_r23\test_viz_contract.py           # 5/5
& $py n3d_shape\train.py --smoke-test                         # 16/16；产物 193,551 B / a1f4772f…（== README 表）
& $py n3d_shape\train.py --smoke-test --fc-dim -1             # 16/16
& $py -m compileall -q n3d_shape                              # 退码 0
& $py n3d_shape\verify_shape.py --quick                       # 78/78，退码 0
& $py checkpoints\n3d_shape\_verify\nosyn_probe\probe_snapshot.py after   # 退码 0，输出与上轮逐行相同
```

**测试资产**：`.lizhu_env/n3d_shape_r23/`（`test_r23fix.py` + 上轮 6 个脚本、`results_r23fix.json`、`r23fix_*.log`、保护基线 `protected_hashes_r23fix_before.txt`）、`.lizhu_env/pw/`（E2E 工程与 `r23fix_e2e_*.log`）。

---

## 七、总评

- **D1**：README §19.2 与 current_spec 的两处字节数已更正为与同行 SHA 自洽的 **193,551 B**（现场冒烟落盘后复核一致，3 次复跑稳定），并新增“修订留档（离朱 R23 发现 D1）”说明；历史名 `smoke.pt` 的 `192,755 B` 以历史留档形式保留。
- **D2**：README §12.2 决策句后已补「第 19 轮修订」就地标注，且明确声明**决策项状态未变**（D2 排序口径与 `_build_edge_groups` 注解仍不在同步范围）；全仓不再有未标注的二期零改动类表述。
- **O1**：两处文档均如实披露 `baseline.json` 的 `buffer_meta[*].persistent` 因探针首版判据写反而不可用、**未参与任何结论**、结论来自 `after` 模式直接复算、其余字段有效，并声明**取证文件保持原样**；实测 `baseline.json` 字节/哈希/mtime 与瑕疵字段状态**均未被改动**，`probe_snapshot.py` 行为未变。
- 代码与受保护产物零变更，全部回归（编译、冒烟、形状验证、`n3d_viz`、E2E）通过。**本轮判定：通过，无遗留问题。**
