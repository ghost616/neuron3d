# 离朱 R47 测试报告：n3d_qa_learn 第 7 轮（bge-m3 新特征下重跑 exp_repr 8 组矩阵 + 同切分对照）

被测模块：`n3d_qa_learn`（`exp_repr.py` 新增特征档/步骤 2 自检索/归因/对照报告；`exp_repr_run.py` 新增 `compare`；`encoders.py` 缓存 memo；`features.py` `encode_matrix`；`train.py` `encoder_config`）
测试时间：2026-10-05（UTC）；脚本：`lizhu_r47_scripts/`；结果：`lizhu_r47_scripts/_tmp/results_r47_*.json`

## 1. 结论：说明中 30 项全部通过

| 测试组 | 覆盖项 | 通过 / 总数 |
|---|---|---|
| A 契约/结构（不加载 HF 模型，`verify_r47_unit.py`） | 1,2,3,9,10,11,12,26,27,28,4(落盘),13-22,23,24,25,30 | **38 / 38** |
| B 真跑（`verify_r47_live.py`：真 bundle + 缩水版对照） | 5,6,7,8,12(现场),4/13/14/15/17/19(现场) | **13 / 13** |
| C 产物纪律复核（`_tmp/artifact_check.py`） | 29,30 | **6 / 6 断言** |
| 编译检查 | `compileall` + 全模块导入 | 通过 |

关键量（现场实测）：`build_step2_bundle(lexical-88)` → `n_pool=2665`（**不是 1999**）、`n_query=666`、`pool_limited=False`、`frozen_split.intersection=0`、键表 2665 行、`det_baseline Recall@1=0.9835 / @5=1.0`；`pool_cap=400` → `n_pool=400 / n_query=99 / pool_limited=True` 且不抛异常；`step2_recall_of_model` → `q_path.n=666`、`n_library=2665`，dim 错配抛 `ValueError`。

## 2. A. 特征档与切分一致性（items 1-4）

- **item 1** ✔ `FEATURE_PROFILES` 恰为 `lexical-88` / `bge-m3-1024` 两档；`profile_by_name("")` 回落 `DEFAULT_PROFILE == 'lexical-88'`；未知档名抛 `KeyError` 且报文含可用档列表。两档关键字段：词面档 `question=local-hash`、`text_line=local-hash`；语义档 `question/text_line=bge-m3`、`text_line` 生效 `max_length=8192`、`source=models/bge-m3`（Windows 现场为 `models\bge-m3`，规范化后一致）。
  说明中的 `question.max_length=512`：dataclass 字段为 `0`（0 = 用角色冻结口径），`resolved_max_length()` 与 `to_dict()["max_length"]` 均报 **512**（见第 5 节观察 1）。
- **item 2** ✔ `split_qids_digest` 返回 `per_split_sha256`（四键、各 64 位小写十六进制）与 `n_per_split`；`split_qids_equal` 对四子集有序比较，调换两条（含同长度换序）即 `False`。
- **item 3** ✔ `assert_same_split` 对某一条换序抛 `AssertionError`，报文含子集名 `train_unknown` 与首个不同位置 `i=0`；完全一致时不抛。
- **item 4** ✔ 落盘报告 `exp_repr_compare.json`：`split.cross_profile_checks` 恰 **8 项**（bge-m3 档 8 组逐组 vs 参照档），`identical_across_profiles is True`，8 项 `identical` 全 True。现场缩水版对照（8 组）也独立复现「8 项全 identical」。

## 3. B. 步骤 2 自检索接入（items 5-8）

| 项 | 实测 |
|---|---|
| 5 | `n_pool=2665`（≠1999）、`n_query=666`、`pool_limited=False`、`frozen_split={library 1999, query 666, intersection 0, union 2665}`、`dim==profile.expect_dim`(88)、键表 2665 行、`det_baseline` 含 `recall_at_1=0.9835`/`recall_at_5=1.0`；构建耗时 1.8 s |
| 6 | `pool_cap=400` → `n_pool=400`、`pool_limited=True`、`n_query=99(<666)`、键表 400 行、**不抛异常**（限批专用路径，0.5 s）；bge-m3 档限批（pool_cap=256）同样成立（`n_pool=256`、`dim=1024`） |
| 7 | `step2_recall_of_model(model88, bundle)` → `q_path.n == len(query_index)==666`、`q_path.n_library == 2665`、`det` 与 bundle 一致；换成 `dim=1024` 模型 → **`ValueError`** |
| 8 | **口径一致性（关键）**：`bundle.det_baseline` 与现场直调 `step2.deterministic_baseline`（同池/同查询/同键表）**逐键相等、容差 0**（`n / n_library / note / recall_at_1 / recall_at_5` 全等） |

## 4. C/D. 维度代价与归因（items 9-16）

- **item 9** ✔ `D=88`：`W_in [88,55]/4840`、`W_out [88,64]`、`backbone_total=10642`、`head_total=7833`、`total_parameters=18475`、answer_table buffer `[11,88]` 且 `trainable is False`、冻结档可训 `n_trainable==1`。
- **item 10** ✔ `D=1024`：`W_in [1024,55]/56320`、`W_out [1024,64]/65536`、`backbone_total=122026`、`head_total=1049601`、`total_parameters=1171627`、buffer `[11,1024]`、开表示档 `n_trainable_head==1049601`。
- **item 11** ✔ 两档 `topology` 的 `E/K/S_in/S_out` 完全相同 = `106 / 7 / 55 / 53`。
- **item 12** ✔ `params_per_sample` = `28.4669`(=18475/649) 与 `1805.2819`(=1171627/649)，容差 0.1 内。
- **item 13** ✔ 归因三节 `feature_only`/`training_only`/`combined` 各自独立且非空（落盘报告 1/4/2 条；现场缩水对照同口径 1/4/2 条）。
- **item 14** ✔ `feature_only[0]` 的 `group=='A1_baseline'`，五项 delta 与「其它档 A1 − 参照档 A1」**逐项相等（容差 0）**：`macro_acc` / `step2_recall_at_1` / `step2_det_recall_at_1` / `gap_over_sigma` / `refusal_rate`（现场复核同样成立）。
- **item 15** ✔ `training_only` 恰 4 条（两档 × `A1→A2`、`A1→A3`），且**同档内 A1→A2 与 A1→A3 的 delta 相同**（落盘报告与现场缩水对照均逐项相等，`raw` 口径下骨干不在计算图）。
- **item 16** ✔ `combined` 恰 2 条（bge-m3 档 `A2`/`A3` vs 参照档 `A1`，`reference="lexical-88/A1_baseline"`）。

## 5. E. 主判据与报告结构（items 17-22）

- **item 17** ✔ `verdict` 七键齐全；`len(both)+len(not_both)==n_groups`（8）；`any == bool(groups_both_better)`；`all == (n_groups>0 and not groups_not_both_better)`；现场缩水对照独立复现该恒等式（both=[A1_baseline, C2_staged, C3_supcon]）。
- **item 18** ✔ 每行含 `reference`/`others`；8 行逐行核对 `delta_macro_acc == others.metrics.macro_acc - reference.macro_acc`、`delta_step2_recall_at_1` 同式成立、`both_better == (Δmacro>0 and ΔR@1>0)`，**无一处不符**（容差 0）。
- **item 19** ✔ `report["runs"]` 两档、各 8 组；**16 组的 `gate.passed is True`、`gate.names_consistent is True`、`gate.missing_from_snapshot == []` 全部成立**；现场缩水对照的 16 组门禁同样全 PASS。
- **item 20** ✔ `cost` 含两档 `dimension_cost`（dim 88 / 1024）；`seconds.per_profile_total` / `per_group`（每档 8 组）/ `grand_total` 齐备，`grand_total ≈ Σ per_profile_total`（差 0，容差 1e-6）。
- **item 21** ✔ `render_comparison_markdown` 含标题「n3d_qa_learn 特征档对照实验」、「逐组对照（主判据…）」、「归因分解（…）」三子标题各 1 个（只换特征 / 只打开表示训练 / 两者叠加）、「D 的代价（现场构造实测）」、「CPU 耗时（秒）」。
- **item 22** ✔ `write_comparison_report(report, out_dir)` 写出 `exp_repr_compare.json` + `exp_repr_compare.md`（均**无 BOM**，UTF-8）并返回两路径；`load_comparison_report(json)` 读回后 `verdict` 与三节条数一致；缺文件抛 **`FileNotFoundError`**。

## 6. F. 零回归（items 23-28）

- **item 23** ✔ `run_group(A1_baseline)`（不传 profile）与 `run_group(..., profile="lexical-88")` 的 `metric_step1` / `refusal` / `geo` / `gate` **四项逐位相同**（JSON 规范化后相等，`profile` 字段同为 `lexical-88`）。
- **item 24** ✔ `exp_repr_run summary` 退码 0（锚点对账全落在容差内）；`cli probe` / `cli drill --output-mode index` / `cli selftest --model …` / `cli guard --model …` 全部退码 0，无 Traceback。
- **item 25** ✔ `encoders_run boundary --encoder zh-bag` 退码 0（`n_passed=3`、`n_inapplicable=1`）；`step2_run verify-vectorizer --encoder zh-bag --n-texts 4` 退码 0 且 `evidence.passed=True`。
- **item 26** ✔ 同一 key 连续两次 `get` 返回的数组**逐位相同**；`read_bytes(key)` 与 `get` 结果的 float32 序列化**逐字节相同**；`memo_stats()` 含 `memo_entries` / `memo_hits` / `disk_hits`（首读 `disk_hits==1`，二次 `memo_hits≥1`）。
- **item 27** ✔ `features.TextVectorizer.encode_matrix` 返回 `float32[5,88]`，与 `np.asarray([encode(t) …], dtype=np.float32)` **逐字节相同**，且覆盖空文本 / 仅空白 / 超长文本（空行仍为全零）。
- **item 28** ✔ `build_training_data(cfg)`、`(cfg, encoder_name="")`、`(cfg, encoder_config=EncoderConfig())` 三者 `vectorizer.dim` 均 88、指纹相同；`run_training` 签名含 `encoder_name`（默认 `""`）与 `encoder_config`（默认 `None`）两个可选参数。

## 7. G. 产物纪律（items 29-30）

`checkpoints/qa_learn/` 顶层 9 个 `qa_*.pt.zip` 的**字节数 / mtime / sha256** 与测试前基线完全一致、顶层条目集合无增删；`proxy_step1/` 4 个既有锚点 zip 尺寸未变（未新增任何 zip）；`git status --porcelain -- n3d_qa n3d_shape n3d_sphere n3d_proto` 为空；`requirements.txt` sha256 未变（`f69450d55d148a52…`）。`exp_repr_run drill --profile lexical-88 --out-dir <临时目录>` 退码 0，目录内**只有 `drill.json`，无任何 `.zip` / `.pt`**。现场报告 `_verify/exp_repr/exp_repr_compare.json` 保持原样（`n_groups=8`、`epochs=40`、`grand_total=617.1`，未被本次测试覆盖）。测试全部落 `lizhu_r47_scripts/_tmp/`。

## 8. 观察（不构成失败）

1. **`EncoderConfig.max_length` 字段语义**：档定义里 `question.max_length` 的**字段值**是 `0`（0 = 用角色冻结口径），说明 item 1 写的 `512` 是**生效口径**——`resolved_max_length()` / `to_dict()` / `HFTextEncoder` 构造期均报/用 512。全仓只有 `encoders.py:1414` 直接读该字段并原样透传给 `HFTextEncoder`（后者内部按角色解析），故无静默错配；仅提示「字段=0」与「文档=512」的表述差异。
2. **bge-m3 档的逐组开销**：`run_group` 内部按 `profile` 现构造编码器，故 bge-m3 档**每组都会重新加载一次模型权重**（现场 8 组 ≈ 600 s；词面档 8 组 ≈ 87 s）。这与说明里「缓存预热下全量 compare ≈ 10 分钟」一致，但若后续要频繁重跑，可考虑在 `run_experiment` 层复用同一编码器实例。非本轮缺陷。
3. `source` 字段在 Windows 现场存为 `models\bge-m3`（反斜杠），比较时需规范化；报告 `as_dict()` 里保留平台原样值。

## 9. 测试方法与环境

- 解释器 `.venv\Scripts\python.exe`（Python 3.12.10 / torch 2.14.1+cpu / numpy 2.5.3 / transformers 5.18.0）；全程离线（脚本内强制 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`），一律 `source=models/bge-m3`。
- HF 编码器构造 **2 次**（≤ 说明建议的 3 次）：bge-m3 档步骤 2 共用件（`pool_cap=256`，87 s）+ 缩水对照训练（600 s）；词面料用例不加载模型。
- 结构类断言优先使用**现场只读报告** `checkpoints/qa_learn/_verify/exp_repr/exp_repr_compare.json`（8 组 × 2 档、40 epoch 的真实产物），并对 items 4/13/14/15/17/19 额外做**缩水版真跑**（8 组 × 2 档、epochs=2）独立复现。
- 未执行（无对应载体）：HTTP 接口测试与 E2E（无路由、无前台 UI）、mypy（未配置）；虚拟环境无 pytest 且模块内无既有测试文件，故用自带断言运行器执行等价单元测试。

## 10. 证据文件

脚本：`lizhu_r47_scripts/verify_r47_unit.py`（38 断言）、`verify_r47_live.py`（13 断言）、`_tmp/artifact_check.py`（产物纪律）。
结果：`_tmp/results_r47_unit.json`、`_tmp/results_r47_live.json`、`_tmp/artifact_check.py` 输出。
只读依据：`checkpoints/qa_learn/_verify/exp_repr/exp_repr_compare.json` / `.md`。