# 离朱 R46 测试报告：n3d_qa_learn 第 6 轮**修复轮**（R45 缺陷 D1/D2/D3/N1/N2/N3 回归）

被测模块：`n3d_qa_learn`（`encoders.py` / `encoders_run.py` / `step2.py` 等）
上一轮报告：`test_reports/lizhu_n3d_qa_learn_encoders_r45_report.md`（本次为修复轮的独立复测）
测试时间：2026-10-05（UTC）；测试脚本：`lizhu_r46_scripts/`；结果：`lizhu_r46_scripts/_tmp/results_r46_*.json`

## 1. 结论：46/46 项全部通过

| 测试组 | 覆盖项 | 通过 / 总数 | 结果 |
|---|---|---|---|
| A 单元/契约（不加载模型，`verify_r46_unit.py`） | 4,5,6,7,8,13-17,20,21,22,23,24,25,26,29,30,31,32,33,34,35,36,46 | **68 / 68** | 全通过 |
| B CLI 契约（`run_cli_r46.py`） | 1,2,3,9,10,11,18,19,37,38,39,40,41,42,43 | **17 / 17** | 全通过 |
| C HF/模型权重（`verify_r46_hf.py`，离线） | 12,27,28,30(指纹),44 | **8 / 8** | 全通过（44 项首轮为测试脚本解析缺陷，修正后 2/2 通过） |
| D 产物纪律（`final_baseline_r46.py`） | 45,46 | **4 / 4** | 全通过 |
| 编译检查 | `compileall` + 全模块导入 | 通过 | — |

**R45 六项缺陷全部修复并经独立复现确认**（见第 2 节），上一轮已通过项**零回归**（见第 3 节）。

## 2. R45 缺陷回归（逐项证据）

| 项 | 上轮缺陷 | 本次实测 | 判定 |
|---|---|---|---|
| **D1-1** | `boundary --encoder zh-bag` 崩 AttributeError、无报告 | 退码 **0**，stderr **无 Traceback / 无 AttributeError**，`boundary.json` 落盘，`n_cases=4 / n_applicable=3 / n_inapplicable=1 / n_passed=3`，不适用项 = `['overlong_text_truncated']`，stdout `[OK] G4：边界处置自检 3/3 PASS（共 4 项）` | ✅ 修复 |
| **D1-2** | `drill --encoder zh-bag` 崩溃 | 退码 **0**，`drill.json`：`dim=192`、`cache_file_written=true`、`second_bitwise_equal=true`、`second_cached=true`，`[OK] G2：D=192…二次读取逐位一致` | ✅ 修复 |
| **D1-3** | — | `drill --encoder zh-bag --reuse-only` 退码 **0**（新进程、不加载模型）：`phase=reuse`、`cache_hit=true`、`bytes=768`、`matches_first_run=true`，且 `vector_sha256` 与冷相位完全一致 | ✅ |
| **D1-4** | `ZhBagVectorizer` 缺 `encode_with_stats` | 返回对象含 `vector/n_tokens/n_truncated`，`n_tokens==0 and n_truncated==0`（如实口径），`vector` 与 `encode()` **逐字节一致**；非 `str` → `TypeError`；空文本 → 零向量 | ✅ |
| **D1-5** | — | `encode_with_stats_of(最小实现, text)`：对**只有** `encode` 的对象走回退路径，`n_tokens/n_truncated==0`；非 `str` → `TypeError`；对原生实现走原生路径（字节与 `encode_with_stats` 一致） | ✅ |
| **D2-6~8** | 无 `applicable` 语义 | 每项都含 `applicable` 与 `passed`；`summarize_selftest` 返回 4 键且 `n_cases == n_applicable + n_inapplicable`、`n_passed` **只统计 applicable&passed**（定点用例 {4 项混合} → `n_passed=1`）；`failed_selftest_cases` 只返回 applicable&!passed（定点 `["c"]`） | ✅ |
| **D2-9** | zh-bag CLI 退码 1 | `boundary --encoder zh-bag` 退码 **0**，`n_cases=4 / n_applicable=3 / n_inapplicable=1`，不适用项名 = `overlong_text_truncated` | ✅ |
| **D2-10** | local-hash 有 False 项 | 退码 **0**，`n_applicable=4 / n_passed=4 / n_inapplicable=0`，`inapplicable_cases=[]` | ✅ |
| **D2-11** | — | `boundary --encoder zh-bag --no-cache` 退码 **0**，3/3 | ✅ |
| **D2-12** | — | `boundary --encoder bge-m3 --source models/bge-m3 --local-files-only` 退码 **0**，`n_applicable=4 / n_inapplicable=0 / n_passed=4`（4 项全 applicable&passed）；`missing_model` 明细含 `requires_model_files=true` | ✅ |
| **D3-13/14** | hash 家族静默忽略 source | `local-hash` / `zh-bag` + 不存在目录 → **`EncoderUnavailableError`**（报文含"hash 家族…不使用模型文件"与 `bge-m3` 引导） | ✅ |
| **D3-15** | — | + **存在**目录（`models/bge-m3`）→ **`EncoderError`**（报文：不使用模型目录、请清空 source 或改用 HF 编码器）。注：`EncoderUnavailableError` 是 `EncoderError` 子类，本项由实现**主动选择**抛基类 | ✅ |
| **D3-16** | — | `source=""` → `local-hash` dim=88、`zh-bag` dim=192 正常构造（`declared_dim` 一致） | ✅ |
| **D3-17** | 模型缺失检测失效 | 两家族 `boundary_selftest` 的 `missing_model_readable_error` 均 `passed=true`，`detail.type=EncoderUnavailableError` | ✅ |
| **N1-18** | `verify --encoder local-hash` 假阴性退码 1 | 退码 **0**；`check2.passed=true` 且 `first_sha256 == second_sha256`（`2ca947cfc984b3e9…`，352 字节）；`check1=true`、`check3.applicable=true&passed=true`、`passed=true` | ✅ |
| **N1-19** | `--force-cache` 误报 | 退码 **0**，`passed=true`，`check2` 两侧 sha256 相同 | ✅ |
| **N1-20** | list 相等判据 | 挂缓存的编码器「冷→热」两次 `encode`：`E.vector_bytes` 相等（`list` 相等为 False，证明确实不再依赖列表相等）；`step2.verify_vectorizer_contract` 的 check2 `passed=true` 且含 `n_bytes` | ✅ |
| **N2-21** | 非法 role 静默回退 | `EncoderConfig(role="bogus")` → **`ValueError`**（报文列合法角色并说明"未知角色会静默回退默认实现，故构造期即拒绝"） | ✅ |
| **N2-22** | — | `hash_dim=0` / `hash_dim=-3` / `max_length=-1` → 均 `ValueError`；`max_length=0` 合法（角色口径） | ✅ |
| **N3-23** | key_for 不校验 | `max_length=0` → `ValueError`；`model_id=""` / `pooling=""` / `normalize=""` → 均 `ValueError`（报文说明空值会让不同口径共用条目） | ✅ |
| **N3-24** | — | `revision=""` **不报错**且键为 64 位小写十六进制；`text` 变则键变 | ✅ |

## 3. 零回归复跑（上一轮已通过项）

| 项 | 内容 | 结果 |
|---|---|---|
| 25 | 注册表：三名字；未知名 `KeyError` 且报文含合法集合；`bge-m3` expect_dim=1024 / pooling=cls / revision=40 位固定提交号；新增 `truncation_stats`（local-hash=True、zh-bag=False、bge-m3=True）与 `requires_model_files`（仅 bge-m3=True）如实登记 | ✅ |
| 26 | `declared_dim`：默认 88 / text_line 192 / bge-m3 1024；`use_cache_effective`：bge-m3 True、两 hash 家族 False | ✅ |
| 27 | HF 构造期维度错配：`EncoderDimMismatchError`，报文含 `hidden_size=1024` 与 `expect_dim=768` | ✅ |
| 28 | HF 三类可读报错（不存在目录 / 缺文件并列出缺失名 / `local_files_only` 非本地）均 `EncoderUnavailableError` | ✅ |
| 29 | 编码边界：空/空白→零向量且 `len==dim`（两家族）；超长→local-hash `n_truncated>0`（4500→截 4244）、zh-bag 如实 0；非 `str`→`TypeError`；`encode_batch` 与逐条逐字节一致；`encode_matrix` float32 `[4,192]`；L2 范数 1.000000 | ✅ |
| 30 | 指纹 64 位十六进制且稳定（`d4a4881aaa8dfa8c…`）；512 与 8192 不同（HF）；开关缓存不改指纹（含 `CachedVectorizer` 包装）；`declaration()` 不含 use_cache/cache_dir 而 `describe()` 含；`encoder_from_declaration` 指纹不符 → `ValueError`（给两侧前缀）、缺键 → `ValueError`（列缺失键）、非 hf kind → `ValueError` | ✅ |
| 31 | `vectorizer_from_meta`：`kind="hf"` 走注册表；无 kind 委派历史路径（缺键 `KeyError`、指纹不符 `ValueError`、正常重建 dim=88） | ✅ |
| 32 | 嵌入缓存：键不变量（max_length/text/revision 敏感）；put/get/read_bytes；损坏 `.bin` 后 `get→None`；元数据 12 键齐全；`stats()` 与磁盘一致 | ✅ |
| 33 | `CachedVectorizer`：二次 `cached=True` 且逐位一致；指纹与内层一致；`cache_dir` 返回缓存根；dim 委派 | ✅ |
| 34 | `build_training_data(cfg)` 与 `(cfg,"")` 一致且 dim=88；`build_step2_vectorizer(None).dim==192`；bge-m3(text_line) 声明 1024；未挂缓存时 `check3.applicable=False` 且 `passed=True` | ✅ |
| 35 | `build_registry(input_dim=192).input_dim==192`；与 `vectorizer_config` 冲突 → `ValueError` | ✅ |
| 36 | `end_to_end_drill(88, encoder="bge-m3")` → `ValueError`；`(96, encoder="zh-bag", role="text_line")` → `ValueError` | ✅ |
| 37-41 | `encoders_run registry` / `cli probe` / `cli drill index` / `cli drill pointer` / `cli selftest --model …`（自检 7/7）/ `cli guard --model …`（3/3 注入被拒）全部退码 **0** | ✅ |
| 42 | `step2_run verify-vectorizer --encoder zh-bag --n-texts 4`（无 `--force-cache`）退码 **0**、`evidence.passed=true`、`check3.applicable=false`；加 `--enc-force-cache` 退码 **0**、`check3.applicable=true&passed=true`（check2 两侧 sha256 相同） | ✅ |
| 43 | `step2_run probe --out-dir …` 退码 **0**；`probe.json`：`dim=192`、`feature_check.bag_dim=192`、`feature_spec_hash` == `ZhBagVectorizer().fingerprint()`、`rows_over_atol=0` | ✅ |
| 44 | `encoders_run info --encoder bge-m3 --source models/bge-m3 --local-files-only` 退码 **0**，`info.json`：`hidden_size=1024`、`max_position_embeddings=8194`、`pooling="cls"`、`weight_sha256` 64 位十六进制、revision 固定；`verify --encoder bge-m3` 退码 **0**、`passed=true`、`check2` 两侧 sha256 相同（4096 字节）、dim=1024 | ✅ |
| 45 | 顶层 9 个 `qa_*.pt.zip` 的**字节数 / mtime / sha256** 与测试前基线完全一致；`checkpoints/qa_learn/` 顶层无新增/删除条目 | ✅ |
| 46 | `git status --porcelain -- n3d_qa n3d_shape n3d_sphere n3d_proto` 为空；`requirements.txt` sha256 与基线一致（`f69450d55d148a52…`） | ✅ |

## 4. 观察（不构成失败，供决策）

1. **hash 家族 + 显式 `--source` 的组合**：`boundary --encoder zh-bag --source models/bge-m3` 会走"编码器不可用"兜底分支，4 项 `applicable=true, passed=false`，退码 1，报文可读（`EncoderError: hash 家族编码器 'zh-bag' 不使用模型目录（source='models/bge-m3' 存在但无意义）…`）。这是**配置错误下的 fail-closed**，行为可接受；唯一不够精确处是该兜底分支把 `zh-bag` 的 `overlong_text_truncated` 也标成 `applicable=true`，与第 9 项的 applicable 口径略有出入。建议（非阻塞）：兜底分支按 `spec.truncation_stats` / `spec.requires_model_files` 标 applicable。
2. **第 15 项的异常类型取舍**：实现选择"不存在目录 → `EncoderUnavailableError`（与 HF 同形，使第 17 项成立）、存在目录 → 基类 `EncoderError`"。说明中已明确允许（"不含 EncoderUnavailableError 语义上也无妨"），实测符合。
3. **统一接口存在两种既有结果类型**：`local-hash` 原生返回 `features.VectorizeResult`、`zh-bag`/HF 返回 `encoders.EncodeResult`。`encode_with_stats_of` 只依赖契约面（`vector`/`n_tokens`/`n_truncated`），故安全；若下游想 `isinstance` 判定需注意这一点（本次测试已按契约面断言）。
4. **默认嵌入缓存目录的既有条目**：本次所有 CLI 均显式传 `--cache-dir <临时目录>`；默认缓存（`checkpoints/qa_learn/_cache/emb`）中 02:13:32 / 02:13:34 的两条条目来自**修复执行方**的验证脚本（`_verify/encoders/fix_*.log`），不是本次测试写入。本次测试期间该目录**无新增**（最新条目时间戳仍为 02:13:34，早于本次测试开始的 02:31）。

## 5. 环境与方法说明

- 解释器 `.venv\Scripts\python.exe`：Python 3.12.10 / torch 2.14.1+cpu / numpy 2.5.3 / transformers 5.18.0；`models/bge-m3` 权重齐全（`pytorch_model.bin` 2271145830 字节，含 `1_Pooling/config.json`）。
- **全程离线**：所有用例在脚本内强制 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`（首次尝试未设时，一处 HF 调用在本机触发 huggingface.co 连接超时重试 >3 分钟，属本机网络限制而非产品问题；设置后 HF 用例全部走本地目录，单次模型加载 43~46 s，HF 组全程 188 s）。
- 未执行（无对应载体）：HTTP 接口测试（模块为库 + 进程内 CLI）、E2E/Playwright（无前台 UI）、mypy（仓库未配置）。虚拟环境未安装 pytest，且模块内无既有测试文件，故使用自带断言运行器执行等价单元测试。
- 测试写入范围：全部产物落在 `lizhu_r46_scripts/_tmp/`（boundary/drill/verify/registry/info/probe 的 JSON 与临时缓存）；未写 `checkpoints/qa_learn/` 顶层。

## 6. 证据文件

- 脚本：`lizhu_r46_scripts/verify_r46_unit.py`（A 组）、`run_cli_r46.py`（B 组）、`verify_r46_hf.py` + `rerun_r46_hf_44a.py`（C 组）、`baseline_r46.py` / `final_baseline_r46.py`（D 组）
- 结果：`lizhu_r46_scripts/_tmp/results_r46_unit.json`（68/68）、`results_r46_cli.json`（17/17）、`results_r46_hf.json` + `results_r46_hf_44.json`（8/8）、`results_r46_baseline.json`（4/4）、`baseline_r46.json`（测试前基线）