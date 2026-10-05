# 离朱 R45 测试报告：n3d_qa_learn 第 6 轮（可插拔特征生成接口 + bge-m3 适配器 + 嵌入缓存）

被测模块：`n3d_qa_learn`（新增 `encoders.py` / `encoders_run.py`，改动 `train.py` / `cli.py` / `probe.py` / `backends.py` / `step2.py` / `step2_run.py`）

测试时间：2026-10-05（UTC），仓库工作副本 `E:\neuron3d`
测试脚本：`lizhu_r45_scripts/`（`verify_encoders_r45.py`、`verify_encoders_r45_hf.py`、`run_cli_r45.py`、`run_cli_extra_r45.py`）
结果 JSON：`lizhu_r45_scripts/_tmp/results_r45_groupA.json`、`results_r45_groupB.json`、`results_r45_cli.json`、`results_r45_cli_extra.json`

## 1. 测试概览

| 测试组 | 覆盖内容 | 通过 / 总数 | 结论 |
|---|---|---|---|
| A 组（不加载模型） | 注册表/维度声明、hash 家族构造期错配、编码边界、指纹与声明重建、嵌入缓存（纯 Python 部分）、步骤 1/2 接入点、CLI 无关的非法入参 | 68 / 71 | 2 项真缺陷（1 项含 2 个断言）+ 1 项不适用 |
| B 组（bge-m3 权重，本地离线） | HF 构造期维度错配、模型缺失/不全/离线可读报错、指纹敏感性与稳定性、声明重建、HF 编码边界、HF 缓存与 CachedVectorizer、boundary 自检 4/4、步骤 2 的 bge-m3（D=1024） | 30 / 31 | 1 项真缺陷（zh-bag 崩溃） |
| C 组（CLI 契约） | 第 35 项既有 CLI 回归（probe / drill index / drill pointer）、第 36 项 registry、第 37 项 boundary、第 38 项 verify-vectorizer | 8 / 9 | 1 项真缺陷（`--encoder zh-bag` 崩溃） |
| C 组补充（信息性） | `encoders_run drill/verify` 的 hash 家族路径 | 逐条登记（非门禁） | 1 项缺陷（drill zh-bag 崩溃）+ 1 项虚报 |
| 编译检查 | `python -m compileall n3d_qa_learn`（16 个 .py 全部通过 AST 解析） | 通过 | — |

合计断言：**106 / 111 通过**；失败 5 项 = 3 项真缺陷断言 + 2 项“不适用/口径已在说明中豁免”。

未执行（本仓库无对应载体，非遗漏）：
- 接口（HTTP）测试：本模块是 Python 库 + 进程内 CLI，无 HTTP 路由；
- E2E/Playwright 测试：无前台 UI；
- 类型检查（mypy/pyright）：仓库未配置（仅 `n3d_qa_learn` 源码，无 checks 配置），以 `compileall` 作编译检查；
- pytest：虚拟环境未安装 pytest（`No module named pytest`），且模块内无既有测试文件（已搜索 `test_*.py` / `*_test.py` / `tests/` / `conftest.py`，均无），故用自带断言运行器执行等价单元测试。

测试环境（实测）：`.venv\Scripts\python.exe` Python 3.12.10 / torch 2.14.1+cpu / numpy 2.5.3 / transformers 5.18.0；`models/bge-m3` 权重齐全（`pytorch_model.bin` 2271145830 字节，含 `1_Pooling/config.json`）。所有 HF 用例均 `--source models/bge-m3` + `local_files_only=True` + `HF_HUB_OFFLINE=1`，**未联网**。首个模型加载 7.1 s（进程内后续 3.7~4.1 s；进程首载外部观测约 37.7 s），B 组全程 109 s。

## 2. 缺陷（需修复）

### D1【高】`encoders_run boundary --encoder zh-bag` 直接崩溃，不出自检报告

- 复现：`python -m n3d_qa_learn.encoders_run boundary --encoder zh-bag --out-dir <临时目录> --cache-dir <临时目录>`
- 实测：退出码 1，**stderr 是裸 Traceback**，`boundary.json` 未写出：
  `AttributeError: 'ZhBagVectorizer' object has no attribute 'encode_with_stats'`
  （`encoders.py:1473`，经 `encoders_run.py:533 cmd_boundary`）
- 根因：`zh-bag` 实现 `step2.ZhBagVectorizer` 只暴露 `encode/encode_batch/encode_matrix`；`boundary_selftest` 无条件调用 `enc.encode_with_stats(...)`。`--force-cache` 时被 `CachedVectorizer` 包装才恰好有该成员（`CachedVectorizer` 对其缺失有回退），**默认（不加 --force-cache）即崩溃**。
- 影响：违反第 37 项“`--encoder <hash family>`”的可用性；与既有约定的“可读报错、不裸抛”不符。
- 同一根因的第二个入口（补充实测）：`encoders_run drill --encoder zh-bag`（`encoders_run.py:368`）同样 `AttributeError`，退出码 1、无 `drill.json`；`encoders_run drill/verify --encoder local-hash --force-cache` 正常（退码 0）。
- 建议：`boundary_selftest`/`cmd_drill` 统一走“能力探测”（`getattr(enc, 'encode_with_stats', None)`，缺失时回退 `encode` 并把 `n_tokens/n_truncated` 记为 0 且**标为不适用**），或在 `ZhBagVectorizer` 上补齐 `encode_with_stats`（`n_truncated` 恒 0，合理）。

### D2【中】`boundary_selftest` 对 hash 家族无“不适用”语义，且第 3 项检测失效

- 第 17 项 `boundary_selftest(cfg)` 要求 4 项且 `passed` 全 True。实测：
  - `local-hash`：4 项，`passed` = [True, True, **False**, True]；其中 `missing_model_readable_error` 的 `detail.raised == false`（没抛任何异常）。
  - `zh-bag`（加缓存）：4 项，`passed` = [True, **False**, **False**, True]；`overlong_text_truncated` 的 `n_tokens = n_truncated = 0`。
  - `bge-m3`（HF）：**4/4 全 True** ✔（D=1024、`max_length=512`、超长 1794 token → 截断 1282）。
- 根因 1：`boundary_selftest` 把 `source=<cache_dir>/_selftest_missing_model` 交给 `build_vectorizer`，而 **hash 家族分支忽略 `source`**（见 D3），于是"模型缺失"注定检测不到。
- 根因 2：`zh-bag` 家族**本就不做截断**（字符 n-gram 词袋无 `max_length` 语义），经 `CachedVectorizer` 回退 `encode` 后 `n_truncated == 0`，"超长必截断"对该家族恒不成立；说明第 37 项“对 hash 家族不适用（已知口径，不作为失败项）”在库层没有落实。
- 影响：hash 家族调用 `boundary_selftest` 必然返回 `passed=false` 项；CLI 因此退码 1（`[FAIL] G4 未通过：['overlong_text_truncated', 'missing_model_readable_error']`）。
- 建议：为每项增加 `applicable` 字段（与第 32 项 `check3.applicable` 同口径）：hash 家族将“模型缺失/维度不符”标为不适用、`zh-bag` 将“超长截断”标为不适用；`passed` 仅由适用项决定。

### D3【中】`build_vectorizer` 的 hash 家族分支完全忽略 `config.source`

- 复现：`build_vectorizer(EncoderConfig(name="local-hash", source="E:\nope\nowhere"))` → 正常返回 `TextVectorizer(dim=88)`；`zh-bag` 同理返回 `ZhBagVectorizer(dim=192)`。相对不存在路径同样不报错。
- 代码位置：`encoders.py:1328-1335`（只做 `_build_hash_encoder` + 维度对账，未消费 `source`/`local_files_only`）。
- 影响：说明第 8/9/10 项（模型缺失可读报错）对 hash 家族**无法覆盖**；D2 的“模型缺失”检测因此失效，属于“静默忽略入参”的不一致（HF 分支对同一入参正确抛 `EncoderUnavailableError`）。
- 建议：hash 家族对显式 `source` 报可读错误（或明确记 `note` 说明该入参对确定性词袋不适用），使“不适用”显式化而非静默。

## 3. 值得注意的非缺陷（如实登记）

### N1 `encoders_run verify --encoder local-hash --force-cache` 报 `check2` 失败（虚报）

- 实测：`passed=false`，`①指纹对账 True / ②重复编码逐位一致 False / ③落盘缓存逐位比对 True`，退码 1。
- 但 `check2` 记录的 `first_sha256 == second_sha256`（`2ca947cfc984b3e9…`，352 字节）**完全相同**，即第 32 项口径的“逐位一致”实际成立。
- 根因：`cmd_verify` 的 `check2 = (a_bytes == b_bytes) and (list(a) == list(b))`；`a`（首次实算）为 float64 精度，`b`（缓存命中）由落盘 float32 还原，故 `list(a) != list(b)`（88 维中 34 维有末位差，如 `0x1.08b742c0634e4p-2 vs 0x1.08b742p-2`），而 float32 字节完全一致。`encoders.py` 的 `CachedVectorizer`/`HFTextEncoder` 内部只用字节比对，故不受影响。
- 影响：CLI 在**真实通过**的情形下判失败（假阴性），可能误导验收。
- 建议：`check2` 只用 `E.vector_bytes` 比对（与库内一致），去掉 `list(...)` 弱比对。

### N2 `EncoderConfig` 对非法 `role` 无校验

`EncoderConfig(role="unknown-role").resolved_name()` 静默回退到 `question` 默认实现（`local-hash`）；`resolved_max_length()` 抛裸 `KeyError: 'unknown-role'`；`HFTextEncoder` 构造期则正确抛 `ValueError`。CLI 侧均有 `choices` 兜底，故列为健壮性提示：建议在 `EncoderConfig.resolved_name()/resolved_max_length()` 里对 `role` 做显式校验（可读报文）。

### N3 `EmbeddingCache.key_for` 不校验 `max_length`

`key_for(..., max_length=-1, ...)` 正常返回键（纯哈希计算）。`max_length` 的校验点在 `HFTextEncoder` 构造期（`>= 8`）。如希望缓存层独立防呆，可加最小校验。

### N4 缓存“首次未命中”会因跨进程复用而不成立（测试口径提示）

`CachedVectorizer` 的“首次 `cached=False`”只在条目不存在时成立。因为缓存设计目标就是跨进程复用，**任何历史运行写过的同名条目都会让首轮即命中**（本次即遇到：重复运行同一自检文本时首轮 `cached=true`）。这不是缺陷，但自检脚本应使用独立缓存根 + 唯一文本，否则会误判（本次已按此修正用例）。

### N5 错误报文中的路径是 `repr` 形式

`EncoderUnavailableError` 报文用 `{self.source!r}`，Windows 下路径显示为 `'lizhu_r45_scripts\\_tmp\\no_such_model_dir'`（转义反斜杠）。可读性与异常类型均正确（`EncoderUnavailableError`，非 `FileNotFoundError`），仅提示：若下游要按路径字符串匹配，需按 `repr` 口径处理。

### N6 测试对默认嵌入缓存的副作用（已如实登记）

`boundary_selftest` 用 `config.cache_dir`（默认 `checkpoints/qa_learn/_cache/emb`）做前两项探测，运行后会向该目录**追加**空文本/超长自检文本等条目（本次新增少量条目，均为产品自检文本口径，键完整、可复用）。测试期间我另行写入的 1 条探针条目（`f305d750…`，文本“N3D 嵌入缓存自检文本（HF 家族）”）已删除。除此之外**未触碰** `checkpoints/qa_learn/` 顶层 9 个 `qa_*.pt.zip`（尺寸与 mtime 实测未变，见第 5 节）。

## 4. 逐项验收结果（第 1~40 项）

| 项 | 结论 | 证据要点 |
|---|---|---|
| 1 注册表与维度声明 | PASS | `list_encoders()` = [bge-m3, local-hash, zh-bag]；bge-m3：kind=hf, expect_dim=1024, pooling=cls, revision=5617a9f6…, model_id=BAAI/bge-m3, weight_file=pytorch_model.bin；local-hash=88、zh-bag=192；未登记名 `KeyError` 且报文列出合法集合 |
| 2 `register_encoder` | PASS | 同名同内容幂等返回；同名不同内容 `ValueError`；`expect_dim=0`→`ValueError`；`kind='onnx'`→`ValueError`；失败后注册表不变 |
| 3 `assert_declared_dim` | PASS | strict=True 不符抛 `EncoderDimMismatchError`（报文含实测 768 与声明 1024）；strict=False 不抛；相符不抛 |
| 4 `EncoderConfig` 解析 | PASS | 空名按角色取默认（question→local-hash、text_line→zh-bag）；max_length 512/8192 及显式覆盖；`use_cache=None` 时 hf=True、hash(含 zh-bag)=False，显式覆盖生效 |
| 5 `declared_dim` | PASS | 默认 88；`use_length_features=False`→80；`hash_dim=100`→108；`expect_dim` 优先（77）；zh-bag=192；bge-m3=1024（question/text_line）；`hash_dim=128`（D=136>128）→`ValueError` |
| 6 HF 构造期维度错配 | PASS | `expect_dim=768` → `EncoderDimMismatchError`，报文同时给出 `hidden_size=1024` 与 `expect_dim=768` |
| 7 hash 家族显式错配 | PASS | zh-bag `expect_dim=190`、local-hash `expect_dim=99` → 均 `EncoderDimMismatchError`；相符时正常 |
| 8 `source` 指向不存在目录 | PASS（HF）/ 不适用（hash，见 D3） | HF：`EncoderUnavailableError` 且非泛型异常；hash：静默忽略 source（D3） |
| 9 本地目录缺文件 | PASS | 仅 `config.json` 时抛 `EncoderUnavailableError`，报出缺失 `['tokenizer.json|sentencepiece.bpe.model', 'pytorch_model.bin']` 与实际内容 |
| 10 `local_files_only=True` + 非本地 source | PASS | `EncoderUnavailableError`，报文含 `local_files_only`、source 与 `models\bge-m3` 提示 |
| 11 空/空白 → 零向量 | PASS | local-hash 与 bge-m3：`len==dim`、`max_abs==0.0`、`n_tokens==0`（`''`、`'   '`、`'\t\n '`） |
| 12 超长截断 | PASS | local-hash `max_length=16`：5200 token → 截断 4944，正常文本 0；HF `max_length=512`：20002 → 截断 19490，正常 17 全不截断 |
| 13 非 str → TypeError | PASS | `encode_with_stats(123/None)`、`encode(...)` 均抛 `TypeError` |
| 14 batch == 逐条且顺序保持 | PASS | zh-bag 与 bge-m3：float32 字节逐位一致，顺序保持 |
| 15 `encode_matrix` | PASS | zh-bag (4,192)、HF (4,1024)，dtype 均 float32 |
| 16 L2 归一化 | PASS | 非零向量范数 1.0000000（≤1e-4）；空白向量范数保持 0.0 |
| 17 `boundary_selftest` 4 项全 True | **部分失败** | HF：4 项全 True ✔；local-hash：3/4；zh-bag（无缓存）：**崩溃**；zh-bag（有缓存）：2/4（见 D1/D2） |
| 18 指纹格式与稳定性 | PASS | 64 位小写十六进制；同配置两次构造相同（`1877d80dbef3ece4…`） |
| 19 指纹对 max_length 敏感 | PASS | 512 vs 8192 指纹不同（`1877d80d…` vs `75220c7d…`） |
| 20 指纹对 expect_dim/权重敏感 | PASS | `expect_dim=768` 构造期即报错；`verify_weights=False` 时 `weight_sha256=unverified(...)`，指纹随之不同 |
| 21 `declaration()` 不含缓存设置 | PASS | `declaration()` 无 `use_cache`/`cache_dir`；`describe()` 含二者（另含 interface/weight_path/device 等） |
| 22 指纹对缓存开关不敏感 | PASS | 开启/关闭缓存及 `CachedVectorizer` 包装后指纹一致，`declaration()` 逐键相等 |
| 23 `encoder_from_declaration` | PASS | 指纹一致→可用（dim=1024、可编码）；不一致→`ValueError` 且给出两侧 16 位前缀；缺键→`ValueError` 并列出缺失键（revision/max_length/dim/pooling 逐项验证） |
| 24 `vectorizer_from_meta` 分派 | PASS | `kind="hf"` → 走注册表重建；无 `kind` → 委派历史路径（缺 `vectorizer_fingerprint`→`KeyError`，指纹不符→`ValueError`，正常→`TextVectorizer(dim=88)`） |
| 25 缓存键不变量 | PASS | 同字段同文本同键；`max_length`/`text`/`revision` 不同则键不同；64 位小写十六进制 |
| 26 put/get/损坏处置 | PASS | put 后 get 逐位相同；`read_bytes` = dim*4（16 字节 / HF 4096 字节）；不存在键→None；改字节后→None |
| 27 条目元数据字段 | PASS | 12 项齐全（key/dim/bytes/sha256/model_id/revision/pooling/max_length/normalize/weight_sha256/n_tokens/n_truncated/created_utc） |
| 28 `stats()` | PASS | root/n_entries/bytes，且 `n_entries` == 实际 `.bin` 数（3/3、15/15）；目录不存在→全 0 |
| 29 `CachedVectorizer` | PASS | 第二次 `cached=True`、两次 float32 逐位一致、指纹与内层一致、`cache_dir` 指向缓存根、未定义成员委派（`config`/`dim`）；HF 与 local-hash 均验证 |
| 30 步骤 1 默认档不变 | PASS | `build_training_data(cfg)` 与 `(cfg, encoder_name="")` 指纹相同（`d4a4881aaa8dfa8c…`）、dim=88、具备 encode/encode_batch/fingerprint |
| 31 步骤 2 入口 | PASS | `None` 与 `zh-bag` 均 dim=192 且指纹相同；bge-m3(text_line) dim=1024、matrix (2,1024) float32 |
| 32 `verify_vectorizer_contract` | PASS | 结构 4 键齐全；zh-bag（挂缓存）三项全 True；未挂缓存的 zh-bag：`check3.applicable=False`、`passed=True`（不把不适用当失败/通过）；空文本→`ValueError`；声明指纹不符→check1 False、passed False |
| 33 `backends.build_registry(input_dim=D)` | PASS | `input_dim=1024` → `registry.input_dim=1024`（三后端可用）；与 `VectorizerConfig` 冲突 → `ValueError`；一致（88）通过 |
| 34 `probe.end_to_end_drill` 维度门禁 | PASS | `dim=96` + `local-hash`（声明 88）→ `ValueError`；未知编码器→`KeyError`；`dim=88` 正常出结果（logits (8,5)、无零梯度） |
| 35 既有 CLI 回归 | PASS | `cli probe` 退码 0（3/3 后端可用）；`cli drill --output-mode index` 退码 0；`cli drill --output-mode pointer` 退码 0 |
| 36 `encoders_run registry` | PASS | 退码 0，stdout 含三个注册名，`registry.json` 的 `n_encoders=3`、维度 1024/88/192、角色默认与 max_length（512/8192）全对 |
| 37 `boundary --encoder <hash family>` | 部分失败 | local-hash：退码 1，4 项、空文本/超长 True、维度不符 True、模型缺失 False（不适用口径，非失败）；**zh-bag：崩溃（D1）**；bge-m3：4/4 PASS（HF 组已验证） |
| 38 `step2_run verify-vectorizer` | PASS | 退码 0；`verify_vectorizer.json` 的 `evidence.passed=true`，三项检查全 True，dim=192，n_texts=4，encoder_config.name=zh-bag |
| 39 非法入参 | PASS | 负 `hash_dim`→`ValueError`；未知注册名（`get_spec`/`declared_dim`/`build_vectorizer`）→`KeyError` 且列合法集合；非法 role → HF 侧 `ValueError`（`EncoderConfig` 侧见 N2）；`key_for` 负长度不校验（N3） |
| 40 不写顶层 checkpoint | PASS | 全部测试只写 `lizhu_r45_scripts/_tmp/`（及默认嵌入缓存目录的合法条目，见 N6）；顶层 9 个 `qa_*.pt.zip` 实测尺寸/mtime 未变 |

## 5. 产物完整性核对（第 40 项）

`checkpoints/qa_learn/` 顶层 `qa_*.pt.zip` 共 9 个，测试前后尺寸与 mtime 完全一致（41893/41904/41907/53915/60632/40898/40865/40946/33886 字节，mtime 分别为 1791183208/…/1791183548），未被覆盖。验证/取证文件一律写在 `lizhu_r45_scripts/_tmp/`（`cli_boundary/`、`cli_registry/`、`cli_verify_vectorizer/`、`cli_extra*/`）与各临时缓存根，未在 `checkpoints/qa_learn/` 顶层新增文件。

## 6. 环境问题与修复建议

本次**无环境阻塞**：解释器、torch/numpy/transformers、`models/bge-m3` 权重（含 `1_Pooling/config.json`）均就绪；HF 用例全部离线跑通。仅两点环境性事实需知晓：

1. 虚拟环境**未安装 pytest**（`python -m pytest` → `No module named pytest`），且模块内无既有测试文件，故采用自带断言运行器（等价单元测试）；如需以 pytest 形式复跑，先执行 `.venv\Scripts\python.exe -m pip install pytest`（属环境构建，需在 `.lizhu_env/` 内进行）。
2. 无 UI / 无 HTTP 服务，E2E 与接口测试不适用；`module_agent_testing(action="check_playwright")` 未触发（无前台页面可测）。

建议修复顺序：
1. **D1**（`boundary`/`drill` 对 `zh-bag` 崩溃）——影响 CLI 可用性，优先；
2. **D2**（`boundary_selftest` 引入 `applicable` 语义，落实第 37 项的“已知口径”）；
3. **D3**（hash 家族 `source` 静默忽略）——与 D2 同源，一并处理；
4. **N1**（`encoders_run verify` 的 `check2` 弱比对虚报）——单行改动即可消除假阴性。