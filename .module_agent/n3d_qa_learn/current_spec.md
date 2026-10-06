N3D 问答学习框架模块（n3d_qa_learn）。

【目标】把「问答对」学习为「问题 → 匹配学过的答案」的闭集匹配任务，并支持泛化推理：对未见问法 / 多个问题合并的输入，最终推理出的答案与原答案一致即视为学习成功；对「正确答案不在已学答案集合内」的问题显式输出「不相关」。不做自回归生成。

【连接契约代理层】只承担「数据集 → 模型」的连接参数职责：把数据侧的特征维 D 与答案类别数 C 映射为后端 N3D 结构的构造参数（Config 装配），并以注册表形式支持对接不同结构的 N3D 后端；后端模型通过只读 import 各模块的 model/config 取得，不修改任何上游模块。

【数据装配】加载 n3d_qa 产出的问答对产物，构建统一答案空间（全局答案表）、训练/测试切分、多问题重叠集，以及与推理端共用的确定性文本向量化。

【学习与推理】自建训练循环（交叉熵 + 显式不相关类 + 负样本/未学过样本处置），提供推理 API：问题文本（含 choices）→ 学过的答案文本 或「不相关」。

【评估】三套评估集：主测试集（同答案留出问法）、多问题重叠集、泛化改写集（力牧生成并冻结）；按题型分别报告答案一致率与不相关类的 F1 与占比偏差；跨 3 个 seed（42/43/44）报告均值与极差。

【范围】首轮落地五类任务：Math1 判断题、Math1 选择题、Math1 填空题（答案≥5 次子集）、Math1 解答题（答案≥5 次子集）、TriviaQA top-100 答案类；框架的数据层与答案空间需保持通用，便于后续接入其他 QA 数据集。
## 开发状态

### 状态：步骤 1 骨架已实施并完成验收（准确率项未稳定达标，已如实登记）

本模块的**步骤 1 骨架**已落地为可运行代码（11 个文件，见 module_definition），
并完成全部验收项的**真实执行**：

| 验收项 | 实测 |
| --- | --- |
| `python -m compileall -q n3d_qa_learn` | **退码 0** |
| P0 探针 `... cli probe` | **退码 0**，`3/3` 后端可用，`unavailable = {}` |
| 单条端到端演练 `... cli drill`（index / pointer / n3d_proto） | **三组均退码 0**，`zero_grad_params = []` |
| 加载守卫拒绝证明 `... cli guard` | **index 3/3 被拒**；pointer **2/2 被拒**（第 3 项显式标注跳过） |
| 边界处置自检 `... cli selftest` | **7/7 PASS、退码 0** |
| 步骤 1 准确率（`seed=42,43,44` 同超参） | **未稳定通过**：`macro = 0.30 / 0.20 / 0.10`，**均值 0.200 恰等门槛 0.200**，单 seed 3 选 2 |
| 上游零改动 | `git status --porcelain -- n3d_shape n3d_sphere n3d_proto` → **空** |

**口径声明（不得当作能力引用）**

- **步骤 1 准确率的判定口径是"同命令同超参跑三 seed、按均值判定"**，故本项判为
  **未稳定通过**，而非"三选二即通过"。根因经实测定位为**数据规模**（训练侧每类仅 6~22 条、
  主测试集仅 20 条），而非实现缺陷 —— 同一批特征同一批切分下 **1-NN 宏平均准确率**也只有
  `0.25 ~ 0.375`。后续达标方向是**增加每类训练样本量**，不是继续调模型。
- **「`macro >= 10/C`（随机基线 x10）」这条门槛恒不可满足**：准确率 `<= 1` 要求 `C <= 10`；
  本批 `C = 10` 时阈值恰为 `1.0`，`C > 10` 时阈值 `> 1`。验收实现把该条标为
  `applicable = False` 并**如实报出**，真正生效的是「`>=` 多数类基线 + 10pp」与
  「`>=` 多数类基线 + 5pp」两条。
- **指针 Softmax 模式：结构成立、准确率不可用**（同配置 `macro = 0.05`）。根因实测为
  候选键（答案展示文本的确定性编码）彼此几乎不可分，`final_loss = 2.871 ≈ log 11 = 2.398`
  的均匀分布交叉熵。如实登记，不当作可用路径。
- **`n3d_shape` 与 `n3d_sphere` 在 `shape="sphere"` 同配置下数值同构**（参数量 `10642`、
  `E = 106` 逐位相同），二者**只靠类身份区分**；可区分性断言因此用
  「名 / 类 / 参数量 / `E`」四元组而非三元组。
- **并行会话占用**：`checkpoints/qa_learn/` 顶层已被另一个并行会话写入
  （`dataset/` / `step2/` / `_snapshot/`，时间戳 12:21~14:45），`n3d_qa/` 的
  `git status` 改动同属该会话、**不属于本批**。为避免互相覆盖，本批产物写在
  `checkpoints/qa_learn/proxy_step1/` 子目录，`n3d_qa_learn/` 之外**未新增任何目录**。
- 本批**未实现**：多问题重叠集、泛化改写集（力牧生成并冻结）、Math1 多任务适配。

### 已确认并实施的核心口径（细则见 `n3d_qa_learn/README.md`）

- **业务路由固定三级**：步骤 1（QA 数据集 `top-1`，命中即返回、**不再查文本**）→
  步骤 2（文本数据集词袋余弦）→ 「无匹配」。**命中判定 = 输出「不相关」类即视为未命中**；
  路由结果一律携带**来源标记**（`qa` / `text` / `none`）与**分数**。
- **连接契约代理层**只承接特征维 `D`（接到后端 `input_dim` / `output_dim`），
  **不承载训练编排**、不干预后端内部超参；三后端**只读 import**，上游零改动。
  两条实测约束：① `n3d_shape` 的 `Config` 带数据集通用层，自定义维度**必须**显式给
  `dataset="npz"`（占位维语义，不加载数据）；② `K`（递推层数）二期 / 三期**没有**
  `num_layers` 属性，须由 `neuron_pos[:, flow_axis_index]` 去重计数现场算出。
- **两种候选打分实现（全局开关，不做级联）**：`index`（候选键表 `A in R^[C+1, D]`，
  末位固定「不相关」）与 `pointer`（候选键 `K in R^[B, L, D]` **由输入提供**，
  `hasattr(model, "answer_table")` 为假）。两模式**共用同一个 `q` 头**。
- **候选键表默认 `centroid` 口径**（训练样本逐类质心、L2 归一化、固化为 buffer）：
  同规模下 `free` 口径（自由可学习）会退化为"一律拒绝"。该口径不改变
  「答案表 → 候选键 → `q @ A^T`」的打分结构，只改候选键的来源。
- **`q` 的取值口径默认 `raw`**（原始 `D` 维文本特征）；实测 `concat`（与 N3D 读出凸混合）
  与 `n3d`（只用 N3D 读出）均显著更差。即 N3D 后端在本框架里是**被挂载的特征提取器**，
  其读出向量不是本任务主判据 —— 这一点**必须**按实测读，不得按直觉反转。
- **`q` 头默认冻结**（`train_head=False`），仍有真实可训练参数 `logit_scale`
  （不影响 `argmax`），故训练循环真实执行；打开 `--train-head` 会退化为"一律拒绝"。
- **`D` 的规模纪律**（历史纠正记录 #12）：默认 `D = 88`（`hash_dim = 80` + 8 个长度特征），
  强制 `<= 128`。
- **产物自包含 + 确定性**：自写 zip（`meta.json` + `model_state_dict.pt`
  + `answer_table.pt`），zip 条目时间戳固定 `1980-01-01`，同参数重复运行**逐字节一致**；
  `meta` 的 `D` 落点三处（`dim` / `backend.input_dim` / `model.dim`）。
- **两道加载守卫**：答案表指纹（答案键顺序 + 展示文本 + 「不相关」位 + 候选键张量字节）
  与向量化口径指纹（归一化链 + 哈希盐 + 维度 + 长度特征口径 + 词元上限），
  任一不一致**立即报错**（拒绝证明见 `... cli guard`）。

## 文本数据集匹配与分项验收

步骤 2（两级业务路由的第二级：文本数据集匹配）由 `n3d_qa_learn/step2.py` 与 `n3d_qa_learn/step2_run.py` 承载，驱动脚本为 `python -m n3d_qa_learn.step2_run {probe|drill|eval|guard}`。

**候选来源（全局开关二选一，与 heads 的 output_mode 同一开关）**
- `index`（索引式）：候选键表由**库行文本的确定性特征**构造、**冻结不参与梯度**，固化进本模块自写 zip 产物（`meta.json` + `key_table.pt`），加载时执行「键表 SHA256」与「特征口径 spec_hash」两道守卫，篡改即拒。
- `pointer`（指针式）：候选键表 `[B, L, D]` 由输入提供，产物内不存固定候选表。
- 两模式共用同一个训练出的 `q`；走同一条 q 通路（后端特征 → q 头）后做 `q @ K^T` → softmax → top-1 行原文。

**冻结口径**：连接参数 `D=192`（`n3d_qa` 中文特征口径，桶数 64/阶 × 阶数 (1,2,3)，只取词袋块，与产物 `doclines_rows.jsonl` 的 `feature` 同 salt / 同归一化）；文本行、库/查询划分、统一答案表与五个任务（judge/choice/blank/solve/triviaqa）全部只读消费 `n3d_qa` 冻结产物（`checkpoints/qa_learn/dataset/`）。

**自检索留出法（步骤 2 主判据）**：产物冻结的查询行（666）经同一 q 通路在检索池中检索，命中自身行即正确，报告 `Recall@1/@5`；同时报告**不经 N3D** 的纯确定性特征检索 `Recall@1/@5` 作为**参照下限**（不作门槛）。

**开集协议**：因 `n3d_qa` 冻结产物只导出已入表答案的正样本，本模块以**按任务分层的留出类**构造显式「不相关」类（留出比例与 seed 冻结，留出类清单落 SHA256，不随运行重抽）。

**分项 5 项指标**：① 步骤 1 答案准确率 ② 步骤 2 `Recall@1` ③ 路由正确率（命中/未命中判定）④ 无匹配类精确率与召回 ⑤ 端到端最终答案正确率；按任务分别产出，并按 seed 42/43/44 报**均值 ± 极差**。跨 seed 的极差**只含训练随机性**（三个后端 Config 由代理层统一给出且后端自身 seed 恒为 42，故拓扑在所有运行中相同）。

**验收入口**：`probe`（划分复核 + 特征重算 + 留出明细，不训练）、`drill`（单条端到端演练：梯度非零硬门禁）、`eval`（跨 seed × 双模式全量）、`guard`（产物守卫拒绝证明）。实测数字、未达标项与不可复现项统一登记在 `n3d_qa_learn/README_step2.md` 与报告产物中。
**产物格式与可复算（第 2 轮修复后）**
- 模型产物 zip 的成员为 `meta.json` + `model_state_dict.pt`（头与缓冲）+ **`backbone_state.pt`（N3D 骨干）**。原因：`N3DQA.adapter` 是普通 Python 对象而非 `nn.Module`，`N3DQA.state_dict()` **不含骨干权重**（实测仅 5 个键），只存 `state_dict()` 会产出「加载后骨干随机初始化」的假产物；加载侧对缺 `backbone_state.pt` 的产物**直接报错拒绝**，不静默出数。
- `meta` 必带 `head_config`（`head_input_mode` / `normalize_query` / `answer_table_mode` / `logit_scale_init` / `learn_logit_scale` / `mix_logit_init`）与 `source_manifest`（本模块 + `n3d_qa` 全部 `.py` 的 SHA256）；报告同落这两项，用于回答「这份数字是哪一版源码跑出来的」。
- `q` 头输入口径由本模块**显式固定**为 `head_input_mode="concat"`（`step2.DEFAULT_HEAD_INPUT_MODE`），**不吃上游默认值**：上游默认 `raw` 会让 N3D 骨干整体脱离计算图（骨干参数恒零梯度）。
- 第五个验收入口 `replay`：只读加载落盘产物（键表 + 双模式模型）→ 严格重建模型 → 复算自检索 `/` 双模式一致性 `/` 按任务分项 5 项指标 → 与 `step2_report.json` 逐项对账并断言全匹配，用于证明「训练后量可由产物复算」。
**第 3 轮（审查问题修复）**
- **口径分离**：`③ 路由正确率` = 宽松口径（`source == "qa"` 即算命中，不校验答案对错）；`⑤ 端到端最终答案正确率` = **严格口径**（已知题须命中**且**返回答案文本等于金标展示文本），恒有 `⑤ ≤ ③`。另新增到达诊断 `step2_reached_total`（落到文本匹配分支的样本数）。
- **路由候选空间不变量**：`cmd_eval` 在构造期硬断言 `len(router.answer_keys) == model.n_answers`、`router.irrelevant_index == model.answer_index()`、`model.output_dim == irrelevant_index + 1`；现场实测三者同宽（230 / 230 / 231），路由一律使用**模型候选空间（保留类表）**而非统一答案表全量。
- **报告可复算**：`eval` 先写正式目录取得路径 → 回填 `report_paths` → 重写正式报告并写验证目录副本；指标指纹排除 `created_utc` **与** `report_paths`（后者含指纹自身，参与即自指循环），`replay` 读到的 `report_metrics_sha256` 不再为空。
- **日志编码**：所有子命令支持 `--log-file`，由 Python 以 UTF-8（无 BOM）自写；禁用 PowerShell `Tee-Object`（PS 5.1 默认 UTF-16LE）。
- **双模式一致性**：报告 Markdown 渲染层按 seed 遍历嵌套键并输出逐用例表，同时 `cmd_eval` 写扁平汇总键，保证 JSON / MD / README 三处一致（修复此前恒渲染 `None`）。
- 废弃轮次（`smoke` / `smoke2` / `_smoke3`）与脚手架文件分别归档到 `_verify/step2/_deprecated/` 与 `_verify/step2/legacy_scaffold/`。
**第 4 轮（离朱 R42 非阻断项修复）**
- `replay` 的**输入 / 输出语义已分离**：新增 `--report-dir` 指定对账基准 `step2_report.json` 所在目录；`--verify-dir` 只作输出目录。未指定 `--report-dir` 时按 `[--report-dir] → [--verify-dir] → [--out-dir] → [正式产物目录]` 顺序自动回退，并在日志首行打印实际采用的基准报告路径。修复前 `--verify-dir` 兼作输入语义，导致在全新空目录首次执行必然 `FileNotFoundError` 退码 1（离朱 R42 现场复现）。
- 因 `step2_run.py` 变更，已重跑 `eval`（`eval_full_v7`）刷新产物与报告内的 `source_manifest` 源码指纹，并重跑 `replay --full-metrics`（24/24、`all_match=True`）与 `guard`（`all_tampered_rejected=True`）；`details` 与产物 SHA256 同步刷新到 `README_step2.md`。
- **产物 SHA 稳定性口径（如实说明）**：产物 `meta.json` 含 `created_utc`，故同参重跑**整包 SHA256 会变**；逐成员 `key_table.pt` / `model_state_dict.pt` / `backbone_state.pt` 在同参重跑间逐字节一致。与步骤 1 产物在 `README.md` 第七节的如实口径同源。
## 表示训练对照实验

对照实验入口由 `n3d_qa_learn/exp_repr.py`（核心）与 `n3d_qa_learn/exp_repr_run.py`（CLI）承载，驱动脚本为 `python -m n3d_qa_learn.exp_repr_run {drill|run|summary}`。目的是把「默认档下唯一可训参数是 `logit_scale`、而正标量缩放不改变 argmax ⇒ 训练不改变任何预测」这一结构事实变成可量化结论，并量化「表示训练」本身对类内/类间可分性（gap/σ）的影响、定位/修复 `train_head=True` 的「一律拒绝」退化解。

**切分 seed 与训练 seed 分离（`TrainConfig.split_seed`）**
- 新增字段 `split_seed`，默认 `-1` = 沿用 `seed`（历史行为，逐位不变）；唯一取用点是 `train.split_seed_of(cfg)`，`build_training_data` 用它驱动 `make_splits`。
- `cli.py train` 新增 `--split-seed`（默认 `-1`）；`cmd_eval` 以 `.get("split_seed", -1)` 向后兼容读取旧产物 meta；`artifact_name` 仅在 `split_seed >= 0` 时追加 `_sp<k>` 后缀，默认档文件名逐字不变。
- 现场实测：默认档与显式 `split_seed=42` 切分逐位相同；显式 `split_seed=43` 在同样规模下给出不同切分（`train_known` 交集 132/149）。

**被测维度与对照矩阵（`exp_repr.MATRIX`，8 组，逐维单独测、每次只换一项）**
- 维度 a 表示冻结范围 = {`logit_only`（基准）/ `head` / `head_backbone`}；维度 b 头输入口径 = {`raw`（基准）/ `concat` / `n3d`}；维度 c 目标修法 = {`ce`（基准，落在退化态 `head` 上）/ `no_irr_centroid` / `staged` / `supcon`}。
- 所有组共用同一份切分（`split_seed` 固定），并逐位断言 `train_known` / `test_known` / `train_unknown` / `test_unknown` 的 qid 有序序列相同（`assert_same_split`），不一致即该对照判无效。

**指标口径（每组必给）**：`gap`/`σ`（评估池 = `train_known + test_known`，样本与自身类质心余弦为 within、与其余类质心余弦均值为 cross，`σ = std(within, ddof=1)`；**转导诊断量，不参与任何选择**）、`1-NN(raw)`（确定性原始特征，锚点口径）与 `1-NN(repr)`、`macro`/`top1`、`refusal_rate`、以及位移量（`q` 位移、答案表位移、`argmax` 变化率）。

**逐 epoch 退化诊断**：记录 `cos(q, 答案表末位不相关键)`、`cos(q, 金标类质心)`、`cos(q, 冻结参考方向 = train_unknown 原始特征归一化质心)`、训练侧拒绝占比与 known top-1 的逐 epoch 轨迹。现场实测**否证**了「把所有 q 推向不相关质心」这一根因假设（A2 的该余弦从 +0.5960 **净下降**到 +0.1710；**口径如实登记：该序列不是逐 epoch 严格单调** —— 40 个 epoch 内 11 次小幅回升，最小值 +0.1349 出现在 epoch 23，epoch 15 之后在 +0.13~+0.20 的窄带内震荡，净方向明确向下约 −0.425），并给出可观测的替代机制：`cos(q, 金标类)` 下降更快（+0.4566 → −0.0063，首个穿零点在 epoch 18，最小值 −0.0616 在 epoch 37），且 `cos_irr` 在**全部 40 个 epoch 上都高于** `cos_gold`（无符号翻转，最小差 +0.1261 出现在 epoch 12），叠加训练侧损失质量被「不相关」类主导（known 权重质量 38.468 vs unknown 500.0，即 7.14% : 92.86%）。

**门禁与结构事实**：每组必须通过「可训参数更新量 > 0」断言（不通过则该组判无效），同时校验 `names_consistent`（`missing_from_snapshot` 为空）与「`requires_grad=True` 但不进优化器的参数名必须落在显式允许名单 `ALLOWED_UNGROUPED_TRAINABLE = ('head.mix_logit',)` 内」。现场实测两条结构事实：① `head_input_mode="raw"` 下 `adapter.features` 从不被调用，故 `head_backbone` 与 `head` 的全部指标逐位相同、4 个骨干参数全程零更新；② `concat` 口径的 `mix_logit` 是 `nn.Parameter` 却从不进优化器（恒为初值）。

**结论归因（不得合并）**：维度 b（只换特征、不打开训练）把 `gap/σ` 从 0.8249 降到 0.4323/0.4248、`macro` 从 0.4500 降到 0.1000（随机基线）、`refusal_rate` 降到 0；维度 a（只打开训练、不换特征）把 `gap/σ` 抬到 2.2097，但 `macro` 掉到 0.2500、`refusal_rate` 升到 0.9298 —— 即 **gap/σ 的提升与实际可用性相反**。三种修法（`no_irr_centroid` / `staged` / `supcon`）没有一种能在保持拒绝能力的同时把 `macro` 抬到基线之上（`no_irr_centroid` 得 `macro 0.5000` 但 `refusal recall = 0.0008`，开集口径不可用）；如实给出负结果：在当前词面特征上表示训练无法把 gap/σ 抬到可用量级。

**零回归口径（审查收口后更正，**不得再引用旧的「顶层零改动」表述**）**：实验产物一律写 `checkpoints/qa_learn/_verify/exp_repr/`（`exp_repr_report.json` / `exp_repr_report.md` / `drill.json` / 日志 / `calibration_sweep.py` / `stray_report_forensics.md`），`exp_repr_run` 的训练路径不落盘任何 zip 产物；`n3d_qa` / `n3d_shape` / `n3d_sphere` / `n3d_proto` 源码与产物零改动。**如实声明一处例外**：本批的离朱 R44 接口回归曾以 `python -m n3d_qa_learn.cli train ...`（**未加 `--verify`**）执行，因 `cli.py` 的 `out_dir = DEFAULT_VERIFY_DIR if args.verify else DEFAULT_ARTIFACT_DIR`，`_write_report` 把报告写到了 `checkpoints/qa_learn/` **顶层**，产生副产物 `report_n3d_shape_index_s.json`（1793 B，SHA256 `53d1d410fc9e17e550c4d177c0454a83e1b381036f074ad666456ea05efa1949`，内容为 `artifact = E:\neuron3d\.lizhu_env\r44\cli_tmp\sp42.pt.zip`、`epochs = 1`、`step1.macro_acc = 0.3125` 的接口回归报告，**不是正式产物**）；该文件已按审查意见**删除**，取证留档见 `checkpoints/qa_learn/_verify/exp_repr/stray_report_forensics.md`。既有 9 个 `qa_*.pt.zip`（LastWriteTime 均 ≤ `2026-10-05 14:59:08`）与 `_verify/` 下既有文件**始终未被写入或覆盖**（删除前后两次 sha256 + 字节数 + mtime 快照逐项一致，顶层差异仅该 1 个副产物）。`run` 全量矩阵的耗时为 **273.1s**（现场从 `checkpoints/qa_learn/_verify/exp_repr/run_full.log` 的 `[run] 全部组完成（273.1s）` 回读，逐组 32.8+34.3+35.3+36.9+34.8+32.8+33.1+33.2 = 273.2s 与之相符）。模块级依赖清单唯一落点是 `n3d_qa_learn/requirements-qa.txt`（当前零新依赖），根 `requirements.txt` 属 `framework` 模块管辖，QA 侧依赖不写入根文件。

**审查收口登记（第 5 轮，纯文档更正 + 清理 1 个杂散产物；不重跑训练/矩阵、不产生新实验数字）**
- **error（杂散副产物）**：删除 `checkpoints/qa_learn/report_n3d_shape_index_s.json`；删除前已记录 SHA256、字节数、CreationTime/LastWriteTime 与内容摘要（含 `artifact` 路径与 `step1.macro_acc`），删除后快照与差异一并落档于 `_verify/exp_repr/stray_report_forensics.md`。
- **warning（耗时数字）**：README 12.12 的 `265.6s` 系更早一次被覆盖的全量运行数字，已更正为现场日志实测的 `273.1s` 并注明来源（`run_full.log` 末段 `[run] 全部组完成（273.1s）`）；`265.6` 已不再作为当前值出现。
- **info ①（未使用形参）**：`setup_answer_table` 的 `cfg` 形参在函数体内从未使用，**选择「删除形参」**并同步唯一调用点（`run_group`，全仓现场枚举仅此 1 处）；理由：保留未使用形参会造成「答案表取值依赖训练配置」的误导性签名，而该函数实际只由 `data` 与 `loss_mode` 决定。docstring 已登记该决定及现场枚举结论。
- **info ②（PEP8 空行）**：`exp_repr.py` 两处模块级定义之间补足为 2 个空行（仅空白变更，无任何语义改动）。
- **info ③（`drill` 的 epochs 口径）**：**选择「显式标注」**而非暴露 `--epochs`；理由：`drill` 的职责是「通路成立 + 与 `run_training` 逐位等价」这条门槛，多 epoch 只增加运行时间与误读空间，矩阵档预算由 `run --epochs` 承担。标注落在运行日志首行（`epochs = 1（演练口径，非矩阵档）`）与子命令 `--help`/`description` 文本中。
- **info ④（`param_deltas` 前缀前提）**：docstring 已登记前提 —— `head.` / `backbone.` 前缀仅当 `N3DQA` **不把骨干持有为 `nn.Module` 子模块**时成立（现状下两次 `state_dict()` 遍历键集合不相交、不重名）；若日后 `adapter.model` 被登记为子模块，键会重名且 `make_optimizer` / `ungrouped_trainable_names` / `update_gate` 的名字口径会失配（门禁会把真实更新的参数判为 `missing_from_snapshot`），届时必须同步改口径。
**口径澄清（对上一段 warning 项的补充，避免 `265.6` 被误当作当前值）**：本登记（及 README 12.12）中出现的 `265.6` **一律是历史值引用**（来自一次已被覆盖的早期全量运行），**当前值为 `273.1s`**，来源为 `checkpoints/qa_learn/_verify/exp_repr/run_full.log` 的 `[run] 全部组完成（273.1s）`（逐组 32.8+34.3+35.3+36.9+34.8+32.8+33.1+33.2 = 273.2s）。README 全文 `265.6` 出现次数为 **0**（已直接更正为 `273.1`），本 spec 中的出现均带历史值标注并同处给出 `273.1s`。
**第 2 期：bge-m3 语义特征（D=1024）下的 8 组矩阵与同切分对照（`exp_repr_run compare`）**

- **特征档注册表**（`exp_repr.FEATURE_PROFILES`）：`lexical-88`（词面：`local-hash`，D=88）与 `bge-m3-1024`（语义：`BAAI/bge-m3`，`D=hidden_size=1024`，`source=models/bge-m3`）。每档同时声明**步骤 1 题面口径**（`role=question`，`max_length=512`）与**步骤 2 文本行口径**（`role=text_line`，`max_length=8192`），二者**必须同维**，否则同一条 q 通路不可用。`FeatureProfile` 是唯一注册点，`profile_by_name` 未知名立即报错。
- **`run_comparison`**：两档共用同一 `split_seed`/`train_seed`，逐组跑 8 组矩阵；跨档逐组断言四子集 qid 有序序列逐位相同（不一致抛 `AssertionError` 并判该对照无效）；每档的步骤 2 共用件（文本行向量化 + 冻结键表 + 纯特征参照下限）只建一次并复用于该档全部组。
- **步骤 2 自检索接入**（`build_step2_bundle` / `step2_recall_of_model`）：口径与 `step2_run eval` **逐字一致** —— **检索池 = `n3d_qa` 冻结行表全量**（2665，`label_rule`: candidate library row IS the query row），查询集 = 冻结划分出的 query 行 ∩ 检索池（666）；查询行经**同一条 q 通路**检索、命中自身行即正确；另报**不经 N3D** 的纯特征余弦检索作为参照下限。`Step2Bundle` 携带 `pool_index`/`query_index`/`key_table`/`det_baseline`/`evidence`；限批（`pool_cap>0`）时走 `_self_retrieval_limited`（口径不变，只换池下标来源），报告显式登记 `pool_limited`。
- **维度代价实测**（`dimension_cost`）：现场构造骨干与 `N3DQA`，逐参数枚举形状与元素数，给出 `W_in`/`W_out`/骨干合计/`q` 头合计/总参数/每样本参数/`answer_table` buffer 字节，以及**两种情形的可训参数量**——「冻结嵌入 + 质心答案表」（`logit_only`，可训恒为 `head.logit_scale` 1 个，`answer_table` 是 buffer 不入优化器）与「打开表示训练」（`head` / `head_backbone`）。
- **归因分解**（`_attribution`）：**三节分开报** —— 只换特征（`A1_baseline` 跨档，该组只训 `logit_scale` 这个不改 argmax 的正标量，故差异只来自特征）、只打开表示训练（同档 `A1→A2`/`A1→A3`）、两者叠加（其它档 `A2`/`A3` vs 参照档 `A1`）。
- **现场实测（train_seed=42，epochs=40）**：G1 组内 16 组 + 跨档 8 项切分一致性断言**全部通过**；G2 词面档重跑与 12.6 登记值**在 4 位小数上逐项精确一致**（MATCH×8），锚点对账 `all_within_tolerance=True`（语义档下为 False，属**正确行为**：锚点只对词面口径标定）；G5 **16 组门禁全部 PASS**（`names_consistent=True`、`missing_from_snapshot=[]`），`A3` 的 4 个骨干参数在两档下都零更新、`B1` 的 `head.mix_logit` 在两档下都落在显式允许名单内。
- **主判据（macro 与步骤 2 自检索 Recall@1 同时更优）**：`any_group_both_better=True`、`all_groups_both_better=False`，**4/8 通过**（`A2_head` / `A3_head_backbone` / `C2_staged` / `C3_supcon`），4/8 不通过（`A1_baseline` / `B1_concat` / `B2_n3d` / `C1_no_irr_centroid`）。**「只换特征」最强的 `A1_baseline` 不通过**：`Δmacro +0.0500` 但 `ΔR@1 −0.0075`。`B1`/`B2` 语义档下同时退化（`macro 0.1000→0.0000`、`R@1 0.0015→0.0000`、`refusal 1.0000`）为**真实负结果**；`C1` 反向（`ΔR@1 +0.3438`、`Δmacro −0.0500`）。
- **D 的代价**：`W_in [88,55]→[1024,55]`（4840→56320）、`W_out [88,64]→[1024,64]`（5632→65536）、骨干 10642→122026、`q` 头 7833→1049601、总参数 18475→1171627（63.42×）、每样本参数 28.5→1805.3（训练样本 649）；骨干拓扑 `E/K/S_in/S_out = 106/7/55/53` **完全相同**（与 D 无关）。
- **CPU 耗时**：词面档 287.5s、语义档 329.5s、两档总计 **617.1s**（只含矩阵本身）；一次性嵌入编码（2449 题面 + 2665 文本行）落 `checkpoints/qa_learn/_cache/emb/` 跨进程复用 —— 冷缓存单组 1 epoch 演练 1556.8s，缓存预热后同组 ~34~43s。
- **零回归**：`checkpoints/qa_learn/` 顶层 9 个 `qa_*.pt.zip` 的 SHA256/字节/mtime 逐项不变；`git status --porcelain -- n3d_qa n3d_shape n3d_sphere n3d_proto` 为空；`compare` 路径**不落盘任何 zip 产物**（全部落 `checkpoints/qa_learn/_verify/exp_repr/`）。
- **未达标项（如实登记）**：① 主判据未全组通过；② 12.13 第 1 条的「同时重建答案表」**未做**（两档答案表口径完全相同，12.9(1) 的空间错配风险未被排除）；③ 步骤 2 文本行口径为与步骤 1 同维而改用 `local-hash`/`bge-m3`，**不可**与 README_step2 的 `zh-bag` 自检索数字直接比较；④ 固定 `train_seed=42` 单 seed，Δ 不含训练随机性区间。
**第 8 轮：N3D 结构开关（形状 / 全连接层 / 容量 N / 几何权重场）× 对齐机制 的学习对照实验**

**口径（逐轮确认，不得擅自变更）**：逐轴单换（非全交叉）；公共基线 = `head_input_mode="concat"` + `freeze_scope="head_backbone"` + 结构默认 + 对齐关闭（矩阵组 `S0_base`）。被测轴：① 载体组（`S0_base` / `B1_concat` / `B2_n3d`）② 形状（`sphere` / `cube` / `cylinder λ=1.0`）③ `fc_dim`（`0` / `-1` / `128`）④ `N`（`64` / `256`）⑤ `geo_field`（`none` / `additive`）⑥ 对齐（关闭 / 机制 B=投影头+SupCon / 机制 C=显式蒸馏 λ∈{0.1,1.0}）⑦ 特征档（`lexical-88` / `bge-m3-1024`，由 `compare --profiles` 承载）。对齐损失 = `ce + λ·align`；投影头 = 线性 `D→D`（`proj_dim=-1` 跟随 `D`）。主判据 = 步骤 1 `macro` 与步骤 2 自检索 `Recall@1` **同时**优于公共基线（**相对 Δ 为主**，绝对量只作可用性附加判定）。单 seed（`split_seed=42`/`train_seed=42`），**所有 Δ 为单点差、无跨 seed 极差**。

**新增对外接口**
- `backends.BackendStructure`（frozen dataclass，字段 = `shape` / `cyl_aspect` / `fc_dim` / `N` / `y_in` / `y_out` / `geo_field`，含 `is_default` / `to_kwargs` / `as_dict` / `from_dict`）、`STRUCTURE_DEFAULTS` / `STRUCTURE_AWARE_BACKENDS`（现场枚举 = `("n3d_shape",)`）/ `SHAPE_CHOICES` / `GEO_FIELD_CHOICES`；`recommended_config(name, input_dim, structure=None)` 与 `BackendRegistry(input_dim, structure=None)` / `build_registry(..., structure=None)` 透传结构开关。`BackendAdapter.structure` 与 `structure_manifest()`（**`describe()` 的键集合保持不变**）。默认档（`structure=None` 或全默认）下三后端 `Config` 构造实参与改动前逐字符相同、`describe()` 与参数量逐位一致；对不支持结构开关的后端传非默认结构**显式报错**（不静默丢弃）。
- `train.TrainConfig` 新增 7 个结构字段（`structure_shape` / `structure_cyl_aspect` / `structure_fc_dim` / `structure_N` / `structure_y_in` / `structure_y_out` / `structure_geo_field`）与 4 个对齐字段（`align_mode` / `align_lambda` / `proj_dim` / `proj_init`），全部进 `to_dict()`；新增唯一装配点 `train.structure_of(cfg)`；`_build_model` 透传结构 + 对齐；`build_meta` 新增顶层键 `backend_structure`；`rebuild_model` 一律用 `.get(..., 默认)` 读 `backend_structure` 与对齐字段（旧产物向后兼容）。
- `heads`：`ALIGN_MODES = ("off", "proj_supcon", "distill")`、`PROJ_INIT_MODES`、`PROJ_INIT_SEED`、`ALIGN_TARGETS`，`SUPCON_TEMPERATURE` 与 `supcon_loss` 的**唯一实现上移到 heads**（`exp_repr` 只做再导出，保证机制 B 与 `loss_mode="supcon"` 不可能漂移）。`N3DQAConfig` 新增对齐四字段与构造期不变量（`off` 时禁止投影头与 λ≠0；开启时要求 `head_input_mode ∈ {concat, n3d}`；`proj_dim>0` 必须等于 `D` 且只属于机制 B）。`N3DQA` 新增可选线性投影头 `self.proj`（`nn.Linear`，在 `torch.random.fork_rng` 内构造 ⇒ **不消耗全局 RNG**；用局部 generator 固定种子初始化，**不复用 q_head 的近恒等先验**）、`n3d_branch(features, readout=None)`、`alignment_representation`、`alignment_parameters()`、`alignment_loss(...)`（机制 B = supcon；机制 C = 0.5·[(1−cos(z, 本样本编码器特征)) + (1−cos(z, 批内所属类质心，detach))]，两项分别可报），`query(features, n3d_branch=None)` 新增可选的已算支路表示（让 `query` 与对齐损失共用同一次骨干前向）。**关闭对齐时不创建任何模块、不消耗 RNG、`query()` 路径逐字符不变**。
- `exp_repr`：`ReprGroup` 扩展 11 个字段（7 结构 + `align_mode`/`align_lambda`/`proj_dim`）与 `structure()` / `structure_is_default()` / `align_is_off()`；`MATRIX` 扩到 **18 组**（既有 8 组逐字不动 + `S0_base` / `S1_shape_cube` / `S2_shape_cylinder` / `S3_fc_follow` / `S4_fc_128` / `S5_N256` / `S6_geo_additive` / `S7_align_B` / `S8_align_C_l01` / `S9_align_C_l10`）；`DIMENSIONS` 扩到 9 条轴（新增 `d_shape`/`e_fc_dim`/`f_capacity_N`/`g_geo_field`/`h_align`/`i_carrier`）；`CARRIER_GROUPS` / `AXIS_FIELDS` / `custom_group(...)`（CLI 逐轴单换入口）；`construction_probe` / `precheck_constructibility`（可构造性预检 + 失败显式登记）；`effective_rank`（容差口径显式固定为 `σ > σ_max·1e-6`，不中心化；报有效秩 / 参与比 `(Σσ)²/Σσ²` / 占 `R^D` / 占样本数）/ `EFFECTIVE_RANK_REL_TOL` / `alignment_degree`（N3D 支路表示与答案表各类质心的平均余弦：逐类 + 总体 + 「不相关」行单独报；另报对齐目标 (i)(ii)）/ `spectral_diagnostics`；`batch_loss` 签名扩展（`feats` / `n3d_branch` 关键字参数）并**返回含分项的字典**（`loss`/`base`/`align`/`parts`），对齐关闭时路径逐字符不变；`apply_freeze` / `make_optimizer` / `ungrouped_trainable_names` 适配投影头（随 `head` / `head_backbone` 档可训，**允许名单 `ALLOWED_UNGROUPED_TRAINABLE` 无需扩充**）；`param_deltas` / `update_gate` / `dimension_cost` 按现场 `named_parameters()` 枚举处理 `fc_dim≠0` 的参数名替换（**不假设 `W_in` / `W_out` 存在**）；`run_experiment` 新增 `groups=`（显式组定义）与 `construction_precheck` / `construction_failures` / `axis_attribution`；`run_comparison` 新增 `groups=` / `axis_attribution`（**逐轴六节分开归因**：形状 / FC / 容量 N / 几何权重场 / 对齐 / 载体，禁止合并）/ `construction_failures`；报告新增「可构造性预检与失败清单」「逐轴归因」「对齐度与有效秩」「单 seed 声明」四节（单档版与对照版都有）。
- `exp_repr_run`：新增轴参数 `--carrier` / `--shape` / `--cyl-aspect` / `--fc-dim` / `--struct-N`（别名 `--N`）/ `--y-in` / `--y-out` / `--geo-field` / `--align-mode` / `--align-lambda` / `--proj-dim`；**给出任一轴参数即只跑 1 个自定义组**（自 `S0_base` 逐轴单换），一个都不给时行为与历史逐位一致；`drill` 支持新轴并新增**对齐专项门禁**（对齐项必须参与 ≥1 个 batch 且全部有限；此时"与 `run_training` 逐位等价"如实记 `applicable=false`，因为 `run_training` 只优化交叉熵）；`run` / `compare` 在存在构造失败组时**退码 1**（不以成功状态落账）。
- `README.md` 新增第十四节（14.1 口径 / 14.2 矩阵 / 14.3 全矩阵实测 / 14.4 主判据与逐轴 Δ / 14.5 对齐度与有效秩 / 14.6 参数量与耗时 / 14.7 边界可行域 / 14.8 门禁 G1~G8 / 14.9 未达标项 / 14.10 命令与零回归）。

**现场实测（单 seed，`epochs=40`）**：`compare` 18 组 × 2 档，跨档切分一致性 18 项全通过、构造失败 0 项、36 组门禁全 PASS；词面档 648.5s、语义档 867.9s、合计 1516.4s（`compare` 全流程 1557.7s）；单独 `run` 全 18 组词面档 635.1s。**主判据逐轴通过 = `[]`（空集）**：公共基线 `S0_base` 本身落在"一律拒绝"退化态（`refusal ≈ 0.92`、步骤 2 自检索 `R@1 = 0.0000`，两档都是 0.0000），而 11 个逐轴变体的 `R@1` **全部为 0.0000** ⇒ 该轴上不可能有变体满足"`R@1` 严格更优"。逐轴 Δ（相对 `S0_base`，单点差）：换形状 cube 在 bge 档 `Δmacro +0.0500`（相对 +14.3%）但 `ΔR@1 0`，cylinder λ=1.0 两档均不优于基线；加 FC（`-1` / `128`）**两档 `macro` 全部下降**（lexical −0.0500、bge −0.2500）且 `gap/σ` 大降（−1.02~−3.13）⇒ 明确负贡献；扩容量 `N=256` 把有效秩从 44 抬到 87（lexical，`D=88` 饱和）/147（bge，占 `R^1024` 14.36%）但 `Δmacro = ΔR@1 = 0.0000`；几何权重场 `additive`（仅 +14 参数）在 bge 档 `Δmacro +0.0500`、`ΔR@1 0`、`gap/σ` 反降；对齐机制 **C（显式蒸馏）把对齐度从 −0.0277/−0.0742 抬到 +0.5480/+0.5862（lexical）与 +0.9033/+0.9391（bge）**，但 `ΔR@1` 恒为 0，机制 **B（投影头 + SupCon）在词面档直接退化**（`macro 0.0000`、`refusal 1.0000`），bge 档 `macro 0.1000`（低于基线 0.3500）⇒ **"对齐度"与实际可用性方向不一致**（与 12.9 的 `gap/σ` 结论同族，且机制 C 同时把「表示 ↔ 不相关键」余弦也抬到 0.46~0.82，类间区分度未被拉开）。绝对量（附加判定）：`macro` 变体最优 0.4000（bge 档 `S1`/`S6`/`S8`，基线 0.3500），步骤 2 `R@1` 变体最优 0.0000（= 基线本身）。既有 8 组的 legacy 主判据仍为 4/8（`A2_head`/`A3_head_backbone`/`C2_staged`/`C3_supcon`），与上一轮登记**逐字一致**。

**新增诊断量（口径必须显式）**：有效秩 = `#{σ > σ_max·1e-6}`（**不中心化**，池 = `train_known + test_known` = 169）；参与比 = `(Σσ)²/Σσ²`。现场复核风后现状口径：`sphere/fc=0/N=64/D=1024` 下**有效秩 = 44、占 `R^1024` = 4.30%**（与登记的"44 / 4.3%"逐位对上）。**如实登记的未复现项**：风后现场值「参与比 25.9」无法用本实现的口径复现（实测 6.47；另试 6 种候选定义并在 `a_in`/`a_up`/`h_S_out` 三个中间张量上复算，全部落在 1.00~57.49，无一命中），`N=256`/`fc=128` 的有效秩数值也不同步（实测 147 / 66 vs 现场值 163 / 106），方向一致但数值不可逐位对齐——报告一律以本实现现场实测为准，**不把差异包装成一致**。

**边界可行域（现场实测）**：`N=64`（`E=106,K=7,S_in=55,S_out=53`）与 `N=256`（`E=482,K=9,S_in=202,S_out=197`）均可构造；`y_in=y_out=2` 被连通性下限校验拒绝（`E=48 < N=64`）⇒ 固定 `y=4`；`cylinder λ=0.15` 失败（FCC 半径内仅 61 点）、`λ=0.25/0.5` 得 `K=3`（过浅）、`λ=1.0` 得 `K=6`、`λ=4.0` 失败（`R_max=0.321698 < R_min=0.364972`）；`geo_field="class_tied"/"mlp"` 构造期显式拒绝（未实现）；`shape="cube"` 且 `cyl_aspect≠1.0` 报错。

**门禁 G1~G8 现场结果**：G1 切分一致性通过（组内 18 + 跨档 18）；G2 默认档逐位复现通过（`run` 8/8 组、`compare` 8 组 × 2 档的**既有键差异 0 处**，锚点 7 项全在容差内）；G3 全组如实报告（11 个逐轴变体全记 FAIL）；G4 零回归（顶层 9 个 `qa_*.pt.zip` SHA256/字节/mtime 逐项不变、上游 `git status` 为空、根 `requirements.txt` 未改、零新增依赖）；G5 可训参数更新量 > 0（36 组全 PASS，`head.proj.weight`/`head.proj.bias` 进入优化器并发生更新）；G6 构造失败显式登记（两档均 18/18 可构造，显式写「0 项」；有失败即退码 1）；G7 align 损失有限性（`S7` 适用 406 / 跳过 34（批内无同类正样本对，计数可见）、`S8`/`S9` 适用 436 / 跳过 4，非有限计数 0）；G8 先 drill 再全量（`drill` 默认组 5 项训练后量与 `run_training(save=False)` 逐位一致；`--align-mode proj_supcon` / `distill` 与 `--shape cube` / `--struct-N 256` / `--fc-dim 128` / `--geo-field additive` 六条 drill 全部退码 0）。
## 可插拔特征生成接口

把「文本 → D 维特征」从具体实现里抽出为**可插拔接口 + 编码器注册表**，使步骤 1（features 侧）与步骤 2（step2.ZhBagVectorizer 侧）接到**同一接口**，上层评估 / 训练 / 推理 / CLI 的特征调用点零改动，只改「选择实现的入口」。

**统一接口（与既有向量化器逐字同形）**：`dim`（连接参数 D）/ `fingerprint()` / `encode(text)` / `encode_batch(texts)` / `encode_with_stats(text)`；步骤 2 侧另有 `encode_matrix(texts)`。

**编码器注册表**（唯一注册点，注册时声明 `expect_dim`）：`bge-m3`（kind=`hf`，expect_dim=1024，pooling=`cls`，model_id=`BAAI/bge-m3`，revision 固定 `5617a9f61b028005a4858fdac845db406aefb181`，weight_file=`pytorch_model.bin`，mirror_endpoint=`https://hf-mirror.com`；**首个 HF 默认项**）、`local-hash`（kind=`hash`，expect_dim=88，步骤 1 现状口径）、`zh-bag`（kind=`hash`，expect_dim=192，步骤 2 现状口径）。角色默认实现保持现状：`question -> local-hash`、`text_line -> zh-bag`，切 HF 必须显式选择（`--encoder`）。

**HF 编码器适配器**：传模型名或本地路径即可用；从模型 config 读 `hidden_size` 作为连接参数 D；**手工 CLS pooling + L2 归一化**，不引 sentence-transformers / FlagEmbedding；`config.hidden_size` 与声明维度不符在**构造期**抛 `EncoderDimMismatchError`（拒绝静默错配）。模型缺失 / 权重不全 / 缺依赖 / 无网络抛 `EncoderUnavailableError`（可读报文）。**可复算**：固定 revision + 权重文件 SHA256 全部折进 `fingerprint()`。

**编码口径**：题面（role=`question`）`max_length=512`；文本行（role=`text_line`）`max_length=8192`。不同口径的指纹与缓存键都不同。

**嵌入缓存**（`checkpoints/qa_learn/_cache/emb/`）：键 = `sha256(schema, model_id, revision, pooling, max_length, normalize, weight_sha256, text_sha256)`（即「模型名 + revision + pooling + max_length + 文本哈希」）。条目为 `float32` 小端裸字节 `.bin` + 元数据 `.json`（充当提交标记），可跨进程复用；缓存**不参与**口径指纹（`fingerprint()` 原样委派），只影响速度。`hf` 家族默认开启，`hash` 家族默认关闭（可用 `--force-cache` 开启）。

**接入点**：`train.build_training_data(cfg, encoder_name=)` / `run_training(..., encoder_name=)`；`cli` 全局 `--encoder` 与 `_resolve_dim`（`--dim` 优先，否则取注册表声明维度）；`probe.end_to_end_drill(..., encoder=, role=)`（D 的唯一来源 = 编码器注册表）；`step2.build_step2_vectorizer(cfg)`；`step2_run` 的 `--encoder` 等开关与 `verify-vectorizer` 子命令；`train.load_artifact` / `rebuild_model` 与 `cli` 的 `ask` / `selftest` 一律走 `encoders.vectorizer_from_meta`（按口径分派：HF 走注册表，其余委派历史路径逐位不变）。**默认档行为逐位不变**。

**边界处置**（与 features 契约对齐）：空 / 仅空白 → 零向量；超长 → 截断且 `n_truncated` 可见；模型缺失 / 权重不全 / 无网络 → 可读报错；hidden size ≠ 注册维度 → 构造期报错。自检入口 `encoders.boundary_selftest`。

**步骤 2 的验证口径（重建）**：由 `step2.verify_vectorizer_contract` 承担三条 —— ① `fingerprint()` == 注册表 / 落盘声明；② 同文本重复编码**逐位一致**；③ 与**落盘缓存**逐位比对。原「与 `n3d_qa` 冻结产物逐元素比对」（`verify_features_against_product`）降级为 **hash 家族的旁证**，不再作为本接口的验证口径（HF 与该产物不在同一特征空间）。

**验收入口**：`python -m n3d_qa_learn.encoders_run {registry|fetch|info|drill|verify|boundary}`（G1 结构量与来源核实 / G2 单条端到端演练含缓存写入与跨进程二次读取 / G3 三检查 / G4 边界自检）；`python -m n3d_qa_learn.step2_run verify-vectorizer`。取证报告写 `checkpoints/qa_learn/_verify/encoders/`，缓存写 `checkpoints/qa_learn/_cache/emb/`，日志一律由 Python 以 UTF-8（无 BOM）自写。

**现场实测（第 6 轮）**：`hidden_size=1024` / `max_position_embeddings=8194` / `pooling_mode_cls_token=true` / 权重 `pytorch_model.bin` 2271145830 字节 SHA256 `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`；下载须 `HF_ENDPOINT=https://hf-mirror.com` 且 `HF_HUB_DISABLE_XET=1`（xet CAS 在镜像下 401）。口径指纹：question/512 = `67a0e6cdde04cf57274fa5115a2383da5b1b8e76aefc3cab997d86525a278ebd`，text_line/8192 = `bc643ccc98b4a9a2dd755683505e8d4c56ef8a79ff91184e4e5cec0302334228`。G1~G4 全部退码 0（G4 4/4 PASS，含超长 `n_tokens=1794 / n_truncated=1282`）；步骤 2 验证口径在 `zh-bag`（192 维 × 8 条）与 `bge-m3`（1024 维 × 4 条，`max_length=8192`）上均全通过。`checkpoints/qa_learn/` 顶层 9 个 `qa_*.pt.zip` 的 SHA256 / 字节数 / mtime 逐项不变，顶层除新增 `_cache/` 外无新文件；`git status --porcelain -- n3d_qa n3d_shape n3d_sphere n3d_proto` 为空。

**如实登记的未做项**：本批**未**用 `bge-m3` 跑完整步骤 1 / 步骤 2 训练（`D=1024` 与历史纠正记录 #12 的「样本 1e3 量级词袋维度 10^1~10^2」正面冲突，且本轮门禁未要求），故 `bge-m3` 在步骤 1/2 上的准确率**本轮无任何实测数字**。依赖：默认路径仍零新依赖；可选 HF 路径新增 `transformers 5.18.0` / `tokenizers 0.23.2` / `huggingface_hub 1.33.0` / `safetensors 0.8.0`，清单与下载口径登记在 `n3d_qa_learn/requirements-qa.txt`，根 `requirements.txt` 未改动。
**审查修复轮（离朱 R45：106/111 断言通过，3 真缺陷 + 1 假阴性，全部已修）**

- **D1（高）** `zh-bag` 直接调 `encode_with_stats` 抛 `AttributeError`，导致 `encoders_run boundary / drill --encoder zh-bag` 退码 1 + 裸 Traceback。修复：① `step2.ZhBagVectorizer` 补齐 `encode_with_stats`（截断统计如实恒为 0）；② 新增 `encoders.encode_with_stats_of(encoder, text)` 统一入口（无该成员时回退 `encode`），自检/演练一律走它；③ `drill` / `verify` 的缓存默认开启（这两个子命令的职责就是演练缓存链路）。
- **D2（中）** 边界自检没有「不适用」语义，hash 家族被误判失败。修复：`EncoderSpec` 新增能力声明 `truncation_stats` / `requires_model_files`；每项用例带 `applicable`；新增 `summarize_selftest` / `failed_selftest_cases`，CLI 只对**适用项**判失败并报出不适用清单（「不适用」既不伪装成「通过」也不算「失败」）。
- **D3（中）** `build_vectorizer` 的 hash 分支完全忽略 `config.source`。修复：`source` 非空且不存在 → `EncoderUnavailableError`（与 HF 分支同形）；存在 → `EncoderError`（明确拒绝而非静默忽略）。修复后「模型缺失 -> 可读报错」对两家族都成立。
- **N1（假阴性）** `check2` 叠加 `list` 相等判定，缓存还原的 float32 与首次实算的 float64 末位差导致「字节相同却判失败」。修复：`encoders_run.cmd_verify` 与 `step2.verify_vectorizer_contract` 的 ② 一律**只用 float32 裸字节**判定。
- **N2/N3（健壮性）** `EncoderConfig.__post_init__` 校验 `role ∈ ROLES` / `hash_dim >= 1` / `max_length >= 0`；`EmbeddingCache.key_for` 校验 `max_length >= 1` 且 `model_id` / `pooling` / `normalize` 非空（`revision` 允许为空：hash 家族无 revision 概念，HF 侧构造期已拒绝空 revision）。

**修复后现场复测**：`boundary --encoder zh-bag` = 退码 0（3/3 适用项 PASS，1 项不适用）、`zh-bag --no-cache` = 0、`local-hash` = 4/4 退码 0、`drill --encoder zh-bag` = 0、`drill --encoder zh-bag --reuse-only` = 0、`verify --encoder local-hash` = 0；含 `bge-m3` 的全套 G1~G4 + 步骤 2 两条验证口径共 9 条命令**全部退码 0**；`cli probe / drill(index) / drill(pointer) / selftest / guard / ask` 与 `step2_run probe` 退码全 0。修复轮**未**改变任何默认口径指纹（`local-hash` `d4a4881aaa8dfa8c…` / `zh-bag` `8f2523e41484adf3…` / `bge-m3` question `67a0e6cdde04cf57…`、text_line `bc643ccc98b4a9a2…`），G5（顶层九产物 SHA256/字节/mtime 逐项不变）与 G6（上游零改动）保持通过。