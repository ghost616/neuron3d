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
**嵌入缓存跨进程命中诊断与分片修复轮（轮 A；只改性能，取值与产物零变化）**

根因判定与上一轮登记不符（如实登记差异）：上一轮登记「`_cache/emb` 磁盘嵌入缓存几乎不命中」经现场取证**否证**——① 两个独立新进程用真实 `HFTextEncoder` 对同一批真实 (行文本 5 + 候选词 5) 编码，`decl` 逐字段相同（`model_id=BAAI/bge-m3` / `revision=5617a9f6…` / `pooling=cls` / `normalize=l2` / `max_length=512` / `weight_sha256=b5e0ce34…` / `dim=1024`）、`fingerprint` 同为 `77e9071ba2a149bc…`、10 个缓存键**逐位相同**，且 **10/10 `existed=True` + `cached=True`、hits=10 / misses=0**；② 全量工作负载「所需键 vs 落盘键」集合比对（键由代码路径枚举，存在性用一次 `os.listdir` 的内存集合判定）得 **10,464 候选词 10464/10464 命中、2,624 行文本 2624/2624 命中，miss=0**；③ 落盘条目的键自洽性 10/10。

**真实根因 = 扁平大目录的逐路径查找成本**：缓存根为单目录 **67,592 条目**（33,795 条目 × bin+json），该目录内单次 `isfile` 实测 **58~83 ms**、`open` **40~46 ms**、「不存在」查找 **129.5 ms**，而 40 / 100 / 2,000 / 8,000 条目目录只要 **0.34~0.76 ms**；目录顺序读 3,000 个文件的累计耗时 1.1 / 2.8 / 5.4 / 9.2 / 13.5 / 19.0 s（成本随条目在目录中的位置**线性增长** —— exFAT 按名查找是自目录头起的线性扫描，无索引）。一次缓存**命中** = 2 stat + 2 open ≈ **177~239 ms**（逐条实测 0.226~0.398 s，同进程第二次 0.0016 s 走 memo）⇒ 全量 `probe` 需 ~13,088 次查找 × ~0.2 s ≈ **2,600~3,000 s**，与上一轮登记的 `2983.5 s` 吻合；上一轮与离朱观察到的「新进程 `encode_matrix(10 词)` = 6.99 s（0.699 s/词）」正是**命中**在扁平目录里的代价，**不是**重算代价。

**修复（只改条目落盘位置）**：新增冻结常量 `encoders.EMB_CACHE_SHARD_DEPTH = 2`，条目落 `emb/<key[:2]>/<key>.bin|.json`（256 个子目录、每片约 132 条），查找回到亚毫秒级。**`key_for` 逐字未改、条目字节与读出校验逐字未改、`fingerprint()` 仍原样委派且缓存不参与任何口径指纹**（未放宽 / 未删除任何键分量；改 `max_length` / `normalize` / `revision` / `weight_sha256` / `model_id` / `pooling` 任一分量必换键，空文本与极长文本不串键）。读路径分片优先、缺失时**只读回退**历史扁平位置（用惰性 `os.listdir` 索引 `flat_keys()` 判断该不该去大目录找，避免白付 129 ms 的「不存在」扫描）；`read_bytes()` 两处皆认。写路径一律写分片并顺手清理同名扁平副本。`_read_pair` 去掉 `isfile` 预检查改用 `open` + `OSError`（语义逐字等价、每次读少 2 次大目录查找）；`stats()` 改为逐片统计。新增一次性维护入口 `EmbeddingCache.migrate_flat_keys(keys=None, mode='rename'|'copy')`，且 `auto_migrate` 默认 `False`（实测惰性「读+改名」≈ **0.7 s/键**，改名会改动那个大目录使其线性扫描缓存反复失效，会让全量 probe 从分钟级涨到小时级；`copy` 实测 ≈ **0.11 s/键**）。

**边界与损坏处置（逐项现场复验，`cache_layout_selftest.py` 41/41 PASS）**：`.json` 缺失（只有 `.bin`）、`.bin` 缺失（只有 `.json`）、零字节 `.bin`、JSON 语法损坏、元数据缺 `dim`、`dim` 与字节长度失配、`sha256` 失配、`.bin` 被截断、完全不存在的键 —— **一律返回「不命中」并回退实算，绝不抛未捕获异常**；同批健康条目仍可正常读出。取证脚本与日志落 `checkpoints/qa_learn/_verify/emb_cache/`；README §13.11 记录根因、修复、冷/热对照与边界处置，§13.5 同步更新布局口径。
**收口修复轮（皋陶 approved=false：2 error + 2 warning + 1 info；只修表述、容差定性与一处连带读路径）**

边界：**分片方案 / 缓存键分量 / 条目字节格式 / `fingerprint()` 原样委派关系一律未动；编码器对同一输入的取值未变**（唯一行为修改是 `encoders_run` 的 meta 读取，只影响取证侧读不读得到元数据）。

- **error 1（根因表述与现场证据相悖）**：原定性「CPU 前向按批组成有 float32 级不确定（噪声、不可归因）」⇒ 改写为「**确定性、可归因的批组成（padding 配置）依赖**」。最小复现（真实 `models/bge-m3`、`text_line`/512、两个独立进程）：目标文本单独编码（`batch_size=1`，`n_tokens=11`）`sha=52c47064eba2`；放进 `rows[0:3]` 批（`batch_size=8`，批内 `n_tokens=[17,11,22]`、**批内最长 22 ⇒ 短句被 padding 到 22**）`sha=dcab8d33c52bc946`；`max|alone−in_batch|=7.823e-08`；同批重复逐位相同；**A/B 两进程四项哈希全部逐位相同**；落盘条目 `13e0a819f4166885`（`created_utc=2026-10-09T20:26:42Z`）实测 `stored == batch` 而 `!= alone`。措辞扫描：§13.11 段内三词均 0、README 全文「不确 定」「不 可归因」均 0；「噪 声」另有 48 处**全在 §13.11 之外**（§15 高斯扰动实验与 噪 声底标定，另一技术概念、有各自冻结口径）⇒ 判定口径差异已如实登记。
- **error 2（容差自指 + 硬约束被实质放宽）**：① `1.3e-07` 明确重定性为「**由本轮实测最大值 1.229e-07 加余量得到的自证容差，仅作诊断参考、不是契约判据**」；② 新增两条硬句 —— **「本轮的产物零变化以『所需键全量命中、零重算』为前提」**（现场：10,464 词键 + 2,624 行键 100% 命中、`miss=0`）与 **「本轮未证明『重算不改值』」**；③ 新增**风险登记**：回退重算会以**当时的批组成**写入并**回填同一个键**（键不含批组成），`read_bytes` 逐位比对会观察到变化、下游浮点排序在极端接近的候选对上理论上可能换序（本轮未观察到）。
- **warning 1（真实功能缺陷，布局变更连带面未同步）**：`encoders_run` 的 G2 取证按历史扁平布局直拼 `<cache_dir>/<key>.json` ⇒ 分片条目上**永远读空**。新增 `EmbeddingCache.read_meta(key)`（**分片优先 + 扁平只读回退**，缺失/不可解析/非对象一律返回 `None` 不抛），G2 改用它。**前后反例**：修复前 `cache_entry_meta` 13 键全 `null`、`cache_record` 全空；修复后全部非空（`dim=1024`/`bytes=4096`/`model_id=BAAI/bge-m3`/`max_length=512`/`normalize=l2`/`weight_sha256=b5e0ce34…`/`n_tokens=19`/`created_utc`）。**全仓扫描**：直接拼缓存条目路径的写法**只**命中 `encoders.py` 的布局所有者本身（`_entry_paths` / `_flat_entry_paths`），无其它扁平假定读取方；全部读取方走 `read_bytes` / `read_meta`。
- **warning 2（非本轮引入，登记为未决发现，本轮不修）**：`encode_batch_with_stats` 按「**当时命中状态**」切块实算 ⇒ 一次实算的批组成取决于缓存状态 ⇒ 同一文本在不同缓存状态下落入不同批、得到不同向量（7.823e-08~1.229e-07）并作为该键**唯一取值**持久化。本质矛盾 = 「**键只含文本与口径、不含批组成**」与「取值依赖批组成」。影响面：回退重算后的条目取值会变、`read_bytes` 取证会观察到差异、下游浮点排序在极端接近候选对上理论上可能换序。可选修法与代价（A 键里加批组成签名 —— 改键分量且既有条目全失效；B 未命中时逐条实算 —— 失批处理吞吐且仍改变历史条目口径；C 固定 padding 到 `max_length` —— 计算量不可行；D 只登记不修 —— 零风险）见 README §13.11 第十节。
- **info 5**：模块 docstring 的缓存键分量补齐为 8 个（含 `schema` / `normalize` / `weight_sha256`）并注明「唯一权威实现见 `EmbeddingCache.key_for`，冲突时以实现为准」。
- **info 6**：`boundary_selftest` 的 `dim_detail` 采纳「显式标注口径」方案 —— 保留 `actual`（= `spec.expect_dim`，**不是实测值**），新增 `actual_measured`（**从构造期报文解析出的实测 `hidden_size`**：现场 `declared=1025` / `actual=1024` / `actual_measured=1024`）与 `actual_source` 说明；`boundary` 现场 4/4 PASS、退码 0。
- **info 7**：`migrate_flat_keys` docstring 逐分支写全计数语义（`attempted`/`migrated` 以**键**为单位、`failed` 以**文件**为单位且可能大于 `attempted`、同一键可同时计入两者、**目标已存在时两种 `mode` 都直接覆盖、不计 failed**、`shard_depth=0` 为安全 no-op）。

## 分档鲁棒性考卷（第一步 1a）

设计文档三步走的**第一步 1a**：**不改任何网络结构、不做任何训练**，只建一把「考卷」并**验证这把尺子真的有区分度**（背景：上一轮 18 组结构改动全为 Δ0，根因是评价口径饱和）。由 `n3d_qa_learn/entry_table.py` 与 `n3d_qa_learn/robust_eval.py` 承载，入口为 `python -m n3d_qa_learn.step2_run robust {probe|run|calibrate|report}`。

**统一条目特征表（`entry_table.EntryKeyTable`）**：`keys [N, D] float32` **逐行 L2 归一化** + `entry_ids[N]` + `outputs[N]`（QA 侧 = 答案展示文本；文本侧 = 行原文）+ 元数据（`dim` / `norm_spec` / `kind` / `source` / `key_table_sha256` / `encoder_profile` / `encoder_fingerprint`）。构造期不变量：二维、`dtype==torch.float32`、C 连续、行范数 ∈ `[1-1e-6, 1+1e-6]`、`len(entry_ids)==len(outputs)==N`、id 唯一、`shape[1]==dim`；违反即抛。**逐位精确查表**：`lookup(i)` 返回该行 **float32 裸字节**与绑定输出，`verify_bitwise_lookup` 断言裸字节**逐字节相等**（不是近似）。**指纹守卫**：`key_table_fingerprint`（裸字节 hex + ids + outputs + dim 的规范化 JSON SHA256）与 `fingerprint_guard`（键表 SHA256 + 编码器口径指纹，任一不一致立即报错）。

**两侧构造器**：QA 侧只读 `n3d_qa` 冻结 QA 缓存，经 `data.build_answer_space` + `data.make_splits` 复现切分（与 `exp_repr` 基线档同口径：`max_classes=10` / `min_questions=8` / `test_every=3` / `test_per_class=2` / `unknown_train_cap=500` / `split_seed=42`），现场实测 `train_known 149` / `test_known 20` / `train_unknown 500` / `test_unknown 2393` / `C=10`；`train_known` 建表、`train_unknown+test_unknown` 作「未识别」负样本、`test_known` 作辅助查询，并携带三份 `answer_key` 供金标映射（金标 = 同答案键在条目表中的**行集合**，非自身行）。文本侧**只读包装**既有 `step2.TextRowKeyTable`（键表字节与 `sha256()` 原样透传、不写任何文件、**落盘格式零改动**），检索池 = 冻结行表全量 2665、查询集 = 冻结划分 query 行 666，并现场断言包装视图第 0 行裸字节与既有键表一致。**特征档注册表**：`lexical-88`（local-hash）与 `bge-m3-1024`（BAAI/bge-m3，本地 `models/bge-m3`），每档同时声明 `question` / `text_line` 两口径且必须同维；编码器一律经 `encoders.build_vectorizer` 取值（不自造）。

**扰动构造（三种 × 三档，确定性）**：`noise` σ ∈ {0.05, 0.10, 0.15}、`mask` 随机维度遮蔽 ∈ {10%, 30%, 50%}（逐行 `randperm`）、`nmag` 幅度缩放+平移 ∈ {±10%, ±15%, ±20%}（`NMAG_SCALE_FACTOR=0.5`）。**扰动后一律重新 L2 归一化**（唯一实现 `entry_table.l2_normalize_rows`），零范数行**显式报错**（不静默产出 NaN）。随机数一律用**局部** `torch.Generator`（**不消耗全局 RNG**），派生种子公式 `seed*1000 + 100*type_index + 10*level_index + variant`（现场实测 `noise` 三档种子 = 42000/42010/42020）。每档每条目 1 条变体作主判据；另**重新构造 K=5 个真实不同变体**（`variant=0..4`）报均值 ± 极差作稳定性佐证。

**检索与基线**：归一化余弦 top-1 / top-5，名次口径 `rank = 1 + #{j : cos(q,k_j) > 该查询金标集合内最大余弦}`（并列按超出条数计，与 topk 顺序无关；金标集合为空的查询 `rank=0`、恒判未命中并计入分母）。**欧氏基线轴已移除**并现场实证登记：对 L2 归一化向量 `‖a−b‖² = 2 − 2·cos(a,b)`，故 argmin 距离与 argmax 余弦必然同解 —— 现场实测文本侧 666 条与 QA 侧各 512 条查询的「余弦 top-1 vs 欧氏 top-1 不一致条数全部为 0」，恒等式最大残差 4.768e-07；**不把「余弦/欧氏对比」当作有信息量的对照**。

**阈值现场标定（两阶段）**：① 实测**噪声底** = 「可忽略强度（`NOISE_LEVEL_RATIO = 0.02` × 该扰动弱档强度，即 σ=0.001 / 遮蔽 0.002 / 幅度 ±0.002）下，查询与键表用**两次独立随机绘制**（变体 0 / 变体 1）时相对档 0 的最大掉点」—— **必须两次独立绘制**：同一噪声向量施于两侧对自检索是**恒等变换**，命中率恒 1.0，会把噪声底误标为 1.0（现场实测该构造下文本侧三档 R@1 全为 1.000000）；该量现场实测**对强度平直**（ε=0.001 与 ε=0.01 同为 R@1 = 0.983483），说明它反映键表**近重复行**导致的自匹配歧义。② 标定 `τ = factor × 噪声底`（`factor = THRESHOLD_FACTOR = 2.0`，即观测噪声的约 2 倍），逐侧 / 逐扰动 / 逐编码器给出；判据用**严格大于**（`弱档 − 强档 > τ`），因为 τ 在无近重复行的格上可能为 0，`≥` 会把「零落差」误判为有效。

**「未识别」档**：以 `train_unknown 500 + test_unknown 2393 = 2893` 条对条目表取最高余弦，报均值 / 中位数 / 90% / 99% 分位；**本批不标定阈值**（属第二步变体 B 的范围），τ 未给定时未识别率与误召回率如实标「不适用」。空 / 仅空白问题沿用现有契约（归一化后为空 → 返回「无匹配」，不计入分母）。

**CLI**：`robust probe`（表构造与口径取证，不评测；编码器不可用如实登记不静默跳过）/ `run`（全量矩阵，含 `--dry-run` 单组合演练）/ `calibrate`（读 run 结果产出标定阈值与依据）/ `report`（JSON + Markdown 渲染，读不到标定产物时该节标「尚未标定」）。参数：`--side {qa,text,both}` / `--profiles` / `--perturb` / `--seed` / `--variants` / `--stability-k` / `--topk` / `--batch-size` / `--verify-dir` / `--out-dir` / `--run-json` / `--calibration-json` / `--factor` / `--primary-side` / `--dry-run` + `common()` 的 `--product-dir` / `--backend` / `--log-file`。全部子命令支持 `--log-file`（Python 以 UTF-8 无 BOM 自写）。产物一律写 `checkpoints/qa_learn/_verify/robust/`（`robust_probe.json` / `robust_run.json` / `robust_calibration.json` / `robust_report.{json,md}` / 日志），**不落盘扰动后的特征矩阵**（只落报告与指纹）。

**现场实测（真实执行）**：四张表的**逐位精确查表 `exact_frac` 全部 = 1.0**（`text:lexical-88` 2665/2665、`text:bge-m3-1024` 2665/2665、`qa:lexical-88` 149/149、`qa:bge-m3-1024` 149/149），键表指纹分别为 `a4b6d9dda0852b2e…` / `b973d07bfc56fabb…` / `d5a3cf991036f9c9…` / `9fe268a53d046158…`，行范数区间 ∈ `[0.99999982, 1.00000012]`。**主表（文本侧档 0 R@1 = 1.000000 = 逐位自匹配的结构结果）**：`lexical-88` noise 弱/中/强 = 1.000000 / 0.984985 / 0.789790（落差 0.210210）、mask = 0.990991 / 0.947447 / 0.770270（0.220721）、nmag = 1.000000 / 0.998498 / 0.992492（0.007508）；`bge-m3-1024` noise = 0.998498 / 0.980480 / 0.758258（0.240240）、mask = 1.000000 / 1.000000 / 1.000000（**0.000000**）、nmag = 1.000000 / 0.996997 / 0.983483（0.016517）。QA 侧（`test_known` 20 条，**统计意义弱**）：`lexical-88` 档 0 = 0.300000、`bge-m3-1024` 档 0 = 0.650000，三扰动落差 ∈ [−0.100000, 0.200000]。QA 侧未识别负样本 2893 条对表的最高余弦：`lexical-88` 均值 0.742432（中位 0.746753 / 90% 0.807588 / 99% 0.855186）、`bge-m3-1024` 均值 0.450777（中位 0.435551 / 90% 0.554938 / 99% 0.670910）。**阈值标定**：逐格噪声底 ∈ [0.009009, 0.325000]，逐格 τ = 2 × 噪声底，**主判据侧有效扰动 3/6**（`lexical-88` 的 noise 0.210210 > 0.030030 与 mask 0.220721 > 0.018018、`bge-m3-1024` 的 noise 0.240240 > 0.036036 判**有效**；`lexical-88` nmag 0.007508 < 0.028028、`bge-m3-1024` mask 0.000000 < 0.031031 与 nmag 0.016517 < 0.028028 判**无效**），**QA 侧 12 格全部无效**（20 条查询的噪声底 0.141667~0.325000 本身就高于任何观测落差）。**稳定性佐证**：36 行中 32 行极差 > 0（如文本侧 `lexical-88` noise 强：均值 0.780480、极差 0.049550、命中数区间 501~534 命中/666）。**G3 可复现**：同命令三次运行的 114 格 cells 与 36 行 stability 在剔除 `created_utc` / `seconds` 后规范化 JSON 的 SHA256 **完全相同**（`e4f592c0571857ae…`，其中一次为「文本侧包装路径重构为复用 `entry_table_from_text_row_table`」后的复跑 ⇒ 重构零行为改动）。**G1~G6 全部现场执行**：G1 逐位精确查表 100%；G2 零回归（顶层 9 个 `qa_*.pt.zip` 的 SHA256/字节/mtime 逐项不变、顶层无新增文件、上游四目录 `git status` 为空、根 `requirements.txt` 未改、既有子命令 `probe`/`drill`/`guard`/`verify-vectorizer` 与 `cli probe` 全部退码 0 且数字与登记值一致）；G3 逐位一致；G4 三档 × 三扰动全组合可构造、无异常、无 NaN（`n_zero_norm` 全 0）；G5 全组如实报告负结果；G6 先 `--dry-run` 单组合演练再放全量。**现场不变量检查 14 条全部通过**（文本侧档 0 自检索 = 1.0 且 `n_no_gold = 0`；QA 侧已知查询「金标可定义」= 20 > 0；QA 侧未识别负样本「金标恒空」`n_no_gold == n == 2893` 且 `hit_at_1 == 0`；四张表 `exact_frac == 1.0` 与行范数在允差内）。**实测耗时**：全量 run = 110.9s / 105.7s / 110.3s（嵌入缓存预热后；首次冷缓存 ≈ 290s）。

**如实登记的负结果与结构性限制（不得包装）**：① 主判据**仅 3/6 有效**，`bge-m3-1024` 的「随机维度遮蔽」落差为 **0.000000**、两档的「幅度缩放+平移」均**无效**；② **QA 侧 12 格全部无效**，根因是 20 条查询（单样本 = 0.05），其噪声底本身高于任何观测落差；③ **文本侧掉点含「近重复行自匹配歧义」成分**（可忽略强度下 R@1 已降到 0.983~0.992），**不得**把 0.21/0.22 直接读成「语义理解被破坏」；④ **文本侧「弱档」多为天花板效应**（`bge-m3-1024` 在 50% 遮蔽下仍 1.000000），只属于「该编码器 + 该 ID 检索协议」；⑤ 「未识别」档**不标定阈值**（属第二步变体 B）；⑥ 本轮**不做档 1（同答案另一问题）与档 2**：现成数据不存在「同一问题的不同问法」（Math1 的 id 在 4 题型间零重叠、n3d_qa 每个 `question_id` 只有 1 种 `question_text`）；⑦ 文本侧档 0 = 1.000000 与 `step2_run` 的纯特征参照下限 `0.9834834834834835` **是两种口径**（后者为库 1999 / 查询 666 的留出划分 + `zh-bag` D=192），不可直接比较；⑧ **踩坑登记**：稳定性表首版从主判据 cell 按 `variant` 筛选（而主判据只跑 `variant=0`），导致 K=5 的 36 行极差**全为 0.000000**（假稳定性），已改为显式重跑 K 个变体；⑨ **环境边界**：本机系统 Python 缺 `transformers`，含语义档的全量现场用仓库自带 `.venv\Scripts\python.exe` 跑，系统 Python 下该档抛 `EncoderUnavailableError`（可读报错，不静默跳过）；零新增依赖（仅 torch / numpy + 标准库，`bge-m3` 为可选 HF 路径）。
**CLI 错误边界（离朱 R49 条目 33 修复，现场实测）**：`--profiles` / `--perturb` 的取值校验前移到参数解析（`step2_run._csv_names(raw, allowed, label)`，未登记值报「`--profiles` 含未登记的值 [...]；可用 = [...]」），并在 `cmd_robust` 内以分派表 + `try/except (ValueError, KeyError)` 兜底 —— `--profiles bogus` / `--perturb bogus` 现为**单行可读报文 + 退码 2**，不再抛裸 Traceback（原先 `RobustConfig.__post_init__` 的 `KeyError`/`ValueError` 穿透到解释器，stderr 30~33 行）；`calibrate` / `report` 指向不存在的 run 产物时为**可读报文 + 退码 1**。捕获范围**只包住 robust 子命令族**，既有子命令的分发逻辑与参数表逐字符未改；四条正常路径（`probe` / `run --dry-run` / `calibrate` / `report`）仍退码 0，本轮**未重跑全量、数字未变**。

**离朱 R49 验收结果（现场）**：**316 项检查，314 通过 / 2 失败**（同一处缺陷，即上段的 CLI 错误呈现层）。逐组：编译 2/2；A 组（`entry_table.py` 单元/契约，条目 1–8）56/56；B 组（`robust_eval.py`，条目 9–21）115/115（含 9 组合确定性、「调用前后 `torch.get_rng_state()` 逐位相同 ⇒ 不消耗全局 RNG」、派生种子全量公式核验、`effective == (gap > tau)` 严格性含 `gap == tau` 边界判「无效」、UTF-8 无 BOM 首字节核验）；C 组（生产侧契约 22–24）31/31（`train_known 149`/`test_known 20`/`train_unknown 500`/`test_unknown 2393`/`C=10` 与登记值逐项一致；`keys.shape==(149,88)`、`unknown.n==2893`、`known_queries.n==20`；`text` 表 `(2665,88)`/`n_query==666`/`intersection==0`/`reproduced_equals_index is True`；维度错配抛 `ValueError`；调用前后 `checkpoints/qa_learn/step2/` 与顶层**逐文件 SHA256 零新增零改动**）；C 组续（小规模 `run_evaluation`）11/11；D 组（CLI 26–33）66/68；E 组（零回归 34–36）35/35（既有子命令 9 项关键数字逐项一致、顶层 9 个 `qa_*.pt.zip` 零差异、`git status --porcelain -- n3d_qa n3d_shape n3d_sphere n3d_proto requirements.txt` 为空、正式取证目录字节数零变化）。

**离朱登记的 5 条观察项（不判缺陷，如实登记）**：① `drill` 现场文案为 `[drill] 参与前向但零梯度 = []；不在计算图上 = []`（与测试说明引用的「未学到的参数梯度 = []」措辞不同，实质要求已满足 —— 两类未学到参数均为空、零梯度参数 0 个）；② `guard` 的 JSON 字段名为 `all_tampered_rejected`（`all_rejected=True` 只出现在 stdout）；③ `entry_table.encode_statements` 对空/纯空白文本产出零范数行（`local-hash` 返回全零向量）且**不返回取证字典**，下游 `build_key_table_from_statements` 以 `ValueError` 兜底（符合「零范数显式处置」声明，真实数据未触发）；④ `topk_search(k > N)` 的 `scores`/`indices` 尾部静默零填充（`k_eff = min(k, N)`，当前路径 `k=5 ≪ N` 无影响）；⑤ `bge-m3-1024` 档**端到端编码路径未执行**（按测试说明建议只用 `lexical-88`，冷缓存数分钟），该档的声明维度/配置构造/未知名报错/维度错配拒绝均已覆盖。E2E（Playwright）不适用（本轮无前台 UI 交互）。

**环境限制（离朱登记，不属于被测代码，力牧侧同样遇到）**：本仓库 `E:` 卷为 **exFAT**，不支持硬链接，而 `write`/`edit` 工具的原子落盘走「临时文件 + 硬链接」，故报 `EISDIR: illegal operation on a directory, link ...`；力牧侧绕过方式是先用 `New-Item -ItemType File` 建空文件再写入，离朱侧改用 `pwsh` + `UTF8Encoding($false)` 落盘脚本；测试执行与结论不受影响。
**皋陶审查修复轮（4 warning + 3 info，2026-10-06；只做修复与重新取证，不扩大范围）**

- **W1（`nmag` 扰动机制与口径不符，已修，结论方向因此改变）**：改前实现是 `base + ε·NMAG_SCALE_FACTOR·randn`（`NMAG_SCALE_FACTOR=0.5`）即**加性高斯噪声**，与报告表头的「幅度缩放+平移」不符（静默替换机制）。改后为真正的幅度缩放/平移且**闭式无随机数**：`x ← x·NMAG_SHARED_SCALE + ε·(−1)^(i+j+1)`（`NMAG_SHARED_SCALE = 0.5` 三档恒定，平移矩阵由 `nmag_shift_matrix(n_rows, dim, eps)` 给出）；middle 档原口径未定义，显式取 `ε = 0.15` 并登记于 `PERTURB_GRID` 注释与产物 `grid.nmag_middle_note`。新增常量 `PERTURB_FORMULAS`（三种扰动的公式文本，进产物 `grid.perturb_formulas`）与 `NMAG_NEGLIGIBLE_SHIFT = 1e-3`（`nmag` 噪声底用「不缩放、只做可忽略平移」，否则 0.5 倍整体缩放不可忽略、会把噪声底抬到 0.521021 从而把「有效」误判成「无效」）。**影响**：文本侧 `nmag` 落差由 0.007508/0.016517（无效）变为 0.342342/0.438438（有效）；`noise`/`mask` 全部数字逐位不变。
- **G7（本轮新增门禁）扰动公式逐位断言**：`robust_eval.assert_perturb_formulas_bitwise` 用固定已知输入矩阵 `FORMULA_CHECK_MATRIX`（3×8）+ 三种 × 三档共 **9 例**，每例独立按公式重算（`nmag` 闭式、`noise` 重放同派生种子的 `randn`、`mask` 重放同派生种子的 `randperm`），判定分两路：① 扰动后**未归一化**矩阵的 float32 裸字节 SHA256 相等（`raw_bitwise_equal`）② 重新 L2 归一化后裸字节逐字节相等（`bitwise_equal`）。现场 **9/9 全通过**，结果进 `run`/`probe` 产物与 `invariants`（不通过则退码 1）。
- **W2（QA 侧噪声底总体错配，已修，QA 侧结论整体反转）**：改前 `calibrate_thresholds` 的噪声底只按 `(side, profile, cell_role='noise_floor', kind)` 过滤，而 `noise_floor` cell 只由 `cell_role='main'` 那次调用产出 ⇒ QA 侧 `qa_known`（20 条）的噪声底取自 `main`（2893 条**未识别**负样本）这一完全不同总体（旧值 0.141667/0.183333/0.325）。改后 `_run_side_cells` 按 `cell_role` 各自生成 `noise_floor` cell 并写入 `population_role` 与 `population{cell_role, query_n, table_n, min_granularity}` 溯源；`calibrate_thresholds` 过滤条件加 `population_role == 该侧主判据角色`（QA 同总体噪声底为 0.0 / 0.066667）。新增**粒度守卫**：产物与报告显式写出「最小可分辨粒度 = 1/n_queries（QA 侧 20 条 ⇒ 0.05）」与「粒度 ≥ 落差者不得作为判定依据」，落差小于该格粒度者一律判无效并标注 `below_granularity`。**影响**：QA 侧 12 格由「全 12 格无效」变为「12 格全部有效」，但其中 5 格噪声底为 0、n=20 时 R@1 粒度仅 0.05，证据强度弱必须连带登记。
- **判据口径的显式补充**：主口径仍为「弱档 − 强档 严格大于 τ」；当档序列**非单调不增**时（QA 侧 2 格出现）改用「整段落差 `max(档) − min(档)`」并以 `gap_used_basis`（`weak_minus_strong` / `level_range_max_minus`）显式标注，不视为放宽判据。
- **W3（标定产物非确定性，已修）**：产物内**移除** `created_utc` / `seconds` / `seconds_run` / `calibrated_at_utc` 与输入产物的绝对路径（改为只写日志；`robust_report.json` 只留 `*_basename` 与 `*_sha256`），并新增 `deterministic: true` 与 `excluded_fields_note` 显式说明排除清单。**双跑逐字节证据**：同一命令在两个不同输出目录各跑一遍，四产物逐字节一致（`robust_run.json` 412979 B `2ec4df9f7447f5b7…`、`robust_calibration.json` 22885 B `8408460e05db5e15…`、`robust_report.md` 23402 B `734b7fb3e7f580d7…`、`robust_report.json` 437163 B `9387b156b1ec120d…`），`robust_probe.json`（23244 B `bcfe9251bad4d117…`）与 `run --dry-run` 亦然；report 内嵌的 run/calibration 指纹重跑后仍闭环。
- **W4（`calibrate` / `report` 参数校验缺失，已修）**：新增 `step2_run._validate_robust_selector_args` 校验 `--side`/`--profiles`/`--perturb`，在四个子命令入口统一生效（改前 `calibrate`/`report` 会静默忽略这些参数并退码 0）。现场实测四个子命令的 `--profiles bogus` 与 `report --perturb bogus` 全部为**单行可读报文 + 退码 2**、无裸 Traceback。
- **I1**：删除 `euclidean_equivalence_evidence` 从未使用的形参 `neg_top1`（两处调用方同步）与 `unrecognized_report` 中算完未用的 `best_idx`。**I2**：新增 `entry_table.question_is_blank`（沿用 `features.normalize_text` 口径）与 `_drop_blank_questions`，在 `build_qa_entry_tables` 显式过滤「空问题/仅空白」并登记 `n_in/n_kept/n_dropped/dropped_ids_head` 于 `evidence.blank_question_filter`（本批真实数据 0 条被过滤，149/2893/20 全保留，下游规模与全部指标不变）。**I3**：`PERTURB_GRID` 注释改写为「三种扰动都只有唯一标量参数」并逐条写清公式。
- **重新标定后的现场结论**：全局 τ = **0.133333**（= 2 × 噪声底上界 0.066667），主判据侧（`text`）**5/6 有效** —— `lexical-88` 三扰动全有效（noise 0.210210 / mask 0.220721 / nmag 0.342342）、`bge-m3-1024` 的 noise 0.240240 与 nmag 0.438438 有效、**mask 落差 0.000000 判无效**（三档 R@1 全 1.000000）；QA 侧 12 格全部有效但证据强度弱。不变量检查 **24 条全通过**（含 G7 的 1 条汇总 + 9 条逐例）。`README.md` 15.3/15.5/15.6/15.7/15.8/15.9/15.10 已同步修订，15.11 为本轮完整差异清单（含改前/改后逐项对照表）。
**离朱 R50 三项失败的修复（只改 `robust_eval.py`；数值结论逐位未变）**

离朱 R50 现场 329 项检查 **326 通过 / 3 失败**，三项均不影响数值结论（F1 涉及门禁可信度）：

- **F1（中）G7 断言对手算侧共享层无区分度（已修）**：改前 `assert_perturb_formulas_bitwise` 的手算侧调用与被测实现**同一份** `nmag_shift_matrix` / `derived_seed` / `NMAG_SHARED_SCALE`，破坏共享层时两侧同步变化故断言对该层**恒真**（离朱实测 monkeypatch `nmag_shift_matrix` ×2、`NMAG_SHARED_SCALE = 0.25`、`derived_seed + 1` 三种破坏均检不出）。**改后**手算侧把**型别下标**（`list(PERTURB_TYPES).index` / `list(PERTURB_LEVELS).index`）、**派生种子**（`seed*1000 + 100*type_index + 10*level_index`）、**缩放系数 0.5**、**棋盘符号矩阵**（`np.where(((ii+jj) % 2) == 0, -1.0, 1.0)`）全部**内联手写**，新增返回字段 `manual_side_isolated: True` 与 `manual_side_note`。**修复后现场反向验证**：上述三种共享层破坏全部令 `all_bitwise_equal` 变 `False`、恢复后回到 `True`。
- **F2（低-中）`noise_floor_of_cell` 的 `scale_override` 取证字段与实际口径不符（已修）**：改前 `evidence["scale_override"]` 写 `float(eps)`（0.001）而实际传 `1.0`。**改后**如实记录实际传入值并新增 `evidence["scale_used"]`（`nmag` 为 `1.0`、其它两种为 `NMAG_SHARED_SCALE`）；`negligible_eps` docstring 中「测噪声底时把 `s` 也一起压到同量级（`scale_override = ε`）」这句自相矛盾的表述改为「**不缩放**、只做 `NMAG_NEGLIGIBLE_SHIFT = 1e-3` 的可忽略平移」。现场核对 `nmag` 的 `eps_used = 0.001` / `scale_override = 1.0` / `scale_used = 1.0`。
- **F3（低）`noise_floor_population["cell_role"]` 与 `entry["cell_role"]` 口径不同（已修）**：改前写 `EvalCell.role`（`self_retrieval` / `known_queries`），entry 用的是运行级角色（`main` / `qa_known`）。**改后** `noise_floor_of_cell` 新增 `cell_role` 形参（由 `_run_side_cells` 传入运行级角色）：`population["cell_role"]` 记运行级角色、`population["eval_cell_role"]` 另存 `EvalCell.role`，并补 `population["side"]` / `["profile"]`，cell 顶层保留 `population_role`。现场核对 QA 侧 `qa_known` / `qa_known` / `query_n = 20` / `min_granularity = 0.05`、文本侧 `main` / `main` / `666` / `0.0015`（**实质**同总体约束在改前已成立 —— 离朱用假 run 交叉验证未跨总体借用；本项为字段口径对齐）。

**修复后的产物与数值**：因新增取证字段，四个产物字节数变化并已重建 —— `robust_run.json` 419088 B `b293b7783e3a7800…`、`robust_calibration.json` 24328 B `5736b859f78e5ec6…`、`robust_report.md` 23483 B `250be2fe7cbd33e8…`、`robust_report.json` 444769 B `cee33a856b90516a…`、`robust_probe.json` 23692 B `77c6496a0a526bdc…`（F1/F2/F3 修复前的中间值 `2ec4df9f…` / `8408460e…` / `734b7fb3…` / `9387b156…` / `bcfe9251…` 已作废，不再引用）。**跨目录双跑逐字节一致**在修复后**重新验证通过**（四产物 + `probe` 全部 `bytes_equal=True`）；`invariants` 仍 **24 条全通过**、G7 仍 **9/9**、全局 τ 仍 `0.133333`、主判据仍 **5/6 有效**（`bge-m3-1024` 的 `mask` 落差 0.000000 且 `below_granularity = true`），`noise`/`mask`/`nmag` 三轴与 QA 侧全部 R@1 数字与修复前**逐位相同**。`README.md` 15.11 新增「离朱 R50 三项失败的修复（本轮收口）」小节并更新产物 SHA 表。
**离朱 R51 复测（182 项检查 181 通过 / 1 失败）与字面项收口**

R51 确认 **F1/F2/F3 三处修复全部收口**：F1 —— R50 检不出的三种共享层破坏（符号矩阵 helper ×2、共享缩放常量 → 0.25、派生种子 +1）现在**全部被检出**为 `all_bitwise_equal is False`、恢复后回到 `True`，且断言仍非恒真（只改实现侧公式 → 两路均 False；只改归一化链 → `bitwise_equal` False 而 `raw_bitwise_equal` True，两路正确分离）；F2 —— `scale_override` 如实报 `1.0`、新增 `scale_used == 1.0`，离朱用**独立位移验证**（不看自述字段）确认实际口径为「**不缩放** + 可忽略平移」，docstring 已同步；F3 —— 现场 `robust_run.json` 驱动的 `calibrate_thresholds` 中**全部 6 条**满足 `population["cell_role"] == entry["cell_role"]`（R50 的字面断言现已成立）。

**唯一失败项（R51 条目 5.3，信息级 / 纯文案）已收口**：静态扫描要求 `assert_perturb_formulas_bitwise` **函数体内**不出现三个共享层标识符 —— `nmag_shift_matrix(` 与 `derived_seed(`（含尾括号）零命中，但 `NMAG_SHARED_SCALE` 在函数体内的 docstring / 注释 / `manual_side_note` 文案中命中 6 处（命中的是**描述「已去共享化」这件事的文字本身**）。离朱给出三条实质证据表明修复到位（AST 代码层引用为零、手算侧确已内联、行为证据三种破坏全被检出）。本轮按最小改动把该字面项**清零**：函数体内这三处文案中的裸常量名一律改写为「共享缩放常量」「符号矩阵 helper」「派生种子 helper」等描述性表述。**现状实测**：函数体内四个共享层标识符（`nmag_shift_matrix` / `derived_seed` / `NMAG_SHARED_SCALE` / `_perturb_type_index`）的**文本命中数与 AST `Name`/`Call`/`Attribute` 引用数均为 0**，G7 仍 9/9 全通过。

**因 `manual_side_note` 文案变化重建产物并重新验证跨目录双跑逐字节一致（最终值）**：`robust_probe.json` 23706 B `6436f0569a428359…`、`robust_run.json` 419102 B `173942833ae979e9…`、`robust_calibration.json` 24328 B `5736b859f78e5ec6…`、`robust_report.md` 23483 B `250be2fe7cbd33e8…`、`robust_report.json` 444783 B `bf6167a2fb24caa1…`（修复过程中产生的中间值 `2ec4df9f…` / `8408460e…` / `734b7fb3…` / `9387b156…` / `b293b778…` / `cee33a85…` 已作废，不再引用）。**数值结论逐位未变**：`invariants` 24 条全通过、G7 9/9、全局 τ `0.133333`、主判据 **5/6 有效**（`bge-m3-1024` 的 `mask` 落差 0.000000 且 `below_granularity = true`），36 格主判据 R@1 与登记值完全一致。`README.md` 15.11 新增「离朱 R51 复测（182 项检查 181 通过 / 1 失败）与字面项收口」小节并更新产物 SHA 表。
**皋陶第二次审查修复轮（1 warning + 3 info，2026-10-06）**

- **W5（唯一 warning）· 粒度守卫等号边界漏判 + 四处口径不一致（已修）**：文档侧（`MIN_GRANULARITY_NOTE` / 产物 `criterion` / README 正文）写「粒度 ≥ 落差者不得作为判定依据」（**含等号**），实现侧却是 `abs(gap) < min_granularity`（**严格小于**）⇒ 最脆弱的「落差恰为 1 个样本」情形未被守卫覆盖。修复：① 守卫改为在**比值空间**判定 `ratio = abs(gap_used) / min_granularity`，`ratio <= 1 + GRANULARITY_REL_TOL`（新增常量 `GRANULARITY_REL_TOL = 1e-9`）⇒ 判 `effective = False`；② 新增 `at_granularity` 标记（`|ratio − 1| <= 1e-9`），该格 verdict 显式改为「落差 = 粒度（= 1 个样本），**不构成判定依据**」，判定表新增「粒度标记」列；③ 新增常量 `GRANULARITY_GUARD_RULE` 作为**唯一规则文本来源**，被 `MIN_GRANULARITY_NOTE` 与产物 `criterion` **同时包含**（四处口径同源），`probe`/`run`/`calibrate` 三处产物均带该文本。**现场实测的坑（如实登记）**：第一版用浮点字面比较**仍然漏判** —— `qa/bge-m3-1024/nmag` 的 `gap_used = 0.050000000000000017` 与 `min_granularity = 1/20 = 0.050000000000000003` 数值相等但**非逐位相等**，字面 `<=` 判为「高于粒度」（`at_granularity = 0`）；改用比值空间后 `ratio = 1.0000000000000002` 正确命中。④ **G8 边界用例证据**（本轮新增门禁）：合成 4 例（`gap == 粒度`、`gap == 0.20−0.15` 真实浮点值、`±1e-6` 相对）全部 `match=True`；现场 operative 视图命中 **1 格**（`qa/bge-m3-1024/nmag`：`below=True`、`effective=False`、verdict 含「不构成判定依据」）；辅助视图（弱−强口径）另登记 **1 格**（`qa/lexical-88/nmag` 的 `gap_weak_to_strong` 恰为 1 样本，但其档序列非单调、operative 判据是整段落差 = 2 样本，故**只登记不否决**）。⑤ 新增 `granularity_boundary_evidence` 进 calibration 产物，五项一致性检查现场**全 True**（`synthetic_boundary_matches_rule` / `boundary_instances_are_ineffective_and_annotated` / `all_below_granularity_are_ineffective` / `rule_text_in_artifact_criterion` / `rule_text_in_min_granularity_note`）。**与审查描述的差异（如实登记）**：审查称 QA 侧有 **3 格** `gap_used` 恰等于 `min_granularity`，但**现场枚举当前产物**只有 **1 格**在 `gap_used` 空间命中（`qa/bge-m3-1024/nmag` 比值 = 1）；`qa/lexical-88/nmag` 的 `gap_used` 比值 = 2（其弱−强比值为 1，已登记为辅助视图）、`qa/bge-m3-1024/noise` 的 `gap_used` 比值 = 2（弱 0.45 → 强 0.35 = 2 个样本）。本轮处置覆盖**该机制本身**（含等号 + 比值空间 + 辅助视图登记），不依赖格数口径。
- **I4（docstring 措辞误导，已改）**：`entry_table.question_is_blank` 原写「沿用 `data.check_question` 的口径」，但该函数 docstring 明确「超长 -> 立即报错；**空 / 空白放行**」、**不做**空问题过滤。已改为「沿用 `features.normalize_text` 的口径（`data.check_question` 本身对空/空白放行，不做过滤）」。**过滤行为不变**。
- **I5（README 笔误，已改）**：`* **`qm` 的单调性风险已单独登记**` → **`nmag`**。
- **I6（补机器可读口径锚点，已补）**：新增常量 `NMAG_FORMULA_SPEC_DEVIATION`（357 字），随产物落盘于 `run` / `probe` 的 `grid.nmag_formula_deviation`，明确登记「任务书选甲为 `x <- x*(1+eps) + b`，本实现为 `x <- x*0.5 + eps*(-1)**(i+j+1)`，缩放系数取**三档恒定的 0.5** 而非 `1+eps`」及三条设计理由（量纲单一性 / 唯一自变量可归因 / 恒定正缩放对余弦检索不敏感）。

**产物逐字段对账（G5 / 零回归硬要求）**：`cells`（114 格）**逐位未变**、`entry_tables` / `euclidean_axis` / `invariants` / `stability`（36 行） / `unrecognized` **全部逐位未变**；`calibration` 新增 `at_granularity` / `granularity_ratio` / `weak_minus_strong_*` / `granularity_boundary_evidence` / `n_at_granularity_*` 等字段，并把 `qa/bge-m3-1024/nmag` 由「有效」改判为「**无效（落差 = 粒度）**」；**主判据侧仍 5/6**、全部 36 格 R@1 未变、`invariants` 仍 24 条全通过、G7 仍 9/9、全局 τ 仍 `0.133333`。旧→新 SHA256（前 24）：`robust_probe.json` 23706/`6436f0569a428359…` → 24515/`3e8c5aefed85d1b0…`；`robust_run.json` 419102/`173942833ae979e9…` → 420416/`4adb290e8e3fcea4…`；`robust_calibration.json` 24328/`5736b859f78e5ec6…` → 33641/`349e61e28267e404…`；`robust_report.md` 23483/`250be2fe7cbd33e8…` → 25898/`38fb583915ad42fd…`；`robust_report.json` 444783/`bf6167a2fb24caa1…` → 455605/`4a7668dcffd9227e…`。**跨目录双跑逐字节一致（G3）**在重建整条链后重新验证 **5/5 `bytes_equal=True`**（上一版曾因 `honest_notes` 内嵌规则文本与旧 `run` 产物不同步而短暂不一致，已用「当前源码重建整条链」消除）。`README.md` 15.6 与 15.11 已同步修订。
## 变体 B：可学变换与冻结特征库（第二步）

设计文档三步走的**第二步**：正面检验「**表内嵌输出层、权重即特征库**」这一架构能否让 `argmax` 真正可学。由 `n3d_qa_learn/variant_b.py` 承载，入口 `python -m n3d_qa_learn.step2_run variantb {probe|drill|train|eval|report}`（与 `robust` 平级，既有子命令与参数表逐字符未改）。**只变换、不生成**：单层 `D→D` 变换 `T` + **冻结**特征库内积评分，只学变换层。

**模型（`variant_b.DToDTransform` / `VariantBModel`）**：`T(x) = x @ W.T + b`，`W` 初值 = 单位阵、`b` 初值 = 全零（`torch.eye`/`zeros`，**不消耗 RNG**）；可训参数现场枚举恒为 `['weight','bias']`（元素数 `[D², D]`）。**恒等快路径**：`torch.equal(W, I) and (b == 0).all()`（**逐位判定、非容差**）时走**零加性**写法 `x + x@(W−I).T + (b − mean(b))` —— 两项在恒等时逐位为 `+0.0`，故 `T(x)` 与 `x` **逐位相同**、`norm(T(x)) ≡ norm(x) = 1` 在 float32 下逐位成立，且两项**可微**（`∂/∂W = x`、`∂/∂b = 1 − 1/D`）。**`normalize_query`** 是查询侧**唯一**归一化实现（float64 求范数；逐元素 float32 求和恰等于其 float64 精确值 ⇒ 与 `entry_table.l2_normalize_rows` **逐位一致**；零范数行显式抛 `ValueError`）。

**恒等门禁（硬门禁，五条现场证据）**：训练前 `T = I` 时，clean + 9 个扰动档的 top-1 必须与**余弦最近邻逐条一致**。证据 = ① 逐格 `n_mismatch == 0`；② `norm(T(x))` 与 `norm(x)` 逐位相等比例（现场 `1.0`）；③ `T(x)` vs 显式 `x @ W.T + b`（**训练态与评测态各一次**，现场均逐位相同）；④ 恒等态仍在计算图上且两参数梯度非零（`gradient_flow_evidence`）；⑤ **自洽判据** `score_path_evidence`：`model.score(q)` 必须逐位等于 `normalize_query(transform(q)) @ keys.T`。**现场实测两档均 10/10 格通过、五条证据全 True**（`lexical-88` 与 `bge-m3-1024`）。

**训练器（`variant_b.train_transform`）**：**扰动自监督** —— 训练查询 = 库行 **1999**（由「全量下标 − 查询行下标」求**补集**，不修改 1a 代码），标签 = 该行在**全量键表 2665** 中的行下标，`CrossEntropyLoss(logits = T(q) @ keys.T)`，Adam（**唯一优化器**，只优化 `T`），每步轮转 **9 个扰动格**；打乱用**局部** `torch.Generator(seed=SHUFFLE_SEED=4242)`，扰动走 1a 的 `derived_seed`，**不消耗全局 RNG**；同参重跑**逐位一致**（跨目录双跑产物逐字节相同，现场已验证）。**`train_mode="clean"`** 为 A 组（干净自监督）对照档。**可训参数更新量门禁** `update_gate`：逐参数裸字节 SHA256 前后比对，**只要有任一变化即通过**（不设阈值）。**训练侧零范数行边界（现场真实触发 1 次）**：批内某行被扰动后整行归零（`mask/strong` 遮蔽 44/88 维且该行仅 9 个非零维），处置为「原样保留为全零向量 + **在损失中掩码**（其 logits 恒为 0）+ 逐批计数进产物；整批全零则显式报错」；取回「扰动后未归一化矩阵」的路径是 `_perturb_allow_zero` —— 仅在 1a 抛「零范数」错时启用，用 **1a 自己产出的 `raw_perturbed_sha256` 逐字节反查**候选矩阵，**命中不了即 fail-closed 抛出 1a 的原始报错**，**1a 的 `perturb_matrix` 契约未改动**。

**评测（`eval_cell` / `cell_grid`）**：逐（特征档 × 扰动类型 × 档位）比较变体 B 与 **KNN**（= 冻结键表上的归一化余弦，与训练完全无关），报 `R@1`/`R@5`/Δ/命中条数/分档落差（弱 − 强）；名次口径与 1a 逐字一致（`rank = 1 + #{j : cos > 金标余弦}`；top-1 用 `argmax`，并列取最小下标，故逐条一致性判定是确定性的）。**决定性对照臂 `clean_eval_of_perturb_arm`**（扰动训练 × **clean 评测**，复用主臂**同一个训练后模型**，权重只在进程内传递、**不进产物**）把「`T` 本身的方向偏移」与「扰动泛化落差」分开。**主判据 `primary_criterion`**：在 1a 判**有效**的 15 格上 R@1 均不低于 KNN 且至少 1 格严格更高；1a 判无效的格（`bge-m3-1024` 的 mask 三档）仍报数字但**不参与**。**`anchor_1a_check`** 与 1a 的 `robust_run.json` 文本侧主判据格逐格对账（容差 `1e-6`，现场 **18/18** 全在容差内）。

**现场实测（单 seed 42，`epochs=30` / `batch=256` / `lr=1e-2`）**：**主判据通过 = False** —— 15 格中**严格更高 2 / 不劣于 2 / 严格更低 13**，Δ 区间 `[−0.794294, +0.133634]`（仅 `lexical-88` 的 `nmag/middle` +0.133634 与 `nmag/strong` +0.060060 为正）。**决定性对照臂**：`lexical-88` clean 格 `0.519520` vs KNN `1.000000`（**Δ −0.480480**，346/666）、`bge-m3-1024` `0.986486`（Δ −0.013514，657/666）⇒ **掉点主因是变换层本身**，不是扰动泛化落差。**A 组（干净自监督）**同量级为负（clean 格 −0.484985 / −0.006006）⇒「干净训练 ≈ 无变化」在本架构下被实测否证。训练侧自命中 top-1 = `0.342171`（lexical）/ `0.632816`（bge），说明线性 `D→D` 的表达力/优化前提尚未落实。**第三步方向依据**（`third_step`）：先落实「为什么连训练集都拟合不到 1」，再改**候选粒度**（逐行键 → 类/簇级候选，或让键表随表示重算）—— clean 格是**逐位自匹配**（`cos = 1.0`，结构性不可超越），故任何非恒等变换在该粒度上**只能变差**；**不要**继续加大变换层容量。**空间不足格**单列（`bge-m3-1024` 的 clean/noise-weak/三档 mask 的 KNN 错误条数 = 0/1/0/0/0；`lexical-88` 的 clean = 0、mask/weak = 6）。

**两条踩坑登记（均已修，写入代码注释与 README 16.2）**：① 恒等快路径首版写成 `return x` ⇒ 输入脱离计算图，训练第一步 `backward()` 报 `element 0 of tensors does not require grad`；② **`score` 首版漏掉 `self.transform(x)`** ⇒ 评测侧退化成「未训练模型 = 余弦 KNN」、全部 Δ **结构性恒为 0**，而恒等门禁**照样通过**；由新增的自洽判据 ⑤ 现场抓出并修复，本节全部数字为**修复后**的现场值。

**产物纪律与零回归**：一律写 `checkpoints/qa_learn/_verify/variant_b/`（`variantb_probe.json` 13098 B `537814d95d933227…`、`variantb_drill.json` 107049 B `5fe2e2f31e7feab7…`、`variantb_eval.json` 340526 B `ad7d90c2fefc5c2d…`、`variantb_report.md` 14237 B `0bd4b2fde9dbfd67…`、`variantb_report.json` 340913 B `233838e81a1e9c3d…` + 三个 UTF-8 无 BOM 日志），**不落盘特征矩阵、不产 zip 产物**，产物内不含挂钟时间/耗时；**跨目录双跑逐字节一致**（probe 与 eval 均现场验证）。`variantb` 四个子命令入口统一校验 `--profiles` / `--train-mode`（非法值单行可读报文 + **退码 2**，无裸 Traceback）；`eval` 是验收入口，**恒等门禁**与**可训参数更新量门禁**任一不过即**退码 1**，而**主判据不参与退码**（否证条款要求如实报负结论）。`entry_table.py` / `robust_eval.py` **零改动**；顶层 9 个 `qa_*.pt.zip` 的 SHA256/字节/mtime 逐项不变、顶层无新增文件；上游四目录与根 `requirements.txt` 的 `git status` 为空；**零新增依赖**；既有入口 `step2_run probe` 与 `cli probe` 退码 0。**如实登记的自我更正**：`ROBUST_1A_KNN_REFERENCE` 初版按 README 15.5 正文抄了 `bge-m3-1024` 的 `nmag` 三格（`1.000000/0.996997/0.983483`），与 1a **run 产物**（`0.481982/0.144144/0.043544`）不同源；本模块一律以**产物为唯一现场来源**并已按产物更正，**不静默对齐**、不改动 1a 任何文件。**未做项**：多 seed、λ/轮数/lr 扫描、对齐/SupCon 辅助损失（口径明确排除）、类/簇级候选（属第三步）。
**离朱 R57 验收（第 1 轮）**：7 个套件、**101 项断言 101 通过 / 0 失败 / 0 跳过**。关键现场证据：① **`score` 自洽判据有负向区分度** —— 现场构造「`score` 只归一化、不走变换层」的假模型 ⇒ `bitwise_equal is False`（证明它不是恒真空断言，能拦住首版那类回归）；② `is_identity()` 是**逐位**判定（`weight[0,0]` 加 1 个 ULP、`bias[3]` 取最小次正规数 `1.4e-45` 都判 False）；③ 恒等门禁负向用例：`bias += 0.5` ⇒ `0/10` 格一致，非等比 `weight[0,0] *= 3` ⇒ `3/10` 格一致；④ **零范数真实触发点被复现**（`n_steps=240`、`epoch 21 / step 168`、`mask/strong`、`table_rows=[23]`、`zero_norm_rows.total=1`）；⑤ `_perturb_allow_zero` fail-closed 反向用例成立；⑥ 测试前后 8 个正式产物逐字节一致（未污染）。**R57 提出的 6 条观察项收口**：**F2（warning，已修）** 把「一处不同源」的登记补进 `ANCHOR_RULE`，使该文本同时进入 `evidence.knn_reference_rule`（`train` 产物）与 `anchor_1a_check.rule`（`eval`/`report` 产物），并重建 `variantb_eval.json` / `variantb_report.json`（**数值零变化**）；**F5（info，已修）** 出处更正 —— 旧值 `1.000000/0.996997/0.983483` 实际出自 README **§15.10「修复前后完整对照」表的「改前」列**（W1 修复前历史值），而 §15.5 主表已是产物值 `0.481982/0.144144/0.043544`；**F1/F3/F4/F6 如实登记不改代码**（F1 属说明措辞；F3 的 `weight *= 2` 是保序缩放、门禁靠另两条拦下；F4 的 drill 登记指纹对应**两档**命令；F6 的 `anchor_1a_check` 为 18 行口径、不含 clean 格，另做含 clean 格的 20/20 对账）。README 新增 16.7（一处「不同源」的登记）与 16.8（R57 验收与观察项收口）。

**离朱 R58 复测（收口轮）**：7 个套件、**105 项断言 105 通过 / 0 失败 / 0 跳过**（新增 `t9_closure.py` 专项 14 项）。逐项确认：F2 收口**可见**（`ANCHOR_RULE` 485 字符同时逐字符出现在 `anchor_1a_check.rule` 与 `evidence.knn_reference_rule`，文本含新值 3 个 + 旧值 3 个 + 出处 `§15.10`/「改前」+「唯一现场来源」+「不静默对齐」）；F2 收口**不改数值**（`anchor_1a_check` 18/18 在 1e-6 内、最大 `abs_diff` `4.984984984801599e-07`，现场两档重算与产物逐位一致）；F2 收口**不改 probe 产物**（仍 13098 B / `537814d95d933227`，且不含该规则文本）；F5 收口（常量注释 / `ANCHOR_RULE` / README 三处均不再把旧值出处写成「§15.5 正文」，全文对该词组的提及全部处于「更正说明」语境）；回归全绿（恒等门禁五条证据、`score` 自洽判据负向区分度、`TrainConfig` 校验、不消耗全局 RNG、同参重跑逐位一致、`update_gate`、划分与键表指纹、主判据 2/2/13、决定性对照臂 −0.480480 / −0.013514、CLI 错误边界、零回归）。**R58 新发现 F7（info，已修）**：README §16.8 初稿把 `train_top1_self` 写成 `0.34217108554277193`，产物实际为 `0.34217108554277137`（差 5.55e-16），已按产物更正并把「与产物逐位一致」改写为可核对形式（写明取自哪个产物的哪个字段），**不涉及代码、口径与数值结论**。README 新增 16.9（R58 复测与 F7 更正）。

**产物最终指纹**（`checkpoints/qa_learn/_verify/variant_b/`，8 个文件）：`variantb_probe.json` 13098 B `537814d95d933227`、`variantb_drill.json` 107049 B `5fe2e2f31e7feab7`（**两档** drill 命令）、`variantb_eval.json` 341179 B `d094c9bd9200120d`、`variantb_report.md` 14237 B `0bd4b2fde9dbfd67`、`variantb_report.json` 341566 B `33874ccd128fd221` + 三个 UTF-8 无 BOM 日志。`ANCHOR_RULE` 变更前的 `variantb_eval.json` `ad7d90c2fefc5c2d…`（340526 B）与 `variantb_report.json` `233838e81a1e9c3d…`（340913 B）**已作废**不再引用。

**训练前提校准轮（第二步的后续诊断轮）**

> 判定对象 = 第二步否证的**归因**：是「优化前提未落实」还是「线性 `D→D` 表达力不足」。
> **不以提升指标为成功标准。** 触发点 = 第二步的冲突证据（恒等初始化训练行干净自命中 100%，
> 训练 30 epoch 后训练侧自命中只剩 `0.342171` ⇒「训练在自己的目标上就失败了」，否证不干净）。

**口径（用户已确认，不得擅自变更）**：① 本轮特征档**只跑 `lexical-88`**（诊断轮成本优先），
`bge-m3-1024` 未跑、其归因**待补**；② 划分不变（键表全量 2665 / 训练行 1999 / 查询行 666 /
交集 0）；③ 单 seed 42、确定性、局部 RNG；④ **1a 考卷零改动**（`entry_table.py` /
`robust_eval.py` 未修改，扰动与归一化一律复用）；⑤ 产物一律写
`checkpoints/qa_learn/_verify/variant_b/`，不落盘特征矩阵、不产 zip、无挂钟字段；
⑥ 产物内所有数字必须现场真实执行；⑦ 观测量粒度：训练行 1999 ⇒ 1/1999 ≈ 5.0e-4。

**新增对外接口**
- **训练侧打分口径开关** `TRAIN_SCORERS = ("normalized", "raw")`、`DEFAULT_TRAIN_SCORER = "normalized"`、
  `TRAIN_SCORER_RULE`（唯一规则文本，进 `TrainConfig.as_dict()`、产物 `evidence` 与 Markdown）；
  `TrainConfig.train_scorer`（构造期校验；CLI `--train-scorer`）；
  `VariantBModel.train_score(x, scorer)`（训练侧**唯一**打分入口）与
  `VariantBModel.normalize_query_allow_zero(x)`（零范数行保持全零、不报错，**仅训练侧**；
  与严格的 `normalize_query` 在无零行时逐位相同，由 `train_scorer_evidence` 现场断言）。
  `normalized` = `normalize_query(T(x)) @ keys.T`（**与评测侧 `score` 同一口径**）；
  `raw` = `transform(x) @ keys.T`（现状未归一化口径，**逐位复现第二步**）。
  **评测侧口径不变**；**零范数行掩码口径不变**（保留全零 + 损失掩码 + 逐批计数，整批归零显式报错）。
- **恒等参照锚点** `TRAIN_ANCHOR_RULE` / `identity_anchor_grid(model, data)`：未训练（`T = I`）在
  **训练行 1999** 上 clean + 9 格的 top-1（含 R@5 与 KNN 对照），即「训练必须至少不劣于此」的
  硬性下限；调用前硬断言模型处于恒等态。`VariantBData.features_of(rows)` / `row_label(rows)`
  为其唯一取值与口径标签点；`eval_cell` / `cell_grid` 新增 `rows=` 形参
  （`None` = 查询行 666，历史行为逐位不变）使锚点与评测**共用同一条代码路径**。
- **训练无害下限** `TRAIN_HARMLESSNESS_RULE` / `TRAIN_HARMLESSNESS_EXCLUDED_CELLS = ("clean",)` /
  `train_harmlessness(anchor, post)`：训练后同口径**逐格不低于**锚点；`clean` 格**不纳入判据**
  （逐位自匹配、锚点结构性 1.000000）但仍在表中报出；另给 `aggregate_reading`（整段平均读法，
  **不参与判定**）与 `per_cell_including_clean` / `all_not_worse_including_clean`。
- **训练前提校准** `CalibrateCombo`（`scorer:lr:wd`，含 `label` / `parse`）、
  `build_calibration_grid`（`CALIBRATE_LR_GRID = (1e-3, 1e-2)`、`CALIBRATE_WD_GRID = (0.0, 1e-2)`、
  `CALIBRATE_SCORERS = TRAIN_SCORERS`，顺序 = scorer 外层 / lr 中层 / wd 内层）、
  `select_best_combo`（按 `CALIBRATE_SELECTION_RULE` 选最优超参，只在 `normalized` 内选）、
  `attribution_verdict`、`run_calibration(..., on_progress=...)`（每完成一个组合回调一次
  ⇒ 扫描可中断、可单组合复跑）、`render_calibrate_markdown`、常量 `CALIBRATION_SCOPE_NOTE`。
- **机器可读归因** `ATTRIBUTION_REQUIRED_CONDITIONS = ("a_train_harmless", "b_eval_not_worse")`、
  `ATTRIBUTION_CONDITION_TEXT`、`ATTRIBUTION_RULE`（由前两者经 `_build_attribution_rule()` **派生**，
  与判定实现同源）、`ATTRIBUTION_VERDICTS = {optimization_premise, linear_expressivity, mixed_per_profile}`。
  结论单点来源 = `selection.<档>.selected` 指向的组合的 `train_harmlessness` 与 `eval_primary`。
- **W2 修复（取回路径）** `RETRIEVAL_RULE` / `RETRIEVAL_ZERO_TOKEN` / `RETRIEVAL_MAX_ROUNDS` /
  `_zero_rows_from_error` / `_sentinel_row` / `_with_sentinel_rows` / `_manual_perturb_inlined` /
  `perturb_retrieval_evidence` / `perturb_retrieval_reverse_check`；`_perturb_allow_zero` 重写为
  「成功分支 = 1a 本尊（`direct_1a`）／失败分支 = 替换零范数行输入后**重新调用 1a 本尊** + 按下发
  约定置零（`repaired_via_1a`）+ 两个哨兵的行独立性逐字节交叉验证」；**删除**按公式重放的
  `_detail_from_failed_perturb`（其 docstring 承诺未实现）。
- **W1 修复（字段改名）** `TOP1_DISCREPANCY_KEY = "top1_discrepancy_vs_knn"`：`cell_grid` 内嵌的
  top-1 逐条差异**统计量**改名并显式标注 `model_is_identity` / `computed_after_training`；
  `identity_gate` 侧新增 `computed_on` / `discrepancy_key` 与第 ⑥ 条自洽判据（`train_scorer`）。
- **范数诊断** `train_transform` 新增 `norm_diagnostics`（逐 epoch：`mean_logit_abs`、
  `mean_row_norm_x` / `mean_row_norm_Tx`、`weight_fro` / `weight_minus_identity_fro` / `bias_norm`，
  探针 = 库行前 256 行的干净特征，固定无随机）与 `norm_diagnostics_summary`
  （含 `norm_ratio_last_over_first`；**只作机制取证、不参与任何判定**）。

**CLI（`step2_run` 的 variantb 子命令族，与 probe/drill/train/eval/report 平级，纯新增）**：
`variantb calibrate` + `--calibrate-scorers/--calibrate-lr/--calibrate-wd/--grid-combo/--dry-run`
（`--dry-run` 只跑网格第一个组合；`--grid-combo "scorer:lr:wd"` 单组合复跑）；新增
`--train-scorer`（`choices`）；`_float_list` 解析数值网格（空串回退默认）；非法值一律
**单行可读报文 + 退码 2**。`calibrate` 的恒等门禁 / 更新量门禁 / 取回路径（正向 + 反向验证）
任一不过退码 1，**归因结论不参与退码**。

**现场实测（单 seed 42，`epochs=30`，`batch=256`，`lexical-88`，全量耗时 56.5s）**
- **恒等锚点表**（训练行 1999，top-1）：`clean 1.000000`、`noise/weak 1.000000`、
  `noise/middle 0.986993`、`noise/strong 0.774887`、`mask/weak 0.992996`、`mask/middle 0.956978`、
  `mask/strong 0.763382`、`nmag/weak 0.355178`、`nmag/middle 0.101051`、`nmag/strong 0.037519`；
  锚点与 KNN 逐格一致 **10/10**；恒等门禁 **10/10 格逐条一致、通过 = True**（六条证据全 True，
  含新增的训练侧打分口径自洽判据）。
- **扫描表（8 组合）**：`训练侧自命中 / 训练无害格数 / ‖T(x)‖末首比 / 评测侧不劣`：
  `normalized|lr=0.001|wd=0` **0.887444 / 3-9 / 1.0009 / 3-9（严格更高 3）**；
  `normalized|lr=0.001|wd=0.01` 0.885943 / 3-9 / 0.8076 / 3-9；
  `normalized|lr=0.01|wd=0` 0.608804 / 0-9 / 1.9477 / 0-9；
  `normalized|lr=0.01|wd=0.01` 0.437219 / 0-9 / 0.5208 / 0-9；
  `raw|lr=0.001|wd=0` 0.517259 / 3-9 / 2.3116 / 3-9；
  `raw|lr=0.001|wd=0.01` 0.587794 / 2-9 / 1.1136 / 2-9；
  `raw|lr=0.01|wd=0`（**第二步现状档**）**0.342171 / 3-9 / 8.4166 / 2-9**；
  `raw|lr=0.01|wd=0.01` 0.199600 / 0-9 / 0.8319 / 0-9。更新量门禁 8/8 通过。
- **口径对齐前后（同 lr/wd 配对）**：`Δ训练侧自命中 = +0.370185 / +0.298149 / +0.266633 / +0.237619`；
  `raw|lr=0.01|wd=0` 的训练侧自命中 = `0.34217108554277137`，与第二步产物字段**逐位相同**
  ⇒ 重构**未改动 `raw` 档行为**；同档 `‖T(x)‖` 末/首 = **8.4166**、`mean|logit|` 末值 1.5046
  ⇒ **范数膨胀退化解通道现场证实**（`normalized` 同超参下 1.9477、最优组合 1.0009）。
- **最优超参**（按先声明的选取规则）＝ `normalized|lr=0.001|wd=0`（`raw` 四组合标
  `eligible = false`、只作对照）。
- **归因结论**：`a_train_harmless = False`（不劣 3/9、最差余量 −0.125063）、
  `b_eval_not_worse = False`（不劣 3/9、严格更高 3、严格更低 6）
  ⇒ **`verdict_key = "linear_expressivity"`**（线性 `D→D` 表达力 / 架构不足成立）。
  逐格取舍：`nmag` 三格 `+0.644822 / +0.898949 / +0.962481`，`noise`/`mask` 六格
  `−0.000500 ~ −0.125063`；取舍与锚点高度相关（锚点最低的三格正是唯一变好的三格）。
  **整段平均读法**（不参与判定）：训练侧 `mean_margin +0.242232`、评测侧 `+0.232733`，
  该读法会翻成 `optimization_premise`，产物以 `alternative_reading.agrees_with_primary = false`
  并列登记，**不得只引用其中一种**。
- **W2 反向验证**：四类共享层破坏（`derived_seed(+1)` / `NMAG_SHARED_SCALE=0.25` /
  `nmag_shift_matrix→全零` / `PERTURB_GRID["mask"]["strong"]=0.25`）**4/4 全部被检出**，
  基线 True、恢复后 True、总判定 True；手算侧 `manual_side_isolated = True`。
- **W1 现场核对**：同一产物内 `identity_gate.passed = True`（`computed_on = fresh_model(T=I)`）
  与 `cell_grid_discrepancy.*.top1_discrepancy_vs_knn.all_cells_equal = False`
  （`computed_after_training = True`）并存且**不同名**（同名冲突存在 = False）。
- **确定性与产物**：两次不同输出目录的全量跑 `variantb_calibrate.json`（478975 B
  `669bef989008f44f`）与 `variantb_calibrate.md`（22915 B `04ee69123b3e11c4`）**逐字节相同**；
  增量落盘 8 次（每组合一次）；`calibrate_dry/variantb_calibrate.json` 111579 B
  `21a1ebb8ea0e2a0a`、`calibrate_drill/variantb_drill.json` 64021 B `4feec1283da3e438`。
- **零回归**：`entry_table.py` / `robust_eval.py` 零改动；顶层 9 个 `qa_*.pt.zip` 的
  SHA256/字节/mtime 逐项不变、顶层无新增文件；上游四目录与根 `requirements.txt` 的
  `git status` 为空；零新增依赖；第二步的 4 件既有产物
  （`variantb_probe.json` 13098 B `537814d95d933227`、`variantb_drill.json` 107049 B
  `5fe2e2f31e7feab7`、`variantb_eval.json` 341179 B `d094c9bd9200120d`、
  `variantb_report.json` 341566 B `33874ccd128fd221`、`variantb_report.md` 14237 B
  `0bd4b2fde9dbfd67`）**未重跑、逐字节不变**，且仍可被新版 `report` 渲染（现场退码 0）。
- **如实登记（不得包装）**：① 本轮**没有**通过「训练无害」下限（最优组合仍 3/9），
  如实报「**训练仍有害**」；② 判据读法（逐格 vs 整段平均）会**翻转**结论，产物已并列登记；
  ③ `bge-m3-1024` 的归因**待补**；④ 单 seed、无跨 seed 极差；⑤ 网格为最小网格，
  未扫 epochs / batch_size / 优化器。

**离朱 R59 验收与 F1~F4 收口（训练前提校准轮）**

**R59 结果**：8 套件 / **357 项断言全部通过（0 失败）**，另发现 2 个真实缺陷 + 2 处文案不符。
- **F1（严重·回归·已修）**：本机 `sys.stdout.encoding = cp936`（GBK），而 GBK **没有**
  `U+2212`（`−`）与 `U+21D2`（`⇒`）。本轮新增的 **3 处运行时日志字面量**含这些字符
  （`variant_b.py` 的 epoch 日志、`step2_run.py` 的两条 `calibrate` 日志），使 `print` 抛
  `UnicodeEncodeError`；该异常**继承 `ValueError`**，被 `cmd_variantb` 的
  `except (ValueError, KeyError)` 兜住 ⇒ 报成「非法参数 + 退码 2」、**一次跑完的长跑成果被整条丢弃**
  （`train` / `eval` / `calibrate` 均受影响）。**修法三层**：① 字面量改 ASCII（`−`→`-`、`⇒`→`=>`）
  并新增常量 `step2_run._ASCII_SAFE_LOG_NOTE` 固化编码纪律；② `step2_run._log` 对不可表示字符做
  `backslashreplace` 转义后再输出（兜底，不改数值与产物）；③ `cmd_variantb` 新增
  `except UnicodeEncodeError`（**置于 `(ValueError, KeyError)` 之前**）报「输出编码错误 + 退码 1」。
  **修复后现场实测（不设 `PYTHONIOENCODING`）**：`calibrate` / `train --epochs 1` / `eval --epochs 1`
  / `drill` **全部退码 0**，取证产物在 `_verify/variant_b/gbk_runtime_check/`（`--epochs 1`，非矩阵档）。
- **F2（中等·健壮性·已修）**：`--calibrate-scorers raw` / `--grid-combo "raw:..."`（合法：`raw ∈
  TRAIN_SCORERS`）原先在**训练全部完成之后**由 `select_best_combo` 抛 `ValueError` ⇒ 误报
  「非法参数 + 退码 2」、`selection` / `attribution` / 指纹缺失、`.md` 不渲染、只留 `status="partial"`。
  **修法**：`select_best_combo` **不再抛异常**，改为返回 `applicable = False` + 可读 `reason`
  （`candidates` 全量保留）；新增 `attribution_not_applicable(reason, candidates)` 与
  `ATTRIBUTION_VERDICTS["not_applicable"]`；`run_calibration` 写
  `selection.<档>.applicable = false` / `attribution.<档>.verdict_key = "not_applicable"`，**退码 0**。
  **现场实测**：`--calibrate-scorers raw` 退码 0（102949 B `445211c1a98b65df`）、
  `--grid-combo "raw:0.01:0"` 退码 0（102949 B `692116c1e6ad62e7`，`train_top1 = 0.342171`）。
- **F3（文案·已修）**：「既有子命令参数表逐字符未改」不准确 —— parser 另触及 `--epochs` 的 help
  与 `variantb_cmd` 的 `choices`（旧选项集合 ⊆ 新选项集合，语义未变）。已改为精确表述。
- **F4（文案·已修）**：「零范数行 logits 恒为 0」只在 `T = I` 时成立（训练后 `T(0)=b≠0`，
  实测 logits 量级 0.3487 / 0.2163）。功能无缺陷（掩码取自**变换前**的 `q`，该行一律不参与损失与梯度）；
  `zero_norm_mask` / `train_transform` docstring、产物 `zero_norm_rows.rule`、`_honest_notes`
  与 README 措辞统一更正为「该行一律被掩码、与 logits 取值无关」。
- **产物指纹变化（如实登记）**：F2 使 `selection.<档>` 新增 `applicable` / `reason` 两键 ⇒ 校准类产物重建：
  `variantb_calibrate.json` **478975 B `669bef989008f44f` → 479903 B `981ee3cbfaa15c92`**、
  `calibrate_dry/variantb_calibrate.json` **111579 B `21a1ebb8ea0e2a0a` → 111730 B `cf5d97c3dc7a7371`**；
  `variantb_calibrate.md`（22915 B `04ee69123b3e11c4`）与 `calibrate_drill/variantb_drill.json`
  （64021 B `4feec1283da3e438`）字节未变。**数值与结论零变化**（8 组合自命中、锚点 10 格、
  最优组合 `normalized|lr=0.001|wd=0`、`verdict = linear_expressivity`、跨目录逐字节一致）。
- **R59 独立复现的规格事实（全部通过）**：锚点 10 格逐值一致；8 组合自命中逐值一致；
  `raw|lr=0.01|wd=0 == 0.34217108554277137` 逐位相同；W2 正向 True / 反向 4-4 detected /
  `_detail_from_failed_perturb` 确已删除 / 手算侧 AST 零共享层引用 / 行独立性用自建哨兵独立复核；
  门禁第 ⑥ 条确实接入 `passed`（强制 False 后门禁失败）；非恒等态锚点 fail-closed 且不改模型、
  不耗全局 RNG；归因不参与退码（`verdict = linear_expressivity` 仍退码 0）。

**离朱 R60 复测与 3 项 info 级观察项（收口轮）**

**R60 结果**：10 套件 / **469 项断言 / 0 失败**，未发现新缺陷；R59 的 F1~F4 **全部独立复现为已修复**。
- **F1 三层全验**：AST 全模块扫描 **0 处**非 cp936 可编码的运行时日志字面量（R59 为 3 处）；
  `_log` 在 cp936 / ascii 严格档下注入 `U+2212` / `U+21D2` 均**不抛异常**（转义为 `\u2212` / `\u21d2`，
  其余文本保全，纯 ASCII 输入与 `print` 逐字节相同）；AST 确认 `except UnicodeEncodeError` 排在
  `(ValueError, KeyError)` **之前**，注入时**退码 1 + 「输出编码错误」**、不含「非法参数」，
  普通 `ValueError` / `KeyError` 仍退码 2。**不设 `PYTHONIOENCODING`（本机 cp936）下**
  `calibrate` / `train --epochs 1` / `eval --epochs 1` / `drill` **全部退码 0 并产出产物**。
- **F2**：`select_best_combo` 无 `normalized` 时不再抛（`applicable=False` + `reason` + `candidates` 全留，
  空列表同样不抛）；`--calibrate-scorers raw`（4 组合）与 `--grid-combo "raw:0.01:0"` 均**退码 0**、
  `status=complete`、含 selection/attribution/fingerprint、`.md` 已渲染、扫描数字全保留
  （0.517259 / 0.587794 / 0.342171 / 0.199600；单组合 `0.34217108554277137` 逐位），
  且与作者现场产物 `calibrate_rawonly/`、`calibrate_rawonly_single/` **逐字节相同**；
  `not_applicable` **不驱动退码**。
- **F3 / F4**：目标位置表述已改为精确版（选项集合 = 旧集合 ⊂ 新集合；零范数行「一律被掩码、
  与其 logits 取值无关」），且「恒为 0」的任何出现均带更正语境。**F4 功能实质亦被验证**：
  掩码取自**变换前**的输入 `q`；训练后 `T(0)=b≠0`（探针实测该行 logits `0.9219`）；
  把该行 logits 改写为任意值后**损失逐位不变**。
- **规格现场事实逐项一致**：`variantb_calibrate.json` 479903 B `981ee3cbfaa15c92`（F2 新增
  `applicable`/`reason` 两键所致）、`variantb_calibrate.md` 22915 B `04ee69123b3e11c4`、
  `calibrate_repro/` 与主产物逐字节相同、`calibrate_dry/` 111730 B `cf5d97c3dc7a7371`、
  `calibrate_drill/variantb_drill.json` 64021 B `4feec1283da3e438`；恒等锚点 10 格与 8 组合自命中
  逐值一致；最优组合 `normalized|lr=0.001|wd=0`；`verdict = linear_expressivity`；
  耗时 55.5/59.6/60.3s；增量落盘 8 次；R59 旧指纹 478975 B `669bef989008f44f` **已作废**。
- **零回归（R60 复核 + 测试后二次「临时目录外零漂移」）**：1a 考卷两文件零改动；
  顶层 9 个 `qa_*.pt.zip` 的字节/SHA256/mtime 三项不变、顶层无新增（本轮新增的
  `calibrate_*/`、`gbk_runtime_check/` 均在 `_verify/variant_b/` 子目录内）；第二步 5 件既有产物
  逐字节不变且指纹逐项吻合；4 条 `git status` 路径为空；`compileall` 退码 0；零新增依赖；
  仅声明 4 个文件被修改；CLI 错误边界与向后兼容保持。

**3 项 info 级观察项（非缺陷；本轮如实登记、不修改）**：
- **O1**：`step2_run._ASCII_SAFE_LOG_NOTE` **自身**以 `−` / `⇒` 作反例插图 ⇒ 该常量**不可 cp936 编码**，
  且 AST 核验**当前无任何引用**。因它**从不进日志路径**（`_log` 兜底层也不会崩）故不影响运行，
  但「固化 ASCII 纪律」的常量自身违反纪律属实。**建议（未执行，避免再次变更已验收源码）**：
  插图改用 `U+2212` / `U+21D2` 文字描述或标注「仅供文档」。
- **O2**：两件**冻结产物** `variantb_eval.json` / `variantb_report.json` 仍含 F4 修复前的旧措辞
  （零范数行 logits「恒为 0」一类）—— **零回归要求逐字节不变**，故不得改动。
  本模块「**零回归 ↔ 统一措辞**」的固有张力，**明确接受残留并登记**；该句**只在 `T = I` 时成立**。
- **O3**：「逐字符未改」在**非目标位置**仍有残留（`step2_run.py` 的 `cmd_variantb` docstring 说的是
  「分发逻辑」、robust 族两处，其中一处与 HEAD 逐字符相同、属本轮范围外）；**目标位置已按 F3 更正**。

**口径收口修复轮（皋陶 approved=false：2 error / 2 warning / 2 info；只修文本与口径，不改数值结论）**

**硬约束（现场断言，全部满足）**：`attribution_summary.verdict_key == "linear_expressivity"`、
`selection.<档>.selected == "normalized|lr=0.001|wd=0"`、`candidates_ranked` 顺序不变、
`alternative_reading.agrees_with_primary == false`、恒等锚点 10 格逐值不变、
8 组合批内滚动平均逐值不变、逐组合无害格数 / `min_margin` / 评测侧三项计数逐值不变。
**未重跑超参搜索**；重建产物用的是同参复跑（seed 42 / epochs 30 / batch 256 / lexical-88）。

**逐条落地**
1. **error 1（已作废的 W2 机制描述）**：README §16.3 的「训练侧用 `raw_perturbed_sha256` 反查取回
   1a 的中间矩阵，命中不了即 fail-closed 抛出 1a 的原始报错」——该路径已在 §16.11.7 删除
   （反查恒命中、`raise` 分支不可达）。现就地改写为**现实现**（成功分支 = 1a 本尊 `direct_1a`；
   失败分支 = 哨兵替换零范数行输入 → **重新调用 1a 本尊** → 只丢弃被替换行并置零
   `repaired_via_1a` → **两个不同哨兵的行独立性逐字节交叉验证**），并显式标注旧描述**已作废**、
   指向 §16.11.7 与 `RETRIEVAL_RULE`（唯一口径来源）。
2. **error 2（归因写反）**：`_calibration_honest_notes` 的逐格取舍 note 面向被选中的
   `normalized` 组合却写成「`raw` 口径下的『训练失败』」，且「弱扰动格被强扰动格牺牲」与现场相反。
   **修法**：note 改为**按组合归属** —— 本最优组合下**变好 = `nmag` 三格**（锚点最低，
   `+0.644822 / +0.898949 / +0.962481`）、**变差 = `noise`/`mask` 六格**（锚点最高，
   `−0.000500 ~ −0.125063`），并显式否证「弱被强牺牲」；`raw` 的结论**下移为独立一条**，
   只引用 `raw` 组合自己的逐格读数（按 `min_margin` 取 `raw|lr=0.001|wd=0.01`：
   变好 = `nmag/weak +0.249125`、`nmag/middle +0.057529`；变差 = `noise`×3 + `mask`×3
   与 `nmag/strong −0.001001`）。README §16.11.6 同段更正并标注更正来源（皋陶 error 2）。
3. **warning 3（主报量口径）**：原先把「训练行 1999 上 clean + 9 格的 top-1」当作训练侧观测量，
   但产物主打的 `train_top1_self` 实为**批内滚动平均**（每步单格轮转、批内 256 行、
   用该步**更新前** logits），与锚点不是同一观测量（同组合 `0.887444` vs 逐格均值 `0.905453`）。
   **修法**：① `TRAIN_HARMLESSNESS_RULE` / `TRAIN_ANCHOR_RULE` / `ATTRIBUTION_CONDITION_TEXT`
   改为明确「**逐格 R@1**（跑一次完整 10 格网格）」并声明**不得**代入批内滚动平均；
   ② 报告主表（`render_calibrate_markdown` §2）与 README §16.11.3/§16.11.4 表头改为
   **逐格 R@1 为主报量**、批内滚动平均显式标注「**仅诊断**」；
   ③ 新增产物字段 `train.running_vs_grid`（现场差值 + **逐格一致性硬校验**：
   `train_side_cells[].variant_b_recall_at_1` 与 `train_harmlessness.per_cell[].post_top1`
   必须逐格相等，不等即抛 `ValueError` 拒绝出数；现场 8/8 组合 `per_cell_consistent = true`）。
   逐格 R@1 **直接取自产物既有字段，未重算**。
4. **warning 4（选取规则自相矛盾）**：`CALIBRATE_SELECTION_RULE` 既写「clean + 9 格中…」又写
   「同 10 格的平均值」，而 `train_harmlessness()` 已排除 `clean` ⇒ 格数与数字两义。
   **修法**：规则文本改写为**与实现逐字对齐** —— ① 明确「**9 个扰动格**（`clean` 已按
   `TRAIN_HARMLESSNESS_EXCLUDED_CELLS` 排除）」；② 明确「整段平均的格集合 = **10 格
   （`clean` 计入）**，与 ① 不同」；③ 新增常量 `CALIBRATE_CANDIDATE_CELL_SET` 并逐项写入
   `selection.<档>.candidates[].train_post_top1_mean_cell_set`。
   **如实登记一处与皋陶建议的差异**：皋陶给出两条等价修法（统一为 9 格 / 声明 ② 同样排除 clean），
   二者都要求**改实现**（② 的值会从 `0.914907` 变成 `0.905453`）；本轮硬约束是
   「只修文本与口径、不得改变任何数值」，故选择**把文本对齐到实现**（② 保持 10 格并显式标注），
   **不修改任何数值**。
5. **info 5（行集合不可互校 + calibrate 缺 1a 对账）**：新增常量 `ANCHOR_ROW_SET_NOTE`
   （唯一来源）并写入 `identity_anchor.row_set_note` 与 `anchor_1a_check_row_source`；
   **`calibrate` 现在也产出查询行 666 侧的 `anchor_1a_check`**（取第一组合的评测网格；
   KNN 与模型无关，现场新增不变量 `knn_invariance_across_combos.all_equal = true`，
   `n_distinct_knn_vectors = 1` / 8 组合）；现场对账 `9/9` 行在 1e-6 内。
   README 锚点表加行集合警告（`nmag/weak` 训练行 `0.355178` vs 查询行 `0.391892`）。
6. **info 6（同名概念互串）**：`train_top1_self` → **`train_batch_running_top1`**
   （`history[]` 与 `train` 两处；现场 `'train_top1_self' in combos[0]['train'] == False`）；
   新增唯一口径文本 `TRAIN_TOP1_SELF_RULE` 并在 `train_harmlessness` / 锚点 / 选取规则 /
   `third_step_evidence`（`train_self_top1_rule` / `train_self_top1_note`）四处**互相交叉引用**；
   `render_markdown` 与 CLI 日志标签同步为「批内滚动平均（仅诊断）」；
   报告新增 §2.2「批内滚动平均 vs 逐格 R@1（两个观测量，不得互相代入）」逐组合差值表。

**产物（同参重建；判定量逐项未变）**：`variantb_calibrate.json` **508313 B `9ed2d843cafbf389`**、
`variantb_calibrate.md` **29850 B `5f73285c0cd53204`**、`calibrate_repro/` 与之**逐字节相同**、
`calibrate_dry/` 121479 B `525f7ef833f4f1ff`、`calibrate_drill/variantb_drill.json`
64021 B `4feec1283da3e438`（**字节未变**）、`calibrate_rawonly/` 278761 B `73230bafc072ba64`、
`calibrate_rawonly_single/` 112400 B `84f6f5389c0c6983`、`gbk_runtime_check/` 两份按新 schema 重建；
`render_calibrate_markdown(json)` 与磁盘 `.md` **逐字符相同**。
**旧值作废（如实登记）**：`variantb_calibrate.json` 479903 B `981ee3cbfaa15c92`（R60）与
509151 B `3161259bc6e2567f`（本轮中间态）、`.md` 22915 B `04ee69123b3e11c4` / 30688 B
`1f58486031eb3fe2`、`calibrate_dry/` 111730 B `cf5d97c3dc7a7371`、`calibrate_rawonly/`
102949 B `445211c1a98b65df`、`calibrate_rawonly_single/` 102949 B `692116c1e6ad62e7`、
`gbk_runtime_check/variantb_run.json` 195782 B `14d700d2a1f9b746`、`variantb_eval.json`
186624 B `85847d4938424103` —— 均产生于字段改名与口径文本更正之前。
**零回归**：1a 考卷零改动；顶层 9 个 `qa_*.pt.zip` 的 SHA256/字节/mtime 逐项不变、顶层仍 9 个文件；
上游四目录与根 `requirements.txt` 的 `git status` 为空；第二步 5 件既有产物逐字节不变；
`compileall` 退码 0；零新增依赖；GBK 控制台下 `calibrate` / `train` / `eval` / `drill` 全部退码 0。

**离朱 R61 复测与收口（1 处 F1 残留 + 4 项 info 观察；只改渲染层与文本，判定量逐项未变）**

**R61 结果**：15 套件 / **550 项断言 / 549 通过 / 1 失败**；唯一失败为 F1 的**一处未标记残留**；
其余 5 条修复、全部硬约束与零回归逐项通过。离朱的独立复现（不看文字自述）：F2 取舍列表与
`per_cell` 正负号逐格相等；F3 `train_side_cells` 与 `per_cell` 逐位相等、**注入不一致后
`run_calibration` 如期抛 `ValueError`**（证明硬校验比较的是两个独立视图）；F4 ① 复算 = 9 格判据计数、
② 复算 = 10 格均值且**仍精确为 `0.9149074537268633`**；F5 `row_set_note` 逐字符等于常量、
666 侧 `anchor_1a_check` 逐行与 1a 登记值精确相等；F6 6 件新产物旧名出现 **0** 次、
`TRAIN_TOP1_SELF_RULE` 四处交叉引用、§2.2 表 8 行回产物（5e-7 内）。

**收口处置**
1. **F1 残留（唯一失败）**：README **§16.8**（R57 收口节）仍以现在时把**已删除**的指纹反查机制
   当作现行 fail-closed 行为（「候选指纹对不上 ⇒ `ValueError`；零范数路径返回的未归一化矩阵
   裸字节 SHA256 == `raw_perturbed_sha256`」）。**就地更正**：加「口径收口修复轮就地更正
   （离朱 R61 F1 残留）」标记，写明该机制**已删除**（`detail` 是 1a 函数内局部变量 ⇒ 反查恒命中、
   `raise` 分支不可达），列出**现行为**（报文不可解析 / 已替换行集合一致仍报错 / 超轮次 /
   行独立性交叉验证不一致）与 `RETRIEVAL_RULE` 的两分支 + 4/4 反向验证；
   同段补注 `train_top1_self` 为历史字段名（现名 `train_batch_running_top1`）。
2. **O1（标签字面量）**：训练循环日志 / `_honest_notes` / 校准报告 §2 表头 / §2.2 表头 / CLI 日志
   五处统一为**逐字**「**批内滚动平均（仅诊断）**」；`批内滚动平均 top-1` 全仓出现次数 = **0**。
3. **O2（评测侧条件用词）**：`ATTRIBUTION_CONDITION_TEXT["b_eval_not_worse"]` 补为
   「**逐格 R@1** **全部不低于** KNN 基线」，与条件 a 的用词对齐。
4. **O3（md 表格未转义 `|`）**：新增渲染层助手 `_md_table_cell()`（`|` → `\|`），
   用于校准报告 §2 / §2.2 / §3 三张表的组合标签列；**JSON 键名不受影响**。
   现场核对：三张表无任何表格行残留未转义标签；示例行「转义竖线 2 / 真实分隔 12」与 11 列表头一致。
5. **O4（冻结产物旧文本）**：与 §16.11.11 的 O2 同源，**明确接受残留并登记**
   （两件受冻结保护的产物仍含旧字段名 186 次 + R59 F4 旧句；不得据此推断旧口径仍现行）。

**判定量前后对照（R60 → R61，逐项一致）**：`verdict_key = linear_expressivity`；
`selected = normalized|lr=0.001|wd=0`；`candidates_ranked` 四组顺序不变；
`alternative_reading.agrees_with_primary = false`；条件 a 不劣 `3/9`、最差余量 `−0.125063`；
条件 b `(3, 3, 6)`；锚点 10 格与 8 组合批内滚动平均逐值不变；取回路径正向 True / 反向 4-4 / passed True；
恒等门禁 10/10；更新量门禁 8/8。

**产物（R61 收口后重建）**：`variantb_calibrate.json` **508585 B `ca0b1135dabc5f6a`**、
`.md` **30059 B `ad47440cc688934f`**、`calibrate_repro/` 与之**逐字节相同**、
`calibrate_dry/` 121548 B `6c8796ec950bceaa`、`calibrate_drill/variantb_drill.json`
64021 B `4feec1283da3e438`（**始终未变**）、`calibrate_rawonly/` 278917 B `17bf4f6223b26d00`、
`calibrate_rawonly_single/` 112469 B `8015a8141192ec36`、`gbk_runtime_check/` 重建为
200820 B `3903db3b78fab699` / 191424 B `8c48c6b0f72de012`；`render_calibrate_markdown(json)`
与磁盘 `.md` **逐字符相同**。**旧值作废**：R61 收口前的 508385 B `eafdcd5e187cb2e0` /
29850 B `5f73285c0cd53204` 及更早各批（见 README §16.11.2 的倒序清单）。
**零回归**：1a 考卷零改动；顶层 9 个 `qa_*.pt.zip` 的 SHA256/字节/mtime 逐项不变、顶层仍 9 个文件；
上游四目录与根 `requirements.txt` 的 `git status` 为空；第二步 5 件既有产物逐字节不变；
`compileall` 退码 0；零新增依赖；GBK 控制台下 `calibrate` / `train` / `eval` / `drill` 全部退码 0。

**字段改名的向后兼容回归修复轮（旧键回退）与离朱 R64 验收**

**缺陷（皋陶唯一 error）**：口径收口修复轮把 `train.train_top1_self` 改名为
`train.train_batch_running_top1` 时，**读取侧只写** `.get(新键, 0.0)`；而**冻结的第二步产物**
（只有旧键、按零回归要求不得重建）⇒ `third_step_evidence()` / `_honest_notes()` 在旧产物上
**静默返回 `0.0`**。**修法：最小改动、只加回退，不改变任何判定量。**

**单点取值实现（新增，导出）**：常量 `TRAIN_TOP1_KEY = "train_batch_running_top1"`（现行键名唯一来源）、
`LEGACY_TRAIN_TOP1_KEY = "train_top1_self"`（旧键名）、`TRAIN_TOP1_KEY_FALLBACK_RULE`（回退规则文本）；
函数 `train_batch_running_top1_of(block) -> (值, 命中的键名)`：优先级 **新键 → 旧键 → `0.0`**，
**回退只在旧键存在且新键缺失时生效**（⇒ 新产物路径读数逐位不变），**两键同时存在 ⇒ 新键优先**，
两键都缺失 ⇒ `(0.0, "missing")`（键名可判断真值不可得），**非映射输入不抛异常**。
`third_step_evidence()` 与 `_honest_notes()` 均改用该助手；前者在 `third_step` 块内新增
`train_self_top1_key_used`（逐档键名）、`train_self_top1_legacy_fallback_profiles`、
`train_self_top1_key_fallback_rule`，并扩写 `train_self_top1_note`；后者**新键路径文本逐字不变**，
仅新键缺失时追加「（取值键 = `train_top1_self`：旧产物沿用旧键名…本值经**回退**取到真值）」注记。

**R64 现场验收（20 套件 / 703 断言 / 703 通过 / 0 失败）**
- **双向验证 4/4**：① 冻结 `variantb_eval.json`（341179 B `d094c9bd9200120d`，每档 `train` 块**只有旧键**）
  ⇒ `train_self_top1 == {bge-m3-1024: 0.632816408204102, lexical-88: 0.34217108554277137}` **逐位相等**
  （修复前同输入为 `0.0`/`0.0`）；② 新 schema `gbk_runtime_check/variantb_eval.json` ⇒
  `0.6803401700850426` **逐位相等**且与 `train.train_batch_running_top1` 一致、`legacy_fallback_profiles = []`；
  ③ 对旧产物重跑 `variantb report`（隔离目录）**退码 0**，输出 `third_step.train_self_top1` 不再为 `0.0`；
  ④ 旧产物路径 `_honest_notes` 不再出现「= 0.000000」。
- **边界**：两键并存 ⇒ 新键优先；两键都缺 ⇒ `(0.0,"missing")`（键名已断言）；8 类非映射输入均不抛异常；
  新产物路径注记**逐字不变**（以改动前的归档产物做前后对照）。
- **两项如实登记经核实均属实**：① 皋陶所述「产物层面对旧产物重跑 `report` 会写出 `0.0`」**现场不成立**
  —— 冻结产物**自身内嵌**的 `third_step.train_self_top1` 已是真值（产生于改名前的代码），
  且 `cmd_variantb_report` **不重算** `third_step`（输出块与输入块逐字段相同）⇒ **CLI 路径从未受影响**，
  真实受影响面只有按 README 去**复算该块**的两条 API 路径，本轮已修；
  ② 「JSON 键集合为封闭集」**非现场事实** —— 对模块 `.py` + **全部历轮 `lizhu_r*_scripts/*.py`** +
  `.module_agent/**` 定向扫描（行内同时含 `third_step`、`==` 与 `keys()/set(/sorted(`）**0 处**
  等号固定键集合的断言（历轮均为存在性检查）；新增 3 键**只出现在重算块**（重建产物 `third_step` 10 键，
  冻结两件仍 **5 键**且逐字节不变）⇒ **未放宽/删除任何断言**。
- **硬约束逐项零变化**（`verdict_key` / `selected` / `candidates_ranked` 顺序 / `agrees_with_primary=false` /
  条件 a `3/9`·`min_margin −0.125063` / 条件 b `(3,3,6)` / 锚点 10 格逐值 / 8 组合批内滚动平均逐值 /
  `candidates[0].train_post_top1_mean = 0.9149074537268633`）。
- **产物**：冻结 5 件**逐字节不变**；**本轮唯一重建** = `gbk_runtime_check/` 两件
  （`variantb_run.json` **202194 B `121ce3b6eaae5fc5`**、`variantb_eval.json` **192798 B `fefbb213ef2c9de1`**，
  各 +1374 B 恰为 3 个新键体量）；校准族产物未被触碰；渲染器逐字符复现 md；E2E 逐字节复现；
  零回归（1a 考卷零改动、顶层 9 个 zip 三项不变、顶层无新增、`git status` 为空、`compileall` 退码 0、
  零新增依赖、GBK 不设 `PYTHONIOENCODING` 下 `train/eval --epochs 1` 退码 0、CLI 边界与向后兼容保持）。
- **离朱登记的 3 项 info（非缺陷，未再改动以保持「已验收状态 = 交付状态」）**：
  **O-a** 新键存在但值为 `None` 时会回退到旧键（规格未规定；若要求严格「仅按键是否存在」可只判 `key in data`）；
  **O-b** 模块内仍有一处 `.get('train_batch_running_top1', 0.0)`（校准报告 §2.2 渲染 `rvg.get(...)`），
  它读的是校准报告自身的 `running_vs_grid` 块（与改名同轮引入、**不存在旧 schema 版本**，不可能静默归零）；
  **O-c** 重建产物内嵌 `honest_notes` 与朴素重算在其他注记上有**调用约定差异**（被测的那一条逐字相同），
  建议在 README 标注 gbk 产物的生成命令以便复算对齐。
- **离朱方法学自纠（测试代码侧，已如实登记）**：R61 期「产物文本旧名出现 0 次」在本轮**必然不成立**
  （旧键名按设计出现在 `third_step` 登记字段）⇒ 该断言按 R64 规格收窄为**可核对的作用域形式**
  （每个 `train` 块不得含旧键 + 旧名只能出现在 `third_step` 登记字段，用「全文计数 == `third_step`
  序列化计数」断言，现场每件 2 次恰在两处登记字段），并已核验不存在被放宽/删除的等号式键集合断言。
## 文本侧数据改造（第三步 Phase 1：多表述 -> 原文）

把文本侧从「**自匹配恒等任务**」改造成「**多表述 -> 原文**」的真实检索任务，并引入难负例；**本轮零训练**（不上任何模型、不训练），只用纯余弦检索判定改造是否有效。新增 `n3d_qa_learn/text_variants.py`（核心）与 `n3d_qa_learn/text_variants_run.py`（CLI），并在 `step2_run` **纯新增** `textvariants` 子命令。产物一律写 `checkpoints/qa_learn/_verify/text_variants/`（不落盘特征矩阵、不产 zip、无挂钟字段）。

**4 支变体（问题侧；答案恒为行原文，与 1a / 步骤 2 的 `outputs = 行原文` 一致）**
- `原文`（锚点，恒等）；`关键词`（KeyBERT + 仓库自有 `models/bge-m3` 薄适配 `RepoEncoderEmbedder`）；`摘要`（抽取式**三级阶梯**：句级 -> 子句级 -> 兜底截断，结果恒为原文**连续子串**）；`自适应`（按 `char_len` 分档：short 0-24 关键词 / mid 25-64 首子句上限 24 / long >=65 上限 42）。
- 冻结口径常量：`KEYWORD_TOP_K=3`、`KEYWORD_NGRAM_RANGE=(1,3)`、`KEYWORD_USE_MMR=False`、`KEYWORD_POS_KEEP`、`KEYWORD_MIN_DOC_FREQ=2`、**`KEYWORD_MAX_CANDIDATES=32`**（R67 由 5 改为 32：5 会使 `TOP_K=3` 退化，见下）、`SUMMARY_MAX_SENTENCES=2`、`SUMMARY_MAX_CHARS=60`、`SUMMARY_CLAUSE_MAX_CHARS=24`、`SUMMARY_HEAD_MAX_CHARS=12`、`SUMMARY_HEAD_FALLBACK_BELOW_CHARS=12`、`SUMMARY_MIN_CHARS=1`、`ADAPTIVE_TIERS`、`ADAPTIVE_MID_MAX_CHARS=24`、`ADAPTIVE_LONG_MAX_CHARS=42`；纯确定性、不消耗全局 RNG；极短行 / 空文本**显式回退原文并计数登记**，不产空串 / NaN / 零范数。
- 语料级候选词表：`build_keyword_vocabulary` 从**全量行表**构建并**落成产物** `keyword_vocab.json`（R67 由 `_cache/` 迁移为产物；`keyword_vocabulary_path()` 仅作直接调用 API 的缓存回退，为唯一登记例外）。`KEYWORD_ENCODER_MAX_LENGTH=512`（`char_len` 最大 293 ⇒ 不截断）。
- **现场实测驱动的修正**：① 2092/2665 行只有单句，句级抽取对它们等于原样返回 -> 必须下沉到子句级；② `RepoEncoderEmbedder` **必须**继承 `keybert.backend.BaseEmbedder`（否则 keybert 0.9.0 的 `select_backend()` 会**静默回落到 `paraphrase-multilingual-MiniLM-L12-v2` 并尝试下载**）；基类**惰性求值**（缺 keybert 时退回 `object`，模块导入不被阻断，R67）；③ keybert 0.9.0 的 `doc_embeddings` 是 `np.ndarray`（传 `True` 抛 `TypeError`）。

**变体产物与指纹**：规范化 JSONL（`sort_keys` + 紧凑分隔符、无挂钟字段）+ 逐支 SHA256（`variant_fingerprint` / `variant_product_files` / `serialize_variant_records`）+ **覆盖度** `coverage`（`n_rows` / `source_total_rows` / `limit` / `coverage_ratio` / `is_full_corpus`）。

**难负例**：`same_file`（同 `file` 他行，**按 |line_no 差| 最近**、并列按 `row_id` 升序，取前 `HARD_NEGATIVE_CAP=16`）/ `adjacent_line`（`line_no ± 1`）/ `encoder_neighbor`（编码器归一化特征余弦 top-n，**全序键 `(-cos, row_id)`**）/ `random`（局部 `random.Random(TEXT_VARIANTS_SEED)`，`RANDOM_NEGATIVE_COUNT=16`）。产物落 `counts` / `fingerprint` / `encoder_neighbor_skipped` + `encoder_neighbor_skip_reason`。

**特征差异门禁（R67 起为真实断言）**：`feature_gate` 对**每一个可约简行**（`row_is_irreducible == False`）**逐支断言**其 float32 特征与原文支**逐位不同**（违反即 `AssertionError`，fail-closed）；**不可约简行**（任何抽取算子都只能返回自身，由 `row_is_irreducible` **独立于抽取实现**枚举得出）的重合**如实登记**。产物含 `hard_gate.{passed, assertion_passed, reducible_rows, reducible_rows_with_identical_features, irreducible_rows, violations_head, rule}`、`per_variant.cos_distribution`、`per_tier`、`short_tier_identity_registration`、`identity_registration`、**`coverage`**。

**零训练评测（R67 起为三设定，含真对照）**：`evaluate_retrieval`（全量池）/ **`evaluate_retrieval_pools`（逐查询池）** + `zero_training_eval`：
- A **`full_pool_all_rows`**：池 = 本次评估的全部行，**不消费 `negatives.json`**（主指标基线）；
- B **`small_pool_hard`**：每查询行池 = { 自身 } ∪ 难负例（≤ `NEGATIVE_POOL_SIZE=16`）；
- C **`small_pool_random`**：同查询集、**池大小与 B 完全一致**，负例换 `random` 类；
- **难负例对照 = B − C**（同池大小、只差负例内容），口径常量 `SMALL_POOL_RULE` / `FULL_POOL_RULE` 与 `negatives_participation` 一并落产物（显式声明 A 不消费负例、不得当作对照基础）。**修正前的"对照"两种设定共用同一池与同一函数调用，结构性恒为 0**。
- 分档报 `R@1`/`R@5`（**不混平均**）；公共子集在同一批查询行上**真正分别评测三次**。**金标集口径 `GOLD_SET_RULE`**：`金标集 = 自身行 ∪ 原文逐字节相同的其他行`，修复「重复原文档住 top-1 导致锚点 `R@1` 非结构性 < 1.0」的测量缺陷。

**机器可读判定**：`render_verdict` —— 规则文本由常量**派生**（`_build_verdict_rule()`，导入期 `assert` 与 `VERDICT_RULE` 逐字相同），结论键 `anchor_not_structural` / `variant_too_close`（`1-R@1 <= 0.01`）/ `variant_too_far`（`R@1 <= 1e-9`）/ `task_valid_not_identity`（`1-R@1 >= 0.05` 且 `R@1 - 1/pool >= 1e-6`）/ `undecided` / `mixed`（R67 补 `all(undecided)` 分支）。**粒度守卫**（`GRANULARITY_GUARD_RULE`，含等号）。

**产物内「声明口径」与「实测口径」分离（R67）**：`retrieval_eval.json` 的 `retrieval_pool_size` / `random_baseline` / `granularity` **均为本次运行实测值**，另落 `declared_constants`（全语料常量 + 显式注记「不得直接当作本次实测基线」）、`coverage`、`measured_scope.{n_rows, source_total_rows, limit, granularity_measured, random_baseline_measured_full_pool, per_setting}` ⇒ 下游仅凭产物即可正确复算随机基线与粒度。

**QA 侧数据组织取证**：`qa_answer_groups` / `candidate_group_map` 把「同答案的多问题」显式成组并出规模、组内问题数分布与指纹；**本轮只取证、不接任何损失**（`QA_GROUPING_NO_LOSS_NOTE`）。

**CLI**：`python -m n3d_qa_learn.text_variants_run {probe|gate|eval|report}` 与 `python -m n3d_qa_learn.step2_run textvariants {probe|gate|eval|report}` 两条入口**共用同一份参数表函数**；支撑 `--limit`、`--log-file`（UTF-8 无 BOM）、`--no-encoder-neighbors`、`--repro-dir`、`--hard-cap`（0 = 用冻结常量默认）。产物目录守卫：只允许写在 `checkpoints/qa_learn/_verify/` 之下（**相对与绝对路径同口径**）。缺 `negatives.json` 时 `gate` / `report` / `eval` **统一**为可读报错 + 退码 1。`step2_run` 既有 8 个子命令参数表**逐字符未改**。

**全量 2665 现场实测（R67，`probe --no-encoder-neighbors` -> gate -> eval -> report 全部退码 0）**：耗时 probe **7068.5 s** / gate 61.2 s / eval 107 s / report 63.8 s；覆盖度 2665/2665（`is_full_corpus=true`）。**设定 A**：`R@1` = 原文 **1.000000** / 关键词 **0.445403** / 摘要 **0.956848** / 自适应 **0.768480**；`R@5` = 1.0 / 0.775610 / 0.989869 / 0.931332；`verdict_key = mixed`，逐支 `[task_valid_not_identity, undecided, task_valid_not_identity]`。**设定 B（难负例小池，池恒 17）**：0.694934 / 0.982364 / 0.897561；**设定 C（随机小池）**：0.970356 / 0.999250 / 0.986867；**对照 B−C** = 原文 0.000000 / 关键词 **−0.275422** / 摘要 −0.016886 / 自适应 −0.089306。门禁 `passed=True`、`assertion_passed=True`、`reducible_rows=2387`、`violations=0`、`irreducible_rows=278`。产物指纹：`variants.jsonl` 2416229 B `92b1ddbd382cc43b`、`variants_meta.json` 6618 B `82d2e8690ccb54c0`、`negatives.json` 2282329 B `e96878e7a228a5d3`、`feature_gate.json` 7155 B `3b58efb156b1b6f8`、`qa_grouping.json` 1430 B `7de71d4f47be6866`、`keyword_vocab.json` 241001 B `990027709ab78b20`。**现场反向验证**：① 全量上互换 `random` 与 `same_file` 清单后，设定 A 四支 `R@1` **完全不变**、B−C **符号翻转**（−0.275422 → +0.275422）⇒ 对照量真依赖 `negatives.json`；② 在真实产物上把一条可约简行的摘要/关键词支改写成原文后重跑门禁，**`AssertionError` 如期抛出**。**双跑**：`gate`+`eval` 在第二个独立进程重算后 6 件内容产物逐字节相同，`retrieval_eval.json` / `gate_summary.json` 归一化内嵌路径后逐项相等；如实登记「variants.jsonl 等 4 件为带入对照目录而非第二进程重算，真正独立重算的是 feature_gate.json」。

**依赖**：`keybert 0.9.0` / `jieba 0.42.1`（含 `scikit-learn` / `scipy` / `joblib` / `threadpoolctl` / `narwhals` / `cloudpickle` / `sentence-transformers` 传递依赖），登记在 `n3d_qa_learn/requirements-qa.txt`；**根 `requirements.txt` 未改动**。安装口径：`pypi.org` 下载 `scipy` wheel 长时间零字节增长 ⇒ 改用 `pypi.tuna.tsinghua.edu.cn` 镜像；`jieba` 只有 sdist。离线可用：`HF_HUB_OFFLINE=1` + 本地 `models/bge-m3`。

**如实登记的局限（不得包装）**：① 抽取式关键词/摘要属**信息减少**、**不是**真 paraphrase，结论不得包装成「解决了换说法泛化」；② `--no-encoder-neighbors` 使三类难负例只落地两类（产物已落 skip 标记）；③ 四个 md 文件行数并不均等（`a.md` 51 / `b.md` 2403 / `c.md` 167 / `d.md` 44），任务描述「每文件约 666 行」现场不成立；④ 单 seed、无跨 seed 极差；⑤ 全量 `probe` 的 117.8 min 主要耗在候选词 bge-m3 嵌入（CPU）。

**零回归**：`entry_table.py` / `robust_eval.py` / `variant_b.py` 零改动；顶层 9 个 `qa_*.pt.zip` 的 SHA256/字节/mtime 与顶层文件清单逐项不变；第二步 5 件冻结产物逐字节不变；根 `requirements.txt` 未改；四个上游模块 `git status --porcelain` 为空；`compileall` 退码 0。

**R68 小收口轮（渲染层 / 文案 / 取证字段，**判定量零变化**）**
- **报告渲染**：`render_markdown` 的第 5 节现为 `5.1/5.2/5.3`（三设定）-> **`5.4` 公共子集（按三个池键分别三张表，含池 min/中位/max）** -> **`5.5` 难负例对照** -> **`5.6` 小池前置条件**；小节编号不重复，报告全文不再出现 `None`（此前公共子集表整表为 `None`：`zero_training_eval` 在 `common_subset` 下产出的是 `variants_full_pool` / `variants_small_pool_hard` / `variants_small_pool_random` 三个分池键，没有 `variants` 键）。
- **小池前置条件**：新增常量 `SMALL_POOL_PRECONDITION_RULE`；`zero_training_eval` 在评测前 **fail-closed 断言**「B/C 查询集相同 且 逐查询行池大小相等」（B − C 只有在此前提下才只是「负例内容」之差）；`report` 新增 `small_pool_precondition`（只读 `measured_scope.per_setting`，**不向 `retrieval_eval.json` 新增字段**）并渲染到 §5.6。
- **截断取证**：`build_keyword_backend` 的 `word_max_length` 改取**编码器实际生效**的 `max_length` 并由 `_build_generator` 透传；`probe` 新增 `_truncation_evidence()`（`encode_batch_with_stats` 对最长 8 条行文本实测）把 `truncation_observed`（`n_tokens_max` / `n_truncated` / `verdict`）写进 `keyword_backend`（落 `variants_meta.json` + `probe_summary.json`）。现场：`max_length=512`、`longest_char_len=293`、`n_truncated=0`。
- **确定性证据可读性**：`determinism_check.json` 新增 `n_content_common_files` / `content_common_files` / `content_files_bitwise_equal`，4 件内嵌 `out_dir` 的报告收敛为常量 `PATH_DEPENDENT_FILES`（不再被误读为确定性失败）；`report` 的 `determinism_evidence` 同步分列「内容产物」与「path-dependent」两类。
- **语料级候选词表**：回退路径 `KEYWORD_VOCAB_FILE`（`_cache/text_variants/keyword_vocab.json`）的**旧文件已删除**（R68），仅保留为直接调用 API 时的回退；`probe` 一律使用产物目录下由**全量行表**构建的 `keyword_vocab.json`。
- **`negatives.json.rule`** 文案按 `NEGATIVE_KINDS` 顺序重写（各类只描述一次；「抽出即全部入池」）。该修复**必须**改动 `negatives.json` 字节（报告为硬约束例外，判定量零影响）。
- **`honest_notes`** 与三设定一致（A 才是全量池；B/C 是逐查询行小池；对照 = B − C），并新增 `determinism_evidence` 使「证据缺位」类文字的出现条件可审计（仅在证据文件确实缺失时出现）。

**R69 轮 B（`encoder_neighbor` 第三类难负例 + 设定 D + 自指修复 + 口径收口；判定量逐位不变）**

- **第三类难负例落地**：`ENCODER_NEIGHBOR_TOPK` 由 `3` 改为 `16`（= `NEGATIVE_POOL_SIZE`）。理由：设定 D 的池 = {自身} ∪ `encoder_neighbor`，必须与既有 B / C **逐查询行池大小严格一致**（1+16=17）。B 的池由 `same_file` 前 16 条填满、`encoder_neighbor` 从不进 B 的池，故该变更**只影响** `negatives.json` 的 `encoder_neighbor` 类，B / C 数值逐位不变。`negatives.json.kinds = ['same_file','adjacent_line','encoder_neighbor','random']`（**三类齐备**），`encoder_neighbor_skipped` 转为 `False` + 空理由（旧的「只落地 two 类」标记作废）。
- **新增第四种设定 D `small_pool_hard_encoder`**：`NEGATIVE_SETTINGS` 增至 4 项；`negative_setting_payload` 新增 `encoder_pools` / `encoder_eligible_row_ids`；`zero_training_eval` 新增设定 D 与 **`encoder_vs_random_delta`（D − C）**。`D − C` 是**新实验量**，与 `B − C` **并列报告**，产物 note 与 README **显式禁止**用它反向解释或覆盖 `B − C`。
- **三侧 fail-closed 前置断言**：`SMALL_POOL_PRECONDITION_RULE` 扩展为「B / C / D 查询行集相同 且 逐查询行池大小两两严格相等」；缺 `encoder_neighbor` 类时**不产 0 值假数据**，改为落 `encoder_setting_applicable=false` + `_ENCODER_SETTING_UNAVAILABLE_REASON`，报告 §5.4 不渲染、§5.7 显式登记。
- **同源性登记（新增常量 `ENCODER_NEIGHBOR_HOMOLOGY_RULE`，必须显式）**：现场事实是**评测口径 = `step2.ZhBagVectorizer`（哈希词袋 D=192）**，`bge-m3`（D=1024）**只造邻、不参与指标** ⇒ 主档与指标**不同源**；「用被评测的那个编码器造邻 ⇒ 难度自证、只作诊断」的风险已用 `zh-bag`（同源档）与 `local-hash`（同族档）**两档现场对照**分别登记（`_verify/text_variants/homology_zhbag/` 与 `homology_localhash/`）。`text_variants_run._neighbor_encoder` 相应按注册表 `kind` 分派 `source`（hf 用本地 `models/bge-m3`；**hash 家族必须空 source**），使该对照可由 CLI 现场复现。
- **`determinism_check.json` 自指修复**：`_determinism_block` 枚举两目录时**跳过证据文件自身**（新增产物键 `excluded_self_file` / `self_exclusion_rule`），使其成为**输入目录内容的确定性函数** —— 修复前是「自指哈希链」（自录项 = 上一轮留在盘上的自己），修复后**连跑两次逐字节相同**。口径变更：`files` 少自身一项 ⇒ `n_common_files` / `all_bitwise_equal` 的含义收窄为「除本证据文件外的共有文件」；`content_common_files` 本就不含自己、**不变**。
- **报告渲染（R69）**：第 4 节增 `kinds` 渲染；第 5 节为 `5.1 A / 5.2 B / 5.3 C / 5.4 D` → `5.5 公共子集（四种池设定）` → `5.6 B−C 与 D−C 并列` → `5.7 三侧前置条件`；**小节编号改为动态**（按实际渲染的设定数递增，D 不可用时不出现空号）。
- **全量 2665 现场实测（R69）**：`probe` 126.9 s / `gate` 66.7 s / `eval` 129.5~130.1 s（含双跑证据）/ `report` 完成，四步退码 0；覆盖度 2665/2665。设定 D `R@1` = 1.0 / 0.54296435272045029 / 0.97335834896810502 / 0.82551594746716694；**`D − C`** = 0.0 / −0.4273921200750469 / −0.02589118198874296 / −0.16135084427767354。三侧逐行池大小去重集合 = `{(16,16,16)}`（池恒 17）、三侧不等的行数 = 0。**现场反例**：把 `encoder_neighbor` 截断到 3 条后调用 `zero_training_eval` ⇒ `AssertionError` 如期抛出。
- **如实上报的硬约束冲突**：`gate_summary.json` 内嵌 `negatives.json` 的 `counts` / `fingerprint`，而任务 ① 要求 `negatives.json` 必增一类 ⇒ 该件**必然变化**（8762 `1286d1b2e848eb9a` → 8793 `dff6a6af3c368ef6`），与「逐项不变」在逻辑上互斥；变化面**只有** `negatives` 块，为最小化改动面**撤回**了原计划在该文件新增的 `kinds` / `n_kinds` 两键（报告改从 `negatives.json` 本尊读取）。另如实登记：任务书给出的两件报告基线（129992 `bac640865b8db74c` / 13818 `9c42e988f52a8031`）与**开工前现场值**（129995 `4f8887446fb06861` / 13818 `ec6463010a27b1f6`）不一致（前者是 §17.18.14 收口值，§13.11 轮已登记该漂移）。
- **产物**：逐项不变 `variants.jsonl` 2416229 `92b1ddbd382cc43b` / `variants_meta.json` 6754 `f3877dd85a01a7e9` / `feature_gate.json` 7155 `3b58efb156b1b6f8` / `qa_grouping.json` 1430 `7de71d4f47be6866` / `keyword_vocab.json` 241001 `990027709ab78b20`；允许变 `negatives.json` **3387028 `5ded5446594613f1`**、`probe_summary.json` **66785 `340c2de4a5338e10`**、`retrieval_eval.json` **148820 `ecec8e7cf5c14d89`**、`determinism_check.json` **7481 `e312ae83504c0717`**、`text_variants_report.json` **172667 `734f64ea60a71d4c`**、`text_variants_report.md` **18065 `635000ccb58fd867`**；`gate_summary.json` **8793 `dff6a6af3c368ef6`**（见上条冲突）。
- **零改动 / 零回归**：`encoders.py` / `encoders_run.py` / `entry_table.py` / `robust_eval.py` / `variant_b.py` 零改动（`git diff` 空）；批组成依赖按 **D 方案「只登记不修」** 结案（README §10.2 + §10.5）；顶层 9 个 `qa_*.pt.zip` 的字节/SHA256/mtime 与顶层文件清单逐项不变；第二步 5 件冻结产物逐字节不变；四个上游模块 `git status` 为空；根 `requirements.txt` 未改；零新增依赖；`compileall` 退码 0。
- **文档口径**：README §13.11 第六条更正为「**上一轮（分片修复轮）未改** `encoders_run.py`；其后收口轮只把 G2 取证的 meta 读取改为 `EmbeddingCache.read_meta`（只影响 meta、不影响向量）」；§10.3 措辞扫描改为「**先剔除强调标记**（`**`/`*`/`` ` ``）再计数 + 同时报 raw 与 stripped + README 以**下标引用** `WORDS` 不写字面量」，使「0」成为**语料属性**而非书写形式的函数（改后仍 PASS、退码 0）。

**R70 收口修复轮（皋陶 2 error + 2 warning + 3 info；**判定量逐位不变**，`retrieval_eval.json` 指纹前后逐字节相同）**

- **error 2｜断开「报告 ←→ 证据」回路**：`text_variants_run` 新增常量 `RERENDERABLE_FILES = ("text_variants_report.json", "text_variants_report.md")`（**后置可重渲件**：在证据之后才由 `report` 写出/重渲）并**整类排除**于 `_determinism_block` 的枚举 —— 既不进 `files`，也**不记录任何 SHA / 字节数**（只登记文件名，新增产物键 `excluded_rerenderable_files` / `rerenderable_exclusion_rule`）；`cmd_report` 内嵌的 `determinism` 由**整块**降级为**摘要**（新增 `_determinism_summary()`：无逐文件 SHA、无 `out_dir`/`repro_dir` 路径）。根因：R69 只排除了证据文件自身，却仍记录两件报告的 SHA，而报告又把 `determinism` 整块内嵌回 `text_variants_report.json` ⇒ 「报告内嵌证据 ←→ 证据记录报告 SHA」回路，使该文件不是输入目录内容的确定性函数（常规工作序 `eval → report → eval` 下必然改变）。
- **error 1｜硬验收重写为可成立的命题**：命题 (i) 固定工作序下 `eval × 2`（其间不跑 `report`）逐字节相同；(ii) **跑 `report` 之后再 `eval`** 仍逐字节相同。**在真实产物目录** `checkpoints/qa_learn/_verify/text_variants`（`--repro-dir …\repro_r69`）现场复现：四次运行均 **7792 B / `edeb0bded509e5a2…`**（①②④⑤，全部退码 0）。R69 登记的 `7481 B / e312ae83…` **作废**；皋陶实测的 `7433 B / f81f04be…` 与 `7433 B / b7759b46…` 保留为「修复前」证据。
- **warning 1｜裸 `assert` 清零 + `-O` 守卫**：新增异常类 `VariantContractError(RuntimeError, AssertionError)`（同时是 `RuntimeError` ⇒ CLI 报「契约 / 门禁失败」退码 1，不被 `(ValueError, KeyError)` 误报成「非法参数 + 退码 2」；同时是 `AssertionError` ⇒ 既有 `except AssertionError` 调用方无需改动）与 `CONTRACT_FAILURE_NOTE`；`text_variants.py` 的 **5 处裸 `assert`**（`VERDICT_RULE` 派生一致、`IDENTITY_GATED_VARIANTS` 门禁、可约简行逐位门禁、摘要支不变量、小池三侧前置条件）**全部改为显式 `if not …: raise VariantContractError(...)`**（报文逐字保留）⇒ 全仓扫描 `^\s*assert\b` 计数 **0**；`cmd_textvariants` 另新增**入口级 `-O` 守卫**（`sys.flags.optimize != 0` ⇒ 可读报文 + 退码 1），两入口（`text_variants_run` 与 `step2_run textvariants`）共用同一分派故同时覆盖。现场反例：`PYTHONOPTIMIZE=1` 下注入异常输入（截断 `encoder_neighbor` 到 3 条）仍**如期抛出 `VariantContractError`**；CLI 在 `-O` 下直接拒绝运行（退码 1）。
- **warning 2｜CLI 帮助与常量一致**：`--random-negatives` 的 help 由「0 = 冻结默认 1」改为「0 = 冻结默认 **16** = `RANDOM_NEGATIVE_COUNT`；与 `NEGATIVE_POOL_SIZE` 同值，故 B/C 池大小一致」。
- **info 6｜移除恒假字段**：`_determinism_block` **删除 `all_bitwise_equal`**（存在天然不同的 `path_dependent_files` 时结构上恒为 `False`、无信息量且易被误读为确定性失败），改为语义明确的 **`determinism_ok`**（= 内容产物两目录共有且逐字节相等）+ `common_but_not_equal_files`（共有但天然不同的件白名单）+ `determinism_scope`（判定范围文本）；`cmd_eval` 日志与 `cmd_report.determinism_evidence`（原 `all_common_files_bitwise_equal`）同步换字段；`render_markdown` §9 由「逐文件 SHA 表」改为渲染**证据摘要**。现场：`determinism_ok = True`、`common_but_not_equal_files = ['probe_summary.json']`。
- **info 7｜D 判据写死**：`_small_pool_precondition` 判「设定 D 是否被评测」由 `if d else []`（键存在但为空 dict 时会与 B/C 假性判等）改为 `bool(measured_scope.encoder_setting_applicable)`，并在产物新增 `d_judged_by = "measured_scope.encoder_setting_applicable"`。
- **info 5｜文档过期表述**：README §17.7 的「对**三种池设定**分别评测」改为「**四种池设定**」并补明 `common_subset` 实际渲染的四个分池键（含 `variants_small_pool_hard_encoder`）。
- **现场实测（R70）**：`probe` / `gate` **未重跑**（产物逐项不变）；`eval` 116.5 / 117.4 / 115.6 / 118.1 s，`report` 58.2 s，全部退码 0；小批演练（`_drillC`，`--limit 20`）`eval → eval → report → eval` 三次均 **7911 B / `2e36e2d9b33d26e0…`**。产物：**逐项不变** `variants.jsonl` 2416229/`92b1ddbd382cc43b`、`variants_meta.json` 6754/`f3877dd85a01a7e9`、`feature_gate.json` 7155/`3b58efb156b1b6f8`、`qa_grouping.json` 1430/`7de71d4f47be6866`、`keyword_vocab.json` 241001/`990027709ab78b20`，且 `negatives.json` 3387028/`5ded5446594613f1`、`probe_summary.json` 66785/`340c2de4a5338e10`、`retrieval_eval.json` 148820/`ecec8e7cf5c14d89`、`gate_summary.json` 8793/`dff6a6af3c368ef6` **未变**；允许变 `determinism_check.json` **7792/`edeb0bded509e5a2`**、`text_variants_report.json` **166341/`a7a526e938c09368`**、`.md` **17800/`7664e8787d7a2f9d`**。零回归：顶层 9 个 `qa_*.pt.zip` 三项与顶层文件数 9 不变、第二步 5 件冻结产物逐字节不变、四上游 `git status` 空、根 `requirements.txt` 未改、零新增依赖、`compileall` 退码 0。
## 合并表零训练对照（轮 C）

把「QA 与文本是**同一个 map**」这条架构命题做成**零训练离线对照**：`merge_table.py`（核心）+ `merge_table_run.py`（CLI），驱动脚本为 `python -m n3d_qa_learn.merge_table_run {drill|eval|report}`。**本轮不改生产路由**（`route.py` 零改动）、不做任何训练。

**合并表（口径冻结）**：`MergedEntryKeyTable` 复用 1a 的 `EntryKeyTable`（键 `[N, D]` float32 逐行 L2 归一化 + `entry_ids` + `outputs` + `meta`），新增的**唯一**结构是「逐条目类型标记」`entry_kinds`（`KIND_QA='qa'` / `KIND_TEXT='text'`），按类型的下标区间由 `half_slice` / `indices_of` **现场推导并交叉校验**（区间必须恰等于该 kind 的条目集合，否则拒绝出数）。类型标记由**两个独立视图**背书：写入的 `entry_kinds` 与「按输出内容归类」的 `outputs_index` —— 不一致即构造期报错。合并顺序 `MERGE_ORDER = ('qa','text')`（QA 半区在前）。两侧元数据经 `MergeProvenance` **完整保留**（dim / norm_spec / encoder_profile / encoder_fingerprint / key_table_sha256 / source）。现场口径：**N = 2814 = QA 侧 train_known 149 + 文本侧全量行 2665**，`D = 88`（`lexical-88`），`merge_key_table_sha256 = e81839673cc60b30...`。

**三条合并前置断言**（`PREMERGE_ASSERTION_KEYS`，**先跑、任一失败即停止**，全过才允许出数）：① `same_dtype_and_norm`（两侧同 D / 同 dtype / 各自逐行 L2 归一化，现场范数区间 qa `[0.99999994, 1.00000012]` / text `[0.99999988, 1.00000012]`）；② `role_bitwise_identity`（`local-hash` 的 `question` 与 `text_line` 两角色对同一文本逐位相同，现场 **2665/2665**，两角色口径指纹同一串）；③ `merge_key_bytes_unchanged`（合并前后各半键字节零变化，qa **149/149** / text **2665/2665**）。**口径纪律**：断言③一律用**键裸字节 SHA256**（新增 `raw_keys_sha256`，与 `TextRowKeyTable.sha256()` 同源）比较，**不得**与 `EntryKeyTable.sha256()` 的**内容指纹**（schema+形状+裸字节+ids+outputs+dim 的规范化 JSON SHA256）互比 —— 二者是不同指纹方案，混比会把天然不同误判为字节漂移（本模块首跑即因此失败，已修复并登记）。

**类型标记契约**：`LookupContract(index, returned_kind, expected_kind, returned_output)`，其中 `returned_kind ∈ RETURN_KINDS = ('qa','text','none')`、`expected_kind ∈ ENTRY_KINDS = ('qa','text')`；`type_contract_ok` 即「返回物类型 == 查询期望类型」；`classify_type_violation` 按 `TYPE_VIOLATION_CLASSES = ('returned_none_expected_entry','returned_qa_expected_text','returned_text_expected_qa')` 归类；`empty_violation_counts()` 给零值计数表。`MergedEntryKeyTable.lookup_typed(i)` 返回 `(该行 float32 裸字节, LookupContract)`；`contract_for(i, expected_kind)` / `miss_contract(expected_kind)` 供路由侧构造契约。

**两臂对照（唯一变量是路由结构）**：`ARM_ORDER = ('qa_priority','merged_topk1')`。`qa_priority` = 现状两级硬优先级（QA 半区 top-1 达 `TAU` 即返回，否则文本半区，两级都低于门禁则「无匹配」）；`merged_topk1` = 合并表全表单次排序取 top-1（低于 `TAU` 则「无匹配」）。两臂**共用**同一批查询（QA 侧 `test_known` **20** 条 + 文本侧冻结 666 查询行 x 4 支变体）、同一批键与**同一个** `TAU = 0.5`（`TAU` 不接受运行期覆盖）。`topk_scores(query_feats, keys, index=None|候选下标, topk, batch_size)` 是两臂共用的打分实现（余弦 = 逐行 L2 归一化后的内积；`argsort(stable=True)` 保证并列确定性）；`arm_top1(query_feats, merged, arm, expected_kind, tau, batch_size)` 返回逐查询的 `top1_index` / `top1_score` / `top1_kind` / `returned_index` / `returned_kind` / `returned_output` / `type_contract_ok` / `type_contract_violation` / `stage`。

**主判据口径**：命中率 / `R@1` / 返回类型正确率一律取**未过滤 top-1**（不受 `TAU` 影响），`TAU` 过滤后的量单列为诊断（带 `_returned` 后缀），以免门限取值污染主对照。文本侧 `R@1` 的金标集 = 自身行 ∪ 与查询行**原文逐字节相同**的其他行（`GOLD_SET_RULE`）；原文支 `R@1 == 1.0` 是**结构性锚点**、单列报告且**不参与判定**。文本侧变体一律读**已冻结**的 `_verify/text_variants/variants.jsonl`（`load_frozen_variants` / `variant_fields_of`），**禁止**本轮重新生成。因两臂 `stage` 语义不同（`qa_priority` 是两级路由的**到达档**；`merged_topk1` 退化为「top-1 的 kind」），文本侧 `R@1` 必须**同时**报总体值与按 `stage` 拆分值。

**现场实测（单 seed，`lexical-88`，`D=88`）**：三条前置断言全过、合并表 2814 条目；`原文` 支 `R@1` = `qa_priority` 0.156156 / `merged_topk1` **1.000000**（结构性锚点）；非锚点三支 `R@1` = 关键词 0.243243→**0.346847**、摘要 0.267267→**0.879880**、自适应 0.267267→**0.659159**；返回类型正确率 0.432191→**0.998882**；QA 侧命中率两臂**并列** 1.000000（基线落在**饱和点 1.0** ⇒ 该指标上「严格优于 / 不劣于」**不可判**，已显式登记），严格命中率两臂并列 0.300000（两臂在 QA 侧逐查询决策 20/20 完全相同）。判定 = **`merged_strictly_better`**（劣化指标 `[]`，改进 4 项；最小增益 +0.103604、最大 +0.612613）。返回类型不一致样本数（两臂合计）= **1527**（全部 `returned_qa_expected_text`、全部来自 `qa_priority` 臂的文本侧查询，按查询侧拆分 `{'qa': 全 0, 'text': 1527}`），已按规定**登记为缺陷**并给出逐样本；`merged_topk1` 臂残留 **3** 条（均在 `关键词` 支，QA 条目 56 / 100 / 11 以 0.576572 / 0.627490 / 0.647529 压过正确文本行）为**真实负结果**。

**确定性**：产物一律写 `checkpoints/qa_learn/_verify/merge_table/`（`_guard_out_dir` 只允许写在 `_verify/` 之下），**不落盘裸特征矩阵、不产 zip、无挂钟 / 耗时字段**；同参双跑（`eval` -> `eval --repro-dir`）现场**共有文件 4、逐字节相同 = True**；`determinism_block` **排除证据文件自身**且**不记录两件报告自身的 SHA**（报告在证据之后才渲染）⇒ 无自指哈希回路；`report` 连跑两次产出的 `.json` / `.md` 逐字节相同。产物路径由 `checkpoint_path_resolution` **现场**调用 `n3d_shape.train.resolve_checkpoint_path`（最终参数 `dim=88 / max_batches=0 / tag='merge_table'`，`status=computed`）复算，`artifact_scope_evidence` 核实该路径**不在**本模块产物目录内（产物保护）。

**零回归**：`route.py` / `entry_table.py` / `robust_eval.py` / `variant_b.py` / `text_variants.py` / `text_variants_run.py` / `encoders.py` 等**零改动**；上游四目录 `git status --porcelain` 为空；顶层 9 个 `qa_*.pt.zip` 与第二步冻结件、第三步 12 件产物的 SHA256 / 字节 / mtime **逐项不变**（逐行 diff 为空）；根 `requirements.txt` 与 `requirements-qa.txt` 未改；零新增依赖（仅 torch / numpy + 标准库）；`compileall` 退码 0。

**如实登记的局限（不得包装）**：单 seed 单切分（`split_seed=42`）、QA 侧查询仅 20 条；`qa_hit_rate` 基线饱和、该指标不可判；`bge-m3-1024` 语义档**未做**（两角色 `max_length` 512 / 8192 不同口径，需先对齐才可与本结论并列）⇒ 结论只在 `lexical-88` 档成立；`TAU=0.5` 未做标定扫描；原文支援性锚点 1.0 不得当作检索能力证据；本结论**仅在 C=10 口径下成立**，扩 C 后合并表规模与异构干扰量都会变化、**必须另行复跑**。
**离朱 R74 测试与三项缺陷修复（判定量逐位不变）**

离朱 R74 结果：单元 85 条 → **82 PASS / 3 FAIL**（均真缺陷）、集成 **9/9 PASS**、`compileall` 退码 0、零回归全过。三项缺陷处置如下（判定量逐位不变：`route_eval.json` 31424 B `bf2444a40f7339bf...` 与 `route_eval_queries.jsonl` 1538962 B `f188d788171537f0...` **逐字节不变**；`drill.json` 2684 B `f5b5d2d7d9592066...` 不变）。

- **D1（一般）`MergedEntryKeyTable.meta()` 顶层缺 `encoder_profile` / `encoder_fingerprint`**：修法为顶层补齐这两项（沿用前半区取值），并新增 `encoder_consistency()`（逐半 `encoder_profile` / `encoder_fingerprint` + `profiles_equal` / `fingerprints_equal` / `top_level_field_source`），明确「两半口径不同时以 `per_half` 为准」。现场两半均 `lexical-88`、指纹同一串 `d4a4881aaa8dfa8c...`。影响面：`merge_table.json` 14701 → **17273 B**。
- **D2（重要）`determinism_block` 把证据文件自身纳入比对**：枚举共有文件时未排除 `determinism_check.json`，与函数自身写的 `excluded_self_file` 约定自相矛盾；且在**全新**对照目录上首跑必然 `determinism_ok=False`（该文件只存在于一侧 ⇒ 计入 `b_only_files`）。修法：两个目录**都**过滤 `excluded_self_file`（不进 `files`、不记 SHA），新增 `excluded_self_present` / `n_files_before_exclusion` 取证，并把「`determinism_ok` 只对两目录共有的非排除文件成立、单侧独有文件不纳入该布尔值」写成新字段 `determinism_ok_semantics` 显式登记（不静默忽略）。现场复验：干净目录 A + 上轮产物目录 B 首跑 ⇒ `excluded_self_present={'in_a_dir': False, 'in_b_dir': True}`、`n_common_files=3`，不再因自身判 False；同参双跑仍 `determinism_ok=True`。影响面：`determinism_check.json` 2099 → **2943 B**。
- **D3（重要）`eval` 在 cp936 控制台下输出 UTF-8（GBK 解码失败）**：根因是 `cmd_eval` 调 `checkpoint_path_resolution` → `import n3d_shape.train` → 该模块**导入期** `_reconfigure_stdio()` 把 `sys.stdout` / `sys.stderr` 重配为 `utf-8` / `errors='replace'`，并 `log_info` 打印一行含中文的产物保护提示 ⇒ 本模块 `_log` 的 backslashreplace 兜底永不触发、GBK 纪律被静默旁路。修法：新增 `_save_stdio_encoding()` / `_restore_stdio_encoding()`，并把「导入 + 复算」整段**重定向到临时缓冲**（`io.StringIO`）——上游改的是缓冲的编码、上游日志落在缓冲里，真实 stdout 的编码与内容都不受影响，随后按进来时口径恢复；取证落 `checkpoint_path_resolution.stdio_encoding.{before,buffer_encoding_after_import,restore,upstream_stdout,rule}`（`upstream_stdout` 记录被重定向内容的字符数 / SHA256 / 首行，供审计「上游确实打印过什么」）。现场字节级复验：`drill` / `eval` / `report` 的 stdout 分别 **123 / 952 / 180 B，三者均为合法 GBK**（对 UTF-8 解码失败）、全退码 0；修复前 `eval` 的 stdout 是合法 UTF-8。

**另 3 条口径提示（非缺陷，登记不修）**：**R1** 未登记子命令由 argparse 写 stderr（`invalid choice`），契约错误（TAU 覆盖 / 产物目录越界）写 stdout，两处退码语义一致（2）；**R2** `--log-file` 为 LF、控制台为 CRLF，归一化后文本一致；**R3** `top1_score` 在 `batch_size` 1 vs 64 存在 ≤1e-6 浮点差异（BLAS 分块），主判据字段（`top1_index` / `returned_kind` / `stage`）严格相同。
**轮 3 收口（P1 / P2）**

- **P1 · `merge_table_report.json` 的唯一终值**：§18.13「产物影响面」此前把它记成**中间态**
  `58017 B` / `5b553c3c...`，而 §19.8 的旧→新表、`merge_table/report.log` 首行、盘上实测
  三方一致给出**终值** `58860 B` / `73805cb62da0a558...`。已改为终值，并显式登记「中间态
  58017 只用于说明指纹演进的中间步骤，**不是终值**」；同段的
  `determinism_check.json` 登记值一并订正为 `3782 B` / `61e627792c6d06b2...`。
- **P2 · W1/E1/W3 三块**渲染进报告**：`qa_scale_run.render_markdown` 新增 **§4.2**
  （`criterion_tables_by_scope` 三格作用域表 + `cross_scope_rule` + `why` + `scopes`）、
  **§4.3**（E1 的 `merged_consistency` 逐类自洽表 + `merged_total_self_consistency`）、
  **§4.4**（W3 的 `task_list_consistency` 四来源 + `missing`/`extra`/`per_task_rows`）。
  此前这三块**只在 JSON 产物里**、报告全文没有作用域表与跨作用域声明，与 §19.4.1 的承诺不符。
  现场渲染值（C=298）：全量类 298/0.003610/0.103610、匹配 275/0.003636/0.103636、
  解题 6/0.166667/0.266667（`10/C`=1.666667 ⇒ `applicable=False`）；E1 两口径 × 两单位
  逐类自洽全 True、合并侧合计 10210 / 13468；W3 四来源逐项相同、`per_task_rows` 合计 50865。
  §19.8 变更原因列改述为「W1/E1/W3 三块**渲染进报告**（P2 修复后）」。
  指纹：`qa_scale_report.json` 136406 `41ecf14f...` → **136406 `fc1037d7...`**、
  `qa_scale_report.md` 10214 `06fd0b6a...` → **13773 `3a960828...`**。
## QA 侧数据规模扩容与任务分层（轮 1）

把 QA 侧数据规模从「每类仅 2~22 条、主测试集仅 20 条」扩到**全量口径**，并把「解题类任务」从匹配任务中拆出。**只改数据口径与任务分层，不改模型结构、不改打分链路、不做任何训练**。新增 `qa_scale.py`（核心）+ `qa_scale_run.py`（CLI），入口 `python -m n3d_qa_learn.qa_scale_run {probe|eval|report}`，产物一律写 `checkpoints/qa_learn/_verify/qa_scale/`（不落盘裸特征矩阵、不产 zip、无挂钟字段）。

**数据来源（现场枚举，不凭记忆）**：只读 `checkpoints/qa_learn/dataset/` 的冻结产物 —— `answer_table.jsonl`（298 行；字段 = `['count','display','index','key','raw_variants','sources']`）、`n3d_qa_all_pairs.jsonl`（50865 行；字段 = `['answer_table_size','answer_text','candidate_answer_index','candidate_answer_key','label','pair_id','question_id','question_text','row_id','task']`）、`n3d_qa_all.npz`（`X [50865,306]` float32 + `y` 二值；**只读 `meta`**）、`manifest.json`。**一处字段陷阱（已实测并登记）**：`n3d_qa_all_pairs.jsonl` 的 `task` 列**恒为 `'all'`**（不承载逐行任务）；**逐行真实任务**取自 `n3d_qa_all.npz` 的 `meta.task_of_row`（与该文件逐行同序，现场核对行数一致）。**为什么不用 QA JSON 缓存**：manifest 的 `classes = 298` 只对冻结产物成立，缓存是另一套答案空间（基数与类集合都不同）；缓存口径的现状基线（C=10 / 149 / 20）作为**另一总体**单独登记，不混为一谈。

**扩 C（同一个总体上的唯一变量是类别数上限）**：`ScaleCorpus` / `build_scale_corpus(product_dir, max_classes)` —— 类别截断按 `(-count, key)` 降序取前 `max_classes`（与 `data.build_answer_space` 的确定性排序键一致）；`max_classes=0` 为全量。截断到 C 类时**金标全部落在保留类之内**的问题才属于该口径，金标含被截断类的问题**整体排除**（不能只删越界的那部分金标 —— 那会把「答案不在表内」偷偷变成「答案在表内」），行下标过滤后**重新编号**保证只有一套「语料行号」。现场：C=10 重建 = 10 类 / 10161 问题 / 39909 行（排除 2566 个越界问题）；C=298 全量 = **298 类 / 12727 问题 / 50865 行**（排除 0）；问题 **x1.253**、行 **x1.275**、类别 **x29.8**。

**按比例留出（边界归属 = 小于）**：`question_side` 是唯一实现 —— `h = int(sha256("qgroup\0{seed}\0{task}\0{question_id}"), 8B big-endian) % 100`，`h < holdout_pct` 进测试侧；作用单位是**问题组**（同一 `question_id` 的全部行同侧，哈希只用 `(seed, task, question_id)` 故构造性成立）。**边界归属 = 严格小于**（左闭右开 `[0, holdout_pct)`，命中概率恰为 `holdout_pct/100`）；`proportional_holdout_evidence()` 给出 `boundary_probe`（`h = pct-1 / pct / pct+1` 三点上「`<`」与「`<=`」的判定对照，`h == pct` 处两者**可区分**）与 `pct ∈ {0,100}` 退化端证据。废掉了 `test_per_class=2` 的逐类封顶语义。现场实际测试问题占比 0.191954 / 测试行占比 0.192313（目标 0.20）。

**判据退化修正**：`macro_criterion_applicability(n_classes)` —— 阈值 `10/C`，`applicable ⇔ 10/C < 1.0`（**严格小于**）；**恰好等于边界的现场用例**：C=10 时 `threshold == 1.0` ⇒ `applicable=False`（`boundary_case.threshold_at_10 / applicable_at_10`）。C=298 时阈值 0.033557 ⇒ `applicable=True`。真正生效的两条是「`macro >= 多数类基线 + 10pp`」与「`+ 5pp`」（阈值均为闭区间下界，判据用 `>=`）；`saturated_baseline_check` 做可行域边界核查（内部 ⇔ `0.0 < baseline < 1.0`），两个口径的多数类基线（0.100000 / 0.003610）都在内部 ⇒ 可判。

**解题类任务拆分（依据现场统计而非直觉）**：`SOLVE_TASKS = ('judge','choice')`、`MATCH_TASKS = ('blank','solve','triviaqa')`；依据 = 现场 per-task 答案类数 `{'blank':153,'choice':4,'judge':2,'solve':78,'triviaqa':100}`（judge/choice 只有 **2 / 4** 类，答案由计算/选项判定而非问题相似性，如「3 是奇数吗」与「3 是偶数吗」问法极相似而答案相反）。`task_side_stats` / `grouped_stats` 给分任务分侧统计；`task_split.impact_on_merged_metric` 显式给拆分对**合入口径**的影响：C=298 下解题任务集基线 macro 0.166667 vs 合入口径 0.003610（**x46**），故不拆分会让解题任务被 298 类长尾淹没。

**确定性参照基线（不训练、不用模型）**：`deterministic_baseline_metrics` 给 `constant_majority`（一律答本侧正样本行数最多的类）与 `row_argmax`（按本次统计行的正样本行数取多数类）两条；micro/macro 口径显式（macro 分母 = 本范围内实际出现的类数）；新增 `class_coverage`（`n_classes_with_gold_rows` vs `n_classes_with_candidate_rows` 分开报，macro 分母只由前者性质决定）与 `n_rows_out_of_scope`（指向本轮未保留类别的行，**逐行计数**、不进分母，全量口径下恒为 0 —— 这是「集合类断言必须显式列出允许的额外项」的落地）。现场：C=10 测试侧 micro 0.233512 / macro 0.100000；C=298 测试侧 micro 0.196074 / macro 0.003610。

**产物与确定性**：`qa_scale_probe.json` 8098 B `dc4e64c856b0ddd5...`、`qa_scale_eval.json` 88154 B `975534da01940a71...`、`qa_scale_report.{json,md}` 102830 B `5eb43f03d056eab4...` / 10152 B `3eebe0b302f2b7fc...`、`determinism_check.json` 2594 B `e21df22b8809ba6e...` + 4 个日志。`eval` 双跑（主目录 -> 拷内容产物到 `_repro/` -> `eval --repro-dir`）`n_common_files = 2`、`not_equal_files = []`、`determinism_ok = True`；`report` 连跑两次逐字节相同；证据块**排除**证据文件自身与两件报告（后置可重渲件）⇒ 无自指哈希回路；单侧独有文件显式登记、**不纳入** `determinism_ok` 布尔值。**如实登记一处非确定性来源**：`a_dir`/`b_dir` 保留命令原样串 ⇒ 绝对路径不同的运行之间不逐字节相同，同路径同参重跑才逐字节相同；另给 `*_normalized` 与 `paths_normalized_equal` 供机器判定。

**生效口径（所有结论受此约束）**：**max_classes = 298（全量）+ 20% 按比例留出 + 单 seed（42）+ 问题组级切分**。

**如实登记的局限（不得包装）**：单 seed（跨 seed / 跨留出比例极差未测）；**样本量只涨 x1.25** —— 冻结产物本身**已经是**「只导出已入表答案」的口径（`questions_in_table = 12727` / `questions_seen = 38364`），扩 C 的收益是「不再预先丢掉 288 个类」而非样本量暴涨（要真正暴涨必须回到 `questions_not_relevant = 25637` 的未入表问题，那需改上游口径，不在本轮范围）；本轮**不训练**，两条基线**不得**当作模型能力；测试零样本类 **51** 个（哈希比例抽稀不保证每类都有测试样本，保证会破坏比例性；这 51 类对 macro 无贡献，macro 分母只含实际出现的 277 类）；`triviaqa` 测试侧仅 7 问题 / 129 行 / 92 类，方差极大，不得据此下任务级结论；解题任务集类数极少（2/4），可行域窄，任何「优于基线」判据必须先做饱和点核查；**C=10 重建不是现状基线**（后者用缓存总体，二者总体不同，不得互相代入）。

**零回归**：`data.py` / `route.py` / `entry_table.py` / `robust_eval.py` / `variant_b.py` / `text_variants.py` / `encoders.py` / `backends.py` / `train.py` 等**零改动**；上游四目录 `git status --porcelain` 为空；顶层 9 个 `qa_*.pt.zip`、`_verify/text_variants/` 12 件、`step2/` 9 件、`_verify/merge_table/` 11 件、`dataset/` 17 件与两个 requirements 文件的 SHA256 + 字节**逐项不变**（基线↔终态共有 42 项比对 `CHANGED = []`）；`compileall` 退码 0；零新增依赖。
**轮 2 审查收口（皋陶 approved=false 的 1 error + 3 warning + 1 info，以及计划 1 遗留 3 项）**

- **E1（error，必修）按行口径合并分布聚合错误**：`qa_scale_run._scale_block` 的 `merged_r` 原为 `zip(q_all, r_all)`（把「训练侧**问题数**」与「训练侧**正样本行数**」相加），得到既不是问题数也不是行数的混合量；现场污染值 `merged_by_row.total` = **C=10 16480（min 310 / max 3632）**、**C=298 21698（min 2 / max 3632）**，且被渲染进 README 第 19.2 节直方图的「按正样本行数」列。修法：改为与 `merged_q` 对称的 `zip(r_all, r_test)`。修复后现场 `merged_by_row.total` = **10210（C=10）/ 13468（C=298）**，与 `train_by_row.total + test_by_row.total`（8240+1970 / 10849+2619）逐项相等。**新增现场断言** `assert_merged_distribution_consistent`：`merged[i] == train[i] + test[i]` **逐类逐项**（question 与 row 两个单位各自独立跑），取证落 `class_distribution.merged_consistency.{question,row}` 与 `merged_total_self_consistency.{question,row}`。**非恒真**：现场反例（任改一个元素 / 用历史污染数组 / 三者长度不一致）**均如期抛 `AssertionError``。
- **W1（warning）判据基线作用域显式化**：`criterion_table_using_test_baseline` 用的是**全量类口径**基线，与**分任务集** macro（解题任务集仅 2/4 类）不同量纲。修法：新增 `criterion_scope_declaration`（作用域 → 基线 → 阈值 绑成一格 + `cross_scope_rule`）与产物 `criterion_tables_by_scope`（`all_classes` / `match_tasks` / `solve_tasks` 各一格，含该作用域自己的类数、基线 macro 与三条阈值）；分侧基线由 `deterministic_baseline_metrics(..., tasks=SOLVE_TASKS/MATCH_TASKS)` **同作用域现场算出**。现场：C=298 的 `+10pp` 阈值 = all_classes **0.103610**（298 类）/ match_tasks **0.103636**（275 类）/ solve_tasks **0.266667**（6 类）；`solve_tasks` 的 `random 10/C` = **1.666667 ≥ 1 ⇒ applicable=False`（与 C=10 全量类同源）。
- **W2（warning）`constant_majority` 命名与实现对齐**：实现取的是**本侧（side + 任务过滤）**正样本行数最多的类，而 `predictor` 字段与 README 曾写「全局最高频类」⇒ 名实不符。取方案①（改述字段与文档，因为「取本侧多数类」是本基线想表达的意思）：`predictor` 改为 **`scope_side_majority_class`**，`row_argmax` 改为 **`scope_rows_positive_row_count_argmax_class`**，两条各增 `scope_of_majority` / `scope_of_majority_detail`（现场 = `side_row_positive_counts`，仅本侧无正样本行时回退 `frozen_table_count_fallback`）；`majority_class_index` docstring 同步改述为「纯 argmax 工具，不隐含全局语义」。
- **W3（warning）`ALL_TASKS` 现场一致性校验**：此前 `ALL_TASKS` 硬编码、文档声称「现场来自 manifest」但**全流程无校验**。修法：新增 `validate_task_list`（`probe` 与 `eval` 两条路径都跑），把**三处来源**（`manifest.tasks` / `npz meta.task_of_row` 去重 / `pairs` 回填后的任务去重）**逐项比对**再与 `ALL_TASKS` 比集合：**现场多出的任务名 ⇒ 抛 `AssertionError`（fail-closed，防静默漏样本）**；**清单多出的任务名 ⇒ 不抛但显式列出**（允许的额外项）。现场三处逐项相同、`missing=[]`、`extra=[]`。**非恒真**：去掉 `triviaqa` 或改名 `triviaqa2` ⇒ 抛；多一个 `math2` ⇒ 通过并在 `extra_in_declared` 列出。
- **I1（info）`__init__.py` 登记本轮新模块**：docstring 模块清单与 `__all__` 补入 `qa_scale` / `qa_scale_run`，与包内既有惯例一致。
- **F1（计划 1 遗留）`build_verdict` 量化字段语义**：`min_gain` / `max_drop` 原都写成 `min(delta)` ⇒ 恒相等且劣化时为负「增益」。修法：新增 `_quantified_deltas` —— 只在「可比且 `baseline_interior == True`」的指标上统计；`min_gain` = **正向 delta 最小值**（无增益时 `None`）；`max_drop` = **负向 delta 最小值**（无劣化时 `0.0`）；新增 `eligible_metrics` / `excluded_metrics` / `n_gains` / `n_drops` / `basis`。现场：`min_gain = 0.1036036036036036`、`max_drop = 0.0`、`excluded_metrics = ['qa_hit_rate']`（基线饱和不可判）⇒ 与 README 18.6 的 **+0.103604（关键词支）** 一致。**主判据逐项未变**（`verdict` / `degraded_metrics` / `improved_metrics`）。
- **F2（计划 1 遗留）`RETURN_TYPE_RULE` 第②类违规**：合同声明两类计数、实现只做第①类。取方案①：新增 `content_view_violations(merged, returned_indices, expected_kinds)` —— 第②类（期望 qa 但返回物**不属于任何 qa 条目的展示文本**）**只看输出内容**（`outputs_index` 内容视图）、**不看 `kind` 字段**，逐臂落 `route_eval.json.return_type_contract.content_view_check`。新增常量 `RETURN_TYPE_VIOLATION_CLASS_2_REACHABILITY` 写明**结构上不可达**（`__post_init__` 强制 kind ↔ 内容逐条一致）。**非恒真**：改 `outputs_index` ⇒ 构造函数当场抛 `ValueError`；用 `object.__new__` 绕过构造器喂伪造内容视图 ⇒ 报出 1 条第②类违规（`content_view_kind=''` 而 `returned_kind_by_field='qa'`）。现场 `n_class2_violations_total = 0`（两臂各 1684 检查项）。
- **F3（计划 1 遗留）`GOLD_SET_RULE` 注释**：原称「逐字抄录」但漏抄 rank 一句。取方案②：改述为**「口径等价摘录（本轮不使用 rank 字段）」**，新增 `GOLD_SET_RULE_SOURCE`（上源权威位置 = `n3d_qa_learn.text_variants.GOLD_SET_RULE`）与 `GOLD_SET_RULE_EXCERPT_NOTE`（明写**不声称逐字一致**）。
- **产物指纹重新基线（旧值 → 新值）**：`qa_scale_probe.json` 8098 `dc4e64c8…` → **9252 `4b2683ea…`**；`qa_scale_eval.json` 88154 `975534da…` → **119973 `7faeda90…`**；`qa_scale_report.json` 102830 `5eb43f03…` → **136406 `41ecf14f…`**；`qa_scale_report.md` 10152 `3eebe0b3…` → **10214 `06fd0b6a…`**；`qa_scale/determinism_check.json` 2594 `e21df22b…` → **2596 `1c90d2d9…`**；`merge_table.json` 17273 `759f3a1e…` → **17542 `f7a678f7…`**；`route_eval.json` 31424 `bf2444a4…` → **36767 `6b8853a5…`**；`route_eval_queries.jsonl` 1538962 `f188d788…` → **1652875 `a01f2cc1…`**；`merge_table_report.json` 52402 → **58017 `5b553c3c…`**；`merge_table_report.md` 19424 → **19693 `28b908ca…`**；`merge_table/determinism_check.json` 2940 `1c4234dc…` → **2943 `ac84a475…`**；`drill.json` 2684 `f5b5d2d7…` **不变**。
- **零回归**：上游四目录 `git status --porcelain` 为空；顶层 9 件 / `step2` 9 件 / `text_variants` 12 件 / `dataset` 17 件与两个 requirements 文件指纹**逐项不变**（11 项核心比对 `CHANGED=[]`）；`compileall` 退码 0；零新增依赖；本模块**只改** `qa_scale.py` / `qa_scale_run.py` / `merge_table.py` / `merge_table_run.py` / `__init__.py` / `README.md`。cp936 控制台下 `qa_scale` 三子命令与 `merge_table report` 的 stdout **均合法 GBK 且退码 0**。
**轮 3 主体：B3 QA 侧变体生成 + B4 开集 τ 标定（`qa_variants*.py` / `open_set_tau*.py`）**

### B3 · QA 侧变体生成（`qa_variants.py` / `qa_variants_run.py`）

**只用数据、不训练。** 复用第三步 `text_variants.VariantGenerator` 的 4 支生成器
（`原文` / `关键词` / `摘要` / `自适应`），**不改动**文本行侧产出路径与其 12 件既有产物
（零回归已逐项确认）。

- **B3-2 绑定输出 = 答案，不是问题原文**：`BINDING_OUTPUT_RULE` 唯一来源；
  现场 `binding_output.n_binding_equals_question_text = 0`（若等于问题数则口径当场失败）。
- **B3-3 严格按 `question_id` 整组切分**：定侧复用轮 1 的 `question_side`
  （`h = int(sha256('qgroup\0seed\0task\0question_id'), 8B) % 100; h < holdout_pct` 进测试侧），
  故「整组同侧」构造性成立；仍由 `verify_group_split` 复核三条不变量
  （`rows_disjoint_and_union_complete` / `questions_disjoint_and_union_complete` /
  `every_question_wholly_on_one_side`）；**权威全集（`expected_row_keys` /
  `expected_question_ids`）必须显式传入**，否则退回自比并标 `authority=self_referential`
  （该模式**不能**发现丢行）。现场 400 题 × 4 支 = 1600 条，训练 332×4 / 测试 68×4，
  三不变量全 True；**四个反例**（翻半组 / 复制一行 / 删一行 / 删整组）全部如期翻面。
- **B3-4 特征确实变化**：`feature_difference_report` 以原文支为参照，给逐位裸字节相同数、
  余弦 min/mean/p50、`n_cosine_below_one`；判定
  `feature_really_changes = 存在余弦<1 或存在裸字节不同`（**只看文本不同不足以断言**）。
  现场（400 题）：关键词 0/400 同文本、cos mean 0.393480；摘要 **4/400 逐字节同原文**、
  cos mean 0.876336；自适应 cos mean 0.681775；原文支 400/400（锚点，应然）。
- **执行范围如实登记**：实际在**按 `question_id` 升序的前 400 题**上全量生成（非 12727 全量），
  因全量外推约 1 小时、超出单次前台执行上限（600 s），且后台执行在本环境会使子进程停滞
  （现场 CPU 时间停在 0.078 s、模型未加载）。
- **词表口径显式偏离**：`QA_VOCAB_MIN_DOC_FREQ = 20`（文本侧为 2）；QA 语料唯一候选词
  163690、mdf=2 保留 49988 词、mdf=20 保留 2849 词；**400 题子集上仅保留 50 词** ⇒
  关键词支更保守。偏离文本落 `generator.vocab_caliber_deviation`。
- 产物写 `checkpoints/qa_learn/_verify/qa_variants/`（**不落盘裸特征矩阵、不产 zip、
  无挂钟字段**）；双跑 `determinism_ok=True`、`n_common_files=3`、`not_equal_files=[]`。

### B4 · 开集「未识别」档的 τ 标定（`open_set_tau.py` / `open_set_tau_run.py`）

**补齐第一步 1a 的遗留。** 此前 5 处「本档**不标定**阈值（属第二步变体 B 的范围）」声明
**已全部订正**为「已实现 + 指向实现入口」（`robust_eval.py` 模块 docstring /
`unrecognized_report` docstring / `threshold_standard` 字段 / `run_evaluation` notes /
`render_markdown` 4 节 / `step2_run.py` 的 `unrecognized_note`）。
实现入口：`robust_eval.calibrate_thresholds(..., unrecognized_scores=...)` 的
`unrecognized_tau` 块；核心 = `open_set_tau.calibrate_tau`。

- **工作点**：τ = 使平衡准确率 `(recall+specificity)/2` 最大的候选阈值（候选 = `sorted(pos∪neg)`
  相邻中点），并列按 `(balanced_accuracy↓, margin↓, τ↑)` 字典序 —— 确定性、无超参。
- **边界归属（显式）**：判据 = **`margin > max(noise_floor, granularity)`，严格大于、等号归无效**
  （`GUARD_BOUNDARY_RULE`）；理由 = 恰好等分辨率的间隔可能只由**一个样本**造成。
  现场：确定性网格扫 37 组**可分**组合，命中 **6 组** `margin == threshold`，**全部 effective=False**；
  严格性对照（负总体降一档 ⇒ margin 0.25→0.5、threshold 不变 ⇒ effective 翻 True）
  证明判据**非恒假**。
- **噪声底同总体、同口径、样本量减半**：`noise_floor_of` 把同一批 pos/neg 确定性对半拆分、
  各自重标定取最大偏差（现场 0.0001197755）；**粒度** = 工作点移动一个样本引起的 τ 变化
  （现场 0.0009683371）+ 比率空间 `1/n_pos` / `1/n_neg`，与落差一并核算。
  `calibrate_tau` **强制**校验 `population` 的 `side` / `profile` / `table_sha256` 齐备且
  `pos_side == neg_side`，缺失或不一致**直接抛**（**禁止跨总体复用**）。
- **现场场景**（真实语料）：键表 = 9227 个训练侧题面向量（`lexical-88`/role=question/dim=88），
  正总体 = 抽掉的 1057 个训练侧问题（**不在表内**），负总体 = 2443 个测试侧问题
  （轮 1 整组切分下从不进表）；`table_sha256 = 858dcb40...`。
- **诚实结论（负结果）**：τ = 0.8392274976、`margin = 0.0001155138`
  **小于** `threshold = 0.0009683371`（主因粒度比 margin 大 8.4 倍）⇒ **`effective = False`**。
  两总体最高余弦几乎完全重叠（mean 0.8544 vs 0.8536）⇒ 单靠全局 τ 无法分到超过观测分辨率；
  **不得**据此宣称「开集识别已可用」，该 τ 仅作参照值且必须随向量化器/键表重新标定。
- 产物写 `checkpoints/qa_learn/_verify/robust/`；**不落盘裸特征矩阵、不产 zip、无挂钟字段**。
- **局限**：题面自检索而非「问题→答案」闭环，不代表任务准确率；两总体样本量不同（1057 vs 2443）；
  单 seed；噪声底用对半拆分近似、**未做** bootstrap（零新依赖约束）。
**轮 4：README 文档收口（纯文档，零代码改动，不重跑产物）**

皋陶遗留 3 项 warning 全部是 README 第 20 节的**文档内部不一致**，其中 D1 是轮 2 已修过的
「同一文件出现两个最终值」同型问题的**复现**。

- **D1 · 同一文件的唯一终值**：§20.3.5 的产物指纹表仍是**中间态**
  （`27066` / `1aa71396…`、`40671` / `70c4b642…`、`7857` / `d45d3814…`、`d9aad608…`），
  与 §20.5.3 的终值（`27863` / `ed889854…`、`41476` / `b3f9e9e1…`、`8109` / `d621888e…`、
  `15109dc6…`）及盘上实测矛盾。处置取**比「标注中间态」更彻底的「改为指针」**：
  §20.3.5 **整表删除**，数值一律以 §20.5.3 为**唯一来源**（理由：标注法仍会让同一文档
  出现两个字节数，后续引用仍可能选错）。§20.6 的引用改为三处**可唯一解析**的目标
  （§20.3.5 = 双跑工作序与证据口径 / §20.4.6 = robust 侧双跑取证 / §20.5.3 = B3 侧指纹终值）。
  **现场校验**：7 个终值字符串在全文**各出现恰好 1 次**；5 个中间态字符串**各出现 0 次**
  （`7857` 在 §17 的浮点数 `-0.13470919324577857` 里有 1 次巧合子串，非指纹引用，已核实）。
- **D2 · 章节编号全局唯一**：`#### 20.4.7 产物` 与 `#### 20.4.7 如实登记的局限` 重号，
  后者改为 **20.4.8**；新增 **§20.6.2**「编号唯一性校验（脚本断言，防再犯）」，
  内含**现场枚举结果**（全篇 `#### 20.x.y` 标题 **19** 个、去重后 **19** 个、重复项 = 空
  ⇒ 断言通过）+ 19 行编号/标题清单 + 可复算的断言脚本（正则枚举 + `Counter` + `assert`）。
- **D3 · 改动面唯一口径**：原 §20.6 写「新增 4 + 修改 6」却只列 5 项、§20.5.2 表列 8 项（称 7+README）。
  以 `git status --porcelain -- n3d_qa_learn` **现场实测**统一为
  **本轮（轮 3）实际动过 = 9 个文件 = 新增 4 + 修改 5**（新增 `qa_variants.py` /
  `qa_variants_run.py` / `open_set_tau.py` / `open_set_tau_run.py`；修改 `qa_scale_run.py`(P2) /
  `robust_eval.py` / `step2_run.py` / `__init__.py` / `README.md`），新增 **§20.6.1** 逐项表，
  与 §20.5.2 的逐文件表逐项对齐。
  **关键澄清**：仓库级 `git status` 是 **修改 4 + 未跟踪 8** —— 差异逐文件解释：
  4 个已跟踪文件显示 ` M`；4 个本轮新建文件是 `??`；**`qa_scale_run.py` 是轮 1 新建、
  至今未提交**，故仓库级只能归入 `??`（`??` 表示**提交状态**，不是「本轮没动过」）；
  `qa_scale.py` / `merge_table.py` / `merge_table_run.py` 建于轮 1 / 轮 C、本轮**未动**。
  据此：凡「本轮改动面」以 **新增 4 + 修改 5 = 9** 为准；凡「零代码改动」以
  **本轮仅改 `README.md`** 为准。
- **验收证据（现场枚举，非记忆）**：① **29 个** `n3d_qa_learn/*.py` 的 SHA256 改动前后
  逐项比对 ⇒ `CHANGED python files: (none)`、`removed: (none)`（任务要求的「16 个」是这 29 个
  的子集，此处给出全覆盖的更强证据）；② **16 件既有产物逐项命中**、无一被重跑；
  ③ 第 20 节内 **62 处** `20.x.y` 引用、**0 处**未解析；④ 冻结产物件数不变
  （顶层 9 / `step2` 9 / `text_variants` 12 / `dataset` 17），上游四目录 git status 为空；
  ⑤ `compileall` 退码 0。**本轮只改 1 个文件**：`n3d_qa_learn/README.md`。
## 关键词抽取器可插拔与 TF-IDF 口径


**离朱轮 5 测试与 4 条观察项处置（收口）**：离朱 **36/36 PASS**（单元 18 / CLI 契约 10 / 编译 1 / 产物可复算与零回归 7，FAIL 0、ERROR 0；E2E 不适用——本模块无 UI）。4 条观察项按「如实登记优先、非必要不改代码」处置，落在 README 新增 §21.10：**O2** tfidf 的「不需要 keybert」只在**运行期**成立（导入期仍有 R67 起的可选 keybert 导入 try/except，缺失回退 object 不阻断）——已补限定语；**O3** 工厂字典键 `extractor` 与两个 CLI 落盘证据块键 `keyword_extractor` 同义不同名（工厂字典只用于构造期分派、不进产物）——已明示；**O1** `keyword_terms` 在 `keyword_backend=None` 且候选为空时返回 `([], [])` 不抛错——**登记不修**（该方法为只读取证/调试用途、主路径走批量；候选为空时「无词可返」是如实结果；该早退分支为**既有实现**、本轮未改动，改成抛错会变更既有可观测行为而不影响任何产物；主生成路径 fail-closed 语义已成立）；**O4** 0 字节 `run.log` 现场定位在**另一个旧失败目录** `_verify/qa_variants_full/run.log`（mtime `2026-10-18 08:56:58`，早于本轮首次命令 `09:08:32`，非本轮产物），`_verify/qa_variants/` 现场恰 11 件、无 run.log、全部 mtime ≤ `2026-10-17 14:38:56`、7 件内容产物 SHA256 与 §20.5.3 登记值逐项一致。收口后 `compileall` 退码 0、`_verify/qa_variants_tfidf_full/` 11 件指纹与收口前逐项一致、上游四目录 git 空、改动面仍为 4 个文件（README 新增 §21.10，**代码零改动**）。


**轮 6 收口（W1/W2/W3 + 7 项 info；不新增功能，全量档 tfidf 产物行为零变化）**
- **W1 候选集同源无条件成立**：`TfidfKeywordExtractor._candidates_of` 对齐 `keyword_candidates` 的 `if vocab:` 语义 —— **只有语料级词表非空才过滤**；词表为空（退化语料）⇒ 两侧都不过滤；词表非空但本行无命中才回退前 `KEYWORD_MIN_CANDIDATES` 个。`fit` 固化 `self._vocab = frozenset(kept)`。**口径常量文本未改**（`TFIDF_CANDIDATE_RULE` 在修正后对所有语料都为真 ⇒ 产物字段不变）。现场退化用例（5 行 × 4 词 + `mdf=20`，`n_docs=5 / distinct=15 / kept=0`）：KeyBERT 9/行、TF-IDF 9/行、**旧口径 3/行**，合计 **45 = 45（旧 15）**，逐行候选（含顺序）全相等。
- **W2 语料 fail-closed**：`_tfidf_extractor()` 删除 `corpus_fallback` 形参；`keyword_corpus` 为空即抛可读 `VariantContractError`（批量 / `keyword_variant` / `keyword_terms` 三入口同规），两 CLI 侧前置同样报错。**选定 fail-closed 而非「保留回退并把 evidence 落产物」**，理由：① 只写在内存属性的偏离等于静默偏离；② 单行语料 `N=1`、`df≡1` ⇒ idf 退化常数，结论不可跨侧比较；③ 入口报错可见且不影响正常路径。现场：三入口 3/3 抛 `VariantContractError`（报文含 `keyword_corpus`）。
- **W3 CLI 不静默忽略**：`--keyword-extractor` **只挂在消费它的子命令**上 —— `qa_variants_run` 的 **probe/eval**；`text_variants_run` 的 **probe**（`_add_textvariants_arguments` 内 `if subcommand in ("", "probe")`）。其余子命令：独立入口由 argparse 报 `unrecognized arguments` **退码 2**；`step2_run textvariants`（单子命令、参数合并）由 `cmd_textvariants` 的**显式守卫**打印可读报文并**退码 2**。现场 5 条证据（text eval / text report / qa report 各 exit 2；`step2_run textvariants eval` 被 argparse 拒 exit 2；直接调 `cmd_textvariants` 返回 2 + 报文，均无 Traceback）。**如实登记既有缺陷**：`step2_run` 的 `textvariants` 子解析器**没有** `textvariants_cmd` 位置参数（只有默认 probe），故该入口现况只能跑 probe，与其 docstring 不符 —— 属上一轮遗留、本轮不改，但守卫已就位。
- **全量档真跑比对**（eval 164.0 s，退码 0）：`qa_variants_eval.json` 31899 `06c6e2ad895b3eb4`、`qa_variants_records.jsonl` 22706291 `1479bb8273aa1e4e`、`keyword_idf.json` 3339 `b043a62f7f956dbc`、`qa_variants_probe.json` 12918 `bc3c4add0b4ed212`、`qa_variants_report.md` 11045 `56b2237c42a1a90f` **全部逐字节不变**；仅 `determinism_check.json` 2699 `30c94046c50986e3` → **2716 `28c2d745ede9f011`** 与 `qa_variants_report.json` 49458 `21804b5496211fbb` → **49476 `7ec5193cc9e3d683`**。原因逐字段登记：前者只变「目录状态取证」两字段（`n_files_before_exclusion` `{7,4}`→`{11,4}`、`a_only_files` +`report.log`），后者只因**内嵌**该证据块；**受控实验**（同代码同数据，只换目录里是否已有报告产物）2655 B vs 2672 B，**差值同为 +17 B**，且 `n_common_files=4`/`determinism_ok=true`/`not_equal_files=[]` 两侧一致。
- **info 快修**：I1 §21.3/§21.8 命令逐行改 `E:\neuron3d\.venv\Scripts\python.exe -m ...`（并注明可用已激活 .venv 的 `python`）；I2 `qa_variants_run.py` 三处字面量改引用 `TV.KEYWORD_EXTRACTOR_TFIDF`；I3 删除只写字段 `self._tfidf`，改只读属性 `keyword_extractor`，并把 `keyword_fit_evidence` 改为只读属性（读 `TfidfKeywordExtractor.evidence`，由 `fit` 落值）；I4 `idf_of` docstring 按实现改述（`df=0` 只在本行不在拟合语料内或词表为空的兜底分支出现；生产调用拟合语料恒为超集 ⇒ `df>=1`）；I5 `_build_generator` docstring 补 TF-IDF 分支与语料 fail-closed 条；I6 `render_markdown` 的口径说明**只输出一次**（`applicable=True` 分支输出逐字符不变 ⇒ 全量档 `report.md` 与轮 5 逐字节相同）；I7 §21.7 新增第 8 条（关键词支输出可含重叠 n-gram 造成的重复词，现场样例 `平面 平面 平面 平面 平面 垂直`，属钉死规则的可读性现象、非缺陷，消除需改口径且会改动全量产物）。
- **验收现场**：`_verify/qa_variants/` **11 件** + `_verify/text_variants/` **12 件**逐项（字节 + SHA256，19 件走登记值、4 件日志走现场值）⇒ `mismatch=[]`，两目录最大 mtime 分别 `2026-10-17 14:38:56` / `2026-10-13 23:37:16`；上游四目录 git 空；两份 requirements 未改（根 `f69450d5…`）；零新增依赖；`compileall` 退码 0；README 新增第二十二节（22.1~22.6）。


**轮 6b 复测收口（I1 文档缺陷 + W3 报文去 Markdown；离朱 20/20 PASS）**：轮 6 首测 24/25，唯一 FAIL 是**文档级** I1 —— `README §21.3` 命令表 4 行中 3 行（`eval`×2 / `report`）仍是 `<venv-python>` 占位符，与「表…逐行带显式解释器」的字面要求不符；§22.5 只写「改表头」也与实现不一致。处置：① §21.3 命令表 **4 行全部**改为带完整 `E:\neuron3d\.venv\Scripts\python.exe -m ...`，表前加注记「下表 4 行命令逐行给出完整解释器；§21.8 代码块 7/7 行同样带完整路径」并明确 `<同上>` 只指目录实参；② §22.5 的 I1 行改述为「表前注记 + 表内 4 行命令全部 + §21.8 7/7 行」；③ §22.4 表前补件数口径「7 件内容产物（另有 4 件日志派生件不参与内容指纹比对，`report.log` 随 `report.json` 变化）」；④ `text_variants_run` 的 W3 守卫报文去掉 Markdown 强调符（`**不接受**`→`不接受`，纯文本便于精确子串断言，逻辑与退码不变）。**复测现场**：20/20 PASS / FAIL 0 / ERROR 0 —— §21.3 表 4/4 行含完整解释器、全文 `<venv-python>` 计数 **0**、§21.8 7/7 行完整、注记与 I1 行措辞一致；守卫报文运行时 `**` 计数 0、`cmd_textvariants(gate/eval/report × tfidf)` **3/3 返回 2** 且含精确子串「不接受 --keyword-extractor」、无 Traceback；回归 W1 **45 = 45**（旧 15）、W2 三入口 3/3 `VariantContractError`、W3 参数落点不变（qa `report` 2/2 与 text `gate/eval/report` 6/6 均 exit 2；两个 `probe` 带 tfidf 均 exit 0 且产物 `keyword_extractor=tfidf`、文本侧 idf N=2665）、`compileall` 退码 0；产物零回归（11 件 + 12 件 + 7 件产物 + 4 日志逐项一致，套件期间 30 个受监控文件前后不变）；上游四目录 git 空、两份 requirements 未改、零新增依赖。**如实登记的观察项（不阻塞）**：§21.8 两处 `<同上>` 仍指代目录（注记已澄清，可后续展开）；轮 6b 的 probe 字节 12,800 vs 12,801 只因临时 `--out-dir` 名短 1 字符（探针产物内嵌 out_dir 字符串，非行为变化）；§22.4 受控实验（2655/2672 B）本轮按说明未复跑、只核对登记值与 `2716−2699=17` 算术自洽；`step2_run textvariants` 非 probe 子命令现况被 argparse 先拒（既有缺陷、已在 §22.3 登记），守卫用直接调用验证。


**轮 7 文档微收口（D1 断链 / D2 四处 docstring / D3 轮 6 测试登记；纯文档与注释，零代码行为变化、零产物重跑，改动 2 个文件）**
- **D1 三处断链已修**：§21.1 的「见 §21.11 的 W3 现场证据」→ **§22.3**、§21.1 的「（见 §21.11）」→ **§22.2**、§21.6 的「逐字段原因见 §21.11」→ **§22.4**。**枚举证据（现场脚本，可复算）**：README 内 `§21.x`/`§22.x` 引用 **24 处**（逐目标 `21.1×1 / 21.3×3 / 21.7×1 / 21.8×7 / 21.10×1 / 22.2×2 / 22.3×3 / 22.4×4 / 22.5×2`），9 个去重目标全部命中现有 17 个节标题（§21.1~21.10、§22.1~22.7）⇒ **未解析 = 0**（脚本 `assert not unresolved` 通过）；修前同脚本报 3 处未解析。§22.7 内**刻意不写出**那个从未创建的节编号，以免枚举脚本把它当成真实引用。
- **D2 四处 docstring 校正（只改注释）**：① `TfidfKeywordExtractor.candidates()` 补「**须先 fit**；未拟合时完全不过滤，与 `extract_terms()` 抛 `RuntimeError` 的语义**不一致**（既有实现差异），本方法**不加** fail-closed 断言（那属行为变更）」；② `min_doc_freq` 参数补「**fit 后修改不生效**（fit 把满足下限的词固化进 `_vocab`，须重 fit）」；③ `fit()` 补「**副作用**」段（固化语料侧统计与 `_vocab`，并把同一份取证块写入 `self.evidence`）；④ `keyword_terms` 原「只读，不改变任何计数器」限定为「不改变 `VariantGenerator.counters`；TF-IDF 分支仍会累加**抽取器自身** `stats` 的打分统计，两者不是同一本账」。**docstring-only 证明（AST 口径）**：`text_variants.py` 的 AST **去掉全部 docstring** 后与本轮改动前备份逐字比较 ⇒ `code_ast_identical = True`（脚本断言通过）。
- **D3 README 新增 §22.7「离朱轮 6 测试登记」**（沿 §21.10 惯例）：**r6 = 25 用例 / 24 PASS / 1 FAIL / 0 ERROR**，唯一 FAIL 为 **D1b · I1 字面要求未落地**（§21.3 命令表三行为占位符、§22.5 只写「表头」）→ 修复动作（4 行全量 venv 路径 + 表前注记 + §22.5 改述 + §22.4 件数口径）与 **r6b 复验 PASS**；同批处置观察项 **O2**（W3 报文去 Markdown，运行时 `**` 计数 0）；**r6b = 20/20 PASS / 0 FAIL / 0 ERROR** 的关键现场值；O1~O5 五条观察项的处置表；并把本轮两项一致性校验写进该节。
- **验收现场**：三目录共 **34 个文件**（`_verify/qa_variants/` 11 + `_verify/text_variants/` 12 + `_verify/qa_variants_tfidf_full/` 11）逐项字节 + SHA256 ⇒ `mismatch=[]`（**未重跑任何产物**）；`compileall -q n3d_qa_learn` 退码 0；上游四目录 git 空；两份 requirements 未改；零新增依赖；本轮改动面由 mtime 佐证（README 17:35:06 / text_variants.py 17:34:06 为本轮，`qa_variants_run.py` 15:37:06 与 `text_variants_run.py` 15:55:14 属轮 6/6b 未提交状态、本轮未触碰）。临时脚手架已删除。
## 答案表同义归并与审计（轮 8）

把统一答案表里「同一实体的拼写变体」在**数据层**收敛为单一 canonical 类，用来隔离 QA 侧不可学的
**标签噪声**成因。**离线、可审计、默认关闭**：只改数据口径，不碰上游 `n3d_qa`、不改模型、不做任何训练。

**四条规则（唯一实现 `answer_merge.py`；规则编号进产物，逐条可追溯）**
- **R1a-case-ws**：小写 + 去首尾空白 + 折叠内部空白（现场命中 **0** 类，冻结表 key 已全为小写无首尾空白）。
- **R1a-decoration**：去**纯装饰性标点**（逗号 / 句点 / 撇号 / 引号，`DECORATION_CHARS`）；**只**在不含数字与运算符的答案上执行（现场 **14** 类）。
- **R1c-paren**：剥离包裹式括号及其内容，**仅当**剥离后非空且「主体字母部分不变」；括号内容含数字/运算符、括号不配平、剥离后为空（如 `(us)`）或剩纯标点 ⇒ **不剥离**（fail-closed；现场 **9** 类）。单步实现 `strip_one_paren_group` 给出 `ok/result/content/reason` 六个原因码。
- **R2-synonym**：小型**冻结**同义表 `SYNONYM_RULES`（`R2-1` us/usa -> united states、`R2-2` united states of america -> united states、`R2-3` uk/u k -> united kingdom），每条带 `basis`；`SYNONYM_EXCLUSIONS` 登记**未采纳**候选与否决理由（great britain / england / america / peking）。**本语料命中 0 条**（只作守门，如实登记）。
- **R3-restricted（禁并硬约束）**：任何含**数字或运算符**（`OPERATOR_CHARS`，**不含** `.` `,` 引号）的答案**只做大小写与空白归一** ⇒ `1` vs `-1`、`1/2` vs `12`、`(1,0)` vs `1,0` 必须保持独立。
- **R3-negation**：否定词元签名（`NEGATION_TOKENS`）不同 ⇒ 不合并（现场阻断 **0** 对）。
- **R4-high-risk（安全阀）**：簇内成员 > `HIGH_RISK_MAX_CLUSTER_SIZE`(4) 或簇内 display 相似度下界 < `HIGH_RISK_MIN_DISPLAY_SIMILARITY`(0.5) ⇒ **仍然合并**但标 `high_risk` 并进人工复核清单。

**对外接口**：`MergeCluster` / `MergePlan`（`as_mapping()` / `canonical_labels()` / `fingerprint()` / `summary()` / `multi_member_clusters()` / `high_risk_clusters()`）、`build_merge_plan(rows)`（构造期 5 条不变量：index 稠密、key 互异、簇覆盖全集、canonical 稠密、簇 count 之和 == 原表 count 之和）、`validate_merge_map(merge_map, n_original=..., merge_labels=...)`（定义域/值域/标签三元一致，fail-closed）、`assert_no_merge_pairs(plan)`（R3 反例现场断言，key 缺失即抛）、`class_records` / `cluster_records`（`merge_map.jsonl` 的两种记录）、`diagnostic_relaxed_merges`（**诊断量，不参与归并**：若再叠加变音符折叠 + 去空格会额外并掉多少类，并标出**违反 R3** 的组）、`rules_documentation()`（规则文本唯一来源）、`normalize_answer` / `has_digit_or_operator` / `negation_signature`。

**语料层接入（默认关闭）**：`qa_scale.build_scale_corpus(product_dir, *, max_classes=0, merge_map=None, merge_labels=None)`。`merge_map=None`（默认）时**不进入**任何归并分支、`source` 不多出任何键 ⇒ 与轮 1 逐位不变（现场 4 件内容产物逐字节相同）。给定时经 `validate_merge_map` 校验后由 `_canonicalize_answer_table` 重建 canonical 表（`index/key/display/count` 与冻结表同形，`count` 聚合）：`answer_keys` / `answer_display` / `class_row_counts` / `class_pos` 全部换成 canonical 空间，`pairs[].candidate_index` 与 `question_gold` **一并改写**为 canonical 下标，`source` 追加 `merge`（含 `n_duplicate_rows_before/after_merge`、`n_gold_pairs_collapsed`、`gold_rows_unique_premise`、`merge_map_sha256`）与 `index_space`；`merge_labels` 缺 `merge_map` 时**报错拒绝**（不静默忽略）。

**零训练对照 CLI**：`python -m n3d_qa_learn.answer_merge_run {probe|eval|report}`，产物一律写 `checkpoints/qa_learn/_verify/answer_merge/`（`merge_map.jsonl` / `audit.md` / `before_after.json` / `determinism_check.json` + 3 个日志；**不产 zip、不落盘裸特征矩阵、无挂钟字段**）。`eval` 用**同 seed 42 / 同 20% 按比例留出 / 同 `(-count, key)` 排序键**对照归并前后，并硬断言「切分四项逐位相同」`assert_same_split` 与「`merged[i] == train[i] + test[i]`」逐类自洽 `assert_train_plus_test_equals_merged`；`report` 从 `merge_map.jsonl` + `before_after.json` **确定性渲染** `audit.md`（规则登记 / R2 表 / 簇明细 / **high-risk 单独章节** / R3 反例 / 放宽诊断 / 前后对照 / 决策门 / 口径变更声明），并可用 `--repro-dir` 落双跑证据（比对 3 件内容产物、**排除证据文件自身**）。

**结构决策门（只判结构，不判可学）**：① 归并后「每类样本数中位数 >= 10」（主口径 = 训练侧问题数，另附冻结表 count / 合并问题数 / 训练侧行数三档）；② 测试侧零样本类**显著下降**（`after < before` 且相对降幅 >= 10% 且绝对减少 >= 1 类）。两条同时成立才 `structural_gate_passed = True`，且 `not_claimed` 显式列出「可学性 / 模型能力 / 跨 seed 稳定性」。

**现场实测（单 seed 42，零训练）**：298 -> **281** 类（减 17，5.70%）；多成员簇 **10** 个覆盖 **27** 类；high-risk **2** 簇（`america` 规模 6、相似度下界 0.4242；`china` 规模 5、0.2174）；簇 count 合计 **13468 = 13468**；问题 12727 / 行 50865 / 训练问题 10284 / 测试问题 2443 / 训练行 41083 / 测试行 9782 **全部不变**；每类样本中位数（训练侧问题数）**5.0 -> 5.0**，测试零样本类 **51 -> 51**（相对降幅 0.0000）⇒ `structural_gate_passed = False`；结构改善的可见量：只有 1 条训练问题的类 **14 -> 8**、训练问题 <= 4 条的类 **133 -> 116**；基线 `constant_majority` micro **0.196074** 不变、macro **0.003610 -> 0.003846**（C 变小 ⇒ `1/C` 变大，**不得**读成模型变好）。R3 反例 4 条（3 必需 + 1 附加 U+2212）全部保持独立，并给两条「非恒真」反证（伪造 `index_map` / 真关 R3 后断言必抛）。**281 与任务书举例 277 的差额**由放宽诊断解释（+6 -> 275，其中 `['1,0','10']` 一组**违反 R3** 不可采纳，剔除后 276）。

**产物指纹**：`merge_map.jsonl` 222070 B `682a742a…`、`audit.md` 25884 B `1ba1d2ba…`、`before_after.json` 35737 B `c89cef10…`、`determinism_check.json` 2451 B `0bf96d66…`；双跑 `n_common_files = 3`、`determinism_ok = True`、`not_equal_files = []`。**验收**：`compileall -q n3d_qa_learn` 退码 0；`probe` 737 B / `eval` 445 B stdout 均**合法 GBK 且非合法 UTF-8**、`report` 46 B 纯 ASCII；上游四目录 `git status --porcelain` 为空；两份 requirements 未改、零新增依赖（只用标准库 + 既有 numpy）；自写验收脚本 **58/58 PASS**。

**口径变更声明（引用前必读）**：归并后类数 = **281**，不再是 298 ⇒ 轮 1（`qa_scale`）的全部「C=298」结论**不再直接适用**，引用须以 `before_after.json` 的 `after` 块为准。**局限**：归并只消标签噪声、**不增加样本总量**；测试零样本类未下降；单 seed、单留出比例；零训练（所有指标为确定性参照基线）；金标对折叠 55 条（`gold_rows_unique_premise = True`）；**如实登记上游既有性质**——冻结问答对里 `(question_id, candidate)` 本来就不唯一（label=0 重复行 9479 条，label=1 侧唯一），该现象**不是**归并造成的。
### 验收口径补充：归并关闭时的「默认路径逐字节不变」分三档（现场实测，勿合并成一档）

- **档 1（全新目录，无任何既有文件）**：`qa_scale_run probe` / `eval` 与交付**逐字节相同**（`qa_scale_probe.json` 9252 B `4b2683ea…`、`qa_scale_eval.json` 119973 B `7faeda90…`）。
- **档 2（同一份双跑证据在场）**：把交付的 `determinism_check.json` 复制进新目录后跑 `report`，`qa_scale_report.json` 136406 B `fc1037d7…` / `.md` 13773 B `3a960828…` 与交付逐字节相同；且两个**目录名不同**的复现目录之间也逐字节相同 ⇒ `report` 的差异只取决于证据文件内容，与目录名无关。
- **档 3（全新目录、无证据文件）**：`report` 少掉「双跑证据」整块（现场 `qa_scale_report.json` 133614 B、`.md` 13116 B，md 195 行 vs 203 行）。这是**轮 1 已登记口径**（`report` 是双跑证据的渲染器，`a_dir` / `b_dir` 原样串经 `determinism_check.json` 进产物），**不是本轮引入的差异**；`qa_scale_run.py` 本轮**零改动**（`git status --porcelain -- n3d_qa_learn` 仅显示 `qa_scale.py` / `__init__.py` / `README.md` 三个修改项 + 两个新增文件）。**不修**该行为的理由：改它必然改动交付的 `qa_scale_report.{json,md}`，与本轮「既有产物一律不动 / 不得覆盖 `_verify/qa_scale` 下任何既有文件」硬约束直接冲突。
- **对照样板**：本轮新增的 `answer_merge` 三件内容产物（`merge_map.jsonl` / `before_after.json` / `audit.md`）在**全新目录**重跑即与交付逐字节相同（不内嵌任何路径串），可作为后续对齐方向。
### 观察项：轮 1 的 `report` 是「probe + eval + 证据」的渲染器（本轮登记、不修）

`qa_scale_run.cmd_report` 的两处既有行为（**轮 1 代码，本轮零改动**），已作为后续轮次候选修法登记：
1. **缺证据 ⇒ 少整块**：没有 `determinism_check.json` 时不写 `determinism` 段与 md 第 7 节表格（现场 `qa_scale_report.json` 133614 B / `.md` 13116 B，md 195 行 vs 交付 203 行）；
2. **残留旧证据 ⇒ 污染报告且无告警**：在残留上一轮证据的目录重渲，会把上一轮的目录路径与双跑指纹写进报告（交付目录现场 `n_files_before_exclusion.a = 9`，即含历史残留）。
**本轮不修**的理由：改它必然改动交付的 `qa_scale_report.{json,md}` ⇒ 与「既有产物一律不动 / 不得覆盖 `_verify/qa_scale` 下任何既有文件」直接冲突。候选修法（下一轮可做）：report/md 不内嵌原始路径串（原样串只留 `determinism_check.json`）、`n_files_before_exclusion` 改为「本次运行产物清单」、缺证据时在 `report.json` 写显式键（如 `determinism.present=false`）并在证据比 `eval.json` 旧时告警、改完重新冻结期望哈希。本轮新增的 `answer_merge` 已按该方向实现（三件内容产物在**全新目录**逐字节相同，`determinism_check.json` 已被排除在比对集合之外）。
### 轮 9：W1 / W2 / W3 三条 warning 的修复（本模块代码 + 产物已同步刷新）

- **W1（断言恒真）**：新增 `all_side_class_counts(corpus, unit)` —— **不看 split**、直接对全部问题/全部行按类计数，作为**独立来源**；`assert_side_sum_equals_all(unit, side_a, side_b, all_counts)` 校验 `side_a[i] + side_b[i] == all_counts[i]`（逐类），`_corpus_block` 的 `merged_by_question` / `merged_by_row` 改为该独立统计值（不再由 `zip(train,test)` 得出）；另加 `side_sum_counterexample`（把某侧某类 +1 ⇒ 必须抛错）并在 `_corpus_block` 内**硬门禁**：自检未抛错即 `AssertionError` 拒绝出数。现场：`10849 + 2619 == 13468` 通过；反例抛错（报文含 `side_a/side_b/all_side` 逐类明细）。
- **W2（只要求 1 件）**：`determinism_block` 改为 `determinism_ok = len(files) == len(expected) and not not_equal`，新增 `n_expected_files` / `missing_files` / `compared_files`；新增 `determinism_missing_file_probe`（TEMP 下复制三件、删一件、再跑同一函数，路径不进产物），结果写进 `determinism_check.json.missing_file_counterexample`，`cmd_report` 对其做硬门禁。现场：齐备 `True`；删 `audit.md` ⇒ `False` + `missing_files=["audit.md"]`。
- **W3（守卫过宽 / 日志无守卫）**：新增 `out_root()` = `checkpoints/qa_learn/_verify/answer_merge`，`_guard_out_dir` 根目录**收窄**到它（旧版只要求 `_verify/` 前缀，可覆盖 `_verify/qa_scale` 等别轮交付产物）；新增 `_guard_log_file`（与产物目录**同等**规则，旧版完全没有守卫）；`main()` 在**建目录之前**校验两者，越界 ⇒ 可读报文 + **退码 2** 且不创建目录；新增 `guard_evidence()`（复刻旧判据与新判据逐路径对照，纯计算不建目录）写入 `before_after.json.out_dir_and_log_guard_evidence`。现场：`changed=True` 用例 **4** 条（`qa_scale` / `step2` / `qa_scoped` / `qa_scale` 默认产物目录），CLI 实测 `--out-dir .../qa_scale` 与 `--log-file .../qa_scale/probe.log` 均 **rc 2** 且未创建目录，`qa_scale` 9 件哈希未变。
- **轮 8 产物已按修复重跑刷新**：`merge_map.jsonl` 222070 B `682a742a…`（未变）、`audit.md` 25884 B `1ba1d2ba…`（未变）、`before_after.json` **41310 B `c7112bc3…`**、`determinism_check.json` **3181 B `963025fb…`**；`determinism_ok=True`、`n_expected_files=3`、`missing_files=[]`。
## 类集收缩与零训练可学性下限（轮 9）

按「每类问题数」收缩类集（A 口径）并给出**零训练可学性下限**；**不动答案表、不训练任何模型**。

**A 口径（引用任何数字前必读）**
- **A 口径 = 仅筛选类集，归并关闭**。轮 8 的答案归并有 2 个 high-risk 簇语义有误（`china(Peking)` → `china`；`america(US)`/`(USA)` → 有歧义的 `america`），**high-risk 未解决前不得启用归并**；`qa_scoped` 不接受 `merge_map`，也不与轮 8 的 281 类口径混用结论。
- **门槛口径 = 每类的「问题数」**（该类的不同 `question_id` 个数，一问只计一次），**不是**正样本行数、也不是冻结表 `count`；门槛用**筛选前**（全量类集、全部问题）的计数判定。
- 语料沿用轮 1 的规则：**金标全部落在保留类内**的问题才进语料（含越界金标的问题**整体**排除）；切分沿用 20% 按比例留出 / `question_id` 整组 / 单 seed 42 / 边界归属严格小于。

**语料层接口（默认关闭）**：`qa_scale.build_scale_corpus(product_dir, *, max_classes=0, min_class_questions=0, merge_map=None, merge_labels=None)`。`min_class_questions=0`（默认）时**不进入**筛选分支、`source` 不多出任何键 ⇒ 与轮 1 逐位不变（现场 4 件 `qa_scale` 内容产物逐字节相同）。开启时：先在**筛选前的每类问题数**上按门槛剔除，再按 `(-count, key)` 截断（两者同给时顺序固定并写进 `source["class_scope"]`）；`source["class_scope"]` 登记 `count_unit="questions"` / `count_rule` / `n_classes_before|after` / `kept_class_question_counts{min,median,max,total}` / `n_question_class_incidences_kept` / `n_questions_covered_any_kept` / `n_questions_excluded_by_scope_filter` / `n_questions_after_scope_filter`；保留类集为空即 `ValueError`。

**对外接口（`qa_scoped.py`）**：`build_scoped_corpus(product_dir, *, min_class_questions=10, holdout_pct=20, seed=42)`（语料 + 切分 + 摘要文档；门槛 <= 0 拒绝）、`verify_split_invariants`（三条不变量逐条复核，违反即抛）、`load_question_texts`（`question_id -> question_text`，同一问题多文本即抛）、`build_question_matrix`（`entry_table.build_vectorizer_for("lexical-88","question")` 编码，返回 `matrix`（**只在内存**）/`dim`/`fingerprint`/`n_zero_norm`/`persisted=False`）、`class_prototypes`（训练侧问题特征均值再 L2 归一化；某类无训练样本或零范数即抛）、`predict_by_cosine`（余弦 argmax，平票取最小下标，报 `n_ties`）、`question_level_metrics`（问题级 micro / macro（分母 = **有测试问题的类数**）/ `macro_all_classes`（分母 = 全部类数）/ 每类 recall 分布 / 零 recall 类）、`majority_predictions`（同口径多数类基线）、`decision_gate`、`evaluate_prototype`。

**决策门与现场结论**：门 = 「原型 macro − 同口径多数类 macro >= +0.10」。现场（A 口径，单 seed 42）：**82 类 / 12005 问题 / 47495 行**（门槛口径问题-类关联 **12374**、覆盖 **12073**、语料内 **12005**；pre-scope 每类问题数 min **10** / median **21.0** / max **2240**，post-scope min 3 / median 20.5）；切分训练 9692 问题 / 38361 行、测试 2313 问题 / 9134 行（测试问题占比 0.192670），三条不变量成立；测试侧覆盖 **77/82** 类（5 类零测试）。原型基线 **micro 0.121055 / macro 0.241276 / macro_all_classes 0.226564**，多数类基线 **micro 0.183312 / macro 0.012987**，`delta_macro = +0.228289` ⇒ **过门（passed=True）**。**必须同时引用的 3 条限定**：① 原型 micro **低于**基线（0.1211 < 0.1833，基线把预测压在单一超大类上）；② **34/77** 个有测试问题的类 recall = 0（recall p25 0.000 / median 0.100 / p75 0.338）⇒ 「下限成立、覆盖面有限，不是全类可分」；③ `macro`（分母 77）与 `macro_all_classes`（分母 82）不同量纲。行级（轮 1 口径）参照 = micro 0.206233 / macro 0.0125（80 个类），**不可**与问题级 macro 直接比较。结论措辞固定为「零训练可学性下限成立，不等于已学会」；不过门时的固定措辞为「特征空间不可分 + 三条后续建议（换 bge-m3 语义档 / 继续上游放量 / 复核门槛）」。

**产物与 CLI**：`python -m n3d_qa_learn.qa_scoped_run {probe|eval}`，产物只写 `checkpoints/qa_learn/_verify/qa_scoped/`（`qa_scoped_corpus.json` 7155 B `d8fdbe5c…`、`qa_scoped_prototype.json` 6605 B `1fe56f33…`、`qa_scoped_gate.json` 3074 B `b37a4fff…`、`determinism_check.json` 2718 B `89abefab…` + 2 日志 + `_repro/`）；**不产 zip、不落盘裸特征矩阵（产物里 `persisted=false`）、无挂钟字段**。守卫与轮 9 修好的 W3 同口径：`--out-dir` / `--log-file` 只允许落在 `_verify/qa_scoped/` 之下，建目录**之前**校验，越界退码 2 且不创建目录；双跑证据按 W2 修好的语义「三件内容产物齐备且逐字节相等」才为 `determinism_ok`，缺件登记 `missing_files` 并附「删一件 ⇒ False」的反例。现场：三件内容产物在**两套全新目录**之间及与交付之间逐字节相同；`compileall` 退码 0；`probe` 379 B / `eval` 351 B stdout 为合法 GBK 且非合法 UTF-8；上游四目录 git 空；两份 requirements 未改、零新增依赖。

**局限（如实登记）**：类集收缩**不增加任何样本**（问题 12727→12005、行 50865→47495，只提高每类密度）；零训练 ≠ 模型能力（无任何可学习参数，也不涉及 N3D 后端训练）；门槛保证 pre-scope 每类 >= 10，整组排除后 post-scope 最小类仅 3 个问题（若要求语料内也达标需再迭代筛一轮，本轮不做）；单 seed / 单留出比例 / 单特征档；micro 与 macro 结论相反（判据是 macro，但 micro 退化必须同时报）；34/77 个类 recall=0 不支持「全类可学」。
### 轮 9 复测修复（离朱 R1 / R2）：两处「断言覆盖不足」

两处都出在**断言/自检的覆盖**上（数据与指标本身没错），已修并附现场证据：

- **R1（W1 自检不可判别）**：旧 `side_sum_counterexample` 把断言**写死**，复制 `side_b` 后某类 +1 再跑同一断言 ⇒ 增量必然破坏 `a+b==all`，`raised` **恒为 True**，因此调用点的 `if not raised: raise` 是**不可达死代码**，无法识别它声称要识别的「恒真」。修法：`side_sum_counterexample(..., assertion=None)` 增**注入参数**，新增 `tautological_side_sum_check`（把 `all_counts` 现场算成 `side_a+side_b` 的**故意恒真**断言）；`_corpus_block` 对 question / row 各跑**两个方向**（真实断言 ⇒ `raised=True`；恒真断言 ⇒ `raised=False`），两者都成立才记 `discriminating=True`，任一不成立即 `AssertionError` 拒绝出数；空序列改为 `applicable=False`（不再 IndexError）。现场：`before_after.json` 的 `counterexample.raised=True / assertion_name="assert_side_sum_equals_all" / discriminating=True`，`vacuous_assertion_probe.raised=False / assertion_name="tautological_side_sum_check"`。
- **R2（第三条切分不变量恒真）**：旧第三条只检查「问题 id 是否只出现在一侧列表」——两个列表本就按 id 切成不相交列表 ⇒ **天然恒真**；「整组挪行（把某测试问题的 4 行挪进训练侧、id 列表不动）」因此**不被发现**。修法：改用 `corpus.question_rows` **反查行归属**，并把不变量拆成**四条**：①行不交且并集完整 ②问题不交且并集完整 ③组内行不跨两侧 ④**问题 id 所在侧与其行所在侧一致**（④ 挡整组挪行）。现场：正例四条全 True（12005 个问题逐个反查，跨侧 0 / 两侧都无行 0 / id 侧不一致 0）；负例三种**全部抛错**（整组挪行 ⇒ `question_side_matches_row_side`，报文含 `declared_side=test / row_side=train`；组内拆开 ⇒ `every_question_wholly_on_one_side`；问题 id 重复 ⇒ `questions_disjoint`）。
- **产物指纹刷新**：`qa_scoped_corpus.json` **7718 B `afca7c80…`**（新增第四条不变量字段）、`qa_scoped_prototype.json` 6605 B `1fe56f33…`（未变）、`qa_scoped_gate.json` 3074 B `b37a4fff…`（未变）、`qa_scoped/determinism_check.json` **2718 B `8720c74f…`**（内嵌 corpus 哈希变化）、`answer_merge/before_after.json` **45516 B `39f37436…`**、`answer_merge/determinism_check.json` **3181 B `ad3f95b1…`**；`merge_map.jsonl`（222070 B `682a742a…`）与 `audit.md`（25884 B `1ba1d2ba…`）**未变**。