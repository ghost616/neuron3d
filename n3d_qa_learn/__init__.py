"""n3d_qa_learn：N3D 问答学习框架（两级业务路由 + 连接契约代理层）。

业务逻辑（固定，不可配置）
--------------------------
用户输入一个问题：

1. 先在 **QA 数据集**匹配，命中则返回答案 ``A``（**不再查文本**）；
2. 未命中则在 **文本数据集**匹配，返回匹配的文本行；
3. 两级都不命中则返回「无匹配」。

**命中判定 = 输出「不相关」类即视为未命中。**

对外能力
--------
* :mod:`n3d_qa_learn.features`：确定性文本向量化器（``D`` 维 + 口径 SHA256 指纹）；
* :mod:`n3d_qa_learn.backends`：连接契约代理层（``BackendAdapter`` 注册表，只承接 ``D``）；
* :mod:`n3d_qa_learn.heads`：``D`` 维 ``q`` 头 + 索引生成式 / 指针 Softmax 两种实现；
* :mod:`n3d_qa_learn.data`：QA 问答对读取、全局答案表、训练/测试切分、文本行语料；
* :mod:`n3d_qa_learn.route`：两级业务路由（携带来源标记与分数）；
* :mod:`n3d_qa_learn.train`：自建训练循环、确定性产物落盘、加载守卫；
* :mod:`n3d_qa_learn.probe`：P0 探针与单条端到端演练（硬门禁）；
* :mod:`n3d_qa_learn.evaluate`：步骤 1 评估协议、守卫拒绝证明、边界处置自检；
* :mod:`n3d_qa_learn.exp_repr`：**表示训练对照实验**（gap/σ 与 1-NN 指标、冻结范围 /
  头输入口径 / 目标修法三维对照、逐 epoch 退化诊断、可训参数更新量门禁）；
* :mod:`n3d_qa_learn.exp_repr_run`：对照实验 CLI（``drill`` / ``run`` / ``summary``，
  只写 ``checkpoints/qa_learn/_verify/exp_repr/``）；
* :mod:`n3d_qa_learn.cli`：纯 CLI 入口（``probe`` / ``drill`` / ``train`` / ``ask`` /
  ``eval`` / ``guard`` / ``selftest``）。

设计文档三步走的两个落地模块
----------------------------
* :mod:`n3d_qa_learn.entry_table` / :mod:`n3d_qa_learn.robust_eval`：**第一步 1a** ——
  统一条目特征表 + 逐位精确查表 + 分档鲁棒性考卷 + KNN 基线台账（**不改结构、不做训练**），
  入口 ``python -m n3d_qa_learn.step2_run robust {probe|run|calibrate|report}``；
* :mod:`n3d_qa_learn.variant_b`：**第二步 变体 B** —— 单层 ``D→D`` 可学变换（恒等初始化）
  + **冻结特征库**内积评分（「表内嵌输出层、权重即特征库」，**只变换、不生成**），
  扰动自监督训练 + 恒等门禁 + 逐格 变体 B vs KNN 对照 + **恒等参照锚点（T=I × 训练行 1999）**
  + 训练无害下限 + ``train_scorer``（``normalized`` 默认 / ``raw`` 对照档）+
  **训练前提校准轮**（``train_scorer × lr × weight_decay`` 最小网格 + 机器可读归因结论），
  入口 ``python -m n3d_qa_learn.step2_run variantb {probe|drill|train|eval|calibrate|report}``，
  产物一律写 ``checkpoints/qa_learn/_verify/variant_b/``（不落盘特征矩阵、不产 zip）。
* :mod:`n3d_qa_learn.text_variants` / :mod:`n3d_qa_learn.text_variants_run`：
  **第三步 Phase 1 文本侧数据改造** —— 把文本侧从「自匹配恒等任务」改造成
  「**多表述 -> 原文**」：4 支变体（``原文`` 锚点 / ``关键词`` KeyBERT / ``摘要`` 抽取式 /
  ``自适应`` 按 ``char_len`` 分档）+ 难负例（同文档他行 / 相邻行 / 编码器近邻）
  + **特征差异硬门禁**（逐位 + cos 分布 + 短行重合比例）+ **零训练**纯余弦检索
  ``R@1``/``R@5``（两种负例设定分开报）+ 机器可读判定结论（规则与判定同源）
  + QA 侧「同答案多问题」成组取证（**本轮只取证、不接损失**），
  入口 ``python -m n3d_qa_learn.text_variants_run {probe|gate|eval|report}``，
  产物一律写 ``checkpoints/qa_learn/_verify/text_variants/``（不落盘特征矩阵、不产 zip）。
* :mod:`n3d_qa_learn.merge_table` / :mod:`n3d_qa_learn.merge_table_run`：
  **轮 C 合并表零训练对照** —— 把 QA 侧 149 条目与文本侧 2665 条目合并为**单张**
  ``EntryKeyTable``（合计 2814），沿用 ``kind`` 字段区分返回类型，并完整保留两侧元数据
  （``dim`` / ``norm_spec`` / ``encoder_profile`` / ``encoder_fingerprint`` /
  ``key_table_sha256``）；三条**合并前置断言**（同 D 同 dtype 同归一化 / local-hash
  两角色逐位相同 / 合并前后键字节零变化）先跑、任一失败即停；两条路由臂
  （``qa_priority`` 现状两级硬优先级 vs ``merged_topk1`` 合并表单次排序取 top-1）
  用**同一批查询与同一批键**对照；新增**类型标记契约** ``LookupContract``
  （返回物类型 / 期望类型 / 类型一致性 / 违规分类计数），杜绝「返回文本却当成答案」，
  入口 ``python -m n3d_qa_learn.merge_table_run {drill|eval|report}``，
  产物一律写 ``checkpoints/qa_learn/_verify/merge_table/``
  （**不落盘裸特征矩阵、不产 zip、无挂钟字段**；同参重跑逐字节一致）。
  本模块**不改生产路由**（:mod:`n3d_qa_learn.route` 零改动），只做离线对照评测。
* :mod:`n3d_qa_learn.qa_scale` / :mod:`n3d_qa_learn.qa_scale_run`：
  **轮 1 QA 侧数据规模扩容与任务分层** —— 把 QA 侧从 C=10 扩到 **C=298 全量**
  （冻结产物 ``checkpoints/qa_learn/dataset/`` 的统一答案表口径），把主测试集从
  ``test_per_class=2`` 的逐类封顶改为**按比例留出**（默认 20%，哈希比例抽稀、
  **问题组级**切分、边界归属 = 严格小于 ``h < holdout_pct``），并把
  ``judge`` / ``choice``（答案类仅 2 / 4 类、答案不由问题相似性决定）拆为
  **解题任务集**与**匹配任务集**分侧报告；同时修正 ``macro >= 10/C`` 判据在 C>=10 时
  恒不可满足的退化（标 ``applicable=False`` 并如实报出），给出**同作用域**的
  ``macro >= 多数类基线 + 10pp / +5pp`` 阈值与确定性参照基线（**不训练任何模型**），
  入口 ``python -m n3d_qa_learn.qa_scale_run {probe|eval|report}``，
  产物一律写 ``checkpoints/qa_learn/_verify/qa_scale/``
  （**不落盘裸特征矩阵、不产 zip、无挂钟字段**）。
* :mod:`n3d_qa_learn.qa_variants` / :mod:`n3d_qa_learn.qa_variants_run`：
  **轮 3 · B3 QA 侧变体生成** —— 把第三步的 4 支变体生成器（原文 / 关键词 / 摘要 /
  自适应）**复用到 QA 问题文本**上；与文本侧改造的本质差异是**绑定输出 = 该问题绑定的
  答案展示文本，不是问题原文**（否则退化成「问题自匹配」）；切分**严格按
  ``question_id`` 整组**（同问题全部变体同侧，三条不变量 + **四个反例**取证，
  含「自指并集」这一真缺陷的修法）；逐支给出与原文支的**特征差异**（逐位裸字节 +
  余弦分布，抓「只改文本标签、特征没变」）并如实登记「与原文一致」的条数；
  入口 ``python -m n3d_qa_learn.qa_variants_run {probe|eval|report}``，
  产物一律写 ``checkpoints/qa_learn/_verify/qa_variants/``
  （**不落盘裸特征矩阵、不产 zip、无挂钟字段**）；**不改动**文本侧既有产出路径与产物。
* :mod:`n3d_qa_learn.open_set_tau` / :mod:`n3d_qa_learn.open_set_tau_run`：
  **轮 3 · B4 开集「未识别」档的 τ 标定**（补齐第一步 1a 的遗留）—— 工作点 = 最大平衡
  准确率；**噪声底同总体同口径、样本量减半**（对半拆分重标定取最大偏差）；
  **粒度**（移动一个样本引起的 τ 变化 + 比率空间 1/n）与落差**一并核算**；
  守卫判据 **`margin > max(noise_floor, granularity)` 严格大于、等号归无效**，
  并给出**恰好等于阈值的现场搜索用例**（全部判无效）与**抬高 margin 的严格性对照**
  （判有效 ⇒ 判据非恒假）；入口 ``python -m n3d_qa_learn.open_set_tau_run
  {calibrate|report}``，产物一律写 ``checkpoints/qa_learn/_verify/robust/``。

上游边界
--------
本包**只读** import ``n3d_shape`` / ``n3d_sphere`` / ``n3d_proto`` 的 ``model`` 与
``config``，以及读取 ``n3d_qa`` 已落盘的 QA JSON 缓存；**不修改**任何上游模块的源码、
产物与 ``current_spec.md``。产物一律写 ``checkpoints/qa_learn/``，验证类运行写其
``_verify/`` 子目录。
"""

from __future__ import annotations

__all__ = [
    "features",
    "backends",
    "heads",
    "data",
    "route",
    "train",
    "probe",
    "evaluate",
    "exp_repr",
    "exp_repr_run",
    "encoders",
    "encoders_run",
    "entry_table",
    "robust_eval",
    "variant_b",
    "text_variants",
    "text_variants_run",
    "merge_table",
    "merge_table_run",
    "qa_scale",
    "qa_scale_run",
    "qa_variants",
    "qa_variants_run",
    "open_set_tau",
    "open_set_tau_run",
    "cli",
]

__version__ = "0.1.0"