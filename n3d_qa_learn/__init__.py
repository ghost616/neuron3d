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
    "cli",
]

__version__ = "0.1.0"