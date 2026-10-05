"""n3d_qa：通用 QA 数据集处理模块（任意问答数据集 -> N3D 数组格式；包声明）。

定位
----
本包是**通用 QA 数据集处理模块**，对外只暴露「QA 数据集 -> N3D npz 数组」的构建与验证能力。
**当前内置 TriviaQA**（证据段落二分类）作为**参考实现**：下述维度 / split / 产物名均为
TriviaQA 口径；后续 QA 数据集按 adapter 挂入（输入解析 / 文档定位 / 答案合并 / split 表），
共用特征层与 npz 落盘层。

对外能力
--------
* :mod:`n3d_qa.build_dataset`：把 triviaqa-rc.tar.gz 流式构建成
  ``X[M, D] float32`` / ``y[M] int64`` 的 npz（键 ``X`` / ``y`` / ``meta``），其中
  ``D = feature_dim(hash_dim, no_bag, features)``：
  缺省 ``--features base`` 且 ``--hash-dim 64`` -> ``64 + 6 = 70``；``--no-bag`` 为 6；
  ``--features rich`` 在附加块上再加 4 列 IDF/TF-IDF 特征 -> ``74``（``rich + no-bag`` 为 10）。
  split 口径：``wiki`` / ``web``（verified 子集，产物名带 ``verified``）与
  ``wiki-dev`` / ``web-dev``（非 verified 全量 dev，按 QuestionId 剔除 verified 子集后构建）。
  原计划口径 1030 已实测不达标，仅作失败对照留档。
  [!] 产物路径与文件名（``checkpoints/triviaqa/n3d_triviaqa_verified_*.npz``）与产物 meta 里的
  ``module`` 字段**冻结为 TriviaQA 参考实现口径**，不随包名变化（否则既有产物 SHA256 会变）。
* :mod:`n3d_qa.verify_dataset`：产物契约 / 幂等 / 标签均衡 / 无泄漏 / 端到端训练验证
  （E0–E8；E7 为 rich 特征判别力 5 折 CV，E8 为 verified 与 dev 的 QuestionId 交集为 0）。

增量扩展（中文口径，与上述 TriviaQA 口径**并存且互不影响**）
------------------------------------------------------------
* :mod:`n3d_qa.zh_features`：**中文文本特征口径**（字符级 1/2/3-gram 哈希词袋 + 6 列覆盖度/
  长度特征）。复用既有 blake2b 落桶与 L2 归一化口径，不消耗全局 RNG；口径参数（n-gram 阶数、
  每阶桶数、盐、归一化方式、分词规则）全部写入产物 meta 并折叠为单一 ``spec_hash``。
  修掉了既有英文口径 ``TOKEN_RE = [a-z0-9]+`` 对中文**丢掉全部字符**（词袋恒全零）的阻塞项。
* :mod:`n3d_qa.adapters`：数据集 adapter（Math1 八文件 QA 解析 / ``data/doc`` 文本行切分 /
  答案归一化 / 跨任务统一答案表 / 负样本采样 / 库-查询留出集划分）。
* :mod:`n3d_qa.build_qa`：新增产物构建（QA 匹配任务 5 个 + ``all`` 合并 + 文本行产物），
  落 ``checkpoints/qa_learn/dataset/``：npz（``X[M,D] float32`` / ``y[M] int64`` / ``meta``，
  复用 :func:`n3d_qa.build_dataset.save_npz_deterministic` 的确定性 zip）+ 文本侧 JSONL
  （问答对 / 文本行表 / 行表索引 / 统一答案表）+ ``manifest.json``（逐文件 SHA256）。
* :mod:`n3d_qa.probe_zh`：中文判别力探针（硬门禁；用 :mod:`n3d_qa.verify_dataset` 的 E7 同口径
  5 折 CV）与逐桶数扫描，读数如实打印。
* :mod:`n3d_qa.verify_qa`：新增产物验证（F0–F9：口径回读 / 向量化确定性 / 产物契约 /
  逐字节幂等 / 文本侧可回读 / 行级重算 / 答案表一致 / 剔除登记 / 库查询无交集 / 零回归）。

本包**不导入、不修改** n3d_proto / n3d_sphere / n3d_shape / n3d_viz / framework 的任何代码
（:mod:`verify_dataset` 与 :mod:`verify_qa` 只**只读**导入 ``n3d_shape.data.load_npz_arrays``
判定产物契约）。
"""

from __future__ import annotations

__all__ = [
    "build_dataset",
    "verify_dataset",
    "zh_features",
    "adapters",
    "build_qa",
    "probe_zh",
    "verify_qa",
]

