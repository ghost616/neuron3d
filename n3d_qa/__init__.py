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

本包**不导入、不修改** n3d_proto / n3d_sphere / n3d_shape / n3d_viz / framework 的任何代码
（:mod:`verify_dataset` 只**只读**导入 ``n3d_shape.data.load_npz_arrays`` 判定产物契约）。
"""

from __future__ import annotations

__all__ = ["build_dataset", "verify_dataset"]
