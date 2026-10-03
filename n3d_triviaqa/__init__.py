"""n3d_triviaqa：TriviaQA 证据段落二分类数据集的构建与验证模块（自包含）。

对外能力
--------
* :mod:`n3d_triviaqa.build_dataset`：从 triviaqa-rc.tar.gz 流式构建
  ``X[M, D] float32`` / ``y[M] int64`` 的 npz（键 ``X`` / ``y``，另含 ``meta``）；
  ``D = feature_dim(hash_dim, no_bag)``：缺省 ``64 + 6 = 70``，``--no-bag`` 为 6；
  原计划口径 1030 已实测不达标，仅作失败对照留档；
* :mod:`n3d_triviaqa.verify_dataset`：产物契约 / 幂等 / 标签均衡 / 无泄漏 / 端到端训练验证。

本包**不导入也不修改** n3d_proto / n3d_sphere / n3d_shape / n3d_viz / framework 的任何代码。
"""

from __future__ import annotations

__all__ = ["build_dataset", "verify_dataset"]