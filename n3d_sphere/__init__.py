"""N3D 二期架构变体包（三维神经元空间 + 球形有向拓扑）。

与一期 `n3d_proto` 的关系
-------------------------
* 本包是**并列存档的二期变体**，自包含（不 import `n3d_proto` 的任何模块）；
* 一期 `n3d_proto/` 完整存档、默认行为与既有产物逐位不变；
* 本变体默认 `topology="cube"` 时的行为与一期**逐位一致**
  （`python n3d_sphere/train.py --smoke-test` 的 `loss` 必须同为 2.419689）。
"""

__all__ = ["config", "utils", "model", "data", "train"]
