"""N3D 神经元空间**形状变体**包（球体 / 立方体 / 圆柱体 + FCC 规则堆积 + 两阶段双副本展开）。

模块定位
--------
* **形状 = 生长度量，不是裁剪掩码**：FCC 放置分两步 —— ① `keep = metric <= rho` 裁剪；
  ② `argsort(metric, stable=True)[:N]` 取离中心最近的 N 个。**第②步才是产生形状分布的
  机制**（已实测：只把第①步换成同尺度立方体裁剪，选取集合与球体逐位相同）。本模块因此
  只替换第②步的排序度量：`sphere: ||p||2` / `cube: ||p||inf` /
  `cylinder: max(||p_xy||2, |p_axis|/lambda)`。
* **自包含**：照"二期从一期拷贝"的既有做法，从 `n3d_sphere` 拷贝 `__init__ / utils /
  data / config / model / train` 后独立演进；**不 import 一期 `n3d_proto` 或二期
  `n3d_sphere`**。两期的源码与既有产物**零改动**，从而彻底规避回归风险。
* **回归锚点**：`shape="sphere"`（默认）必须与二期在相同配置下**张量级逐位一致**
  （`torch.load` 后逐张量 `torch.equal`；依仓库 D1 口径**不比文件 SHA256**）。
* 网络为**分层有向无环图**：连接规则强制 `z_A < z_B`（沿流向轴严格上行），
  并按 z 升序逐层传播（阶段 2），输入层另设阶段 1 驱动 `S_in` 神经元。
  形状**不改变连接判据、突触半球切分、随机放置、训练循环与数据管线**，
  但会改变层数 `K`（架构深度）与 `E / |S_in| / |S_out| / 参数量`。

产物隔离
--------
本模块产物一律写入 `checkpoints/n3d_shape/`；验证类运行写入其 `_verify/` 子目录。
产物指纹**含形状维度**（圆柱还含长径比），保证"同配置不同形状产物名互不相同"。

快速入口
--------
* 冒烟测试：``python n3d_shape/train.py --smoke-test``（形状感知判据 16 条，逐条打印 PASS/FAIL）
* 指定形状冒烟：``python n3d_shape/train.py --smoke-test --shape cube``
  或 ``python n3d_shape/train.py --smoke-test --shape cylinder --cyl-aspect 0.5``
* 正式训练：``python n3d_shape/train.py --shape cube --epochs 10 --seed 42``
* 形状验证：``python n3d_shape/verify_shape.py``（S1-S13 硬断言，退出码 0 / 1）
"""

__all__ = ["config", "utils", "model", "data", "train", "verify_shape"]
