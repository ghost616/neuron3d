"""N3D 二期三维拓扑可视化模块。

把一个已训练产物的 ``model_state_dict`` 中的几何与拓扑信息，渲染成
「自包含交互式 HTML + 点云 PLY + 线框 OBJ」三件套。

设计约束（硬约束，违反即失败）：

* **零新依赖**：仅使用 ``torch`` / ``numpy`` 与 Python 标准库
  （``tkinter`` 属标准库）。不引入 matplotlib / plotly / pyvista / tkinterdnd2
  等任何第三方绘图或 GUI 库。
* **自包含**：只读取 checkpoint 文件的 ``state_dict``，**不 import**
  ``n3d_sphere`` / ``n3d_proto`` 的任何代码；被可视化模块的源码与产物零改动。
* **单实现**：GUI 与 CLI 共享 ``core`` 层的同一套逻辑，本模块内不存在第二份绘图实现。

产物命名由 checkpoint 文件名派生（``viz_<ckpt名>.html/.ply/.obj``），默认写入
``checkpoints/n3d_viz/``；同名产物已存在时明确提示，不静默覆盖。
"""

from __future__ import annotations

from . import core, export_geometry, render_html

__all__ = ["core", "export_geometry", "render_html", "gui"]

__version__ = "1.0.0"
