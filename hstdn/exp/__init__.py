"""hstdn.exp -- H-STDN 实验验证模块（G0 门禁与诊断面板）。

交付物（唯一权威见各文件 docstring）：
    gates.py         G0 十五项断言门禁脚本（python -m hstdn.exp.gates）
    diagnostics.py   诊断面板函数（率/沉默/类间 cos/特征可比性/顶界占比/
                     输入->池 w 均值/延迟直方图），供 G1 与 gates 调试复用

G0 门禁约定：每项独立可执行、独立通过才允许进入 G1；全绿 = 十五项 PASS。
本包 __init__ 保持导入轻量（不拉入 numpy/core），gates/diagnostics 显式导入。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
