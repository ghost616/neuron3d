# H-STDN（混合时空脉冲神经网络）

H-STDN v3.2-final 的工程化 Python 实现：STDP 储备池 + 读出的端到端科研代码库。
以设计文档为**唯一权威规范**，从零搭建 `hstdn/` 包，按 G0–G4 分级验证推进
（G0 十五项断言 → G1 合成 10 类首跑 → G2 MNIST → G3 DVS-Gesture → G4 规模化）。

## 唯一权威规范

《H-STDN 详细设计文档 v3.2-final》：含 D1–D14 设计裁决、B1–B11 勘误、§2 数据布局、
§3 模块设计 M1–M6、§4 超参、§9 实施计划。**仅此文档可作为实现依据**；
四份早期源文档（kimi/ds/智谱/千问）已降级归档，禁止引用。

## 代码布局

```
hstdn/            包根：公共入口 + 轻量 G0 契约断言工具（framework 模块维护）
├── __init__.py   包元信息与公共断言 API（assert_global_ids 等）
├── checks.py     形状 / dtype / ID 域 / 有限性 / 单调性通用校验辅助
├── core/         M1–M6 核心模拟实现（layout/network/spatial_hash/encoder/kernel/
│                 plasticity/features/readout/scheduler）——core 模块维护（待建）
├── configs/      §4 超参唯一契约 default.yaml ——exp 模块维护（待建）
├── exp/          gates.py（G0 十五项断言门槛）、diagnostics.py ——exp 模块维护（待建）
└── data/ main/   数据与训练/评估入口 ——data/main 模块维护（待建）

requirements.txt  免构建依赖安装清单（Python 3.12）
pyproject.toml    构建与依赖权威声明（包名 hstdn）
```

## 环境要求

- Python 3.12（本地基准 3.12.10 / Windows）
- 运行依赖：`numpy>=1.24`、`scipy>=1.10`、`numba>=0.59`（Python 3.12 兼容最低版本）、
  `pyyaml`、`matplotlib`
- **延迟引入**（勿提前安装）：`torch`（G2：MNIST 读出/批量桥接）、`tonic`（G3：
  DVS-Gesture 数据）、`snntorch`（可选：读出对比基线）

## 快速开始

```powershell
# 1) 建虚拟环境并激活（Windows）
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2) 安装依赖与包
python -m pip install --upgrade pip
python -m pip install -e .            # 推荐：依赖以 pyproject.toml 为权威来源
# 免构建替代：python -m pip install -r requirements.txt

# 3) 冒烟验证
python -c "import hstdn; print(hstdn.__version__)"   # 期望输出 0.1.0
```

## G0 门禁运行方式

G0 十五项断言（`hstdn/exp/gates.py`，exp 模块交付）为项目门槛，**全绿方可进入下一
阶段**；每项独立可执行。安装 hstdn 后从项目根执行：

```powershell
python -m hstdn.exp.gates
```

单项运行与诊断参数由 exp 模块的 gates 实现定义（届时见 `hstdn/exp/gates.py` 文档）。
修改既有实现前必须先跑相关 G0 断言。

## 公共 G0 断言工具

由 framework 模块维护、供 `hstdn/core/layout` 与 `hstdn/exp/gates` 等复用的轻量校验：

```python
from hstdn import (
    assert_global_ids,      # 存储层（CSR/CSC/trace）全局 ID 域校验：[0, n_total)
    assert_local_ids,       # 状态数组层池局部 ID 域校验：[0, n_local)
    assert_shape,           # 形状精确匹配校验
    assert_dtype,           # dtype 校验
    assert_finite,          # 有限性校验（NaN/Inf）
    assert_non_decreasing,  # 一维单调不减校验（CSR indptr / 排序 ID 列表等）
)
```

约定：校验失败一律抛出带**可读消息与统计数值**的 `AssertionError`；公共工具仅做通用
契约校验，业务契约断言的实现细节由 core 模块负责。

## 工程纪律（摘要）

- 三件套纪律：每次修改 = 代码 patch + change_history 记录 + G0 断言覆盖。
- ID 契约：存储层全局 ID / 状态数组层池局部 ID；全局→局部转换仅发生在内核投递处
  （`dst_local = dst - N_IN`）。
- G1 前冻结全部超参，`hstdn/configs/default.yaml` 为唯一来源，禁止散落魔法数字。
- L1 Numba 分支禁止 pass 占位——未实现功能须显式 `raise NotImplementedError`。