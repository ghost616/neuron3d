搭建项目根级基础：requirements.txt、pyproject.toml、README、.gitignore、G0 断言公共工具（ID 规范断言、契约断言辅助函数），作为 H-STDN v3.2-final 工程化的地基。
## 依赖与构建配置

项目根级工程配置（framework 模块维护，Python 3.12）：
- requirements.txt：运行时依赖免构建安装清单（numpy>=1.24、scipy>=1.10、numba>=0.59、pyyaml、matplotlib；torch/tonic/snntorch 注释为 G2/G3 按需延迟引入）。
- pyproject.toml：构建与依赖权威声明。包名 hstdn，requires-python>=3.12，setuptools 包发现限定 hstdn*（随 core/exp/data/main 子包建立自动纳入），动态版本读取 hstdn.__version__。
- README.md：项目简介、唯一权威规范（H-STDN 详细设计文档 v3.2-final）引用、代码布局、环境要求、快速开始、G0 门禁运行方式、工程纪律摘要。
- .gitignore：Python 常规忽略 + H-STDN 实验产物目录（outputs/runs/checkpoints/logs）。
## G0 断言公共工具

framework 维护的轻量 G0 契约断言辅助（公共工具），供 hstdn/core/layout 与 hstdn/exp/gates 复用：
- hstdn/checks.py：assert_global_ids（存储层全局 ID 域 [0, n_total) 校验）、assert_local_ids（状态数组层池局部 ID 域 [0, n_local) 校验）、assert_shape / assert_dtype（形状与 dtype 精确校验）、assert_finite（NaN/Inf 有限性校验）、assert_non_decreasing（CSR indptr / 排序 ID 单调不减校验）。
- hstdn/__init__.py：包入口，__version__=0.1.0（pyproject 动态版本来源），重导出 checks 全部公共函数。
- 约定：失败一律抛含可读消息与统计数值的 AssertionError；只做通用校验，业务契约断言实现细节由 core 模块负责。
