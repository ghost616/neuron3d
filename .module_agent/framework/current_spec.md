搭建项目根级基础：requirements.txt、pyproject.toml、README、.gitignore、G0 断言公共工具（ID 规范断言、契约断言辅助函数），作为 H-STDN v3.2-final 工程化的地基。
## 依赖与构建配置

项目根级共用文件（framework 模块维护）。仓库现为 N3D 一期原型项目：代码位于 `n3d_proto/`（二期 `n3d_sphere/`）子目录，根目录只保留跨模块共用文件。

- `requirements.txt`（仓库根）：N3D 运行时依赖安装清单——`torch>=2.2.0`、`torchvision>=0.17.0`、`numpy>=1.26.0`；注释说明本文件位于仓库根目录、安装命令为 `pip install -r requirements.txt`，并指引 GPU 版本按 PyTorch 官方索引安装所需 CUDA wheel。由原 `n3d_proto/requirements.txt` 迁移而来，依赖条目原样保留。
- `README.md`（仓库根）：N3D 项目主页文档。首部「仓库布局」说明明确：**代码位于 `n3d_proto/` 子目录，本说明位于仓库根目录**；正文含项目简介与设计动机、四步闭环架构示意与归一化方向硬契约、安装与运行方式（**全部运行命令保持 `python n3d_proto/train.py ...` 形式不变**）、环境与数据说明、产物保护规则、文件说明表、一期范围、验收标准与阶段 B 实测结果分析。由原 `n3d_proto/README.md` 迁移而来，自指路径表述已按根目录位置校正。
- `.gitignore`（仓库根）：Python 常规忽略（`__pycache__/`、`*.py[cod]`、egg-info、build/dist）、虚拟环境、测试与静态检查缓存、IDE/OS 文件、运行产物目录（`/outputs/`、`/runs/`、`/checkpoints/`、`/logs/`、`/exp/runs/`、`/exp/outputs/`、`*.log`——其中 `/checkpoints/` 忽略使 checkpoint 工件不进入版本控制）、`.lizhu_env/` 本地测试环境，以及「数据集原始文件（运行时复制，不入库）」分节下的 `data/mnist/MNIST/`。

  `data/mnist/MNIST/` 忽略项的作用：`n3d_proto/data.py` 与 `n3d_sphere/data.py` 优先复用工程内 `data/mnist/` 的 4 个原始 IDX 文件（已被版本控制跟踪），首次运行时会把它们复制到 torchvision 期望的布局 `data/mnist/MNIST/raw/`（4 个文件、约 11 MB）。该目录是运行时副作用产物，故整体忽略；忽略粒度收窄到 `data/mnist/MNIST/` 这一级，**不影响 `data/mnist/` 下已跟踪的原始 IDX 文件**。

已移除：`pyproject.toml`（原 H-STDN 构建与依赖权威声明，包名 hstdn）随 H-STDN 代码一并删除；`n3d_proto/README.md`、`n3d_proto/requirements.txt` 已上移为仓库根文件，`n3d_proto/` 下不再残留这两个文件。
## G0 断言公共工具

**（已废弃）** 本节所述 G0 断言公共工具随 H-STDN 代码一并移除，当前不再存在：

- 原 `hstdn/checks.py`（`assert_global_ids` / `assert_local_ids` / `assert_shape` / `assert_dtype` / `assert_finite` / `assert_non_decreasing`）与 `hstdn/__init__.py` 包入口（`__version__=0.1.0`）均已随 `hstdn/` 整目录删除。
- 删除原因：仓库已由 H-STDN 转向 N3D 一期原型（`n3d_proto/`），两套规范互不覆盖，H-STDN 的 ID 契约（全局 ID / 池局部 ID）与 G0 十五项断言不再适用于当前项目。
- 现状：N3D 侧不依赖这些断言工具（已全仓搜索确认无任何 `hstdn` 引用残留）；`n3d_proto/utils.py` 内部提供本项目所需的形状与数值校验辅助。
- 若后续需要重建根级公共断言工具，应按 N3D 契约（张量形状、边级稀疏表示、`tau > 0`、稀疏度口径等）重新设计，而非恢复 H-STDN 的 ID 域断言。
## 仓库根级共用文件

仓库根级共用文件由本模块管理，当前为 `README.md`、`requirements.txt`、`.gitignore`。

- `README.md` 与 `requirements.txt` 原属 `n3d_proto` 模块，H-STDN 项目移除时上移至仓库根目录，所有权移交本模块。
- `pyproject.toml` 已随 H-STDN 一并删除。
- `.gitignore` 由本模块维护：已在末尾新增「数据集原始文件（运行时复制，不入库）」分节，新增规则 `data/mnist/MNIST/`。该规则忽略 `n3d_proto/data.py` 与 `n3d_sphere/data.py` 首次运行时复制出的 torchvision 布局副本（`data/mnist/MNIST/raw/`，4 个文件约 11 MB）；忽略粒度收窄到 `data/mnist/MNIST/` 这一级，**不影响 `data/mnist/` 下已被版本控制跟踪的 4 个原始 IDX 文件**。
