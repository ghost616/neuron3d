搭建项目根级基础：requirements.txt、pyproject.toml、README、.gitignore、G0 断言公共工具（ID 规范断言、契约断言辅助函数），作为 H-STDN v3.2-final 工程化的地基。
## 依赖与构建配置

项目根级共用文件（framework 模块维护）。仓库现含五个模块：`framework`（仓库根级共用文件）、`n3d_proto`（一期原型）、`n3d_sphere`（二期球形有向拓扑）、`n3d_shape`（三期形状变体）、`n3d_viz`（三维可视化工具）；代码位于 `n3d_proto/`、`n3d_sphere/`、`n3d_shape/`、`n3d_viz/` 子目录，根目录只保留跨模块共用文件。

- `requirements.txt`（仓库根）：N3D 运行时依赖安装清单——`torch>=2.2.0`、`torchvision>=0.17.0`、`numpy>=1.26.0`；注释说明本文件位于仓库根目录、安装命令为 `pip install -r requirements.txt`，并指引 GPU 版本按 PyTorch 官方索引安装所需 CUDA wheel。由原 `n3d_proto/requirements.txt` 迁移而来，依赖条目原样保留。
- `README.md`（仓库根）：N3D **工程入口文档**（工程介绍 + 模块功能介绍 + 快速开始），共三节，**不含任何技术细节与实验记录**：第一节说明 N3D 的机制定位（连接由突触间距与阈值 `D` 决定）、核心思想（几何决定连接、边级参数化、不 materialize 稠密权重矩阵）、研究定位，以及**「工程现状」——已明确为「合计五个模块」（含 `framework` 本身）**，与第二节五行模块表一一对应（一期基础原型 / 二期球形有向拓扑 / 三期形状变体 / 三维可视化工具四项工作，加 `framework` 维护仓库根级共用文件）；第二节以表格 + 每模块一段正文介绍这五个模块的定位与关键能力；第三节给出 `pip install -r requirements.txt` 与各模块最简命令（`python n3d_proto/train.py --smoke-test`、`python n3d_sphere/train.py --smoke-test`、`python n3d_shape/train.py --smoke-test`、`python -m n3d_viz -c <训练产物>.pt`、`python -m n3d_viz`），并以行内方式指向各模块 README 供深入阅读（`n3d_proto/README.md` 等），不单设「文档入口」节。**一期详细技术文档（四步闭环架构示意、运行方式、文件说明、一期范围、验收标准与阶段 B 结果分析等全部技术内容与实验记录）已移至 `n3d_proto/README.md`**，根 README 中零残留。由原 `n3d_proto/README.md` 迁移并改写而来。
- `.gitignore`（仓库根）：Python 常规忽略（`__pycache__/`、`*.py[cod]`、egg-info、build/dist）、虚拟环境、测试与静态检查缓存、IDE/OS 文件、运行产物目录（`/outputs/`、`/runs/`、`/checkpoints/`、`/logs/`、`/exp/runs/`、`/exp/outputs/`、`*.log`——其中 `/checkpoints/` 忽略使 checkpoint 工件不进入版本控制）、`.lizhu_env/` 本地测试环境，以及「数据集原始文件（运行时复制，不入库）」分节下的 `data/mnist/MNIST/`。

  `data/mnist/MNIST/` 忽略项的作用：`n3d_proto/data.py` 与 `n3d_sphere/data.py` 优先复用工程内 `data/mnist/` 的 4 个原始 IDX 文件（已被版本控制跟踪），首次运行时会把它们复制到 torchvision 期望的布局 `data/mnist/MNIST/raw/`（4 个文件、约 11 MB）。该目录是运行时副作用产物，故整体忽略；忽略粒度收窄到 `data/mnist/MNIST/` 这一级，**不影响 `data/mnist/` 下已跟踪的原始 IDX 文件**。

已移除：`pyproject.toml`（原 H-STDN 构建与依赖权威声明，包名 hstdn）随 H-STDN 代码一并删除；`n3d_proto/requirements.txt` 已上移为仓库根文件。`n3d_proto/README.md` 为 `n3d_proto` 模块自有文档（一期详细技术文档），不再与仓库根 `README.md` 同源。
  `node_modules/` 忽略项的作用：`n3d_viz/tests/` 下由 npm 安装的 Playwright 测试依赖（`playwright` 1.63.0，183 文件、约 17.7 MB）属可再生产物，位于新增的「Node / 前端依赖（Playwright 测试依赖，不入库）」分节（`.gitignore` 第 29–32 行，规则本体在第 32 行）。该规则生效前 `git check-ignore -v n3d_viz/tests/node_modules/playwright/package.json` 退出码为 1（未命中任何规则），依赖仅因未被 `git add` 才未入库；生效后退出码为 0 且命中 `.gitignore:32:node_modules/`，`git status --porcelain -uall` 中 183 条 `n3d_viz/tests/node_modules/` 未跟踪条目归零，从而消除 `git add -A` / `git add .` 误提交依赖（约 17.7 MB）的风险。规则写作不带前导斜杠的 `node_modules/`，以匹配任意层级的依赖目录；经 `git ls-files | git check-ignore --stdin` 全量校验，被跟踪文件中被忽略者数量为 0，无路径误伤。
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
