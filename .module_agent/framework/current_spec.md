搭建项目根级基础：requirements.txt、pyproject.toml、README、.gitignore、G0 断言公共工具（ID 规范断言、契约断言辅助函数），作为 H-STDN v3.2-final 工程化的地基。
## 依赖与构建配置

项目根级共用文件（framework 模块维护）。仓库现含五个模块：`framework`（仓库根级共用文件）、`n3d_proto`（一期原型）、`n3d_sphere`（二期球形有向拓扑）、`n3d_shape`（三期形状变体）、`n3d_viz`（三维可视化工具）；代码位于 `n3d_proto/`、`n3d_sphere/`、`n3d_shape/`、`n3d_viz/` 子目录，根目录只保留跨模块共用文件。

- `requirements.txt`（仓库根）：N3D 运行时依赖安装清单——`torch>=2.2.0`、`torchvision>=0.17.0`、`numpy>=1.26.0`；注释说明本文件位于仓库根目录、安装命令为 `pip install -r requirements.txt`，并指引 GPU 版本按 PyTorch 官方索引安装所需 CUDA wheel。由原 `n3d_proto/requirements.txt` 迁移而来，依赖条目原样保留。
- `README.md`（仓库根）：N3D **工程入口文档**（工程介绍 + 模块功能介绍 + 快速开始），共三节，**不含任何技术细节与实验记录**：第一节说明 N3D 的机制定位（连接由突触间距与阈值 `D` 决定）、核心思想（几何决定连接、边级参数化、不 materialize 稠密权重矩阵）、研究定位，以及**「工程现状」——已明确为「合计五个模块」（含 `framework` 本身）**，与第二节五行模块表一一对应（一期基础原型 / 二期球形有向拓扑 / 三期形状变体 / 三维可视化工具四项工作，加 `framework` 维护仓库根级共用文件）；第二节以表格 + 每模块一段正文介绍这五个模块的定位与关键能力；第三节给出 `pip install -r requirements.txt` 与各模块最简命令（`python n3d_proto/train.py --smoke-test`、`python n3d_sphere/train.py --smoke-test`、`python n3d_shape/train.py --smoke-test`、`python -m n3d_viz -c <训练产物>.pt`、`python -m n3d_viz`），并以行内方式指向各模块 README 供深入阅读（`n3d_proto/README.md` 等），不单设「文档入口」节；第三节另含一段**数据集说明**（见本节末「数据集目录不入库」）。**一期详细技术文档（四步闭环架构示意、运行方式、文件说明、一期范围、验收标准与阶段 B 结果分析等全部技术内容与实验记录）已移至 `n3d_proto/README.md`**，根 README 中零残留。由原 `n3d_proto/README.md` 迁移并改写而来。
- `.gitignore`（仓库根）：Python 常规忽略（`__pycache__/`、`*.py[cod]`、egg-info、build/dist）、虚拟环境、测试与静态检查缓存、IDE/OS 文件、运行产物目录（`/outputs/`、`/runs/`、`/checkpoints/`、`/logs/`、`/exp/runs/`、`/exp/outputs/`、`*.log`——其中 `/checkpoints/` 忽略使 checkpoint 工件不进入版本控制）、`.lizhu_env/` 本地测试环境、「Node / 前端依赖（Playwright 测试依赖，不入库）」分节下的 `node_modules/`、「数据集目录（不入库；运行时置于本地，见 README 说明）」分节下的 **`/data/`**（见本节末「数据集目录不入库」），以及「模型权重目录（不入库；运行时置于本地）」分节下的 **`/models/`**（见本节末「模型权重目录不入库」）。

已移除：`pyproject.toml`（原 H-STDN 构建与依赖权威声明，包名 hstdn）随 H-STDN 代码一并删除；`n3d_proto/requirements.txt` 已上移为仓库根文件。`n3d_proto/README.md` 为 `n3d_proto` 模块自有文档（一期详细技术文档），不再与仓库根 `README.md` 同源。
  `node_modules/` 忽略项的作用：`n3d_viz/tests/` 下由 npm 安装的 Playwright 测试依赖（`playwright` 1.63.0，183 文件、约 17.7 MB）属可再生产物，位于新增的「Node / 前端依赖（Playwright 测试依赖，不入库）」分节（`.gitignore` 第 29–32 行，规则本体在第 32 行）。该规则生效前 `git check-ignore -v n3d_viz/tests/node_modules/playwright/package.json` 退出码为 1（未命中任何规则），依赖仅因未被 `git add` 才未入库；生效后退出码为 0 且命中 `.gitignore:32:node_modules/`，`git status --porcelain -uall` 中 183 条 `n3d_viz/tests/node_modules/` 未跟踪条目归零，从而消除 `git add -A` / `git add .` 误提交依赖（约 17.7 MB）的风险。规则写作不带前导斜杠的 `node_modules/`，以匹配任意层级的依赖目录；经 `git ls-files | git check-ignore --stdin` 全量校验，被跟踪文件中被忽略者数量为 0，无路径误伤。

### 数据集目录不入库

**全仓数据集目录不再入库**，`data/` 已由 `.gitignore` 整体忽略；克隆后本地不存在该目录，训练前需自备或联网获取。

- `.gitignore` 中的规则为「数据集目录（不入库；运行时置于本地，见 README 说明）」分节（规则本体在 `.gitignore` 第 56 行）下的 **`/data/`**（根锚定，与 `/checkpoints/`、`/outputs/` 等既有分节同风格），替代原先只忽略运行时副本的 `data/mnist/MNIST/`；`data/mnist/` 下 4 个原始 IDX 文件与其 torchvision 运行时副本 `data/mnist/MNIST/raw/` 现同处忽略范围，原「原始 IDX 由版本控制跟踪」的表述已随之删除并失效。
- **锚定写法是硬要求**：无前导斜杠的 `data/` 会匹配任意层级同名目录（实测会使已跟踪的 `.module_agent/.workspaces/neuron3d/executions/data/*.json` 等落入忽略范围）；`/data/` 只作用于仓库根，功能目标等价且无误伤。
- 根 `README.md`「三、快速开始」在依赖说明段之后新增一段数据集说明：数据集不入库、需自备 `data/mnist/` 下 4 个 IDX 文件（`train-images-idx3-ubyte.gz` / `train-labels-idx1-ubyte.gz` / `t10k-images-idx3-ubyte.gz` / `t10k-labels-idx1-ubyte.gz`，合计约 11 MB），或调用各代 `data.py` 的 `ensure_mnist_files(<root>, allow_download=True)` 联网获取（默认 `allow_download=False`，此时缺失抛 `FileNotFoundError` 并列出已检查目录与候选目录）。三份 `data.py`（`n3d_proto` / `n3d_sphere` / `n3d_shape`）签名一致。
- **忽略 ≠ 取消跟踪**：忽略规则不会自动取消对已跟踪文件的跟踪，4 个 `.gz` 在规则生效后仍被 git 跟踪（`git ls-files data` 仍返回 4 条），须执行 `git rm -r --cached data`（本地文件保留）才真正取消跟踪；已提交历史中仍含这约 11 MB，取消跟踪不缩减既有历史体积。**本轮实测更正**：该 `git rm -r --cached data` 已由提交 `09a5b46`（chore(gitignore): 数据集目录 data/ 不再入库）执行完毕，故 `git ls-files data` 现返回 **0 条**（实测 `git ls-files | Select-String '^data/'` = 0 条，`git status --ignored --porcelain -- data` 输出 `!! data/`）；「已提交历史中仍含这约 11 MB、取消跟踪不缩减既有历史体积」的结论不变。
- 自测口径备忘：`git check-ignore` 对**已跟踪**文件一律不报告忽略（返回 1），需加 `--no-index` 才能验证规则命中；`git ls-files -i -c --exclude-standard` 会列出「被跟踪 ∧ 命中忽略规则」的文件 —— 本规则生效后该命令输出恰为那 4 个 `.gz`，属预期结果。另注：`git check-ignore --stdin` 在**不加** `--no-index` 时同样跳过已跟踪文件，故 `git ls-files | git check-ignore --stdin` 恒为空输出，真正的全量误伤校验须用 `git ls-files --cached | git check-ignore --stdin --no-index`。

### 模型权重目录不入库

**预训练模型权重目录不入库**，仓库根 `models/` 已由 `.gitignore` 忽略；克隆后本地不存在该目录，运行 `n3d_qa_learn` 前需自行下载或从本地缓存放置。

- `.gitignore` 中的规则为「模型权重目录（不入库；运行时置于本地）」分节（新增于文件末尾，`.gitignore` 第 58–61 行，规则本体在第 61 行）下的 **`/models/`**（根锚定，与 `/data/`、`/checkpoints/`、`/outputs/` 等既有分节同风格：`# ---- 标题 ----` + 中文注释 + 规则）。
- 用途：`n3d_qa_learn` 引入 `BAAI/bge-m3`（1024 维向量模型），权重下载到仓库根 `models/bge-m3/`（约 2.1 GB），属 GB 级、可重新下载的本地资产，不得入库。分节内两行中文注释即说明此用途。
- **锚定写法是硬要求**：无前导斜杠的 `models/` 会匹配任意层级同名目录；本轮实测反证——用仅含 `models/` 的临时 `core.excludesFile` 时 `git check-ignore -v --no-index n3d_proto/models/probe.txt` 退出码为 **0**（命中该未锚定规则），而现行 `.gitignore` 的 `/models/` 对同一路径退出码为 **1**（未命中）；`/models/` 只作用于仓库根，功能目标等价且无误伤。
- 生效验证（真实执行）：① 正向命中 `git check-ignore -v --no-index models/bge-m3/pytorch_model.bin` → 输出 `.gitignore:61:/models/`，退出码 **0**；② 锚定正确性 `git check-ignore -v --no-index n3d_proto/models/probe.txt` → 无输出、退出码 **1**；③ 全量无误伤 `git ls-files | git check-ignore --stdin` → 输出为空（0 条），其补强形式 `git ls-files --cached | git check-ignore --stdin --no-index`（强制对已跟踪文件套用忽略规则）亦输出为空（0 条，全仓共 301 个已跟踪文件）；④ 改动范围 `git status --porcelain -- .gitignore requirements.txt README.md` 仅 ` M .gitignore`，`git diff --numstat -- .gitignore` = `5	0`（纯新增、零删除、单 hunk `@@ -54,3 +54,8 @@`），HEAD 仍为 `a2f74a2`，全程未执行任何 git 写操作（未 add/commit/push），未修改 `requirements.txt` 与 `README.md`。
- 文件形态：`.gitignore` 由 56 行 / 1241 字节增至 61 行 / 1546 字节（+5 行 / +305 字节），行尾风格保持 CRLF（CRLF 61、无混行尾）、无 BOM、以 CRLF 结尾；改后 SHA256=`E45C2310EF94873F81AE67B670804E387711E0633CFC653BD138F7B129A6B35A`。

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
