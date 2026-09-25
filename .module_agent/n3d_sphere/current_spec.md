# n3d_sphere 功能说明（纯球形分层有向无环架构）

## 项目定位

本模块是 N3D 的**纯球形分层有向无环架构（DAG）**实现：神经元按 FCC（面心立方）规则堆积放置在球空间内，突触按全局流向轴切分到正/负半球，连接规则强制沿流向轴严格上行，因此图天然无环。

- **纯球体几何**：网络上不存在其它几何分支，也不存在几何类型开关（既无几何类型字段，也无边长字段）。
- **自包含**：不 import `n3d_proto` 的任何模块；一期完整存档、默认行为与既有产物逐位不变（`python n3d_proto/train.py --smoke-test` 仍 9/9 PASS、退出码 0、`loss=2.419689`；`git status --porcelain -- n3d_proto` 为空；一期三件产物 SHA256 未变）。
- **与一期产物的物理隔离**：本模块产物一律写入 `checkpoints/n3d_sphere/`，验证类运行写入其 `_verify/` 子目录，绝不触碰一期 `checkpoints/` 根下的 `n3d_model_*.pt`。
- **可执行不变量**：全部架构不变量（几何、FCC、DAG、去重、逐边数值、感受野、判据语义、产物指纹、**设备契约**、**文档登记数字**）都固化为脚本判据，由 `verify_all.py` 一键执行并汇总；文档中出现的每一个实测数字都登记在 `_verify/doc_numbers.json` 并由 `verify_all.py` 现跑比对。
- 上游规格偏离声明：本模块的几何定义（球体分布 + 半球切分 + 全局流向 + 严格上行 DAG）偏离原提示语 §2.1/§2.2/§2.3 的"核心设计，必须严格遵循"要求，故独立成模块。
## 背景结论（一期已证否的维度）

1. 容量维度无效（**以下为一期历史实验记录，其迭代轮数维度在本模块当前架构中已不存在**）：N / y_in,y_out / D / 迭代轮数 全部无效或负收益（D 0.15→0.25 掉 1.94pp，迭代轮数 4→6 掉 0.68pp）。
2. 拓扑种子无效：seed 42/7/2024/123 的 12-epoch 结果为 97.84% / 97.55% / 97.90% / 97.68%，极差 0.35pp。
3. 瓶颈在架构：同等控制变量下普通 MLP（1,628,170 参数）达 98.64%，主模型（1,691,730 参数）仅 97.84%，差 +0.80pp。
4. 四步闭环的信息损耗点：2a 分组 softmax 使输出为邻居凸组合（动态范围压缩）；2d LayerNorm 抹掉激活幅度；2c ReLU 负侧梯度恒零；readout 只取末轮 s_out。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| `n3d_sphere/__init__.py` | 包声明与模块定位：纯球形分层 DAG、自包含、一期不变、快速入口 |
| `n3d_sphere/utils.py` | 通用工具层（可复现性 `set_seed`、设备 `get_device`、日志、参数/梯度统计），自包含实现；四步闭环算子为历史遗留、当前架构不调用 |
| `n3d_sphere/data.py` | MNIST 数据层（IDX 惰性解析 + 归一化 + DataLoader，num_workers>0 时按 (seed, worker_id) 播种） |
| `n3d_sphere/config.py` | 超参配置层：球半径窗口公式、FCC 晶格常数派生量、判据开关、三预设（SMALL/DEFAULT/HIGHACC，**均为 `D = H`**）；`__post_init__` 实施 **`D <= H` 硬校验（G1）**；模块 docstring 的参考值与"关键不变量"清单已与合法口径同步（H1） |
| `n3d_sphere/model.py` | 球形分层 DAG 模型：FCC 规则堆积、半球突触采样、神经元级连接（同对去重取最近突触对）、Kahn 拓扑序与 CSR 分组、**整层向量化的两阶段双副本前向**、**设备契约守卫 `_assert_index_device`**、**连通性下限校验 `check_connectivity_floor`（G3）**、统计与自检接口；另含 `MLPBaseline` 对照基线 |
| `n3d_sphere/train.py` | 训练入口：CLI（flow-axis/space-radius/**d**/input-scope/readout-scope/placement，`--d` 的 help 注明 `D <= H`）、产物指纹与隔离、15 条冒烟判据、checkpoint 元数据；覆盖已有产物前按 `--backup`（默认开）生成 `<产物名>.pt.bak` —— 这是"同名覆盖仍可追溯"的关键机制（K1 修复轮 error①） |
| `n3d_sphere/README.md` | 模块文档：几何与尺度、**`D <= H` 硬约束与连通性下限（含改造前后对照表）**、连接规则、两阶段双副本前向、判据开关、CLI、冒烟判据、实测拓扑统计（标注 seed）、**产物纪律（含 `_verify/legacy/` 历史口径存档的用途与判定口径）**、字节确定性与 SHA256 语义、**第 7.2 节二期首次正式全量训练（5 轮对照表 / 达标结论 / 正式产物来源 / 筛参轮 / 迭代顺序与未探索维度）**、**第 10.6 节 K1 修复轮（皋陶 2 error + 4 warning + 3 info 的逐条修复与两类教训）**、验证脚本、已知边界与各轮修复记录 |
| `checkpoints/n3d_sphere/_verify/verify_sphere_dag.py` | 几何 / FCC / DAG / 去重 / 双副本 / 逐边数值 / 感受野 / 判据 / 产物 验证脚本（R1-R7b，4 组配置）；每次运行把 111 条全精度指标落盘 `sphere_dag_metrics.json` |
| `checkpoints/n3d_sphere/_verify/verify_dh_constraint.py` | **H1-H4**：`D <= H` 硬校验（含负例消息取证）、三预设 `D = H` 一致性、连通性下限校验（含 D 过小负例）、退化实测 |
| `checkpoints/n3d_sphere/_verify/verify_device_regression.py` | **设备契约回归取证（D1-D7）**：类型/注册状态、`vars(model)` 通用扫描、层切分等价性、`meta` 搬运实验与对照、两条守卫负例、CUDA 实测或静态取证、`state_dict` 往返逐位一致 |
| `checkpoints/n3d_sphere/_verify/verify_scope_and_fingerprint.py` | 判据开关语义与产物指纹/冒烟命名验证脚本（S1-S5） |
| `checkpoints/n3d_sphere/_verify/verify_config_contracts.py` | 半径公式 / 窗口校验 / 字段清理 / 零残留验证脚本（C1-C6） |
| `checkpoints/n3d_sphere/_verify/verify_topology_snapshot.py` | 固定实验点（6 个，均 `D = H`）× 10 seed 拓扑量快照脚本（落盘 `topology_snapshot.json`） |
| `checkpoints/n3d_sphere/_verify/run_smoke_matrix.py` | 以当前代码重跑 11 种冒烟组合：解析实际落盘路径并与独立预测的产物名比对、回读 config 与期望逐字段比对、断言同路径多次写入逐位一致；刷新 `smoke_matrix_f9.json` / `log_smoke_matrix_f9.txt` 并重建 `smoke_scope_matrix.json` |
| `checkpoints/n3d_sphere/_verify/run_round.py` | **正式全量训练取证 runner**（K1）：Python 侧写盘把子进程 stdout+stderr **流式**写成 UTF-8 日志（规避 Windows PowerShell 重定向写 UTF-16LE 使日志不可解析），末尾追加真实 `EXITCODE=<n>` |
| `checkpoints/n3d_sphere/_verify/verify_full_runs.py` | **F-L1~F-R1 正式全量训练取证判据**（K1；K1 修复轮加固）：F-L1 台账结构自检；F-L2/F-L3 逐轮三方一致（台账 vs 真实终端日志 vs **该轮冻结快照**，含逐 epoch 轨迹、参数量按重建模型 `count_parameters()` 口径、**快照路径存在性 + SHA256 + 字节数强制比对**、退出码、达标判定）；F-L4 正式产物 `model.pt` 与台账来源轮一致；F-R1 逐位可复现重放（限批写 `_verify/`）。日志按 BOM 自动识别 UTF-8/UTF-16LE。`ledger` 秒级（已纳入 `verify_all`）/ `bounded` 分钟级 |
| `checkpoints/n3d_sphere/_verify/build_full_runs_ledger.py` | 轮次台账构建脚本（K1，**不是判据**）：按每轮显式声明的 `snapshot_source`（`artifact` / `backup`）冻结产物快照到 `artifacts/<轮次>.pt`，冻结时强制校验『来源→快照 SHA256 一致』与『快照 config/test_acc 与本轮终端日志一致』（不一致即中止），执行限批重放取 CE 和，写出 `full_runs.json` |
| `checkpoints/n3d_sphere/_verify/register_full_runs_doc_numbers.py` | 把各轮登记项写入 `doc_numbers.json` 的 `full_run_checks` 类（K1，**不是判据**；产物字段以**冻结快照**为读回对象），并写入两次限批筛参的 `text_checks`；全部现场取数、可重复运行 |
| `checkpoints/n3d_sphere/_verify/verify_snapshots_k1fix.py` | **K1 修复轮的 5 轮快照 × 配置复核脚本**（只读）：打印快照路径/字节数/SHA256/`snapshot_source`/现算 `artifact_available`，并用 `torch.load` 复核 `(epochs, batch_size, lr, readout_bias, lr_schedule, grad_clip, test_acc, E)` 与轮次期望逐项一致；输出落 `log_snapshots_k1fix.txt` |
| `checkpoints/n3d_sphere/_verify/full_runs.json` | **正式全量训练轮次台账**（K1）：5 轮的配置、命令行、日志路径、`exit_code`（含 `exit_code_note` 取证方式与时点）、达标判定、`artifact_available` / `artifact_retention` / `artifact_backup` / `artifact_backup_sha256` / `snapshot_source` / `artifact_snapshot_sha256` / `artifact_snapshot_bytes`、逐 epoch 轨迹、限批重放登记值，以及正式产物 `model.pt` 的来源轮次与内嵌 config |
| `checkpoints/n3d_sphere/_verify/artifacts/` | **轮次产物冻结快照（5 轮各一份，K1）**：`A_baseline_default.pt`（`82C92A4E…`）/ `B_preset_highacc.pt`（`DE1A13E5…`，来源=`.bak`）/ `C_lr2e3_bs64.pt`（`D460F8F2…`）/ `D_capacity_N384.pt`（`63AEC14D…`，来源=`.bak`）/ `E_N384_epochs40.pt`（`995CECD5…`）；同名覆盖时被覆盖轮的产物由 `train.py` 存为 `<产物名>.pt.bak`，快照即从此冻结 |
| `checkpoints/n3d_sphere/_verify/log_full_roundA_baseline.txt` / `log_full_roundB_highacc.txt` / `log_full_roundC_lr2e3_bs64.txt` / `log_full_roundD_N384.txt` / `log_full_roundE_N384_epochs40.txt` | 5 轮正式全量训练的真实终端输出（A/B/C 为 UTF-16LE 且无 `EXITCODE`；D/E 为 UTF-8 且末尾含 `EXITCODE=0`） |
| `checkpoints/n3d_sphere/_verify/screen_y16_b400.txt` / `screen_N384_b400.txt` | 两次**限批筛参**（`--max-batches 400`，**非正式对照**）的真实终端输出与 `EXITCODE=0`：`y=16×16`（末轮 loss 0.0584、E=805、params=135333）被否、`N=384`（0.0457、E=1145、params=233513）入选 |
| `checkpoints/n3d_sphere/_verify/exit_code_roundA.txt` / `log_exitcode_roundB.txt` / `log_exitcode_roundC.txt` | A/B/C 三轮的退出码取证（A 为独立登记文件，其 `$LASTEXITCODE` 时点歧义已在 README §7.2 注明；B/C 为同配置同 seed 的限批重放日志） |
| `checkpoints/n3d_sphere/_verify/log_snapshots_k1fix.txt` / `log_verify_all_k1.txt` / `log_proto_smoke_k1.txt` | K1 轮的取证输出：5 轮快照复核、`verify_all.py` 完整验收（11 条命令 + 文档数字防线 252 项）、一期回归原始输出（9/9 PASS、`loss=2.419689`） |
| `checkpoints/n3d_sphere/_verify/verify_all.py` | 一键验证入口：依次执行 **11 条**验收命令、汇总退出码，并执行**文档数字防线**（`doc_numbers.json` 现跑比对；**五类**数据源：产物字段 / **全量轮次（优先读冻结快照，纯日志字段不依赖产物存在性）** / R1-R7b 指标 / **跨 seed 快照聚合（含按 seed 取单点）** / 命令输出文本；输出含分类明细行） |
| `checkpoints/n3d_sphere/_verify/doc_numbers.json` | **文档数字登记表**（**252 项** = 49 产物字段 + 111 全精度指标 + **14 跨 seed 快照聚合** + **44 文本计数** + **34 全量轮次登记项**），由 `verify_all.py` 现跑比对，不一致即判失败 |
| `checkpoints/n3d_sphere/_verify/legacy_archive_manifest.json` | **J1 归档取证清单**：11 个改造前 `D > H` 产物的逐个 `size` / `sha256_before` / `sha256_after` / `bytes_identical` 与配置字段，两侧总数与总字节守恒断言，以及 `smoke.pt` 未被扰动的记录 |
| `checkpoints/n3d_sphere/_verify/legacy/` | **历史口径存档目录**（不属于当前验收）：改造前 `D > H` 的 11 个冒烟产物（共 10,017,363 字节），仅供历史对照；不被任何验收命令读取或重写 |
| `checkpoints/n3d_sphere/_verify/legacy_dh_baseline.json` | **`D = H` 改造前的历史基线**（`D > H`，现已不可由 `Config` 构造）：改造前两测点的 E / 层数 / `S_in` / `S_out` / params 及其来源产物与快照的 SHA256；J1 后其 records 中的 artifact 路径已指向 `legacy/` |
| `checkpoints/n3d_sphere/_verify/sphere_dag_metrics.json` | `verify_sphere_dag.py` 落盘的全精度实测指标（R1-R7b，111 条） |
| `checkpoints/n3d_sphere/_verify/topology_snapshot.json` | 6 测点 × 10 seed 的拓扑量快照；doc_numbers 的 `snapshot_checks` 现场从此文件聚合跨 seed 区间与按 seed 单点 |
| `checkpoints/n3d_sphere/_verify/smoke_matrix_f9.json` / `log_smoke_matrix_f9.txt` | 11 种冒烟组合的取证记录（产物名 / 退出码 / PASS / FAIL / loss / 梯度范数 / config 核对 / SHA 一致性字段）与完整日志 |
| `checkpoints/n3d_sphere/_verify/log_verify_all_f21.txt` / `log_verify_all_g5.txt` / `log_verify_all_h3.txt` / `log_verify_all_j3.txt` / `log_verify_all_k1.txt` | 五轮（F21 / G5 / H3 / J3 / K1）`verify_all.py` 的完整验收输出：命令退出码与判据计数、文档数字防线（分类明细 + 总计）与总结句 |
| `checkpoints/n3d_sphere/_verify/topology_snapshot.json` / `smoke_scope_matrix.json` | 拓扑量快照与四种 scope 组合的冒烟汇总记录 |
## 验收标准

### 实测验收结果（唯一记录处；全部命令退出码 0）

所有实测数字均**登记在 `checkpoints/n3d_sphere/_verify/doc_numbers.json`**（**252 项** =
49 产物字段 + 111 全精度指标 + **14 跨 seed 快照聚合** + **44 文本计数** + **34 全量轮次登记项**）
并由 `verify_all.py` 在**同一轮现跑**中逐项比对；下表中的数字若与登记表或现跑不一致即判失败。
本轮（`D <= H` 约束 + 预设 `D = H` + 连通性下限校验 + J1 归档清理 + J2 跨 seed 区间登记 +
**K1 二期首次正式全量训练取证（5 轮，全部未达标）**）已把全部受影响数字整体刷新。

> **历史口径归档**：改造前 `D > H` 口径的 11 个冒烟产物已由 J1 移到
> `checkpoints/n3d_sphere/_verify/legacy/`（逐文件 SHA256 前后一致、计数与字节守恒，
> 证据 `legacy_archive_manifest.json`），**不参与当前验收**；根目录只保留 9 个当前口径产物。

| 判据 | 实测证据 |
| --- | --- |
| E1 编译 | `python -m compileall -q n3d_sphere` 退出码 0 |
| E2 冒烟 | `python n3d_sphere/train.py --smoke-test` 退出码 0、**15/15 PASS**；四种 scope 组合均 15/15 PASS 且 **loss 互不相同**（`any/any` = 2.1496479511260986；`any/all` = 2.236379623413086；`all/any` = 2.2899417877197266；`all/all` = 2.337554454803467），各有独立产物可 `torch.load` 复核（汇总记录 `_verify/smoke_scope_matrix.json`，由 `run_smoke_matrix.py` 重建）；默认组合产物 `_verify/smoke.pt` 的 loss = 2.1496479511260986，梯度范数（全精度）W_in 1.5796021223068237 / edge_weight 0.2537662386894226 / neuron_bias 0.18929001688957214 / W_out 0.49458175897598267；E=106、S_in=55、S_out=53、层数 7、可学习参数 43930 |
| E2' 多配置冒烟 | 以当前代码重跑 **11 种组合**（四种 scope × `--seed 7/0/2024` × `--n 32` × `--preset default` × `--flow-axis x` × `--arch mlp`）**全部退出码 0、FAIL=0**，产物名与脚本独立预测的名字**逐条一致**、config 与期望逐字段自洽、`smoke.pt` 未被污染，且**同一产物路径被多条组合写入时逐位一致**（`smoke.pt` 被 3 条组合写入、SHA 去重后 1 种 = `4E11F12F…6A71`）。如实标注：`--seed 0` 按 CLI 约定表示"不覆盖"、`--preset default` 在冒烟路径下的基线即 `SMALL_CONFIG`。取证：`_verify/log_smoke_matrix_f9.txt` 与 `_verify/smoke_matrix_f9.json` |
| R1-R7b 几何/FCC/DAG/去重/双副本/逐边数值/感受野/判据/产物 | `verify_sphere_dag.py all` 退出码 0（10 项判据标签全 PASS），**在 4 组配置上执行**（SMALL / DEFAULT / DEFAULT+`flow_axis=x` / DEFAULT+`space_radius=0.9`，均为 `D = H`），111 条全精度指标落盘 `_verify/sphere_dag_metrics.json`。**R5**：max\|Δa_up\| = 1.047723 / 2.259184 / 1.702276 / 2.259184，受影响下游非 `S_in` 神经元 = 9 / 59 / 61 / 59，`d(sum a_up)/d(a_in)` 非零 = 132/192、653/768、643/768、653/768；**R5b**：最大偏差 = 5.960e-08 / 2.384e-07 / 5.960e-08 / 2.384e-07（阈值 1e-5），M1 形态反例错配 = 106/106、727/736、716/730、727/736 且结果差异 = 8.437e-01 / 1.655e+00 / 1.703e+00 / 1.655e+00；**R5c**：最深层祖先覆盖 7/7、9/9、10/10、9/9 层，第一层扰动传到最深层最大变化 = 7.924e-05 / 2.500e-04 / 2.625e-04 / 2.500e-04；**R7b**：`readout_scope` 两取值 \|S_out\| any=187 / all=14，同一输入下 logits 最大差异 = 0.858180 |
| **H1-H4 `D <= H` 与连通性下限（G1/G3）** | `verify_dh_constraint.py` 退出码 0：H1 负例 `H=0.10/D=0.15`、`H=0.15/D=0.25`、`H=0.10/D=0.1000001` 全部抛 `ValueError`（消息含 H/D/D/H 与约束说明），边界 `D == H`、`D < H` 正常构造；H2 三预设均 `D = H` 且 `describe()` / `to_dict()` 往返一致；H3 三预设通过下限并报出指标（SMALL E=106/K=7/S_in=55/S_out=53/params=43930；DEFAULT E=736/K=9/S_in=193/S_out=187/params=154864；HIGHACC 同 DEFAULT 但 params=154874，因启用 `readout_bias`）；H4 退化实测 D=0.05/0.03/0.02 → E=146/17/1（`E/N` = 0.5703/0.0664/0.0039）全部被下限校验拦下 |
| **D1-D7 设备契约（F16）** | `verify_device_regression.py` 退出码 0（D1-D5 与 D7 共 6 项 PASS，D6 无 GPU 时 SKIP）：D1 `level_edge_reach` / `level_node_reach` 均为已注册 int64 `[K,2]` 张量（SMALL `(7,2)`、`flow_axis=x` `(10,2)`）；D2 `vars(model)` 通用扫描无未注册张量与含张量容器；D3 层切分与独立分层逐位一致、层边区间无缝覆盖 `[0,E)`；D4 `.to('meta')` 后索引张量全部搬到 meta，而对照的普通 Python list 中张量仍停留在 cpu；D5 两条守卫负例均抛 RuntimeError；D6 本机无 GPU → 标注"CUDA 路径为静态取证"并 skip；D7 `state_dict` 往返后前向逐位一致 |
| **文档数字防线（F20/J2/K1）** | `verify_all.py` 现跑比对 `doc_numbers.json`：**一致 252 项 / 不一致 0 项 / 跳过 0 项**；分类明细 artifact 49 / metric 111 / **snapshot 14** / text 44 / **full_run 34**（各类均"不一致 0、跳过 0"）。snapshot 类登记 DEFAULT 测点跨 10 seed 的 `params 146237~164263`、`E 727~750`、`S_in 182~205`、`S_out 185~203`、双副本 167~192、层数 9~9、seed 数 10，以及 SMALL 测点的 seed=42 单点值与其跨 10 seed 区间（`E 96~115`、`params 39235~44710`） |
| **K1 二期首次正式全量训练（5 轮，共享 seed=42 单点运行，全部未达标；K1 修复轮已更正）** | 逐轮命令与结论：**A** `python n3d_sphere/train.py --seed 42`（DEFAULT，退出码 0）→ `test_acc=0.9759`、loss=0.0195、E=736、params=154864、层数 9、137.5s；**B** `--preset highacc --seed 42`（退出码 0）→ **0.9823**、loss=0.0005、params=154874、258.1s；**C** `--preset highacc --seed 42 --epochs 20 --batch-size 64 --lr 2e-3`（退出码 0）→ 0.9816、loss=0.0001、params=154874、495.7s；**D** `--preset highacc --seed 42 --n 384`（退出码 0）→ **0.9828（5 轮最好）**、loss=0.0002、E=1145、params=233523、层数 11、371.7s；**E** `--preset highacc --seed 42 --n 384 --epochs 40`（退出码 0）→ 0.9826、loss=0.0000、E=1145、params=233523、710.1s。达标线 **0.9864**（MLP 对照基线），5 轮**全部未达标**，最好轮 D 距线 **0.36pp**；5 轮均无 NaN/Inf、无崩溃、无下限报错、无 OOM；上限 5 轮用尽后停止并如实汇总。**迭代顺序与未探索维度**：实际为 A 基线 → B 优化 → C 优化 → D 容量 → E 容量+优化（**先做优化组**；该偏离**没有同期依据** —— 轮 B 日志 `01:22:08` 早于 `01:27` 的筛参；`01:27` 两次 `--max-batches 400` 筛参解释的是此后容量维度选 `N=384` 与剩余额度分配，见 README §7.2.3），**几何组（H/D/flow_axis）与判据组（两 scope 四种组合）从未探索**，归因结论不得外推。**产物**：5 轮各有一份可 `torch.load` 复核的冻结快照 `_verify/artifacts/<轮次>.pt`（A `82C92A4E…` / B `DE1A13E5…` / C `D460F8F2…` / D `63AEC14D…` / E `995CECD5…`），来源由每轮 `snapshot_source` 声明（B/D 为 `backup` —— 其产物在同名覆盖时被 `train.py` 的 `--backup` 存为 `<产物名>.pt.bak`，故**未丢失**；`artifact_available=false` 只表示该文件名当前已被同指纹后轮占用）。**正式产物** `checkpoints/n3d_sphere/model.pt` 来源 = **轮 A（DEFAULT_CONFIG 原样）**，内嵌 `config` 为 `N=256/y=8x8/H=D=0.10/bs=64/lr=1e-3/epochs=10/seed=42/Adam`、`test_acc=0.9759`、`epochs=10`、`E=736`、`params=154864`；因**无达标轮**故"达标轮落 model.pt"未触发，`model.pt` **不等于**最好配置（D 的 0.9828）；**轮 A 覆盖了一份既有的同配置产物并生成 `model.pt.bak`**（与轮 A 产物逐位相同 `82C92A4E…`），此后未再被覆盖，**该既有产物的来源不可追溯**（CreationTime `01:13:25` 早于轮 A 的 `torch.save`，但无任何全量训练日志与之对应，全盘亦无第二份 `model.pt*`）。**筛参轮（限批，非正式对照）**：`01:27` 两次 `--max-batches 400`（两侧同为 `preset=default` + `seed=42` + `bs=64`，彼此同预算可比）—— `y=16×16` 的 **epoch 1 平均训练 loss×100 = 41.17**（loss 0.4117）/ **前 100 batch CE 和 = 78.92**（batch100 running_loss 0.7892）、末轮 loss 0.0584、`E=805`、`params=135333` → 被否；`N=384` 的 **36.85** / **68.01**、末轮 0.0457、`E=1145`、`params=233513` → 入选。正式 5 轮同口径参照（**口径一 / 口径二**）：A 27.60 / 75.05、B 26.84 / 52.89、C 23.89 / 59.68、D 24.88 / 48.84、E 24.88 / 48.84；**筛参与 D/E 非同一配置**（筛参 `preset=default`（Adam、lr=1e-3、bs=64、bias 关、无调度/裁剪）vs D/E `preset=highacc`（AdamW+wd=1e-4、cosine、lr=2e-3、bs=128、clip 1.0、bias 开）），**无逐位可比性**。**退出码口径**：仅 D/E 的日志末尾自带 `EXITCODE=0`；A 由 `exit_code_roundA.txt`（mtime `01:21:00`，晚于限批重放的 `01:20:42`，时点歧义如实标注）、B/C 由同配置同 seed 的限批重放日志取证，三份文件均**未改写原始日志**。**复核**：`verify_full_runs.py ledger` 退出码 0（F-L1 结构自检、F-L2/F-L3 逐轮三方一致含**快照路径存在性 + SHA256 + 字节数强制比对**、F-L4 正式产物与来源轮一致）、`verify_full_runs.py bounded` 退出码 0（F-R1 逐位可复现重放：前 200 batch CE 和 A 108.4 / B 78.4 / C 90.44 / D 72.78 / E 72.78）、`verify_snapshots_k1fix.py` 退出码 0（5 轮快照 × `(epochs,bs,lr,bias,sched,clip,acc,E)` 逐项一致）；F-L2 的参数量断言经修复后 **5 轮全部实际执行**（快照重建 `count_parameters()` == 日志 params：154864/154874/154874/233523/233523）。取证：`_verify/full_runs.json`、`_verify/log_full_round*.txt`、`_verify/artifacts/`、`_verify/screen_*.txt`。 |
| S1-S5 判据语义与产物指纹 | `verify_scope_and_fingerprint.py` 退出码 0；`all/any` 下 `d(sum a_up)/d(a_in)` 非零 479/512、`any/any` 下 439/512；`input_scope=all_isolated` 的两种组合双副本神经元数均为 0；指纹变体已全部改用合法 `D`（`D=0.05`、`H=0.12/D=0.12`） |
| C1-C6 半径公式/窗口/字段/零残留 | `verify_config_contracts.py` 退出码 0（C1/C3 用例已改为 `D <= H` 的合法组合；残留扫描覆盖 n3d_sphere 全部源文件 + README + 本功能说明 + 文件定义，真实命中 0 行、上下文豁免 5 行） |
| 拓扑快照 | `verify_topology_snapshot.py` 退出码 0（60 条记录落盘 `topology_snapshot.json`，6 个测点均为 `D = H`，自检 `：PASS`）；DEFAULT 测点跨 10 seed：E 727~750、params 146237~164263、`S_in` 182~205、`S_out` 185~203、双副本 167~192、层数恒 9；SMALL 测点跨 10 seed：E 96~115、params 39235~44710（seed=42 为 E=106 / params=43930） |
| **E5 一期未被触碰** | `python n3d_proto/train.py --smoke-test` **9/9 PASS、退出码 0、loss=2.419689**；`git status --porcelain -- n3d_proto` 输出为空；一期三件产物 SHA256 与本轮开工前完全一致 |

### 阶段 A 冒烟判据（15 条，新架构口径）

旧架构的 `tau > 0`、连接稀疏度（密度 `E/(N*y_out*N*y_in)`）、边级参数数 == E 等判据已随架构失效。现判据集合为 **15 条**（日志逐条打印 PASS/FAIL，文案全 ASCII 以规避 GBK 控制台编码缺陷）；`--arch mlp` 对照基线只有其中 6 条适用：

1. 前向输出形状 == `[B, output_dim]`（**真实断言**：比较实际 logits 形状，非恒真）
2. 反向无错误（全部可学习参数都有梯度；缺失梯度由 `tensor_grad_norms` 记为 -1.0 并逐个核验）
3. 参与 loss 的参数梯度范数 > 0（按 arch 解析输出层参数名：neuron3d 为 `W_out`、MLP 为 `fc2.*`；neuron3d 另断言 `W_out` 非零梯度列数落在 `[1, |S_out|]` —— **区间断言，不是等式**）
4. loss 非 NaN/Inf
5. `S_in` 非空（阶段 1 真正被输入层驱动）
6. `S_out` 非空（readout 真正有信号）
7. 连接数 == 去重后的神经元对数（"同一神经元对只算一条连接"；**不是**"最大入度 <= 1"）
8. 无环 DAG 且每条边严格上行（`z_A < z_B`）
9. 最近邻距 == 2H（FCC 契约，容差 1e-5）
10. 球空间半径落在 `[R_min, R_max]` 内
11. `E > 0` 且平均出度 > 0
12. 不存在 `[N*y_out, N*y_in]` 形状的权重张量（未 materialize dense 矩阵）
13. readout 严格口径：`h` 的非零列**都属于** `S_out`，且 `h` 逐位等于 `a_up * out_scope_mask`
14. 阶段 2 递推顺序 == 流向轴升序（`topo_matches_axis_order == 1`）
15. CPU 单 batch 前向+反向耗时 < 120s

### 验证脚本清单（均在 `checkpoints/n3d_sphere/_verify/`）

> **写入顺序不变量（G6，离朱第 13 轮捕获）**：C6 零残留扫描的**目标**包含
> `.module_agent/n3d_sphere/module_definition.json`，而 `update_definition` 会改写该文件 ——
> 因此**任何元数据 / 文档更新都必须排在最终 `verify_all` 之前**，不得复用更新前的验收结论。

- `verify_sphere_dag.py`：R1 几何、R2 FCC、R3 DAG、R4 去重、R5 双副本、**R5b 逐边数值正确性**、**R5c 感受野覆盖**、R6 判据（含 readout 严格口径）、**R7b readout_scope 生效性**、R7 产物；R1-R7b 均在 4 组配置上执行；运行时落盘 `sphere_dag_metrics.json`（111 条全精度指标）。
- `verify_dh_constraint.py`：**H1-H4** `D <= H` 硬校验（含两条负例）、三预设 `D = H` 一致性、连通性下限（含 D 过小负例）、退化实测。
- `verify_device_regression.py`：**D1-D7 设备契约回归取证**。
- `verify_scope_and_fingerprint.py`：S1-S3 判据语义与双副本随 scope 的行为、S4-S5 产物指纹维度与冒烟命名规则。
- `verify_config_contracts.py`：C1 半径公式、C2-C3 窗口校验与 FCC 容纳性、C4-C5 旧字段清理与取值域、C6 零残留断言（扫描词表与上下文豁免词均在脚本内定义，豁免逐行打印并计数）。
- `verify_topology_snapshot.py`：固定实验点 × 10 seed 拓扑量快照（落盘 JSON；doc_numbers 的 `snapshot_checks` 从此文件的当轮内容现场聚合，**同轮同源、无滞后**）。
- `run_smoke_matrix.py`：11 种冒烟组合重跑与取证刷新（产物名预测比对 + config 校验 + 默认产物未被污染 + 同一路径多次写入逐位一致），并重建 `smoke_scope_matrix.json`。
- `run_round.py`（**K1**）：正式全量训练取证 runner —— Python 侧写盘把子进程 stdout+stderr **流式**写成 **UTF-8** 日志并在末尾追加真实 `EXITCODE=<n>`（规避 Windows PowerShell 重定向写 UTF-16LE 导致日志不可解析）。
- `verify_full_runs.py`（**K1**）：**F-L1~F-R1 正式全量训练取证判据** —— F-L1 台账结构自检；F-L2/F-L3 逐轮三方一致（台账 vs 真实终端日志 vs 可 `torch.load` 的产物，含逐 epoch 轨迹、参数量按"重建模型 `count_parameters()`"口径、退出码、达标判定）；F-L4 正式产物 `model.pt` 与台账来源轮次一致；F-R1 逐位可复现重放（`--max-batches` 限批，产物写入 `_verify/`）。`ledger` 模式秒级并已纳入 `verify_all`；`bounded` 模式分钟级，需单独执行。
- `build_full_runs_ledger.py` / `register_full_runs_doc_numbers.py`（**K1**，**均不是判据**）：前者从真实日志与产物构建轮次台账并冻结产物快照，后者把登记项写入 `doc_numbers.json` 的 `full_run_checks` 类；两者的值全部现场取数、不手填。
- `verify_all.py`：一键依次执行 **11 条命令**（含 `--arch mlp` 冒烟、H1-H4、D1-D7、**F-L1~F-L4 全量轮次复核**与一期回归）并汇总退出码，随后执行**文档数字防线**（**五类数据源**；artifact 按扩展名分派加载器、`full_run_checks` 按轮次/正式产物现场取数且**优先读该轮冻结快照**（`logged.*` 纯日志字段不依赖产物存在性）、`snapshot_checks` 按测点聚合 min/max/range/count 并支持按 `seed` 取单点、`text_checks` 对指定文件/命令输出做正则提取或命中计数；输出含分类明细行）。
- 取证文件：`doc_numbers.json`（252 项）、`full_runs.json`（**5 轮全量训练台账**）、`artifacts/`（**5 轮产物冻结快照**：`A_baseline_default.pt` / `B_preset_highacc.pt` / `C_lr2e3_bs64.pt` / `D_capacity_N384.pt` / `E_N384_epochs40.pt`）、`verify_snapshots_k1fix.py` 与 `log_snapshots_k1fix.txt`、`screen_y16_b400.txt` / `screen_N384_b400.txt`（限批筛参）、`log_full_roundA_baseline.txt` / `log_full_roundB_highacc.txt` / `log_full_roundC_lr2e3_bs64.txt` / `log_full_roundD_N384.txt` / `log_full_roundE_N384_epochs40.txt`（5 轮真实终端输出）、`exit_code_roundA.txt` / `log_exitcode_roundB.txt` / `log_exitcode_roundC.txt`（A/B/C 退出码取证）、`legacy_dh_baseline.json`（`D = H` 改造前基线）、`legacy_archive_manifest.json`（J1 归档逐文件 SHA256 与守恒证明）、`legacy/`（历史口径存档，不参与验收）、`sphere_dag_metrics.json`（111 条）、`topology_snapshot.json`、`log_smoke_matrix_f9.txt` / `smoke_matrix_f9.json`、`log_verify_all_f21.txt` / `log_verify_all_g5.txt` / `log_verify_all_h3.txt` / `log_verify_all_j3.txt`（四轮验收）。
## 纯球形分层有向无环架构

### 球体几何与尺度

神经元是半径 `H` 的球（突触云分布半径）。球空间半径由公式唯一确定（最优堆积系数 `φ = 0.7405`）：

- `R_min = H · (N / φ)^(1/3)`（非重叠容纳下界）
- `R_max = (H + D) · (N / φ)^(1/3)`（保证每个神经元的 D 邻域完整落在球空间内的上界）
- `space_radius` 默认 `0.0` = 取 `R_min`；显式传入时必须落在 `[R_min, R_max]` 内，越界在 `Config` 构造期报错（上下界两侧均有断言）。

**如实口径**：`space_radius` 仅作**半径窗口校验与元数据**，它**不改变神经元放置与拓扑** —— 神经元位置由 FCC 晶格与 `R_max` 决定的搜索半径唯一确定，因此实际 `placement_radius` 允许略超 `R_min`（FCC 格点是离散的，最近 N 个点的最远距离通常大于连续体积下界）。`verify_sphere_dag.py` 的 R1/R2 在含 `space_radius=0.9` 的配置上也验证这一点。

参考值（公式实测，`verify_config_contracts.py` C1 实时计算）：N=256/H=0.10/**D=0.10** → `R_min=0.701840`、`R_max=1.403681`（比值 2.0 = (H+D)/H）；N=256/H=0.10/D=0.05 → `R_max=1.052760`；N=64/H=0.15/**D=0.15** → `R_min=0.663198`、`R_max=1.326395`；N=256/H=0.06/D=0.06 → `R_max=0.842208`。

### 连接半径硬约束 D <= H 与连通性下限

**约束（硬校验，G1）**：`D` 不得超过**接收/发送范围半径** `H`。几何含义：连接判据是"起点神经元的输出突触 `o` 与终点神经元的输入突触 `j` 距离 `<= D`"，而 `o` / `j` 各自落在所属神经元的 `H` 半径球内；`D > H` 表示**连接半径超过突触云自身尺度**，属越界配置。`Config.__post_init__` 在 `D > H` 时抛 `ValueError`（消息含 `H`、`D`、`D/H` 与约束说明）；CLI `--d` 的 help 同步注明。

- **落地口径（G2）**：三个预设一律取 `D = H` —— `SMALL` `H=0.15/D=0.15`、`DEFAULT` `H=0.10/D=0.10`、`HIGHACC` `H=0.10/D=0.10`；数据类字段默认值亦为 `D = H = 0.1`。
- **连带影响**：`R_max` 在 `D = H` 时降为改造前的一半（DEFAULT `1.754601 → 1.403681`）；FCC 放置半径（`0.721110` / `0.670820`）仍远小于新上界，C3 容纳性断言照旧成立。

**连通性下限校验（G3）**：`ThreeDNeuronSpace.__init__` 构图完成后调用 `check_connectivity_floor()`，任一不满足即抛 `ValueError`（消息含实测 `E / N / (E÷N) / 层数K / |S_in| / |S_out| / H / D` 与违反项），通过时指标挂在 `model.connectivity_floor`：

- `E >= N`（平均出度 `E/N >= 1`）；
- 层数 `K >= 2`；
- `|S_in| >= 1` 且 `|S_out| >= 1`。

**退化实测**（N=256/y=8×8/H=0.10/seed=42/any-any，`verify_dh_constraint.py` 的 H4 现跑）：`D=0.10` → `E=736`、`E/N=2.8750`（通过）；`D=0.05` → `E=146`、`E/N=0.5703`（**被拦下**）；`D=0.03` → `E=17`、`E/N=0.0664`（**被拦下**）；`D=0.02` → `E=1`、`E/N=0.0039`（**被拦下**）。

**D = H 改造前后对照**（同测点 N=256/y=8×8/seed=42/any-any）：

| 配置 | \|S_in\| | \|S_out\| | E | 层 K | E/N | params |
|---|---|---|---|---|---|---|
| 改造前 `H=0.10/D=0.15`（违反约束，现已不可构造） | 50 | 45 | 903 | 9 | 3.53 | 42,919 |
| 改造后 `H=0.10/D=0.10` | 193 | 187 | 736 | 9 | 2.88 | 154,864 |
| 改造前 SMALL `H=0.15/D=0.25`（违反约束，现已不可构造） | 13 | 17 | 181 | 7 | 2.83 | 11,077 |
| 改造后 SMALL `H=0.15/D=0.15` | 55 | 53 | 106 | 7 | 1.66 | 43,930 |

改造前两行由 `_verify/legacy_dh_baseline.json` 固化（来源为其登记的改造前产物与 10-seed 快照及 SHA256）；改造后两行来自 `verify_dh_constraint.py` 的 H3 现跑。`|S_in|` 由 50 涨到 193 使 `W_in` 从 `784×50` 撑到 `784×193`，DEFAULT 参数量因此从 42,919 涨到 **154,864（约 ×3.6）**。

### FCC 规则堆积放置

晶格常数 `a = 2√2·H`，基元 `{(0,0,0), (½,½,0), (½,0,½), (0,½,½)}`，故**最近邻距恰为 2H**（相邻神经元的 H 半径突触云恰好相切、不重叠）。取距球心最近的 N 个格点，再按 **(流向轴坐标, 壳层名次)** 的显式字典序做确定性排序，使 `topo_index` 恰为流向轴升序。

- 神经元坐标**与 seed 无关**（完全由 H/N 决定），模块内以断言守护"最近邻距 == 2H"（容差 1e-5）。
- 不存在随机放置分支，也不存在最小间距拒绝采样。

### 突触采样与半球切分

突触在**各自神经元的 H 半径球内按体积均匀**采样：方向在单位球面均匀（正态归一化），半径 `r = H · u^(1/3)`（`u ~ U(0,1)`，立方根保证按体积均匀，**非半径线性采样**）。

输入突触取流向轴**负半球**（-axis）、输出突触取**正半球**（+axis）：把方向的流向轴分量翻转为目标半边（`-|c|` / `+|c|`）。该操作**测度保持**（目标半球内每个方向恰由原始/翻转两种来源各命中一次），等价于半球拒绝采样但不消耗额外随机数、无重试上限。

### 神经元级连接规则

`A -> B` 存在 ⟺ `A ≠ B` 且 `z_A < z_B` 且 `∃ o ∈ out(A), j ∈ in(B): d(o,j) <= D`（`z` 为 `flow_axis` 选定的坐标分量；`D` 受硬约束 `D <= H`）。

- **同一神经元对只算一条连接**：多对突触满足条件时只保留间距最近的一对作为代表连接（`representative_syn_out` / `representative_syn_input`）。
- 判据是"**存在**至少一对"，故必须对合法突触对取 **`amin`** 后与 D 比较（写成 `amax` 语义相反，会把本该连接的神经元对判为不连接）。
- 4D 块视图 `[N_A, y_out, N_B, y_in]` **不可**再 reshape 成 `[N*N, y_out*y_in]` 后按 `flat[A*N+B]` 取值（stride 使轴序相对 2D 视为交换，会产生块内错位）；正确做法是在 `[N_A, N_B, y_out, y_in]` 上分别对两个突触维取 `amin/argmin`。代码内有两处契约断言守护（代表间距必须 == pair_min；不得出现反向边）。
- 全部拓扑量在 `__init__` 预计算并 `register_buffer`，`forward` 不重算 —— 含阶段 2 逐层向量化所用的 `topo_index` / `edge_perm_in` / `neuron_in_edge_reach` / `edge_dst_in` / **`level_edge_reach` / `level_node_reach`**。

### 孤立突触与判据开关

某突触"孤立" ⟺ 其 D 邻域内不存在任何合法连接的对端突触（合法连接 = 对端属于其他神经元且高低关系满足上行约束）。

| 开关 | 取值 | 含义 |
| --- | --- | --- |
| `input_scope` | `any_isolated` / `all_isolated` | 神经元有 ≥1 个 / 全部 y_in 个输入突触孤立 → 进入 `S_in` |
| `readout_scope` | `any_isolated` / `all_isolated` | 神经元有 ≥1 个 / 全部 y_out 个输出突触孤立 → 进入 `S_out` |

实测（DEFAULT 规模 seed=42，**D = H = 0.10**）：`any/any` → S_in=193、S_out=187、双副本 180；`any/all` → 193/14、180；`all/any` → 13/187、0；`all/all` → 13/14、0。`E` 与判据选择无关（恒为 736）。以上数字均登记在 `doc_numbers.json`（键前缀 `r6.` / `r5.`）并由 `verify_all.py` 现跑比对。

### 两阶段前向与双副本展开

- **阶段 1（输入层驱动）**：`a_in[B] = ReLU(x · W_in[:,B] + b_B)`，仅对 `B ∈ S_in` 计算，其余恒为 0（不进入计算图）；`W_in` 形状 `[input_dim, |S_in|]`。
- **阶段 2（单遍逐层递推 + 整层向量化）**：严格按 Kahn 拓扑序（= 流向轴升序）**逐层递推一遍**：
  `a_up[B] = ReLU( Σ_{A→B} w_{A→B} · (a_up[A] + a_in[A]) + b_B )`。
  处理 `B` 时其全部上游（层更小）已算完，故一次前向即完成全部层的传播；**感受野覆盖全部层**（DEFAULT 规模 9 层；R5c 用"祖先层覆盖 + 第一层扰动可传到最深层"两点取证，四组配置的实测最大变化为 0.000079 / 0.000250 / 0.000262 / 0.000250 —— `D = H` 后路径变少，该量比改造前小 1~2 个数量级，但**仍严格 > 0**）。
  **实现形态（F11/F16）**：循环次数 = **层数**（DEFAULT 9 / SMALL 7），而不是神经元数 N；每层只做一次 `index_add`（该层全部入边的消息按目标神经元散加）与一次**非原地** `index_copy`（写回 `ReLU(pre + bias)`，原地赋值会破坏 autograd）。层节点集合由 `topo_index[s:e]` **张量切片**得到。
- **无重复轮数参数**：架构中**没有**迭代轮数（同步迭代的旧设计已移除），故不存在"被固定跳数截断"的问题。
- **双副本展开**：求和项 `(a_up[A] + a_in[A])` 使神经元的"上游版本"与"输入层版本"**都参与后续传播**，两种版本**共享同一套边权** `w_{A→B}`。`a_in` 项作为常量参与每一次聚合；若遗漏则输入层副本被覆盖而永不生效（实测梯度恒为 0）。R5 以"零化 a_in 是否改变 a_up"与 `d(sum a_up)/d(a_in)` 取证：四组配置 max\|Δa_up\| = 1.047723 / 2.259184 / 1.702276 / 2.259184，受影响下游非 `S_in` 神经元 = 9 / 59 / 61 / 59，`d(sum a_up)/d(a_in)` 非零 = 132/192、653/768、643/768、653/768。
- **逐边数值正确性（R5b）**：脚本独立重建阶段 2（按拓扑序逐节点、用原始边表精确递推）与生产实现对比，最大偏差 = 5.960e-08 / 2.384e-07 / 5.960e-08 / 2.384e-07（阈值 1e-5）；同项含 M1 形态**敏感性反例**（错配 106/106、727/736、716/730，结果差异 8.437e-01 / 1.655e+00 / 1.703e+00），确保判据不会退化为恒真。
- **读出（严格口径）**：`h[n] = a_up[n]`（**仅当 `n ∈ S_out`**），否则 `h[n] = 0`；`logits = h @ W_out.T (+ b)`。
  即只有 `S_out` 中的神经元向输出层贡献信号，非 `S_out` 神经元被整体屏蔽 —— 其 `W_out` 列不参与计算图、梯度恒为 0（该口径的直接推论）。`readout_scope` 由此**真正生效**：两取值给出不同 `S_out`（any=187 / all=14），实测同一输入下 logits 最大差异 0.858180（R7b），四种 scope 组合的冒烟 `loss` 互不相同（见"实测验收结果"）。

### 参数集合与稀疏约束

`W_in [input_dim, |S_in|]`、`edge_weight [E]`（每条神经元级连接一个独立标量）、`neuron_bias [N]`（正初值 0.1）、`W_out [output_dim, N]`、（可选）`W_out_bias [output_dim]`。

禁止 materialize dense `[N*y_out, N*y_in]` 权重矩阵：`count_dense_weight_tensors()` 扫描 `named_parameters()` / `named_buffers()`（豁免几何量 `syn_dist`）并断言恒为 0。

**梯度口径（严格读出的直接推论）**：参与 loss 的参数梯度范数必须 > 0（判据 2/3）。neuron3d 的 `W_out` **非 `S_out` 列**梯度恒为 0，故判据断言"非零梯度列数落在 `[1, |S_out|]`"这个**区间**而不是等式：上界来自"只有 `S_out` 列可能非零"，下界 `1` 允许某 `S_out` 神经元的 pre-activation 在整批上非正（ReLU 输出恒 0，其在 `h` 中本来就是零列）—— 这是 ReLU 的正常行为，不是屏蔽错误。MLP 对照基线按 `fc2.*` 参数名断言其输出层梯度 > 0。

### 拓扑统计接口

- `get_connection_stats()`：`num_edges`（神经元级 E）、`num_neurons`、`avg/max_out_degree`、`avg/max_in_degree`、`num_layers`、`num_in_scope`、`num_out_scope`、`isolated_input_syn`、`isolated_output_syn`。
- `get_topology_stats()`：`flow_axis` / `placement` / `space_radius` / `placement_radius` / `lattice_constant` / `nearest_neighbour_dist` / 神经元流向轴高度分布 / 判据编码 / `num_edges` / `edge_dist_mean|min|max` / `dual_copy_count`。
- `connectivity_selfcheck()`：`dag_acyclic` / `all_edges_uphill` / `topo_covers_all` / `topo_matches_axis_order` / `representative_edge_count`。
- `check_connectivity_floor()`（G3，`__init__` 内调用）：校验 `E >= N`、`K >= 2`、`|S_in| >= 1`、`|S_out| >= 1`，返回实测指标字典并挂在 `model.connectivity_floor`。

### 设备契约与静态取证（F16）

**契约**：被 `forward` 当作索引张量使用的一切拓扑量，必须是 `register_buffer` / `nn.Parameter`，
从而 (a) 跟随 `.to(device)` 搬运、(b) 进入 `state_dict()` 持久化。
违反其一即属设备回归：在 CPU 上完全静默（全部 CPU 判据仍全绿），只在 CUDA 上抛 RuntimeError。

- **实现**：`level_edge_reach` / `level_node_reach` 两张 `[K,2]` int64 张量已注册（K = 层数，SMALL 7 / DEFAULT 9 / flow_axis=x 10）；层节点集合由 `topo_index[s:e]` 张量切片得到。历史缺陷是它们曾以**普通 Python `list`** 保存层节点张量 —— `nn.Module.to(device)` **不搬运普通 list 中的张量**。
- **运行时守卫**：`stage2_recurrence` 每次前向调用 `_assert_index_device(a_up)`，逐个校验 `topo_index` / `edge_perm_in` / `edge_dst_in` / `edge_src` / `edge_dst` / `neuron_bias` / `level_edge_reach` / `level_node_reach`（以及启用时的 `W_out_bias`）**既已注册又与激活同设备**，并额外校验层节点切片的设备；任一不符即抛 `RuntimeError`（消息含张力名与两侧设备）。
- **静态取证**：本机 CPU-only（`torch.cuda.is_available() == False`，`torch 2.14.0+cpu`），故 CUDA 路径由 `verify_device_regression.py` 以 `meta` 设备做**搬运实验**（注册 buffer 会搬走、对照的普通 Python list 中张量停留在 cpu）+ **两条守卫负例**（设备不一致 / 未注册均必须抛错）取证；若在具备 GPU 的机器上运行，该脚本自动追加真实 `.cuda()` 前向与 CPU 结果比对（D6）。
- **回归**：`D = H` 改造未触碰该契约，D1-D7 仍全 PASS。

### 实测拓扑规模（可由 `verify_topology_snapshot.py` 复现，引用须标注 seed）

DEFAULT 测点（N=256 / y=8×8 / **H=0.10 / D=0.10** / flow_axis=z / 两个 any_isolated / R=R_min）跨 10 个 seed：

- **E = 727 ~ 750**（均值 ≈ 736.2；seed=42 为 736，seed=7 为 736，seed=2024 为 740）
- 平均出度 = 2.8398 ~ 2.9297（seed=42 为 2.8750）；最大出度与最大入度均为 **4 或 5**（不再出现改造前的 6）
- 层数恒为 9；S_in = 182 ~ 205；S_out = 185 ~ 203；双副本神经元数 = 167 ~ 192
- **可学习参数 146237 ~ 164263**（随 `|S_in|` 变，属"按连接构建"的必然结果；区间取自 `_verify/topology_snapshot.json` 中**该默认测点的 10 个 seed** 记录 —— 全快照含 6 个测点、合并范围为 7874 ~ 164263，因测点规模不同故不混用）
- 改造前对照：同测点在 `D=0.15` 时为 E=900~924、S_in=40~52、params=35094~44489（已由 `legacy_dh_baseline.json` 固化）

SMALL 测点（N=64 / y=4×4 / **H=0.15 / D=0.15** / seed=42）：E=106、平均出度 1.65625、最大出度 4、层数 7、S_in=55、S_out=53、双副本 44、可学习参数 43930。
### 正式全量训练轮次台账与档案（K1；K1 修复轮已更正）

二期首次正式全量训练（K1）的全部轮次都登记在 `checkpoints/n3d_sphere/_verify/full_runs.json`，
每轮只改**一组**参数，判定规则（`test_acc >= 0.9864`）在开工前固定、每轮结束立即判定：

| 轮次 | 参数组 | 关键改动 | `test_acc` | 最终 train loss | E | params | 层数 | 耗时 s | 退出码取值方式 | 产物留存形式 / 冻结快照（SHA256 前 16 位） |
|---|---|---|---|---|---|---|---|---|---|---|
| A `A_baseline_default` | A 容量（默认值未改动） | 无（DEFAULT_CONFIG 原样） | **0.9759** | 0.0195 | 736 | 154,864 | 9 | 137.5 | 独立文件 `exit_code_roundA.txt`（时点歧义见下） | 直接写正式路径 `model.pt`；快照 `artifacts/A_baseline_default.pt`（`82C92A4E…`，来源=artifact） |
| B `B_preset_highacc` | C 优化（预设整体携带） | AdamW + cosine + `grad_clip` + `readout_bias` + bs128 + lr2e-3 + 20 epoch | **0.9823** | 0.0005 | 736 | 154,874 | 9 | 258.1 | 同配置同 seed 限批重放 `log_exitcode_roundB.txt` | 同名被 C 覆盖 → 留存于 `<产物名>.pt.bak`（`DE1A13E5…`）；快照 `artifacts/B_preset_highacc.pt`（`DE1A13E5…`，来源=backup） |
| C `C_lr2e3_bs64` | C 优化（`batch_size` 单点） | 在 B 之上 bs 128→64 | 0.9816 | 0.0001 | 736 | 154,874 | 9 | 495.7 | 同配置同 seed 限批重放 `log_exitcode_roundC.txt` | 当前 `<产物名>.pt` 即 C 产物（`D460F8F2…`）；快照同 SHA |
| D `D_capacity_N384` | A 容量（`N` 单点） | 在 highacc 之上 N 256→384 | **0.9828** | 0.0002 | 1,145 | 233,523 | 11 | 371.7 | `run_round.py` 日志自带 `EXITCODE=0` | 同名被 E 覆盖 → 留存于 `<产物名>.pt.bak`（`63AEC14D…`）；快照 `artifacts/D_capacity_N384.pt`（`63AEC14D…`，来源=backup） |
| E `E_N384_epochs40` | A 容量 + C 优化（`epochs` 单点） | 在 D 之上 epochs 20→40 | 0.9826 | 0.0000 | 1,145 | 233,523 | 11 | 710.1 | `run_round.py` 日志自带 `EXITCODE=0` | 当前 `<产物名>.pt` 即 E 产物（`995CECD5…`）；快照同 SHA |

- **共享 seed=42 前提**：5 轮均为固定 `seed=42` 的单点运行；`seed` 影响突触采样（边集/边数）、
  参数初始化与数据打乱，故全部对照**只在同一 seed 内成立**，不得表述为与 seed 无关；
  与之无关的只有神经元 FCC 放置。本轮**不做**多 seed 方差扫描。
- **迭代顺序与未探索维度（如实标注；K1 修复轮 2 修正时间线）**：实际顺序为
  **A 基线 → B 优化（`--preset highacc`）→ C 优化（bs/lr/epochs）→ D 容量（`--n 384`）
  → E 容量+优化（`epochs=40`）**，**先做了优化组**，偏离"A 容量 → B 几何 → C 优化 → E 判据"
  的优先序。**该偏离没有同期依据**（如实标注，不做事后合理化）：轮 B 的日志于 `01:22:08` 结束，
  早于 `01:27` 的两次限批筛参，故筛参无法解释"为何先做优化"；`01:27` 的筛参实际解释的是**此后**的
  两件事 —— ① 容量维度选 `N=384`（而非 `y=16×16`，见 README §7.2.3）；② 剩余额度的分配
  （额度用于已显现增益的容量与优化两维，几何与判据两组因此再未获得额度）。
  **几何组（`H`/`D`/`flow_axis`）与判据组（两 scope 四种组合）从未探索**，
  故"5 轮用尽"仅指已探索两维内的用尽，归因结论（优化增益最大、容量增益小）**不得外推**。
- **结论**：达标线 = 一期 MLP 对照基线 **0.9864**，5 轮**全部未达标**，最好轮 **D = 0.9828**
  （距线 **0.36pp**）；5 轮均无 NaN/Inf、无崩溃、无连通性下限报错、无 OOM（无跳过轮、无重试掩盖）；
  上限 5 轮用尽后停止并如实汇总，**不伪造达标**。
- **产物纪律（K1 修复轮更正）**：全量产物的文件名指纹为
  `full_N{N}_y{..}_H{..}_D{..}_pl{..}_ax{..}_is{..}_rs{..}_s{seed}[_tag].pt`，
  **不含 `lr` / `batch_size` / `epochs`**，因此 B↔C、D↔E 各自共用同一个文件名、后跑者覆盖先跑者；
  `train.py` 在覆盖前按 `--backup`（默认开）把被覆盖者复制为 `<产物名>.pt.bak`，故**被覆盖轮的
  产物并未丢失**（B 的 `.bak` 实测 `test_acc=0.9823`、D 的 `.bak` 实测 `0.9828`）。
  台账为**每一轮**冻结快照 `_verify/artifacts/<轮次>.pt`（5 轮齐备），来源由每轮
  `snapshot_source` 显式声明（`artifact` / `backup`），并逐轮登记
  `artifact_snapshot_sha256` 与 `artifact_snapshot_bytes`。**`artifact_available` 是现算字段**：
  表示"该轮声明的产物文件名当前是否等于本轮产物"（B/D 为 `false`，**只说明该文件名已被同指纹
  后轮占用，不代表本轮无产物支撑**）。
- **正式产物来源（以产物内嵌 `config` 为准）**：`checkpoints/n3d_sphere/model.pt` 来源 = **轮 A**
  （`DEFAULT_CONFIG` 原样：`N=256`、`y=8×8`、`H=D=0.10`、`bs=64`、`lr=1e-3`、`epochs=10`、
  `seed=42`、Adam、`lr_schedule=none`、`weight_decay=0.0`、`readout_bias=False`、`grad_clip=0.0`），
  内嵌字段 `test_acc=0.9759` / `epochs=10` / `E=736`。因**无达标轮**，"达标轮落 `model.pt`"这一支
  **未触发**，故 `model.pt` **不等于**本轮最好配置（D 的 0.9828）。
  **`.bak` 如实陈述（K1 修复轮更正，替换早期"从未被覆盖、没有产生 `.bak`"的说法）**：
  轮 A **覆盖了一份既有的同配置产物**并生成 `model.pt.bak`（`18072331` 字节、SHA256 与轮 A 产物
  逐位相同 `82C92A4E8396021C…`），此后 `model.pt` 未再被覆盖；该既有产物的
  **来源不可追溯**——其 CreationTime `01:13:25` 早于轮 A 的 `torch.save`（`01:17:06`），
  但 `.module_agent` 时间窗与本轮 `_verify/` 日志中**没有任何全量训练记录**与之对应，
  全盘亦无第二份 `model.pt*`，故只登记"存在过、被备份、与轮 A 逐位相同"三项可核查事实，
  **不给出推测性结论**（该不确定性不影响任何实测数字）。
- **退出码取证口径（K1 修复轮更正）**：**只有 D/E** 的日志末尾自带 `EXITCODE=0`（由 `run_round.py`
  写盘）；**A** 的退出码来自事后登记的 `exit_code_roundA.txt`（mtime `01:21:00`，晚于
  `log_repro_A_bounded200.txt` 的 `01:20:42`，故其 `$LASTEXITCODE` **可能取自随后那次限批重放**
  —— 歧义如实标注）；**B/C** 的退出码取自同配置同 seed 的限批重放日志
  `log_exitcode_roundB.txt` / `log_exitcode_roundC.txt`。三份取证文件均**未改写原始日志**。
  台账的 `exit_code_note` 字段逐轮记录上述方式与时点。
- **取证与判据**：每轮真实终端输出落在 `_verify/log_full_round*.txt`；判据为
  `verify_full_runs.py ledger`（F-L1 结构自检、F-L2/F-L3 逐轮三方一致**含快照路径存在性 +
  SHA256 + 字节数强制比对**、F-L4 正式产物与来源轮一致）与 `verify_full_runs.py bounded`
  （F-R1 逐位可复现重放：5 轮"前 200 batch CE 和"实测 A 108.4 / B 78.4 / C 90.44 / D 72.78 /
  E 72.78，限批产物写入 `_verify/`）；另有只读复核脚本 `verify_snapshots_k1fix.py`
  逐轮打印 `(epochs, batch_size, lr, readout_bias, lr_schedule, grad_clip, test_acc, E)` 比对结果。
  各轮数字同时登记进 `doc_numbers.json` 的 `full_run_checks`（**34** 项）与两个命名口径的
  `text_checks`（**44** 项 = 两次筛参的末轮 loss / `E` / params 6 项 + 筛参与 5 个正式轮的
  「epoch 1 平均 loss×100」与「前 100 batch CE 和」共 23 项 + `full_run.param_assertion_rounds`
  1 项 + 其余 14 项），由 `verify_all.py` 现跑比对。

- **筛参轮（限批，非正式对照；K1 修复轮 2 更正口径）**：`01:27` 两次 `--max-batches 400` 筛参
  （两侧同为 `preset=default` + `--max-batches 400` + `seed=42` + `batch_size=64`，**彼此同预算可比**）：
  `y=16×16` 的 **epoch 1 平均训练 loss×100 = 41.17**（loss 0.4117）、**前 100 batch CE 和 = 78.92**
  （batch100 running_loss 0.7892），末轮 loss 0.0584、`E=805`、`params=135,333` → **被否**；
  `N=384` 的 **36.85**（0.3685）、**68.01**（0.6801），末轮 loss 0.0457、`E=1145`、`params=233,513`
  → **入选**（成为轮 D 的 `N` 取值）。两个口径的**定义与取值来源**：**「epoch 1 平均训练 loss×100」**
  取自 `[epoch 1/N] loss=` 结局行；**「前 100 batch CE 和」** 取自 `epoch 1 | batch 100 | running_loss=`
  行 × 100（该 `running_loss` 本身即前 100 个 batch 的 loss 均值）。正式 5 轮的同口径参照为
  A 27.60 / 75.05、B 26.84 / 52.89、C 23.89 / 59.68、D 24.88 / 48.84、E 24.88 / 48.84；
  **筛参与 D/E 非同一配置**（筛参 `preset=default`：Adam、lr=1e-3、bs=64、bias 关、无调度/裁剪；
  D/E `preset=highacc`：AdamW+wd=1e-4、cosine、lr=2e-3、**bs=128**、clip 1.0、bias 开），
  **二者无逐位可比性**（历史错误更正：早期曾把 B/D 的 epoch1 平均 loss×100 当作筛参的
  "前 100 batch CE 和"，且曾误称"D/E 与 N=384 筛参首 100 batch 完全一致"）。
## 命令行与配置入口

### 几何与判据 CLI

几何相关 CLI（已完全移除几何类型开关）：

- `--flow-axis {x,y,z}`：全局流向轴（缺省沿用预设 z）；输入突触取负半球、输出突触取正半球。
- `--space-radius R`：球空间半径；哨兵 `-1.0` = 未提供、`0` = 取 `R_min`；显式值越界在 `Config` 构造期报错。
- `--input-scope S` / `--readout-scope S`：`any_isolated` / `all_isolated`（空串 = 沿用预设）。
- `--placement P`：缺省沿用预设 `fcc`（当前唯一取值）。
- `--seed S`：覆盖随机种子。**约定：负数报错、`0` 表示"不覆盖"**（`validate_overrides` 与 `--seed` 帮助文本一致）。因此 `--seed 0` 在冒烟矩阵中与默认组合等价（产物同为 `smoke.pt` 且逐位相同），真正的额外 seed 覆盖需用 `--seed 7` / `--seed 2024` 这类正值；`run_smoke_matrix.py` 的记录中已逐条注明。
- `--preset {small,default,highacc}`：冒烟路径的基线由 `build_smoke_config` 决定，`--preset default` 时基线即 `SMALL_CONFIG`（`base = PRESETS[args.preset] if args.preset != "default" else SMALL_CONFIG`），故该写法在冒烟下亦等价于默认组合。
- 全部新参数纳入 `build_smoke_config` 的 explicit 判定（否则 `--smoke-test --input-scope all_isolated` 会被静默丢弃），并同步进 `apply_overrides` 与 `run_full_training` 的兼容模式。
- **值不等价才算覆盖**：`apply_overrides` 构造出的 `Config` 若与基线**逐字段相同**，则**返回基线对象本身**而非新对象（值等价短路），使同一语义配置产出同一字节（见下节）。

### 产物字节确定性与 SHA256 语义（离朱第 11 轮 D1）

**`torch.save` 的产物字节不只由数值决定，还取决于 pickle 的记忆化（memo），而 memo 依赖
对象身份（`id`）而非取值** —— 因此"SHA256 相等"只有在**同一语义配置走同一代码路径**时才
等价于"数值相同"。

- **反例实测（D1）**：`--smoke-test`（走 `SMALL_CONFIG` 单例）与
  `--smoke-test --input-scope any_isolated --readout-scope any_isolated`（scope 取值来自
  argparse 构造的**等值新字符串**）写出的 `smoke.pt`：loss / `config` / `grad_norms` /
  `model_state_dict` 全部张量**逐位相等**，但 `data.pkl` 为 4097 vs 4117 字节、共 1520
  字节不同（单例下两个 scope 字段指向同一个 `"any_isolated"` 字符串对象，第二次出现被写成
  memo 引用；显式传参下两处分别内联写出）。后果：矩阵记录 [0] 登记的 SHA 在矩阵结束后
  **已被同轮后续写入覆盖**，取证文件不自洽。
- **根因侧修复**：`train.py` 的 `apply_overrides` 在"显式覆盖值与基线**逐字段相同**"时
  **复用基线对象本身**，使"同一语义配置 ⇒ 同一字节"。修复后 `--smoke-test` / 显式传等值
  scope / `--space-radius 0.0` / `--preset default` / `--seed 0` 等路径写出的 `smoke.pt`
  SHA256 **全部为 `1A9D68D8…43D7`**；离朱第 12 轮复测另确认 `build_smoke_config(...) is
  SMALL_CONFIG` 为 True（**同一对象**，而非仅字段相等）。
- **日志出现的精确范围（离朱第 12 轮 D2 澄清）**：`apply_overrides` 的短路分支会打印
  `显式覆盖参数与基线取值逐字段一致（值等价）：复用基线配置对象…`。**只有"显式给出覆盖
  参数且其值与基线相同"的形式**才会打印该行，实测为 3 条形式（`--input-scope
  any_isolated`、两个 scope 都显式传等值、`--space-radius 0.0`）；而**裸跑**、
  `--preset default`、`--seed 0` 三种形式在 `build_smoke_config` 的
  `if not explicit: return SMALL_CONFIG` 处**提前返回**（按 CLI 约定它们本就不算"显式覆盖"），
  不会进入 `apply_overrides`，故不打印该行 —— 但**产物字节与前者完全一致**（都走
  `SMALL_CONFIG` 单例）。真正改变取值的覆盖仍打印 `冒烟测试配置已被显式覆盖…` 告警且不打印
  值等价行，语义边界正确（离朱第 12 轮已用 9 组非等值覆盖验证未被过度短路）。
- **判据侧修复**：`run_smoke_matrix.py` 新增"**同一产物路径被多条组合写入时必须逐位一致**"
  的断言，并在每条记录追加 `artifact_sha256_at_end` / `artifact_sha256_stable` /
  `artifact_shared_writers` 字段。实测：`smoke.pt` 被 **3 条组合**写入
  （`scope any/any seed42` / `seed 0` / `preset default`）、SHA 去重后 **1 种**；
  11/11 记录的 `artifact_sha256_stable == true`、`artifact_sha256_at_end == artifact_sha256`
  且与盘上文件实际 SHA 一致。
- **引用规范**：报告引用产物 SHA256 时必须同时给出**配置与调用路径**；跨路径判断"数值
  等价"应以 `torch.load` 后的张量比对（`torch.equal`）为准，而不是文件 SHA。

### 配置字段（已清理）

保留：`N` / `y_in` / `y_out` / `H` / `D` / `flow_axis` / `space_radius` / `placement` / `input_scope` / `readout_scope` / `seed` / `input_dim` / `output_dim` / `hidden_dim`（仅 mlp 基线）/ 训练超参。

已删除（旧架构字段）：几何类型字段与边长字段、温度初始化与残差系数、最小间距与采样重试上限、读出头 dropout、以及**迭代轮数**；同时移除全部与球体无关的几何派生量。`dropout` 相关训练增强一并移除（参数集合中不再有读出头 dropout）。

**当前字段清单**：`N` / `y_in` / `y_out` / `H` / `D` / `flow_axis` / `space_radius` / `placement` / `input_scope` / `readout_scope` / `input_dim` / `output_dim` / `hidden_dim`（仅 mlp 基线用）/ 训练超参。**架构中不存在任何迭代轮数参数**（阶段 2 为单遍逐层递推）。

`to_dict()` / `describe()` / `apply_overrides` 已同步（新增字段缺席会在 `Config(**base.to_dict())` 往返时被静默丢弃，故必须纳入）。

### 产物命名与元数据

- 冒烟产物：**完全默认组合**（neuron3d + SMALL_CONFIG 的 N/y/H/D/seed/**batch_size** + flow_axis=z + 两个 any_isolated + `space_radius=0`）退化为 `_verify/smoke.pt`；其它任何组合（换 arch / 流向轴 / scope / 覆盖 N、y、H、D、seed、batch_size、space_radius）为 `_verify/smoke[_ar{arch}]_{完整配置指纹}.pt`。
- **指纹格式**：`N{N}_y{y_in}x{y_out}_H{H}_D{D}_pl{placement}_ax{axis}_is{scope}_rs{scope}_bs{batch_size}[_R{space_radius}]_s{seed}` —— **指纹维度必须与"默认判定"维度严格对齐**（历史缺陷：默认判定含 `batch_size` / `space_radius` 而指纹不含，导致 `--batch-size 64` 与 `--space-radius 0.9` 落回同名文件、互相覆盖取证产物；`arch` 缺维度亦曾导致 `--arch mlp` 与主模型互覆）。`run_smoke_matrix.py` 会把实际落盘产物名与按此格式**独立预测**的名字逐条比对。
- 限批产物：`verify_<bpe>_N{N}_y{y_in}x{y_out}_H{H}_D{D}_pl{placement}_ax{axis}_is{scope}_rs{scope}_s{seed}[_tag].pt`。
- 全量产物默认路径：与 `DEFAULT_CONFIG` 完全同配置时用 `model.pt`，其它配置 `full_N{N}_y{..}_H{..}_D{..}_pl{..}_ax{..}_is{..}_rs{..}_s{seed}[_tag].pt`（**不**复用限批前缀，避免 `full_verify_0_...` 这类误导名）。
- 指纹维度含 `flow_axis` / `input_scope` / `readout_scope` / `placement` / `seed` / `H` / `D` / `N` / `y`（架构中无迭代轮数，故指纹不含该维度） —— `seed` 影响突触采样（不同 seed 即不同边集），必须进指纹。
- **同一路径的多次写入必须逐位一致**：默认组合专属的 `smoke.pt` 会被多条等值组合写入（裸跑 / 显式传等值 scope / `--seed 0` / `--preset default`），`run_smoke_matrix.py` 对此做强断言并记录 `artifact_sha256_at_end` 等字段（见上节 D1）。
- `model_state_dict` 中的拓扑 buffer：`topo_index` / `edge_offset` / `edge_perm` / `edge_perm_in` / `neuron_in_edge_reach` / `edge_dst_in` / **`level_edge_reach` / `level_node_reach`** / `in_scope_mask` / `out_scope_mask` / 度分布 —— 全部 `register_buffer`，故 `.to(device)` 与 `state_dict()` 都能搬运/持久化（F16 设备契约）。
- checkpoint 元数据新增/更新：`placement` / `flow_axis` / `space_radius` / `effective_space_radius` / `min_space_radius` / `max_space_radius` / `input_scope` / `readout_scope` / `topology_stats`（几何指纹与统计）/ `dag_selfcheck`（无环、严格上行、拓扑序覆盖）。
