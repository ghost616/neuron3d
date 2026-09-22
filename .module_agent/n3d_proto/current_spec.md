# N3D 一期原型：三维神经元空间架构

## 项目定位

受生物大脑启发的神经网络架构：神经元分布在三维空间中，每个神经元带若干输入突触与输出突触，突触自身也有三维坐标；输入突触与输出突触之间只有在距离阈值 D 内才允许传递数据。输入层连接全部神经元的输入突触，输出层连接全部神经元的输出突触。

一期原型采用**静态拓扑**：神经元与突触的三维坐标在初始化后固定不变，仅学习权重参数。目标是验证前向传播与反向传播能否走通，不追求性能。代码位于 `n3d_proto/` 子目录，与工程内既有 `hstdn/` 代码完全隔离。

## 数据结构与张量形状总表

设神经元数为 N，每个神经元有 y_in 个输入突触与 y_out 个输出突触，前向迭代轮数 T，空间边长 L，突触分布范围 H，连接距离阈值 D。默认值：N=256, y_in=8, y_out=8, H=0.1, D=0.15, L=1.0, T=3, input_dim=784, output_dim=10。

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `neuron_pos` | [N, 3] | buffer（固定） | 神经元三维坐标，空间内稀疏均匀随机分布 |
| `input_syn_pos` | [N*y_in, 3] | buffer（固定） | 输入突触坐标，在所属神经元 H 范围内随机 |
| `output_syn_pos` | [N*y_out, 3] | buffer（固定） | 输出突触坐标，在所属神经元 H 范围内随机 |
| `dist` | [N*y_out, N*y_in] | buffer（预计算） | dist[i,j] = ‖output_syn_pos[i] − input_syn_pos[j]‖ |
| `mask` | [N*y_out, N*y_in] | buffer（预计算） | mask[i,j] = 1 若 dist[i,j] ≤ D，否则 0 |
| `edge_index` | [2, E] | buffer（预计算） | edge_index[0]=输出突触索引，edge_index[1]=输入突触索引 |
| `edge_dist` | [E] | buffer（预计算） | 每条边的距离，用于 softmax 权重 |
| `W_conn_sparse` | [E] | Parameter | 每条边一个可学习权重（边级参数化） |
| `tau_raw` | 标量 | Parameter | tau = softplus(tau_raw) + 0.01，保证 tau > 0 |
| `neuron_threshold` | [N] | Parameter | 神经元阈值 |
| `W_in` | [input_dim, N*y_in] | Parameter | 输入层权重 |
| `W_out` | [N*y_out, output_dim] | Parameter | 输出层权重 |
| `scatter_out_to_neuron` | [N, N*y_out] | buffer（预计算） | 将输出突触信号求和到神经元 |
| `broadcast_neuron_to_in` | [N*y_in, N] | buffer（预计算） | 将神经元状态广播到其输入突触 |
| `ln_s_in` | — | LayerNorm(N*y_in) | 输入突触信号层归一化，防止迭代退化 |

## 闭环信息流（四步闭环）

原版设计有两个缺陷：(1) 当 y_in ≠ y_out 时，s_out [B, N*y_out] 无法直接回填为下一轮 s_in [B, N*y_in]；(2) 纯线性迭代会让信号坍缩到主特征向量或数值爆炸。修复方案为**四步闭环 + 残差连接 + LayerNorm**：

1. **步骤 2a 空间连接（稀疏）**：对每个输出突触 o，在其连接的输入突触集合上做 masked softmax；s_out[b,o] = Σ_j (w_oj · s_in[b,j])，其中 w_oj = softmax_j(−dist/tau + W_conn) · mask。
2. **步骤 2b 输出突触 → 神经元（scatter sum）**：neuron_input[b,n] = Σ_{o ∈ 神经元 n} s_out[b,o]。
3. **步骤 2c 神经元激活**：a[b,n] = ReLU(neuron_input[b,n] + neuron_threshold[n])。
4. **步骤 2d 神经元 → 输入突触（broadcast）+ 残差 + LayerNorm**：s_in_new[b,i] = a[b, neuron_of(i)]；s_in_next = LayerNorm(s_in_new + alpha · s_in)，残差系数 alpha 默认 0.1。

步骤 2b 与 2d 通过预计算矩阵 `scatter_out_to_neuron` 与 `broadcast_neuron_to_in` 实现，天然支持 y_in ≠ y_out。

## 前向传播流程

1. **输入编码**：s_in = x @ W_in，形状 [B, input_dim] → [B, N*y_in]。
2. **迭代空间传递（T 轮）**：每轮依次执行 2a 稀疏传播 → 2b scatter sum → 2c 激活 → 2d broadcast + 残差 + LayerNorm，得到新一轮 s_in。
3. **输出汇总**：用最后一轮的 s_out（输出突触信号）接输出层，logits = s_out @ W_out，形状 [B, output_dim]。

## masked softmax 的归一化方向

**必须**对每个输出突触 o，在其连接的输入突触 j 上做 softmax：w_oj = mask[o,j]·exp(−dist[o,j]/tau + W_conn[o,j]) / Σ_{j'} mask[o,j']·exp(−dist[o,j']/tau + W_conn[o,j'])。实现上先算边级 logits，再按 `edge_index[0]`（输出突触索引）分组做 segment softmax。**禁止**对输入突触维度做 softmax，**禁止**对整张矩阵做全局 softmax。

## 稀疏实现约束

1. **禁止 materialize dense [N*y_out, N*y_in] 权重矩阵**——在 N=256, y_in=y_out=8 时该矩阵达 4M 参数且 99% 为零。必须使用边级参数化 `W_conn_sparse` [E]。
2. `sparse_propagate` 必须用 `edge_index` + `scatter_add_` 实现。
3. `segment_softmax` 必须用 scatter 操作实现；优先使用 `torch_scatter`，环境缺失时在 `utils.py` 提供纯 PyTorch fallback（scatter_max + scatter_add）。
4. 全部拓扑量（dist、mask、edge_index、edge_dist、scatter_out_to_neuron、broadcast_neuron_to_in）在 `__init__` 中预计算并注册为 buffer，前向传播中不得重算。

## 文件与职责

| 文件 | 职责 |
|------|------|
| `n3d_proto/README.md` | 项目简介与设计思路、安装方式、两种运行方式、四步闭环架构示意、一期范围说明、后续展望 |
| `n3d_proto/requirements.txt` | 依赖清单（torch、torchvision、numpy 等） |
| `n3d_proto/config.py` | `@dataclass Config` 全部超参；`SMALL_CONFIG`（N=64, y_in=4, y_out=4, T=1, batch=32）与 `DEFAULT_CONFIG` 两套预设 |
| `n3d_proto/model.py` | `ThreeDNeuronSpace(nn.Module)` 模型类，含 `forward`、`sparse_propagate`、`count_parameters`、`get_connection_stats` |
| `n3d_proto/data.py` | MNIST 数据加载，28×28 展平为 784 维，返回 train_loader / test_loader |
| `n3d_proto/train.py` | 两阶段训练主脚本（阶段 A 冒烟测试 / 阶段 B 正式训练），支持 `--smoke-test` |
| `n3d_proto/utils.py` | `set_seed`、`get_device`、`segment_softmax`、`build_edge_index`、`build_scatter_matrix`、日志函数 |

## 验收标准

**阶段 A（必须达到）**：`python n3d_proto/train.py --smoke-test` 在 CPU 上 2 分钟内完成；前向无 shape mismatch；反向无错误且所有可学习参数梯度范数 > 0；loss 不为 NaN/Inf；**连接稀疏度 < 0.1（口径 = 连接密度 `E / (N*y_out * N*y_in)`，见 `get_connection_stats()`）**；tau 初始值 > 0。该命令退出码为 0 即代表全部通过（无需任何豁免开关）。

除上述必判项外，冒烟测试另附两条**语义等价且可执行**的判据（用于验证"未 materialize 稠密权重矩阵"）：
1. 边级参数数 == E（`W_conn_sparse.numel() == num_edges`）；
2. 不存在形状为 `[N*y_out, N*y_in]` 的权重张量（`model.count_dense_weight_tensors() == 0`；同形状的 `dist` / `mask` 属预计算几何 buffer，不计入）。

**阶段 B（尽力达到，不阻塞）**：`python n3d_proto/train.py` 在 CPU 上 30 分钟内完成 10 个 epoch、MNIST 测试准确率 ≥ 90%；若未达 90%，在 README 中分析原因并给出调参建议（增大 N、调整 lr、增加 T 等）。

**产物保护**：冒烟测试（`--smoke-test`）与限批验证跑（`--max-batches > 0`）不得覆盖正式 checkpoint，其产物写入 `checkpoints/_verify/`，并在日志中打印实际写入路径。

阶段 A 是一期核心目标；一期验证的是"机制可行"，不是"性能达标"。
**阶段 B（尽力达到，不阻塞）**：`python n3d_proto/train.py` 在 CPU 上 10 个 epoch、MNIST 测试准确率 ≥ 90%（30 分钟为参考值，非硬性门槛；未达标时须在 README 分析原因并给出调参建议）。

**第 2 轮全量复跑实测（2026-09-22，修复后口径）**：
* 命令：`python n3d_proto/train.py --checkpoint checkpoints/n3d_model_full.pt`（DEFAULT_CONFIG，10 epoch，全量 60000 样本）
* 逐 epoch test_acc：95.12% → 95.89% → 96.48% → 96.71% → 97.28% → 97.32% → 96.69% → 96.95% → 97.43% → **97.20%**（峰值出现在第 9 轮）
* 总耗时 1893.2s ≈ 31.5 min（略超 30 分钟参考值）；最终统计：可学习参数 1691720、连接稀疏度（密度）0.014604、zero_ratio 0.985396、E=61255、平均出度 29.910、tau 0.297663
* 复跑工件：`checkpoints/n3d_model_full.pt`（已用 `torch.load` 校验：`test_acc=0.972`、`epochs=10`、`batches_per_epoch=None`、`connection_stats.sparsity=0.014604`，与 README 记录一致）
## 环境与数据约定

- 运行环境为 Windows + Python 3.12，CPU 训练。
- MNIST 原始文件已存在于工程 `data/mnist/`（train-images-idx3-ubyte.gz 等 4 个文件），应优先复用避免重复下载；若 torchvision 期望的目录结构与现有不一致，可自行解析 IDX 文件或调整数据路径。
- 性能保护：先小规模验证（SMALL_CONFIG）、预计算全部拓扑、稀疏实现、以 batch 为第一维度、冒烟测试打印全部可学习参数梯度范数。
## 实现说明与验收口径澄清

本节记录一期原型落地时对规格的精确化与口径澄清，均已在代码中实现并实测验证。

### T 必须 >= 2（梯度可达性约束）

四步闭环中 2d 的输出（经 `LayerNorm` 的 `s_in_next`）只有进入"下一轮"的 2a 才会影响输出通路。若 `T = 1`，末轮的 `ln_s_in` 与 `neuron_threshold` 不参与 loss 计算，其 `.grad` 恒为 `None`，违反验收标准中"所有可学习参数梯度范数 > 0"。因此 `Config.T` 约束为 `>= 2`；`SMALL_CONFIG` 取 `T=2`，`DEFAULT_CONFIG` 取 `T=3`。

另：输出汇总使用的是**最终轮 2a** 得到的 `s_out`（与规格第 3 条一致）。当 `T = 1` 时该值仍来自唯一的 2a，故模型结构本身无需改动。

### 连接稀疏度口径（已回归规格：即密度定义）

`get_connection_stats()` 规定的"连接稀疏度"公式为 **`E / (N*y_out * N*y_in)`**，即**连接密度**（不是 1 − 密度）。第 2 轮修复已把实现与该口径对齐：

* `utils.connection_density(num_edges, num_output_syn, num_input_syn)` 返回规格公式 `E/(N*y_out*N*y_in)`；`connection_sparsity` 保留为**同义别名**，便于按规格原文检索；
* `utils.zero_ratio(...)` 返回 `1 − 密度`（dense 矩阵的零元素占比），仅用于日志对照，**不参与验收判据**；
* `model.get_connection_stats()` 返回 `sparsity`（= 密度）、`zero_ratio`、`num_edges`、`avg_out_degree`、`tau`；
* 阶段 A 判据为 `sparsity < 0.1`（规格口径）。实测：`SMALL_CONFIG` E=3906 / 65536 -> sparsity=0.0596（PASS）、zero_ratio=0.9404；`DEFAULT_CONFIG` E=61255 / 4194304 -> sparsity=0.0146（PASS）、zero_ratio=0.9854。

配合规格"禁止 materialize dense `[N*y_out, N*y_in]` 权重矩阵"的约束，另设两条**语义等价且可执行**的附加判据：
1. 边级参数数 == E（`W_conn_sparse.numel() == num_edges`；DEFAULT 下 61255 vs 稠密所需的 4194304，相差 68 倍）；
2. 不存在 `[N*y_out, N*y_in]` 形状的**权重张量**——由 `model.count_dense_weight_tensors()` 判定，恒返回 0；同形状的 `dist` / `mask` 是预计算几何 buffer（非权重、不参与训练），已在实现中显式豁免并注释说明。

若真正 materialize 稠密连接权重，可学习参数会从 1691720 膨胀到约 580 万（DEFAULT 配置），这是边级参数化要避免的代价。

### checkpoint 产物保护

`train.resolve_checkpoint_path(checkpoint_override, max_batches)` 统一决定写入路径：
1. `max_batches > 0`（限批/验证跑）-> 一律写入 `checkpoints/_verify/`（冒烟测试为 `_verify/smoke.pt`），**绝不覆盖正式产物**；
2. `--checkpoint PATH` 非空 -> 使用该路径（相对路径按工程根目录解析）；
3. 否则 -> 默认 `checkpoints/model.pt`。

日志会打印实际写入路径，并在限批模式下额外提示"正式产物未被覆盖"。正式复跑工件为 `checkpoints/n3d_model_full.pt`。

### masked softmax 的数值稳定实现

`segment_softmax(values, index, num_segments)` 优先使用 `torch_scatter.scatter_softmax`；环境缺失时走纯 PyTorch fallback：`scatter_reduce(reduce="amax")` 求组内最大值 -> `exp(v - m_g)` -> `scatter_add` 求归一化因子 -> 除以 `Z_g + eps`。分组索引固定为 `edge_index[0]`（输出突触），保证归一化方向为"对每个输出突触，在其连接的输入突触上做 softmax"。

边级的存储形态保证：分组 softmax 天然只在 mask=1 的边上分配概率质量，无需额外乘 mask，也不会因未连接的位置产生概率泄漏。

### 2b / 2d 的高效稀疏算子

除规格要求的稠密标记矩阵 `scatter_out_to_neuron` [N, N*y_out] 与 `broadcast_neuron_to_in` [N*y_in, N] 外，前向传播实际使用两个 y 级（突触分组）算子：
* `utils.sparse_aggregate(s_out, neuron_of_output_syn, N)`：`index_add_` 沿 dim=1 按神经元累加，等价于 scatter sum，复杂度 O(B·N·y_out)；
* `utils.sparse_broadcast(a, neuron_of_input_syn)`：`index_select` 按所属神经元取回，等价于 broadcast，复杂度 O(B·N·y_in)。

两者天然支持 `y_in ≠ y_out`，且不构造任何 `N*y_out × N*y_in` 规模的张量。

### 数据层与可复现性

* MNIST 严格复用工程内 `data/mnist/*.gz`：`data.ensure_mnist_files` 按"已就绪 -> 从候选目录复制（非破坏、不联网） -> 允许时才下载"的顺序处理，torchvision 期望的 `<root>/MNIST/raw/` 布局会被自动补齐；
* IDX 解析为惰性加载（`RawIdxMNIST.images/labels` 为 property）；`__len__` 只解析并缓存标签文件（**不触发图像全量解析**），`get_mnist_loaders` 的日志也改用常量 784，避免为打印一行日志而解压 60000 张图；
* `num_workers > 0` 时由 `data._worker_init_fn` 按 `(torch.initial_seed(), worker_id)` 为每个 worker 独立播种 random/numpy/torch，固定 `seed` 即可复现多进程取数顺序；`num_workers = 0` 时顺序完全由 `torch.Generator`（seed 控制）决定；数据集不含随机数据增强，故不存在跨 worker 的增强多样性差异；
* 神经元/突触坐标与参数初始化由 `Config.seed` 派生的 `torch.Generator` 控制，保证可复现且不扰动全局随机状态。

### 入口健壮性

`train.build_model_and_data` 在装配后立即断言：所有 `named_parameters()` 与 `named_buffers()` 的设备等于目标设备，且首个 batch 的 `x/y` 也位于目标设备；任一不符即抛 `RuntimeError`（含具体参数名与设备），使设备错误在命令行入口附近暴露，而不是在反向传播深处。

### 文档口径同步

* `Config` docstring 标注「验收/生产场景要求 `T >= 2`」（字段级校验仅强制 `T >= 1`，`T >= 2` 由预设常量保证）；`num_workers` 条目补充 per-worker 播种的可复现性前提。
* `train.py` 模块 docstring 同步 `SMALL_CONFIG（N=64, y_in=y_out=4, T=2, batch=32）`，并列出 `--checkpoint` 与产物保护规则。
* `utils.build_edge_index` docstring 的展开顺序表述更正为「行优先（row-major / C order）」，与实现 `flat_idx = o * n_in + j` 一致。
* README 已删除"规格自相矛盾"表述，改为按规格公式说明连接稀疏度（密度）< 0.1 达标；环境说明更正为"torch/torchvision 经 `n3d_pkgs.pth` 自动可见，普通 `python` 命令无需设置环境变量"。

### 离朱回归结果（第 1 轮：2026-09-22，345 条断言全绿）

| 测试类型 | 脚本 | 断言数 | 通过 | 失败 |
|---|---|---|---|---|
| 单元测试（config + utils） | `.lizhu_env/lizhu_tests/lizhu_n3d_cfg_utils_tests.py` | 140 | 140 | 0 |
| 单元测试（model） | `.lizhu_env/lizhu_tests/lizhu_n3d_model_tests.py` | 86 | 86 | 0 |
| 单元测试（data） | `.lizhu_env/lizhu_tests/lizhu_n3d_data_tests.py` | 51 | 51 | 0 |
| 接口/集成测试（train CLI） | `.lizhu_env/lizhu_tests/lizhu_n3d_train_tests.py` | 68 | 68 | 0 |
| 编译检查 | `python -m compileall -q n3d_proto` | — | ✅ | 0 |
| **合计** | | **345** | **345** | **0** |

关键验证点：`sparse_propagate` 在 `W_conn_sparse=0` 时输出与手动参考实现 `Σ_j softmax(-edge_dist/tau)·s_in[:,j]` 逐元素一致（maxdiff < 1e-6），证明确实由 `edge_index` + `segment_softmax` + `index_add_` 实现；`segment_softmax` 的归一化方向已用"按输入突触分组会得到全 1"反证；同 seed 两次构造的 `neuron_pos/dist/W_in/W_out/edge_index` 完全一致。

> 注：上述离朱脚本中关于 `connection_sparsity` 的断言按第 1 轮实现（零元素占比）编写；第 2 轮已把该函数回归为规格口径（密度），调用方与断言需相应更新。
### checkpoint 产物保护（第 2 轮 + 离朱建议落实）

`train.resolve_checkpoint_path(checkpoint_override, max_batches)` 统一决定写入路径：
1. `max_batches > 0`（限批/验证跑）-> 一律写入 `checkpoints/_verify/`；未显式给文件名时按限批量生成 `verify_<max_batches>.pt`（与冒烟测试的 `smoke.pt` **区分开**，避免两者互相覆盖），**绝不覆盖正式产物**；
2. `--checkpoint PATH` 非空 -> 使用该路径（相对路径按工程根目录解析）；
3. 否则 -> 默认 `checkpoints/model.pt`。

此外 `train.backup_existing_checkpoint(path, enabled)` 在覆盖既有 checkpoint 前复制为 `<path>.bak`（CLI 开关 `--backup` / `--no-backup`，默认开启）。原因：`checkpoints/` 被 `.gitignore` 忽略，覆盖后无法按位恢复——离朱按"验收项 6"覆盖 `n3d_model_full.pt` 即触发了该风险（已由离朱内建恢复步骤重训恢复，当前产物 `test_acc=0.9753`、`epochs=10`、SHA256 `888556b0...`）。

冒烟测试写入 `checkpoints/_verify/smoke.pt`，日志打印实际路径与「正式产物未被覆盖」。

**已验证的产物保护行为**（真实执行）：
* 冒烟跑 + 限批跑前后，`checkpoints/model.pt`（SHA256 `d354ed0f...`）与 `checkpoints/n3d_model_full.pt`（`888556b0...`）哈希完全不变；
* 限批跑写入 `_verify/verify_3.pt`（不再覆盖 `_verify/smoke.pt`）；
* 二次覆盖 `_verify/verify_3.pt` 时生成 `_verify/verify_3.pt.bak`；带 `--no-backup` 时不生成。

### 别名弃用说明

`utils.connection_sparsity` 保留为 `utils.connection_density` 的同义别名，代码注释已标注 **deprecated**；`model.py` 内部已改为仅导入并使用 `connection_density`。新代码请统一使用 `connection_density`，避免"稀疏度"一词再次引起口径歧义。
### 当前仓库工件状态（第 3 轮收尾后）

| 路径 | 内容 | SHA256 | 说明 |
|------|------|--------|------|
| `checkpoints/n3d_model_full.pt` | `test_acc=0.9753`、`epochs=10`、`sparsity=0.014604`、`zero_ratio=0.985396`、`avg_out_degree=29.910`、`tau=0.28794`、`num_edges=61255` | `888556B0913C9F46419A674117FD13A99F2C71BA6692367A839DB873A58D8924` | **正式交付工件** |
| `checkpoints/_verify/legacy_model_r1.pt` | `epochs=2`、`test_acc=0.8375`、旧口径 `connection_stats`（无 `zero_ratio`） | `D354ED0F6BD48043BB26775D24A03E9D367145CEA28CB5E4E60130172E67CEAC` | 历史遗留产物，第 3 轮从默认路径移入 `_verify/` 留痕（**不删除**） |
| `checkpoints/model.pt` | — | — | **当前不存在**：默认路径不存放交付工件；该路径仅在正式全量训练（未使用 `--max-batches`）时才会被写入 |
| `checkpoints/_verify/smoke.pt` | `stage="smoke"` 的冒烟产物 | 每次运行重写 | 验证类产物 |
| `checkpoints/_verify/verify_<N>.pt` | 限批验证产物 | 每次运行重写 | 与冒烟产物文件名区分，互不覆盖 |

**数字可比性**：README 中「97.20%」的终端输出块与当前工件**不是同一次运行**（当前工件为 97.53%），二者统计口径完全一致；第 1 轮声称的 97.76% 对应工件已不在仓库中，**不作为可比基线**。

### 设备探针的取数代价（仅说明，逻辑不改）

`train.build_model_and_data` 末尾用 `next(iter(train_loader))` 取一个 batch 作为设备一致性探针。训练集 DataLoader 开启 `shuffle=True`，因此该探针会从打乱流中**先取走一个 batch**（DEFAULT 下 batch_size=64，约占 60000 样本的 0.11%），导致单个 epoch 实际参与训练的 batch 数为 **937 而非 938**。这是"用 0.11% 样本换设备错误在入口即报出"的有意取舍；取数逻辑本身未做任何改动。

### 稠密权重检查的扫描范围（比规格字面更严格）

`model.count_dense_weight_tensors()` 除扫描 `named_parameters()` 外，还扫描 `named_buffers()`：除显式豁免 `dist` / `mask` 两个几何 buffer 外，任何形状为 `[N*y_out, N*y_in]` 的 buffer 也计为违规。这可以捕获"本应是参数却被误注册为 buffer"或"把稠密权重塞进 buffer"的隐蔽写法，因此比规格字面（仅要求"禁止 materialize dense 权重矩阵"）更严格。

### 可复现性的传导前提

`data._worker_init_fn` 的 per-worker 基种子取自父进程的 `torch.initial_seed()`，因此**必须先调用 `utils.set_seed(config.seed)`**（`train.build_model_and_data` 已自动完成），`Config.seed` 才能通过初始种子传导到各 DataLoader worker；`Config.num_workers` 与 `data.get_mnist_loaders` 的 docstring 均已注明该前提。