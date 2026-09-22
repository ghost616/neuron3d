# N3D 一期原型：三维神经元空间架构（静态拓扑）

> **仓库布局**：代码位于 `n3d_proto/` 子目录，本说明位于仓库根目录。
> 依赖清单与运行记录同为仓库根级文件（`requirements.txt`）。

## 这是什么

N3D 是一个受生物大脑启发的神经网络原型：神经元分布在三维空间中，每个神经元带有若干**输入突触**与**输出突触**，突触自身也有三维坐标；只有当某个输出突触与某个输入突触的空间距离不超过阈值 `D` 时才允许传递数据。输入层连接到全部神经元的输入突触（`x @ W_in`），输出层从全部神经元的输出突触读出（`s_out @ W_out`）。

**一期只做静态拓扑**：神经元与突触坐标在初始化后固定不变，只学习权重参数。目标不是刷准确率，而是验证「这个机制能不能跑通前向与反向传播、能不能训练」。

## 为什么这样设计

* **空间稀疏连接**：连接由几何距离决定，天然稀疏（默认配置下边数只占全连接矩阵的约 1%），避免 `[N*y_out, N*y_in]` 这种 4M 参数的稠密权重矩阵——本项目**禁止** materialize 该矩阵，连接权重以边级参数 `W_conn_sparse` [E] 表示。
* **四步闭环 + 残差 + LayerNorm**：纯线性迭代会让信号坍缩到主特征向量或数值爆炸；引入残差系数 `alpha` 与 LayerNorm 后迭代才稳定。同时"输出突触数 `N*y_out` ≠ 输入突触数 `N*y_in`"的问题，通过 scatter（2b）与 broadcast（2d）两个算子天然解决。
* **距离衰减 + 可学习边权**：每条边的权重是 `softmax_j(-dist/tau + W_conn)`，即"按距离的 Boltzmann 分布"再叠加一个可学习的边级修正。

## 四步闭环架构示意

```
x [B, 784]
   │  输入编码：s_in = x @ W_in
   ▼
s_in [B, N*y_in] ──────────────────────────────┐（残差 alpha * s_in）
   │                                            │
   │  2a 稀疏空间连接（masked softmax，按 edge_index[0] 分组）
   │     w_oj = softmax_{j∈conn(o)}( -dist[o,j]/tau + W_conn[o,j] )
   │     s_out[b,o] = Σ_j w_oj * s_in[b,j]        （edge_index + scatter_add）
   ▼
s_out [B, N*y_out]
   │  2b 输出突触 -> 神经元（scatter sum）
   │     neuron_input[b,n] = Σ_{o∈n} s_out[b,o]   （index_add_）
   ▼
neuron_input [B, N]
   │  2c 神经元激活
   │     a = ReLU(neuron_input + neuron_threshold)
   ▼
a [B, N]
   │  2d 神经元 -> 输入突触（broadcast）
   │     s_in_new[b,i] = a[b, neuron_of(i)]        （index_select）
   ▼
s_in_next = LayerNorm(s_in_new + alpha * s_in)  ──┘  回到 2a，共迭代 T 轮

最终轮 s_out ──► logits = s_out @ W_out ──► [B, 10]
```

**归一化方向是硬契约**：必须对每个输出突触 `o`、在其连接的输入突触 `j` 上做 softmax（按 `edge_index[0]` 分组）；禁止对输入突触维度做 softmax，禁止全局 softmax。

## 安装

```bash
pip install -r requirements.txt
```

CPU 版即可运行；有 CUDA 时 `get_device("auto")` 会自动选择 GPU。

## 运行方式

```bash
# 阶段 A：冒烟测试（必须以本命令验收，CPU 上 2 分钟内完成）
# 退出码 0 即代表阶段 A 全部验收项通过，无需任何额外开关
python n3d_proto/train.py --smoke-test

# 阶段 B：正式训练（DEFAULT_CONFIG，10 epoch，保存 checkpoints/model.pt）
python n3d_proto/train.py

# 正式训练并显式指定产物路径（推荐：与默认 model.pt 区分，避免被短跑覆盖）
python n3d_proto/train.py --checkpoint checkpoints/n3d_model_full.pt

# 常用可选参数
python n3d_proto/train.py --epochs 10 --max-batches 20   # 每 epoch 只跑 20 个 batch，CPU 上约 50 秒
python n3d_proto/train.py --device cpu                   # 强制使用 CPU
python n3d_proto/train.py --no-backup                    # 覆盖前不生成 <path>.bak
```

> **环境**：当前环境下 `torch` / `torchvision` 已通过用户 site-packages 中的
> `n3d_pkgs.pth` 自动可见，**普通 `python` 命令无需设置任何环境变量**
> （`python -c "import torch; print(torch.__version__)"` 可直接执行）。
> 唯一已知的环境差异是 `torch_scatter` 未安装——`utils.segment_softmax` 会自动走
> 纯 PyTorch fallback（scatter_reduce(amax) + scatter_add），行为与 torch_scatter 一致。
>
> **数据**：MNIST 使用工程内 `data/mnist/` 的原始 IDX 文件，首次运行会把它们复制到
> `data/mnist/MNIST/raw/`（torchvision 期望的布局，约 11 MB），**不会联网下载**。

> **产物保护**：冒烟测试（`--smoke-test`）与限批验证跑（`--max-batches > 0`）
> **不会覆盖正式 checkpoint**，其产物写入 `checkpoints/_verify/`（冒烟为
> `checkpoints/_verify/smoke.pt`，限批为 `checkpoints/_verify/verify_<N>.pt`），
> 并在日志中明确打印写入位置。此外，正式训练在覆盖已存在的 checkpoint 前会自动
> 备份为 `<path>.bak`（可用 `--no-backup` 关闭）——因为 `checkpoints/` 被
> `.gitignore` 忽略，一旦覆盖便无法按位恢复。
> 因此验证类命令不会覆盖 `checkpoints/n3d_model_full.pt`；默认路径
> `checkpoints/model.pt` 仅在**正式全量训练**（未使用 `--max-batches`）时才会被写入。
> 当前仓库中默认路径**不放任何交付工件**（历史上的一次遗留产物已移到
> `checkpoints/_verify/legacy_model_r1.pt` 留痕），正式交付物为
> `checkpoints/n3d_model_full.pt`。

## 文件说明

| 文件 | 职责 |
|------|------|
| `n3d_proto/config.py` | `@dataclass Config` 全部超参；`SMALL_CONFIG`（冒烟）与 `DEFAULT_CONFIG`（正式训练）两套预设 |
| `n3d_proto/utils.py` | `set_seed` / `get_device` / 日志；`segment_softmax`（torch_scatter 优先，缺失时纯 PyTorch fallback）；`build_edge_index` / `build_scatter_matrix`；`sparse_aggregate` / `sparse_broadcast`；统计 `count_parameters` / `tensor_grad_norms` / `connection_density`（规格口径，`connection_sparsity` 为同义别名）/ `zero_ratio` |
| `n3d_proto/model.py` | `ThreeDNeuronSpace`：坐标与拓扑预计算注册为 buffer，`forward` 实现四步闭环，`sparse_propagate` 实现 2a，含 `count_parameters` / `count_dense_weight_tensors` / `get_connection_stats`（返回 `sparsity` 密度、`zero_ratio`、`num_edges`、`avg_out_degree`、`tau`） |
| `n3d_proto/data.py` | MNIST 加载：**优先复用工程内 `data/mnist/` 的原始 IDX 文件**（必要时复制到 torchvision 期望的 `MNIST/raw/` 位置），完全不联网下载；IDX 惰性加载（`__len__` 不触发图像解析）；`num_workers > 0` 时用 per-worker 播种保证可复现 |
| `n3d_proto/train.py` | 两阶段训练入口：`--smoke-test` 走阶段 A 并逐条打印验收结果（退出码 0 = 通过）；否则走阶段 B 正式训练并保存 checkpoint；`--checkpoint` 指定产物路径，冒烟/限批模式自动隔离到 `checkpoints/_verify/` |

## 一期范围说明

* **纳入**：静态三维拓扑；masked softmax 稀疏空间连接；四步闭环 + 残差 + LayerNorm；MNIST 分类的端到端训练与验收。
* **不纳入**：动态拓扑（突触生长/剪枝/迁移）、结构可塑性、脉冲时序（STDP）、GPU 内核优化。
* **`T >= 2` 的硬约束（实测结论）**：四步闭环中 2d 的输出（`LayerNorm(...)`）只有在"下一轮"的 2a 里才会进入输出通路。若 `T = 1`，末轮的 `ln_s_in` 与 `neuron_threshold` 不参与 loss 计算，其 `.grad` 恒为 `None`，无法满足"所有可学习参数梯度范数 > 0"。因此 `SMALL_CONFIG` 的 `T` 取 2（默认配置为 3）。
* 拓扑量（`dist` / `mask` / `edge_index` / `edge_dist` / `scatter_out_to_neuron` / `broadcast_neuron_to_in`）全部在 `__init__` 中一次性预计算并注册为 buffer，前向传播中不重算。

## 后续展望（二期）

1. **动态拓扑**：让突触坐标与连接掩码可微/可演化（结构可塑性），引入边的生长、剪枝与迁移规则；
2. **多神经元类型与脉冲时序**：引入不应期、膜电位积分泄漏（LIF）与 STDP 局部学习规则；
3. **效率**：引入分块空间哈希（spatial hashing）与 GPU 稀疏内核，支撑更大 N；
4. **性能目标**：在一期机制验证通过的基础上，把 MNIST 准确率推向 ≥ 90%（增大 N、增加 T、调节 lr 与 alpha）。

## 验收标准

**阶段 A（必须达到）**：`python n3d_proto/train.py --smoke-test` 在 CPU 上 2 分钟内完成；前向无 shape mismatch；反向无错误且**所有**可学习参数梯度范数 > 0；loss 不为 NaN/Inf；**连接稀疏度（= 密度 `E / (N*y_out*N*y_in)`，规格口径）< 0.1**；tau 初始值 > 0。该命令退出码为 0 即代表全部通过。

**阶段 B（尽力达到，不阻塞）**：10 个 epoch 后 MNIST 测试准确率 ≥ 90%。若未达标，原因分析与调参建议见下文「阶段 B 结果与分析」。

## 阶段 B 结果与分析

### 实测结果（CPU，torch 2.14.0+cpu）

| 运行方式 | epoch | 每 epoch batch | 最终 test_acc | 耗时 |
|----------|-------|----------------|---------------|------|
| `python n3d_proto/train.py --checkpoint checkpoints/n3d_model_full.pt`（**全量训练**，60000 样本） | 10 | 938（含设备探针提前取走的 1 个 batch，实际训练 937） | **97.53%**（当前工件内记录值；同配置另一次运行实测 97.20%） | ≈ 22–33 min（多次运行区间） |
| `python n3d_proto/train.py --epochs 10 --max-batches 20`（快速验证，产物写入 `_verify/`） | 10 | 20 | 90.70% | 49 s |

连接尺度的对照（同一模型）：**连接稀疏度 = 0.014604**（= 密度 E/(N·y_out·N·y_in) = 61255 / 4194304 ≈ 1.46%，即规格口径），**零元素占比 `zero_ratio` = 0.985396**（≈ 98.54%）。

> **关于 97.53% 与 97.20%**：同一配置的两次全量运行分别得到 97.20%（第 2 轮修复后的复跑）与 97.53%（该产物被测试流程覆盖后按同命令重训恢复所得）。两次均远超 90% 门槛，差异属 CPU 上非逐位确定的正常运行波动（训练种子固定，但底层算子归约顺序等仍可能引入微小差异）。

**以下终端输出块是「97.20% 那次运行」（全量 10 epoch、60000 样本、耗时 1893.2s）的原始记录，与当前仓库中的工件不是同一次运行**（`DEFAULT_CONFIG`：N=256, y_in=y_out=8, T=3, batch=64，边数 E=61255，平均出度 29.91，可学习参数 1691720）：

```
[epoch  1/10] loss=0.2465 test_acc=95.12%
[epoch  2/10] loss=0.1240 test_acc=95.89%
[epoch  3/10] loss=0.0997 test_acc=96.48%
[epoch  4/10] loss=0.0817 test_acc=96.71%
[epoch  5/10] loss=0.0708 test_acc=97.28%
[epoch  6/10] loss=0.0618 test_acc=97.32%
[epoch  7/10] loss=0.0559 test_acc=96.69%
[epoch  8/10] loss=0.0477 test_acc=96.95%
[epoch  9/10] loss=0.0446 test_acc=97.43%   <- 该次运行峰值
[epoch 10/10] loss=0.0385 test_acc=97.20%
checkpoint 已保存：E:\neuron3d\checkpoints\n3d_model_full.pt（总耗时 1893.2s）
最终参数统计：可学习参数总数=1691720；连接稀疏度(密度)=0.014604；零元素占比=0.985396；
             边数 E=61255；平均出度=29.910；tau=0.297663；最终 test_acc=97.20%
```

**当前仓库工件 `checkpoints/n3d_model_full.pt` 的真实统计**（`torch.load` 实测，与上面的终端块不是同一次运行）：

| 字段 | 值 |
|------|-----|
| `test_acc` | **0.9753**（97.53%） |
| `epochs` | 10 |
| `connection_stats['sparsity']`（连接稀疏度 = 密度） | 0.014604 |
| `connection_stats['zero_ratio']` | 0.985396 |
| `connection_stats['avg_out_degree']` | 29.910 |
| `connection_stats['tau']` | 0.28794 |
| `num_edges` | 61255 |
| SHA256 | `888556B0913C9F46419A674117FD13A99F2C71BA6692367A839DB873A58D8924` |


> 上述数字来自第 2 轮修复后的**实际复跑**（`data.py` 取数路径与统计口径均有改动）。按修复计划要求，**以复跑的真实数字为准**；两次运行（97.20% / 97.53%）的统计口径完全一致。
> 说明：**第 1 轮声称的 97.76% 对应工件已不在仓库中**（其 `model.pt` 仅有一次 `--epochs 2 --max-batches 20` 的遗留产物，已移至 `checkpoints/_verify/legacy_model_r1.pt`），**不作为可比基线**。
> 说明：97.20% 那次末轮 test_acc 略低于第 9 轮峰值（97.43%），属于 10 个 epoch 内的正常波动。

tau 从初始 0.808 训练到 0.288~0.298（距离衰减变陡，说明网络倾向于把权重集中到更近的边上）。

### 达标情况

* 阶段 A：**通过**（`--smoke-test` 退出码 0，CPU 单 batch 前向+反向约 5 s；9/9 判据全 PASS：前向无 shape mismatch、反向无错误、7/7 可学习参数梯度范数 > 0、loss 非 NaN/Inf、连接稀疏度 0.0596 < 0.1、tau > 0、耗时 < 120 s，外加「边级参数数 == E」与「无 `[N*y_out, N*y_in]` 权重张量」两条附加断言）。
* 阶段 B：**达到 ≥ 90%**（全量 10 epoch 实测 **97.20% / 97.53%**）。单次耗时 22~33 min（CPU），略超规格中的 30 分钟参考值——该参考值并非硬性门槛（规格中阶段 B 为"尽力达到，不阻塞"），如需压缩可在 CPU 上减少 epoch 或用 `--max-batches` 限批；GPU 环境下单项耗时可大幅下降。

### 结果解读

* 空间稀疏连接（平均出度 ~30 / 2048 个可能输入突触，即每个输出突触只连约 1.5% 的输入突触）已经足以支撑 MNIST 分类到 97%+，说明"距离门控 + 边级权重"的机制本身是可训练的；
* 只用 1 个 epoch 就到达 95.12%，说明该架构的收敛速度很快，瓶颈不在表达力（准确率曲线在 97% 附近趋于平台，属于一次性输入编码 + 无卷积先验的常规上限区间）。

### 继续提升的方向（当前已达标 97.53%）

1. **增大 N**：一期默认 N=256（池内 2048 个输出突触）。空间连接的最大分辨率受 N 限制，增大 N 会同时提高连接数与表达力（代价是 CPU 耗时线性上升）；
2. **增加 T**：`T=3` 时只有 3 次空间传递；增大到 5~8 可让信息在空间上传播更远（梯度会变长，建议同时把 `alpha` 调大一点以增强残差通路）；
3. **调 lr**：`1e-3` 在 Adam 下较稳；收敛慢可试 `3e-3`，出现震荡/diverge 则试 `3e-4`；
4. **调 D / H**：`D=0.15, H=0.1` 对应平均出度约 30（稀疏）。增大 `D` 可提高连接密度与信息通路数，但会削弱"空间稀疏"这一设计初衷，建议同时观察 `get_connection_stats()` 的输出；
5. **增大 batch_size**：64 -> 128/256，可降低梯度噪声（CPU 上内存占用很小）；
6. **训练更久**：一期只跑 10 个 epoch，准确率仍在上升趋势中。

### 连接稀疏度口径与"未 materialize 稠密矩阵"的验证

规格的验收条款是「连接稀疏度 < 0.1」，并在 `get_connection_stats()` 中明确给出公式
**`E / (N*y_out*N*y_in)`**——即**连接密度**（不是 1 − 密度）。代码已按该口径实现。按此口径：

| 配置 | E | 可能连接对 N*y_out*N*y_in | 连接稀疏度（密度，规格口径） | 零元素占比（`zero_ratio`） |
|------|---|--------------------------|------------------------------|----------------------------|
| `SMALL_CONFIG` | 3906 | 65536 | **0.0596**（< 0.1 ✅） | 0.9404（94.04%） |
| `DEFAULT_CONFIG` | 61255 | 4194304 | **0.0146**（< 0.1 ✅） | 0.9854（98.54%） |

`utils.connection_density()` 按上述规格公式实现（`connection_sparsity` 保留为同义别名，**已标注 deprecated**，新代码请统一使用 `connection_density`），另有 `utils.zero_ratio()` 返回 `1 − 密度`，仅用于日志对照、不参与验收判据。`model.get_connection_stats()` 同时返回 `sparsity`（密度）、`zero_ratio`、`num_edges`、`avg_out_degree`、`tau`。

配合规格「禁止 materialize dense `[N*y_out, N*y_in]` 权重矩阵」的约束，冒烟测试还额外校验两条**语义等价且可执行**的判据：

* 边级参数数 == E（例如 `W_conn_sparse.numel() = 61255`，而稠密矩阵需要 4194304 个元素，相差 68 倍）；
* 代码中不存在任何 `[N*y_out, N*y_in]` 形状的**权重张量**（`model.count_dense_weight_tensors()` 返回 0；同形状的 `dist` / `mask` 是预计算几何 buffer，不计入）。

若把连接权重真正 materialize 成稠密矩阵，可学习参数会从 1691720 膨胀到约 580 万（DEFAULT 配置），这正是稀疏边级参数化要避免的代价。
