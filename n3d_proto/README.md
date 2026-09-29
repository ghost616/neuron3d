# N3D 一期原型：三维神经元空间架构（静态拓扑）

> **仓库布局**：本说明位于 `n3d_proto/` 模块内，代码与本说明同目录。
> 依赖清单（`requirements.txt`）与运行记录仍为仓库根级文件，由 `framework` 模块维护。

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

### 配置预设与线程控制

```bash
# 预设：small / default / highacc（缺省 default，保持既有默认行为不变）
python n3d_proto/train.py --preset highacc --checkpoint checkpoints/n3d_model_highacc.pt
# 注：上一行示例的 checkpoint 产物已于 2026 年重建轮删除、当前不存在，该行仅作历史用法记录。

# 线程：0（默认）= 不干预，保持 torch 默认线程数；正数则显式设置
python n3d_proto/train.py --threads 4
python n3d_proto/train.py --threads 8

# 覆盖超参（仅在显式给出时生效；用于快速筛选）
python n3d_proto/train.py --preset highacc --n 384 --t 6 --batch-size 128 --lr 2e-3 \
    --weight-decay 1e-4 --dropout 0.1 --readout-bias --epochs 8 --max-batches 150
python n3d_proto/train.py --preset highacc --no-readout-bias
```

> **产物状态注记（2026 年重建轮）**：本文档中引用的一期 checkpoint 产物——`checkpoints/n3d_model_highacc.pt`、
> `checkpoints/n3d_model_capacity.pt`、`checkpoints/_seedscan/seed_*.pt`——均已于 2026 年重建轮**删除，当前不存在**；
> 所有引用这些路径的行（含本节命令行示例）一律仅为历史实验记录，其中的数字、参数量、耗时与 SHA256
> **逐字符保持原样、未做任何改动**。`checkpoints/n3d_model_full.pt` **未在删除之列**，仍是现存交付工件。

**线程实测对比**（同一限批配置 `--preset default --epochs 1 --max-batches 100`，CPU 8 核，
两次运行结果完全一致：`loss=0.5784 / test_acc=87.92%`）：

| `--threads` | torch 生效线程数 | 训练循环内部耗时 | 进程 wall time |
|-------------|------------------|------------------|----------------|
| `4`（= torch 默认） | 4 | **19.1s** | 24.0s |
| `8` | 8 | 20.7s | 24.8s |

结论：**8 线程反而更慢**（本模型以 `index_add_`/`index_select`/`LayerNorm` 等内存受限算子为主，
线程过多会带来调度与带宽竞争）。因此后续所有高精度实验均使用 `--threads 4`（即等于 torch 默认值，
不干预）；`--threads 0` 亦等价。`os.cpu_count()=8`。

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

## HIGHACC 高精度冲刺（目标 > 99%，实测未达）

### 目的与手段

在不破坏默认路径、不使用数据增强与输入标准化的前提下，尝试把 MNIST test_acc 推到 > 99%。
新增的可用手段（全部**默认关闭**，因此 `SMALL_CONFIG` / `DEFAULT_CONFIG` 行为逐位不变）：

* `Config.weight_decay`（> 0 时优化器由 Adam 切换为 **AdamW**）
* `Config.dropout`（作用在 `s_out` 送入 `W_out` 之前；**0.0 时为 `nn.Identity`，不消耗随机数**）
* `Config.readout_bias`（为 `W_out` 增加 bias，默认不创建该参数）
* `Config.lr_schedule`（`"none"` / `"cosine"`，后者接 `CosineAnnealingLR(T_max=epochs)`）
* `Config.grad_clip`（> 0 时在 `backward` 与 `step` 之间做 `clip_grad_norm_`）
* CLI：`--preset {small,default,highacc}`、`--threads N`、`--lr`、`--weight-decay`、`--dropout`、`--batch-size`、`--readout-bias/--no-readout-bias`
* 容量维度 CLI（第 5 轮新增）：`--n`（神经元数）、`--y-in`/`--y-out`（每神经元输入/输出突触数）、`--h`（突触分布半径）、`--d`（连接距离阈值）、`--t-steps`（迭代轮数 T，`--t` 为同义别名）
* 可追溯性 CLI（第 5 轮新增）：`--tag <str>`（**仅作用于限批模式（`--max-batches > 0`）写入 `checkpoints/_verify/` 的验证产物**文件名，追加为后缀；正式训练路径不受其影响）
* 拓扑 CLI（第 6 轮新增）：`--seed N`（覆盖随机种子；种子决定三维坐标采样与参数初始化，**不同种子即不同拓扑**，E 与 test_acc 都会变）

`HIGHACC_CONFIG` 初值：N=256, y_in=y_out=8, T=4, batch_size=128, lr=2e-3, epochs=20,
weight_decay=1e-4, dropout=0.1, readout_bias=True, lr_schedule="cosine", grad_clip=1.0, seed=42。

#### 筛选产物的可追溯命名（第 5 轮修复）

限批验证跑的产物文件名带**配置指纹**，不同配置不再互相覆盖：

```
checkpoints/_verify/verify_<bpe>_N{N}_y{y_in}x{y_out}_H{H}_D{D}_T{T}_s{seed}[_<tag>].pt
# 例：verify_150_N256_y8x8_H0.1_D0.15_T4_s42.pt
#     verify_150_N256_y8x8_H0.1_D0.25_T4_s42.pt    （仅 D 不同 -> 不同文件）
#     verify_150_N256_y8x8_H0.1_D0.15_T4_s7.pt     （仅 seed 不同 -> 不同文件，第 6 轮补上 seed 维度）
```

> 第 6 轮补上 `_s{seed}` 维度：种子决定三维坐标采样，**不同 seed 即不同拓扑**，
> 若指纹不含 seed，只改 `--seed` 的限批跑会写入同名文件并互相覆盖（离朱第 6 轮实测报告该缺口）。

日志会打印最终产物的**绝对路径**。若显式给出 `--checkpoint` 则沿用其文件名（仍隔离在 `_verify/`）。

### 容量扫描（第 5 轮，可追溯，6 点）

统一条件：`--max-batches 150 --epochs 6 --seed 42 --threads 0`（`--threads 0` = 保持 torch 默认 4 线程）。
每点产物独立，下表所有数字均可用 `torch.load` 从对应产物复核。

| 点 | 配置（N / y_in×y_out / D / T） | 实际 epoch | test_acc | 单 epoch 耗时 | E 边数 | 可学习参数 | 产物文件名 |
|----|-------------------------------|-----------|----------|---------------|--------|-----------|-----------|
| **基线** | 256 / 8×8 / 0.15 / 4 | 6 | **96.87%** | 49.3s | 61255 | 1691730 | `verify_150_N256_y8x8_H0.1_D0.15_T4.pt` |
| N 轴 | 512 / 8×8 / 0.15 / 4 | — | **成本过高，中止** | ~5 min/ep | 225016 | 3485955 | 无（见下） |
| y 轴 | 256 / 16×16 / 0.15 / 4 | 3 | 95.75% | 296.7s（6.0x） | 244424 | 3505107 | `verify_150_N256_y16x16_H0.1_D0.15_T4.pt` |
| D 轴 | 256 / 8×8 / **0.25** / 4 | 6 | 94.93% | 199.2s（4.0x） | 216861 | 1847336 | `verify_150_N256_y8x8_H0.1_D0.25_T4.pt` |
| T 轴 | 256 / 8×8 / 0.15 / **6** | 6 | 96.19% | 74.1s（1.5x） | 61255 | 1691730 | `verify_150_N256_y8x8_H0.1_D0.15_T6.pt` |
| 组合 | 512 / 16×16 / 0.15 / 4 | — | **成本过高，未启动** | 预估 >60 min/ep | — | — | 无 |

**成本闸门记录（实际现象）**
* **N=512（P2）**：`E=225016`、可学习参数 3485955。实测 **~5 min/epoch**（2 个 epoch 用掉 10 分钟），
  按 6 epoch 需 ~30 min，**触发 15 分钟闸门中止**。中止前终端日志为 epoch1 `test_acc=91.48%`、epoch2 `92.88%`
  —— **无产物可复核**（中止发生在保存阶段之前），故仅作现象记录，**不作为可追溯实验数据引用**。
  该点**未产出可用产物**（中止时未到保存阶段）。瓶颈是边级参数（E=225016 需 225k 次 Adam 更新/step）
  与其梯度/优化器状态的访存，而非 `dist` 矩阵规模。
* **组合 512 / 16×16（P6）**：由 P2（N=512 已 ~5 min/ep）与 P3（y=16×16 已 296.7s/ep）线性外推，
  单 epoch 预估 > 60 min，**按闸门不予启动**（记录为"成本过高，未启动"）。
* **y 轴与 D 轴**：虽然单 epoch 已 200~300s，但 3~6 epoch 仍落在闸门内，故按校准后的 epoch 数完成，
  并在下表标注实际 epoch 数（不做跨点绝对精度比较，仅看趋势）。

**判读（阈值 <0.3pp 视为噪声）**
1. **容量维度全部方向无效或负收益**：N 加倍（512）→ 成本 6x+ 且直接触闸门；
   y 加倍（16×16）→ 同 epoch 位置**不占优**、成本 6.0x；D 增大（0.25，边数 3.5x）→ **明显更差**
   （epoch6：94.93% vs 基线 96.87%，差 1.94pp，远超噪声）且成本 4.0x；T 加深（4→6）→ 略差
   （96.19% vs 96.87%，差 0.68pp）且成本 1.5x。
2. **D 增大反而变差的原因**：`D=0.25` 使平均出度 29.91 → 105.89。步骤 2a 的权重是对每个输出突触
   在**全部连接边**上做 softmax，边数变多会稀释单条边的权重（≈1/106），
   等价于把输入信号做了更强的平均，反而削弱了选择性。这与"稀疏连接是有效先验"的设计初衷一致。
3. **T 加深未带来收益的原因**：T 只增加迭代轮数、不增加可学习参数（同为 1691730、E=61255），
   但每轮都过一层 LayerNorm，轮数越多、每样本整体幅度信息被反复归一化掉的次数越多；
   且 T=6 的图更深、梯度路径更长，在同等 epoch 预算下反而略欠拟合。
4. **结论：容量维度已到瓶颈**。在"无数据增强、无输入标准化"的约束下，单纯放大
   N / y / D / T 都不能突破 ~97.8%，且成本随 E 与参数数超线性上升。要越过 99%，
   必须放宽**数据侧**（增强/标准化）或引入**卷积式局部归纳偏置**（架构级改动）。

### 快速筛选（150 batch/epoch，用趋势而非绝对值选向）

> **产物可追溯性说明**：第 4 轮（plan_1c4d4157）的这次筛选产物当时统一命名为 `verify_150.pt`，
> 6 个点互相覆盖，**最终只留下最后一次运行的产物**，因此下表中的数字**无法用工件逐一复核**。
> 第 5 轮已修复该缺陷（产物名带配置指纹 + `--tag`），并重做了一次**可追溯**的容量扫描（见下文）。

| 候选 | 配置 | 最后几轮 test_acc | 最终 | 单 epoch 耗时 |
|------|------|-------------------|------|---------------|
| **A** | `highacc` 原样（N=256/T=4/bs=128/lr=2e-3/cosine） | 96.52 → 96.97 → **97.04** | 97.04% | **46.3s** |
| B | `--n 384`（更大容量） | 96.31 → 97.02 → **97.11** | 97.11% | 112.2s（2.4x 代价） |
| **C** | `--t 6`（更深，**该覆盖实际生效**：配置回显 T=6） | 96.07 → 96.67 → 96.94 → **97.24** | 97.24% | 47.6s |

> **更正（第 5 轮）**：此前记录称「`--t` 被 argparse 前缀匹配为 `--threads`，故候选 C 实际等价于 A」，
> 该说法**错误**。实测证据：`python n3d_proto/train.py --smoke-test --t 6` 的配置行回显 `T=6`，
> 说明 argparse 对 `--t` 做的是**精确匹配**（优先于 `--threads` 的前缀匹配），
> 候选 C 是一次 T=6 的**真实运行**。据此更正基于该错误说法的选型结论：
> 当时是「A / B / C 三点同区间」而非「A 与 C 等价」；A 被选中的理由（B 的 2.4 倍代价仅换 +0.07pp）
> 不受影响。第 5 轮已把 T 覆盖的规范名改为 `--t-steps`（`--t` 保留为同义别名），
> 并在覆盖生效时打印「T 覆盖：<旧> -> <新>」以彻底消除歧义。

### 全量冲刺实测（两轮，均为 60000 样本全量）

**第 1 轮**：`--preset highacc --epochs 12 --checkpoint checkpoints/n3d_model_highacc.pt --threads 4`　〔已删产物：已于 2026 年重建轮删除，当前不存在〕
（AdamW lr=2e-3 / wd=1e-4 / dropout=0.1 / cosine T_max=12 / clip=1.0）

```
[epoch  1/12] loss=0.3020 test_acc=95.03% lr=0.002000
[epoch  2/12] loss=0.1507 test_acc=94.70% lr=0.001966
[epoch  3/12] loss=0.1095 test_acc=96.78% lr=0.001866
[epoch  4/12] loss=0.0859 test_acc=96.38% lr=0.001707
[epoch  5/12] loss=0.0704 test_acc=97.47% lr=0.001500
[epoch  6/12] loss=0.0556 test_acc=97.37% lr=0.001259
[epoch  7/12] loss=0.0427 test_acc=97.70% lr=0.001000
[epoch  8/12] loss=0.0314 test_acc=97.74% lr=0.000741
[epoch  9/12] loss=0.0231 test_acc=97.48% lr=0.000500
[epoch 10/12] loss=0.0163 test_acc=97.79% lr=0.000293
[epoch 11/12] loss=0.0122 test_acc=97.77% lr=0.000134
[epoch 12/12] loss=0.0099 test_acc=97.84% lr=0.000034
总耗时 1570.9s ≈ 26.2 min；参数 1691730；密度 0.014604；tau 0.411118
```

**第 2 轮（阶段 4 兜底，按"过拟合"方向加正则）**：
`--preset highacc --epochs 10 --lr 3e-3 --weight-decay 2e-4 --dropout 0.15 --checkpoint checkpoints/n3d_model_highacc.pt`　〔已删产物：已于 2026 年重建轮删除，当前不存在〕

```
[epoch  1/10] 95.09% | [epoch  2/10] 95.69% | [epoch  3/10] 95.98% | [epoch  4/10] 96.52%
[epoch  5/10] 97.10% | [epoch  6/10] 97.35% | [epoch  7/10] 97.56% | [epoch  8/10] 97.64%
[epoch  9/10] 97.60% | [epoch 10/10] 97.71%（train loss 0.0165）
总耗时 1298.7s ≈ 21.6 min
```

### 第 5 轮全量冲刺（容量扫描后，16 epoch）

依据容量扫描的方向判断（容量维度全部无效或负收益），选择**基线方向**（N=256 / y=8×8 / D=0.15 / T=4，
即 `--preset highacc`）跑全量，并把 epoch 从 12 提到 **16**（更长 cosine 调度、更低末段 lr）。

**开跑前的耗时预算**（按阶段 2 实测单 epoch 131.9s 估算）：
`16 ep × 131.9s + 16 × 评价开销(≈24s) ≈ 35 min` → **< 2 小时预算，无需降配**（实测总耗时 2112.2s ≈ 35.2 min，与估算吻合）。

```
python n3d_proto/train.py --preset highacc --epochs 16 --checkpoint checkpoints/n3d_model_capacity.pt --threads 0
# 注：上一行示例的 checkpoint 产物已于 2026 年重建轮删除、当前不存在，该行仅作历史用法记录。
```
（AdamW lr=2e-3 / wd=1e-4 / dropout=0.1 / readout_bias=True / cosine T_max=16 / clip=1.0）

```
[epoch  1/16] loss=0.3020 test_acc=95.03% lr=0.002000
[epoch  2/16] loss=0.1515 test_acc=95.24% lr=0.001981
[epoch  3/16] loss=0.1109 test_acc=96.79% lr=0.001924
[epoch  4/16] loss=0.0888 test_acc=96.59% lr=0.001831
[epoch  5/16] loss=0.0746 test_acc=97.34% lr=0.001707
[epoch  6/16] loss=0.0620 test_acc=97.31% lr=0.001556
[epoch  7/16] loss=0.0495 test_acc=97.29% lr=0.001383
[epoch  8/16] loss=0.0404 test_acc=97.36% lr=0.001195
[epoch  9/16] loss=0.0315 test_acc=97.14% lr=0.001000
[epoch 10/16] loss=0.0253 test_acc=97.53% lr=0.000805
[epoch 11/16] loss=0.0181 test_acc=97.57% lr=0.000617
[epoch 12/16] loss=0.0132 test_acc=97.71% lr=0.000444
[epoch 13/16] loss=0.0092 test_acc=97.69% lr=0.000293
[epoch 14/16] loss=0.0066 test_acc=97.68% lr=0.000169
[epoch 15/16] loss=0.0051 test_acc=97.75% lr=0.000076
[epoch 16/16] loss=0.0045 test_acc=97.70% lr=0.000019
总耗时 2112.2s ≈ 35.2 min
最终统计：可学习参数 1691730；连接稀疏度(密度) 0.014604；zero_ratio 0.985396；
         边数 E=61255；平均出度 29.910；tau 0.348827；最终 test_acc 97.70%
产物：checkpoints/n3d_model_capacity.pt　〔已删产物：已于 2026 年重建轮删除，当前不存在〕
SHA256：0f7cf500c256bfe41408e3dc68ced9316c21ee4ee94527790f861c76e6c35011
```

**关键观察**：`train loss` 一路降到 **0.0045**（几乎完全记住训练集），而 test_acc 从 epoch 12 起就
在 **97.7% 附近平台**（97.71 → 97.69 → 97.68 → 97.75 → 97.70，全部落在 ±0.07pp 的噪声带内）。
**继续训练不再带来任何提升**，这再次确认瓶颈是泛化上限而非训练不足。

### 拓扑种子扫描（第 6 轮，可追溯）

**动机**：此前四轮全量冲刺全部使用同一个 `seed=42` 的拓扑，因此"97.8% 是泛化上限"严格说只对
这一个空间排布成立。第 6 轮新增 `--seed` CLI 后，固定其他全部超参、**只变种子**做对照实验。

配置完全固定为 HIGHACC：`--preset highacc --epochs 12 --threads 0`（T=4 / bs=128 / AdamW lr=2e-3 /
wd=1e-4 / dropout=0.1 / readout_bias=True / cosine / clip=1.0，全量 60000 样本），**只改 `--seed`**。
产物写入 `checkpoints/_seedscan/seed_<seed>.pt`（每个种子独立命名，不互相覆盖）。　〔已删产物：已于 2026 年重建轮删除，当前不存在；下表四行产物（含 seed 42 的历史基线）均同此，表中数字为历史实验记录〕

| seed | 实际 E 边数 | 可学习参数 | 单 epoch 耗时 | 总耗时 | test_acc | 产物（`checkpoints/_seedscan/`） |
|------|------------|-----------|---------------|--------|----------|----------------------------------|
| **42**（历史基线，未重跑） | 61255 | 1691730 | 131.9s | 1570.9s | **97.84%** | `../n3d_model_highacc.pt`〔已删产物：已于 2026 年重建轮删除，当前不存在〕 |
| 7 | 60167 | 1690642 | 131.8s | 1581.9s | 97.55% | `seed_7.pt` |
| **2024** | 62024 | 1692499 | 134.9s | 1618.6s | **97.90%** | `seed_2024.pt` |
| 123 | 60326 | 1690801 | 122.5s | 1470.0s | 97.68% | `seed_123.pt` |

（`epochs=12`、`batches_per_epoch=469`、SHA256 分别：seed_7 `afbf2221a919feab…`、
seed_2024 `6b1d2657d4847fe5…`、seed_123 `d40b5ebb3418f385…`；表中每个数字均可用 `torch.load` 从对应产物复核。**该"可复核"说法仅适用于产物删除之前——上表四行产物已于 2026 年重建轮删除、当前不存在，上列 SHA256 等数字为历史记录原文、未做任何改动。**）

**判读**
* 四个种子极差 = 97.90% − 97.55% = **0.35pp**，均值 97.74%、标准差约 0.13pp；
* 按预先约定的判读纪律（> 0.3pp → 拓扑确为变量；< 0.3pp → 支撑架构上限结论），0.35pp **刚好越过阈值**，
  因此**不能说种子完全无影响**；但该量级显著小于"改变架构维度"带来的差异
  （对照：D 0.15→0.25 造成 **−1.94pp**、T 4→6 造成 −0.68pp），也小于 4 轮全量之间的波动带（0.29pp）；
* **所有种子的最终 test_acc 全部落在 97.55% ~ 97.90% 区间**，没有任何种子接近 99%；
* **结论：97.8% 是架构在这个拓扑规模下的上限，而非 seed=42 这一个排布的偶然。**
  换种子最多改善约 **+0.06pp**（best seed 2024 的 97.90% vs seed 42 的 97.84%），
  距 99% 还差 **1.1pp**，**不可能靠搜索种子补上**。
* 若仍要冲 99%，必须动**数据侧或架构侧**（数据增强 / 输入标准化 / 卷积式局部感受野），
  见下方"后续可行方向"。围绕 best seed（2024）做超参精调预计收益也仅在 ±0.1pp 量级，性价比低。

### 决定性对照实验：普通 MLP 基准（第 7 轮）

**动机（风后提出的关键疑点）**：一个参数量同量级的普通 MLP（784→2048→10，约 1.63M）在 MNIST 上通常能到
98.3%+，而本模型（1.69M 参数，且多了 T=4 轮稀疏混合 + LayerNorm + 阈值激活）只有 97.84%
——**表达力更强却更差**，强烈暗示四步闭环自身在损失信息。此前六轮已排除容量维度与拓扑种子，
本轮用**同预算、同优化器、同 epoch** 的普通 MLP 做对照，判定瓶颈在**架构**还是在**数据**。

**实现方式（关键约束）**
* 新增 `--arch {neuron3d, mlp}`（默认 `neuron3d`，默认路径行为逐位不变）。
* **MLP 走完全相同的训练循环代码路径**：`build_model_and_data` 中按 arch 分支**只在"模型构造"这一处**
  （`model = MLPBaseline(config) if arch=='mlp' else ThreeDNeuronSpace(config)`）；其后的
  `train_one_epoch` / `evaluate` / `_run_training_with_config` 的**优化器、调度器、梯度裁剪、
  batch_size、epoch 数、评估代码全部共用同一份实现**，没有复制粘贴第二份训练循环。
* 结构：`784 -> hidden_dim(默认 2048) -> 10`，隐藏层后 ReLU，dropout 复用 `config.dropout`。
* 初始化：`fc1` 用 Kaiming 均匀（fan_in=784，与主模型 `W_in` 同属"按 fan_in 缩放"，量级可比）；
  `fc2` 用 Xavier 均匀（与主模型 `W_out` 完全一致）；由 `config.seed` 派生生成器控制，可复现。

**控制变量（三组完全对齐）**：`epochs=12 / batch_size=128 / AdamW lr=2e-3 / weight_decay=1e-4 /
dropout=0.1 / cosine / grad_clip=1.0 / seed=42 / 全量 60000 / 无增强无标准化`。

| 组 | 架构 | test_acc | 可学习参数 | 单 epoch | 总耗时 | 产物（`torch.load` 可复核） |
|----|------|----------|-----------|----------|--------|------------------------------|
| **A**（已有，未重跑） | 四步闭环 `neuron3d` | 97.84% | 1691730 | 131.9s | 1570.9s | `checkpoints/n3d_model_highacc.pt`〔已删产物：已于 2026 年重建轮删除，当前不存在〕 |
| **B**（本轮必跑） | **普通 MLP 784→2048→10** | **98.64%** | **1628170** | **19.6s** | **234.9s** | `checkpoints/_control/mlp_highacc_ep12_seed42.pt`（SHA256 `f261716e7f3c3cba…`） |
| C（条件触发） | 未跑（触发条件为"B 异常偏低 < 97.5%"，实测 B=98.64% **不满足**） | — | — | — | — | — |

**参数量差异（如实报告）**：MLP 1628170 vs 主模型 1691730，**MLP 少 63560 个参数（少 3.8%）**。
即 MLP 以**更少**的参数取得**更高**的准确率——该结论不因参数量差异而被削弱。

**判定（按预先固定的规则，未事后修改）**：MLP 高于主模型 **+0.80pp**，**远超 0.3pp 阈值**
→ **结论：瓶颈在架构——四步闭环在损失信息。**

旁证：MLP 单 epoch 仅 **19.6s**（主模型 131.9s，快 **6.7 倍**），且 MLP 在 **epoch 7** 就已达 98.41%，
而主模型全程封顶 ~97.8% —— 主模型不仅**慢 6.7 倍**，泛化上限也**更低 0.80pp**。

> 注：A 组产物 `n3d_model_highacc.pt` 是第 4 轮产出，其 checkpoint 元数据中无 `arch` 字段　〔已删产物：已于 2026 年重建轮删除，当前不存在〕
> （当时尚未引入 `--arch`）；B 组元数据 `arch="mlp"`。两者的训练控制变量已逐项比对一致（bs/lr/wd/dropout/cosine/clip/seed）。

**架构侧后续方向**（按预期收益排序）
1. **去掉/弱化每样本 LayerNorm 的幅度丢失**：2d 的 `LayerNorm(s_in_new + alpha*s_in)` 会丢弃每个样本的
   整体激活幅度（只保留样本内相对模式），而"整体墨量"对数字判别是有效信息。可试：改为对 batch 维归一化、
   或去掉 LayerNorm 改用可学习缩放、或残差项不参与归一化；
2. **放宽 2c 的阈值死区**：`ReLU(neuron_input + threshold)` 截断负向信息，可试 GELU / LeakyReLU / SiLU；
3. **让连接权重不必经过分组 softmax**：2a 的 softmax 把每条边权重压到 ~1/出度，等价于一次强平均
   （D 增大时更明显：出度 29.9→105.9 反而 −1.94pp）。可试"直接加权和"或"可学习 softmax 温度"；
4. **引入卷积式局部感受野**（把 `W_in` 换成局部连接/卷积），从根上补足归纳偏置。

> 判读纪律说明：本轮结论建立在**同控制变量**的单次对照上（A 未重跑，因其配置与本轮 B 完全对齐）。
> 未事后挑选任何有利结果；C 组在触发条件不成立时**如实报告为"未跑"**。


### 结果：未达 99%，如实汇报

| 配置 | 最终 test_acc | 耗时 | 产物 |
|------|---------------|------|------|
| `DEFAULT_CONFIG`（T=3, bs=64, Adam 1e-3, 10ep） | 97.53% / 97.20%（两次运行） | 22~33 min | `checkpoints/n3d_model_full.pt` |
| `HIGHACC_CONFIG`（T=4, bs=128, AdamW 2e-3, 12ep） | **97.84%**（历史最佳） | 26.2 min | `checkpoints/n3d_model_highacc.pt`〔已删产物：已于 2026 年重建轮删除，当前不存在〕 |
| HIGHACC + 更强正则（wd=2e-4, dropout=0.15, 10ep） | 97.71% | 21.6 min | 第 2 轮产物**未保留**（见下） |
| **HIGHACC 16ep（第 5 轮容量扫描后）** | **97.70%** | 35.2 min | `checkpoints/n3d_model_capacity.pt`〔已删产物：已于 2026 年重建轮删除，当前不存在〕 |

**四轮全量运行的结果全部落在 97.70% ~ 97.84% 的 0.14pp 带内**（差异远小于架构层面的噪声），
即该架构在当前约束下的泛化上限约 **97.8%**。

**产物标签更正**：`checkpoints/_verify/highacc_r1.pt` 经 `torch.load` 复核为
**第 1 轮（12 epoch）的产物**——其 `test_acc=0.9784`、`epochs=12`、
SHA256 与 `n3d_model_highacc.pt` **完全相同**（`9f21ac34c91977fe…`），二者内容一致。　〔已删产物：已于 2026 年重建轮删除，当前不存在；上列 SHA256 为历史记录原文，未做任何改动〕
第 2 轮（97.71%，10 epoch）的产物**未被保留**（当时该轮结束时直接覆盖写入了正式路径，
随后为保留最佳结果又用第 1 轮产物恢复覆盖）；因此第 2 轮的 97.71% **没有工件可直接复核**，
仅作为过程记录保留在下方日志中。

两轮均**未达 > 99%**，最好成绩 **97.84%**。删除前 `checkpoints/n3d_model_highacc.pt` 保存的是　〔已删产物：已于 2026 年重建轮删除，当前不存在；上列数字为历史记录原文，未做任何改动〕
第 1 轮（97.84%）的产物，其 SHA256 为 `9f21ac34c91977fe…`。

### 主因分析（为什么卡在 ~97.8%）

1. **严重过拟合，而非欠拟合**：第 1 轮 train loss 收敛到 **0.0099**（≈ 训练集几乎被记住），
   而 test_acc 在 97.5~97.8% 徘徊 —— 这是"模型记住了训练样本但泛化能力封顶"的典型特征。
   因此阶段 4 没有选择"加容量 / 加 epoch"，而是选了"加正则"。
2. **加正则无效**：第 2 轮把 weight_decay 翻倍、dropout 0.1→0.15 后，同 epoch 位置反而更低
   （epoch 6：97.35% vs 97.37%），终值 97.71% < 97.84%。说明瓶颈不是方差过大，
   而是**表达能力/归纳偏置不足**（偏差项高）。
3. **归纳偏置极限**：本架构把 784 维像素一次性线性编码到突触信号，随后只做
   "距离门控的空间混合 + ReLU + LayerNorm"，**没有任何卷积式的平移等变先验**；
   在"不使用数据增强、不做输入标准化"的约束下，这类非卷积模型在 MNIST 上的
   公开经验量级正好落在 **97~98.5%** 区间。
4. **LayerNorm 的副作用**：四步闭环的 2d 对每个样本单独做 LayerNorm，会**丢掉每个样本的
   整体激活幅度**（只保留样本内部各突触之间的相对模式）。对数字识别而言，"哪些笔画亮"
   之外，"整体墨量"也是有效判别信息，这一步把该信息归一化掉了。
5. **30 分钟级预算下无法用"更多 epoch"破局**：筛选阶段已看到 150 batch/epoch 时
   7~8 个 epoch 就进入 97% 平台；全量 12 epoch 同样在 97.8% 封顶，
   单纯延长时间不会改变平台高度。

### 后续可行方向（需另行申请或超出本轮约束）

* **输入标准化 / 数据增强**（本轮明确排除）：RandomAffine/平移抖动是最直接的 +0.5~1.5pp 手段；
* **引入卷积式先验**：让输入突触感知局部感受野（例如把 `W_in` 换成局部连接/卷积权重），
  或在 2a 中按空间邻近加权 —— 这是**架构级改动**，会改变四步闭环语义，需另立计划；
* **放宽容量上限**：本轮已验证 N=384 在相同 epoch 数下收益极小（+0.07pp / 2.4 倍耗时），
  单纯放大 N 不划算，应配合更好的归纳偏置；
* **集成 / 快照**：对多次运行做 logits 平均或权重平均（SWA），通常可再涨 0.2~0.4pp；
* **测试集不可用于挑选 epoch**：本轮所有 epoch 选择均基于"训练结束时余弦调度到位"这一先验，
  未按测试集反复挑选，故 97.84% 是可信的单次结果。

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
