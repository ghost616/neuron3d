# N3D（三维神经元空间架构）需求设计

## 项目定位
受生物大脑启发的神经网络架构一期原型：神经元分布在三维空间中，每个神经元带 y_in 个输入突触与 y_out 个输出突触，突触自身也有三维坐标；仅当输入突触与输出突触的距离 ≤ D 时才允许传递数据。输入层连接全部神经元的输入突触，输出层连接全部输出突触。

一期采用**静态拓扑**：神经元与突触的三维坐标在初始化后固定不变，仅学习权重参数。目标是验证前向传播与反向传播能否走通，不追求性能。代码位于 `n3d_proto/` 子目录；仓库根目录存放依赖清单与说明文档。

## 权威规范
- 唯一权威规范：`.module_agent/n3d_proto/current_spec.md`，包含数据结构与张量形状总表、四步闭环信息流、masked softmax 归一化方向、稀疏实现约束、验收标准、实现说明与验收口径澄清。

## 首期范围（已交付）
1. `config.py`：`Config` 数据类与两套预设（`SMALL_CONFIG` N=64/y=4,4/T=2/batch=32；`DEFAULT_CONFIG` N=256/y=8,8/T=3/batch=64）。
2. `utils.py`：随机种子、设备选择、日志、`segment_softmax`（含纯 PyTorch fallback）、边表与 scatter/broadcast 映射构建、参数与连接统计。
3. `model.py`：`ThreeDNeuronSpace`，`__init__` 预计算全部拓扑量并注册 buffer，`forward` 实现四步闭环迭代，边级参数化 `W_conn_sparse[E]`，含 `count_parameters` / `get_connection_stats`。
4. `data.py`：MNIST 数据加载（复用仓库 `data/mnist/` 原始 IDX，784 维展平）。
5. `train.py`：两阶段训练入口（阶段 A 冒烟测试 / 阶段 B 正式训练），支持 `--smoke-test`、`--checkpoint`、`--epochs`、`--max-batches` 等参数。

**不在本期**：动态拓扑与结构可塑性（二期方向）。

## 环境
Python 3.12 + PyTorch CPU（torch 2.14.0+cpu / torchvision 0.29.0+cpu）+ numpy；`torch_scatter` 缺失时由纯 PyTorch fallback 承担。

## 工程纪律
- 三件套纪律：代码 patch + change_history 记录 + 冒烟判据覆盖。
- 禁止 materialize dense 权重矩阵，连接一律边级参数化。
- 交付工件 `checkpoints/n3d_model_full.pt` 不得被验证类命令覆盖。
- 运行结果必须真实执行后粘贴，禁止臆造。
