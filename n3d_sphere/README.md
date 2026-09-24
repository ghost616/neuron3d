# n3d_sphere：纯球形分层有向无环架构（FCC 规则堆积 + 神经元级连接 + 两阶段双副本展开）

本模块是 N3D 的**纯球形分层 DAG 架构**实现：神经元按 FCC（面心立方）规则堆积放置在
球空间内，突触按全局流向轴切分到正/负半球，连接规则强制沿流向轴严格上行，因此网络
天然是**有向无环图（DAG）**。本模块不含任何其它几何分支，也没有几何类型开关。

> **与一期 `n3d_proto` 的关系**：本模块**自包含**（不 import `n3d_proto` 的任何模块）。
> 一期完整存档、默认行为与既有产物逐位不变 —— `python n3d_proto/train.py --smoke-test`
> 仍为 9/9 PASS、退出码 0、`loss = 2.419689`（本轮复跑取证见第 7.1 节）。本模块的任何命令
> 都**不会**写入一期产物目录 `checkpoints/` 根下的 `n3d_model_*.pt`。

---

## 1. 几何与尺度

### 1.1 球空间半径由公式唯一确定

神经元是半径 `H` 的球（突触云分布半径）。按最优堆积系数 `φ = 0.7405`，可非重叠放入
`N` 个半径 `H` 的球所需球空间半径为：

```
R_min = H · (N / φ)^(1/3)
R_max = (H + D) · (N / φ)^(1/3)
```

* `R_min`：非重叠容纳下界；
* `R_max`：保证每个神经元的 `D` 邻域仍完整落在球空间内的上界（神经元最远距球心 `R`，
  其突触云再外扩 `H`，连接判据再看 `D`）。

### 1.2 `D <= H` 硬约束（连接半径不得超过接收/发送范围半径）

**约束**：连接半径 `D` 不得超过**接收/发送范围半径** `H`。

几何含义：连接判据是"起点神经元的输出突触 `o` 与终点神经元的输入突触 `j` 的距离
`<= D`"，而 `o` / `j` 各自落在**所属神经元的 `H` 半径球**内（且分别在流向轴正/负半球）。
`D > H` 意味着**连接半径超过突触云自身的尺度**，连接不再受"接收/发送范围"约束，属越界
配置 —— 故 `Config.__post_init__` 直接抛 `ValueError`（G1），不留给下游去发现。

* 落地口径：**硬校验 + 三个预设一律取 `D = H`**（G2）——
  `SMALL` `H=0.15/D=0.15`、`DEFAULT` `H=0.10/D=0.10`、`HIGHACC` `H=0.10/D=0.10`；
  数据类字段默认值亦为 `D = H = 0.1`。
* CLI `--d` 的帮助文本已注明 `D <= H`；判据与负例取证见
  `_verify/verify_dh_constraint.py` 的 **H1-H4**（`D=0.15>H=0.10`、`D=0.25>H=0.15`
  均必须报错；`D == H` 与 `D < H` 正常构造）。
* **连带影响**：`R_max = (H+D)·(N/φ)^(1/3)` 在 `D = H` 时降为改造前的一半
  （DEFAULT：`1.754601 → 1.403681`；SMALL：`1.768527 → 1.326395`）。FCC 放置半径
  （`0.721110` / `0.670820`）仍远小于新的 `R_max`，故 C3 的容纳性断言依旧成立。

### 1.3 连通性下限校验（防止 `D <= H` 下取过小的 `D` 造成图退化）

`D` 变小会显著减少连接：实测（N=256/y=8×8/H=0.10/seed=42/any-any，`verify_dh_constraint.py`
的 H4 现跑）——

| H | D | E | E/N | 层数 K | \|S_in\| | \|S_out\| | 结论 |
|---|---|---|---|---|---|---|---|
| 0.10 | **0.10**（本轮预设口径） | **736** | **2.8750** | 9 | 193 | 187 | 通过下限校验 |
| 0.10 | 0.05 | 146 | 0.5703 | 9 | 256 | 256 | **被下限校验拦下** |
| 0.10 | 0.03 | 17 | 0.0664 | 9 | 256 | 256 | **被下限校验拦下** |
| 0.10 | 0.02 | 1 | 0.0039 | 9 | 256 | 256 | **被下限校验拦下** |

因此 `ThreeDNeuronSpace.__init__` 在构图完成后执行**连通性下限校验（G3）**，
任一不满足即抛 `ValueError`（消息含实测的 `E / N / (E÷N) / 层数K / |S_in| / |S_out| / H / D`）：

* `E >= N`（平均出度 `E/N >= 1`）；
* 层数 `K >= 2`（阶段 2 至少要有跨层传播）；
* `|S_in| >= 1` 且 `|S_out| >= 1`（阶段 1 有驱动、readout 有信号）。

### 1.4 `D = H` 改造前后对照（同一测点：N=256/y=8×8/seed=42/any-any）

| 配置 | \|S_in\| | \|S_out\| | E | 层 K | E/N | params |
|---|---|---|---|---|---|---|
| 改造前 `H=0.10/D=0.15`（**违反 `D <= H`**，现已不可构造） | 50 | 45 | 903 | 9 | 3.53 | 42,919 |
| 改造后 `H=0.10/D=0.10`（本轮预设口径） | **193** | **187** | **736** | 9 | **2.88** | **154,864** |
| 改造前 SMALL `H=0.15/D=0.25`（**违反 `D <= H`**，现已不可构造） | 13 | 17 | 181 | 7 | 2.83 | 11,077 |
| 改造后 SMALL `H=0.15/D=0.15`（本轮预设口径） | **55** | **53** | **106** | 7 | **1.66** | **43,930** |

> “改造前”两行由 `_verify/legacy_dh_baseline.json` 固化（来源为其登记的改造前产物与 10-seed
> 快照及其 SHA256；这些配置因 `D > H` 现已无法由 `Config` 构造，故必须留档）；
> “改造后”两行来自 `_verify/verify_dh_constraint.py` 的 H3 现跑。
> `|S_in|` 从 50 涨到 193 会连带把 `W_in` 从 `784×50` 撑到 `784×193`，DEFAULT 参数量因此
> 从 42,919 涨到 **154,864（约 ×3.6）** —— 这是本约束最直接的成本。

`space_radius` 默认取 `0.0`，表示**使用 `R_min`**；显式传入时必须落在 `[R_min, R_max]`
内，越界在 `Config` 构造期直接报错（两侧越界均有断言，见
`_verify/verify_config_contracts.py` 的 C2）。

实测（`_verify/verify_config_contracts.py` C1 实时计算）：

| N | H | D | R_min | R_max | R_max/R_min | 期望 (H+D)/H |
|---|---|---|---|---|---|---|
| 256 | 0.10 | 0.10 | 0.701840 | 1.403681 | 2.000000 | 2.000000 |
| 256 | 0.10 | 0.05 | 0.701840 | 1.052760 | 1.500000 | 1.500000 |
| 64 | 0.15 | 0.15 | 0.663198 | 1.326395 | 2.000000 | 2.000000 |
| 256 | 0.06 | 0.06 | 0.421104 | 0.842208 | 2.000000 | 2.000000 |

### 1.5 神经元放置 = FCC 规则堆积（确定性，与 seed 无关）

晶格常数 `a = 2√2·H`，基元 `{(0,0,0), (½,½,0), (½,0,½), (0,½,½)}`，因此**最近邻距恰为
`a/√2 = 2H`**：相邻神经元的 `H` 半径突触云恰好**相切、不重叠**。实现取距离球心最近的
`N` 个格点（分布天然呈球形），再按 `(流向轴坐标, 索引)` 做确定性排序，使拓扑序与"沿流向轴
升序"一致。

* 因放置完全由 `H` / `N` 决定，**神经元坐标与 `seed` 无关**（验证脚本 R2 有逐位断言）；
* `seed` 只影响突触采样、参数初始化与数据打乱；
* 实测最近邻距 = `0.299999952`（N=64/H=0.15，期望 `2H = 0.3`）、`0.199999914`
  （N=256/H=0.10，期望 `0.2`），在容差 `1e-5` 内相等。

### 1.6 突触采样（半球内按体积均匀）

每个神经元的突触在**它自己的 `H` 半径球内**按体积均匀采样：方向在单位球面上均匀
（正态归一化），半径 `r = H · u^(1/3)`（`u ~ U(0,1)`，`u^(1/3)` 是 **u 的立方根**，
使半径分布函数为 `F(r) = (r/H)^3`，即球内**按体积均匀**；**不是**半径线性采样）。

半球切分：输入突触取流向轴**负半球**（`−axis`）、输出突触取**正半球**（`+axis`），实现
方式是把方向向量的流向轴分量翻转为目标半边（`−|c|` / `+|c|`）。该操作**测度保持**：目标
半球内任一方向恰由"原始方向"与"翻转后的方向"两种来源各命中一次，等价于半球拒绝采样，
但**不消耗额外随机数、无重试上限**。

---

## 2. 连接规则（神经元级，同一神经元对只算一条）

```
A → B 存在  ⟺  A ≠ B  且  z_A < z_B  且  ∃ o ∈ out(A), j ∈ in(B): d(o, j) <= D
```

其中 `z` 是 `flow_axis` 选定的坐标分量。

* **同一神经元对只算一条连接**：多对突触满足条件时，只保留**间距最近的那一对**作为
  代表连接（`representative_syn_out` / `representative_syn_input`）；
* `z_A < z_B` 使图**严格上行**：每条边都满足 `z_A < z_B`，故环不存在（Kahn 拓扑排序
  覆盖全部 `N` 个节点，验证脚本 R3 有断言）；
* 全部拓扑量在 `__init__` 预计算并 `register_buffer`，`forward` 中不重算。

> 实现注意（易错点，已固化在代码注释与断言中）：判据是"**存在**至少一对突触满足
> `d <= D`"，因此必须对合法突触对取 **`amin`** 后与 `D` 比较；写成 `amax` 会变成
> "全部突触对都 `<= D`"，与规则相反。此外，`syn_dist` 的 4D 视图
> `[N_A, y_out, N_B, y_in]` **不可**再 reshape 成 `[N*N, y_out*y_in]` 后按
> `flat[A*N+B]` 取值 —— 该视图的 stride 使 axis1/axis2 相对 2D 视为交换，reshape 会
> 产生**块内错位**；正确做法是在 `[N_A, N_B, y_out, y_in]` 上分别对两个突触维取
> `amin/argmin`（代码内有两处契约断言守护）。

### 2.1 孤立突触定义

某突触"孤立" ⟺ 其 `D` 邻域内**不存在任何合法连接的对端突触**（合法连接 = 对端属于
其他神经元、且高低关系满足上行约束）。该判据用于选出输入层驱动集合 `S_in` 与读出集合
`S_out`（见第 3.3 节）。

---

## 3. 前向计算：两阶段 + 双副本展开

### 3.1 阶段 1（输入层驱动）

由 `input_scope` 判据选出集合 `S_in`，其中每个神经元从输入层计算：

```
a_in[B] = ReLU( x · W_in[:, B] + b_B )        B ∈ S_in
```

`W_in ∈ R^{input_dim × |S_in|}`。`S_in` 之外的神经元本阶段输出恒为 0（不进入计算图）。

### 3.2 阶段 2（单遍逐层递推 + 双副本展开）

严格按 **Kahn 拓扑序（= 流向轴升序）逐层递推一遍**：

```
a_up[B] = ReLU( Σ_{A→B} w_{A→B} · ( a_up[A] + a_in[A] ) + b_B )
```

处理 `B` 时其全部上游（沿流向轴更靠下的层）已算完，因此一次前向即完成全部层的传播。
**架构中没有迭代轮数参数**（旧设计的同步迭代轮数已移除），故不存在"感受野被固定跳数
截断"的问题 —— 递推沿 DAG 逐层展开，感受野覆盖**全部层**（DEFAULT 规模 9 层）。

> 感受野证据（`verify_sphere_dag.py` 的 R5c，两项可执行判定）：
> ① 最深层神经元的祖先**覆盖全部层**（DEFAULT：9/9 层；flow_axis=x：10/10 层）；
> ② 把**第 1 层**某神经元的 `a_in` 提高 1.0，**最深层**的 `a_up` 必须因此改变
> —— 若递推被截断，跨整个深度的**影响将消失**。四组配置的实测最大变化（`[R5c]` 打印
> 的 6 位小数）：SMALL `0.000079` / DEFAULT `0.000250` / **flow_axis=x `0.000262`** /
> `space_radius=0.9` `0.000250`（`D = H` 后路径变少，故该量比改造前（D>H）小 1~2 个数量级，
> 但**仍严格 > 0**；**以脚本即时输出为准**，全精度值同时登记在
> `_verify/doc_numbers.json` 并由 `verify_all.py` 现跑比对）。

**双副本展开（关键语义）**：求和项 `(a_up[A] + a_in[A])` 表示神经元 `A` 的"上游版本输出"
与"输入层版本输出"**都参与后续传播**，两种版本**共享同一套权重** `w_{A→B}`（每条神经元级
连接一个独立标量权重）。非 `S_in` 神经元的 `a_in` 恒为 0，故只传播上游版本。

> 实现要点：`a_in[A]` 项必须乘上同一套边权后进入**每一次**聚合（它是与层无关的常量项）；
> 否则输入层副本会被"覆盖"而永不生效。`a_up` 的写入必须是**非原地**的
> （`index_copy` 而非 `a_up[node] = ...`）—— 原地赋值会破坏 autograd。
> 验证脚本 R5 用"零化 `a_in` 是否改变 `a_up`"与 `d(sum a_up)/d(a_in)` 两点直接取证。
> 四组配置的实测值（`verify_sphere_dag.py` 的 R5，`[R5]` 即时输出，**以脚本输出为准**）：
>
> | 配置（N=256/y=8×8/H=0.10/D=0.10/seed=42，除注明外） | 双副本神经元 | 零化 `a_in` 后 max\|Δ`a_up`\| | 受影响的下游非 `S_in` 神经元 | `d(sum a_up)/d(a_in)` 非零 |
> |---|---|---|---|---|
> | SMALL（N=64/y=4×4/H=0.15/D=0.15） | 44 | `1.047723` | 9 | 132/192 |
> | DEFAULT（`flow_axis=z`） | 180 | `2.259184` | 59 | 653/768 |
> | DEFAULT + `flow_axis=x` | 180 | `1.702276` | 61 | 643/768 |
> | DEFAULT + `space_radius=0.9` | 180 | `2.259184` | 59 | 653/768 |
>
> （历史登记值 `0.85 / 169 / 722/768`（F11 向量化前）与 `0.941795 / 51 / 166` 等
> （`D > H` 旧预设）均已被本轮 `D = H` 改造后的实测整体替换；全精度值登记在
> `_verify/doc_numbers.json`，由 `verify_all.py` 现跑比对。）

> **逐边数值正确性（R5b，M1 的回归判据）**：脚本**独立重建**阶段 2（按拓扑序逐节点、
> 用原始边表精确递推），与生产实现对比。四组配置的实测最大偏差
> （`[R5b]` 即时输出，**以脚本输出为准**）：
> SMALL `5.960e-08`（E=106）/ DEFAULT `2.384e-07`（E=736）/
> **flow_axis=x `5.960e-08`**（E=730）/ `space_radius=0.9` `2.384e-07`（E=736），
> 全部远小于判据阈值 `1e-5`。
> 同项还带**敏感性反例**：按历史 M1 形态（边已按 `edge_perm` 重排、源激活却按
> `repeat_interleave` 槽位取样）构造的错误映射，实测错配边数
> SMALL `106/106` / DEFAULT `727/736` / flow_axis=x `716/730`，且错配导致的结果差异为
> `8.437e-01` / `1.655e+00` / `1.703e+00` / `1.655e+00` —— 证明该判据对这类缺陷
> **确实敏感**（若反例退化，判据形同虚设；这正是皋陶第 2 轮 F13 的要求）。

> **实现形态与设备契约（F16）**：阶段 2 采用**整层向量化**——`__init__` 预计算并
> `register_buffer` 两张 `[K, 2]` 的 int64 下标表 `level_edge_reach`（每层在入边表中的
> 连续区间）与 `level_node_reach`（每层在拓扑序中的连续区间），前向对每层只做一次
> `index_add` + 一次非原地 `index_copy`，循环次数 = **层数**（DEFAULT 9 / SMALL 7），
> 而不是神经元数 N。层节点集合由 `topo_index[s:e]` **张量切片**得到。
> 历史缺陷（皋陶第三轮 error）：层节点曾用普通 Python `list` 保存 —— `nn.Module.to(device)`
> **不会搬运普通 list 中的张量**，CUDA 前向必然抛 RuntimeError（CPU 上完全静默）。
> 现在 `stage2_recurrence` 每次前向都调用 `_assert_index_device`，显式校验
> **每一个索引张量都已注册为 buffer/parameter 且与激活同设备**，不一致即抛 RuntimeError。
> 本机为 CPU-only（`torch.cuda.is_available() == False`），故 CUDA 路径采用**静态取证**：
> `verify_device_regression.py` 用 `meta` 设备做搬运实验（注册的 buffer 会搬走、
> 普通 Python list 里的张量搬不走）+ 两条守卫负例（设备不一致 / 未注册均必须抛错）；
> 若在有 GPU 的机器上运行，该脚本会自动追加真实 `.cuda()` 前向与 CPU 结果比对。

### 3.3 判据开关与读出

| 开关 | 取值 | 含义 |
| --- | --- | --- |
| `input_scope` | `any_isolated` | 神经元有 **≥1 个**输入突触孤立 → 进入 `S_in` |
| | `all_isolated` | 神经元**全部 `y_in` 个**输入突触都孤立 → 进入 `S_in` |
| `readout_scope` | `any_isolated` | 神经元有 **≥1 个**输出突触孤立 → 进入 `S_out` |
| | `all_isolated` | 神经元**全部 `y_out` 个**输出突触都孤立 → 进入 `S_out` |

读出（**严格口径：只有 `S_out` 中的神经元向输出层贡献信号**）：

```
h[n] = a_up[n]     若 n ∈ S_out
h[n] = 0           否则
logits = h @ W_out.T  (+ b)
```

非 `S_out` 神经元被读出头整体屏蔽 —— 其 `W_out` 列不参与计算图、梯度恒为 0（该口径的
直接推论，冒烟判据据此区分"参数是否拿到梯度"与"是否参与 loss"）。`readout_scope` 由此
**真正生效**：两取值给出不同的 `S_out`（DEFAULT 规模 seed=42：`any`=187 / `all`=14），
实测同一输入下 logits 最大差异 `0.858180`（`verify_sphere_dag.py` 的 R7b），四种 scope
组合的冒烟 `loss` 互不相同（见第 6 节）。

### 3.4 参数集合

| 参数 | 形状 | 说明 |
| --- | --- | --- |
| `W_in` | `[input_dim, |S_in|]` | 阶段 1 输入层权重 |
| `edge_weight` | `[E]` | **每条神经元级连接一个独立权重** |
| `neuron_bias` | `[N]` | 神经元偏置（正初值 0.1） |
| `W_out` | `[output_dim, N]` | 读出层权重（非 `S_out` 列不参与 loss、梯度恒为 0） |
| `W_out_bias` | `[output_dim]` | 可选（`readout_bias=True` 时创建） |

**梯度口径（严格读出的推论）**：参与 loss 的参数梯度范数必须 > 0；`W_out` 的**非
`S_out` 列**梯度恒为 0，故判据断言的是"非零梯度列数落在 `[1, |S_out|]`"而不是"等于
`|S_out|`"（某 `S_out` 神经元 pre-activation 全负时其 ReLU 输出为 0、该列合法地为 0）。
同理，若某 `S_out` 神经元的 pre-activation 非正，其 ReLU 输出为 0、在 `h` 中本来就是
零列 —— 这是 ReLU 的正常行为，不是屏蔽错误。

> **已知参数开销与取舍（如实标注）**：严格 readout 下只有 `S_out` 中的神经元向输出层
> 贡献信号，因此 `W_out` 的**非 `S_out` 列永不参与计算图、梯度恒为 0** ——
> DEFAULT 测点 seed=42 的 `|S_out| = 45`，即 **`W_out` 有 256 − 45 = 211 列（约 82%）
> 属于"结构性静默"参数**：它们占参数预算（`output_dim × N = 10 × 256 = 2560` 个元素）
> 但不接收任何梯度、也不影响前向。
>
> 保留它们的原因与代价：
> * 保留 —— 使 `W_out ∈ R^{output_dim × N}`、`state_dict`/`config` 契约与产物 schema
>   在所有 scope 取值下**形状稳定**（换 `readout_scope` 时产物可直接互相对比，
>   也便于按同一套脚本加载复核）；
> * 代价 —— 每步优化器仍需对这批零梯度列做一次 Adam 更新（虽然 Adam 对零梯度不产生
>   位移，仍有内存与计算开销），且"参与 loss 的参数占比"会随 scope 变化而波动。
>
> 本轮**不改变** `state_dict` / `config` 契约（不改形状、不引入按 `S_out` 裁剪的重参数化）；
> 若后续要消除该开销，需另行评估"按 `S_out` 建立读出短向量"对产物兼容性的影响。

**禁止 materialize dense 权重矩阵**：连接权重一律"按连接构建"，
`count_dense_weight_tensors()` 扫描 `named_parameters()` / `named_buffers()`
（豁免几何量 `syn_dist`）并断言恒为 0。旧架构的边级 `W_conn_sparse`、`tau_raw`、
`neuron_threshold`、`ln_s_in`、`alpha` 残差等结构**已全部移除**。

---

## 4. 文件说明

| 文件 | 职责 |
| --- | --- |
| `n3d_sphere/__init__.py` | 包声明与模块定位（纯球形分层 DAG、自包含、一期不变） |
| `n3d_sphere/utils.py` | 通用工具层（**历史遗留**：四步闭环算子，当前架构只用到 `set_seed` / `get_device` / 日志 / 参数与梯度统计；未调用的算子已在文件头如实标注） |
| `n3d_sphere/data.py` | MNIST 数据层（IDX 惰性解析 + 归一化 + DataLoader） |
| `n3d_sphere/config.py` | 超参配置层：球半径窗口公式、FCC 晶格常数、判据开关、三预设 |
| `n3d_sphere/model.py` | 球形分层 DAG 模型：FCC 放置、半球突触采样、神经元级连接、两阶段双副本前向、统计接口 |
| `n3d_sphere/train.py` | 训练入口：CLI、产物指纹、15 条冒烟判据、checkpoint 元数据 |
| `n3d_sphere/README.md` | 本文档 |

---

## 5. 命令行

```bash
# 阶段 A：冒烟测试（15 条判据逐条打印 PASS/FAIL，退出码 0 表示全通过）
python n3d_sphere/train.py --smoke-test

# 阶段 A：指定判据开关（四种组合均可）
python n3d_sphere/train.py --smoke-test --input-scope all_isolated --readout-scope all_isolated

# 阶段 B：正式训练（默认 DEFAULT_CONFIG）
python n3d_sphere/train.py

# 阶段 B：限批快筛（产物自动写入 checkpoints/n3d_sphere/_verify/，文件名含配置指纹）
python n3d_sphere/train.py --max-batches 50 --tag quick

# 其它几何 / 判据参数
python n3d_sphere/train.py --flow-axis x --space-radius 0.9 \
    --input-scope any_isolated --readout-scope all_isolated --placement fcc
```

CLI 一览（几何相关部分）：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--flow-axis` | `x` / `y` / `z` | 全局流向轴：输入突触取负半球、输出突触取正半球 |
| `--space-radius` | 浮点 | 球空间半径；缺省/`0` = `R_min`；显式值须落在 `[R_min, R_max]`。**仅作半径窗口校验与元数据**，不改变神经元放置与拓扑（实际 `placement_radius` 允许略超 `R_min`，因为 FCC 格点是离散的） |
| `--d` | 浮点 | 连接距离阈值 `D`；缺省取预设值（本轮三预设均为 `D = H`）。**硬约束 `D <= H`**，`D > H` 在构造配置时直接报错；`D` 过小（`E < N`）会触发连通性下限校验（见第 1.2 / 1.3 节） |
| `--input-scope` | `any_isolated` / `all_isolated` | 输入层驱动判据 |
| `--readout-scope` | `any_isolated` / `all_isolated` | 读出判据（严格口径，见 3.3 节） |
| `--placement` | `fcc` | 神经元放置方式（当前唯一取值） |
| `--arch` | `neuron3d` / `mlp` | 主模型 / 对照基线（共用同一训练循环） |

其余覆盖参数（`--n` / `--y-in` / `--y-out` / `--h` / `--d` / `--seed` / `--lr` /
`--weight-decay` / `--batch-size` / `--readout-bias` / `--epochs` / `--device` /
`--threads` / `--max-batches` / `--checkpoint` / `--tag` / `--backup`）的语义与哨兵规则
见 `train.validate_override_args` 的 docstring（负数一律报错、`0` 表示不覆盖）。

> **无迭代轮数参数**：架构中已移除同步迭代轮数（阶段 2 改为按拓扑序单遍逐层递推），
> 故 CLI 中也没有对应的覆盖项。

---

## 6. 阶段 A 冒烟判据（新架构口径）

旧架构的 `tau > 0`、连接稀疏度（密度 `E/(N*y_out*N*y_in)`）、边级参数数 == `E` 等判据
**已随架构失效**（`tau_raw` 参数与边级构图均已移除），现判据集合为 15 条：

| # | 判据 | 说明 |
|---|---|---|
| 1 | 前向输出形状 == `[B, output_dim]` | **真实断言**（比较实际 logits 的形状，非恒真） |
| 2 | 反向无错误（全部可学习参数都有梯度） | 缺失梯度由 `tensor_grad_norms` 记为 -1.0，逐个核验 |
| 3 | 参与 loss 的参数梯度范数 > 0 | 按 arch 解析输出层参数名（neuron3d 为 `W_out`、MLP 为 `fc2.*`）并断言至少一个 > 0；neuron3d 另断言"**非零梯度列数落在 [1, \|S_out\|]**"（严格 readout 下非 `S_out` 列结构性为 0，而全负 pre-activation 的 `S_out` 列也合法为 0）；其余参数（`W_in` / `edge_weight` / `neuron_bias`）断言范数 > 0 |
| 4 | loss 非 NaN/Inf | — |
| 5 | `S_in` 非空 | 阶段 1 真正被输入层驱动 |
| 6 | `S_out` 非空 | readout 真正有信号 |
| 7 | 连接数 == 去重后的神经元对数 | "同一神经元对只算一条连接"的可执行形式 |
| 8 | 无环 DAG 且每条边严格上行 | `z_A < z_B` |
| 9 | 最近邻距 == `2H` | FCC 规则堆积契约（容差 `1e-5`） |
| 10 | 球空间半径落在 `[R_min, R_max]` 内 | — |
| 11 | `E > 0` 且平均出度 > 0 | — |
| 12 | 不存在 `[N*y_out, N*y_in]` 形状的权重张量 | 未 materialize dense 矩阵 |
| 13 | readout 严格口径 | **只断言子集方向**：`h` 的非零列都属于 `S_out`；并**直接结构核对** `h` 逐位等于 `a_up * out_scope_mask`（验证掩码确实作用于 `a_up`）。**不断言** `\|S_out\|` 列全覆盖 —— 全负 pre-activation 的 `S_out` 神经元其 ReLU 输出为 0，该列合法为 0 |
| 14 | 阶段 2 递推顺序 == 流向轴升序 | `topo_matches_axis_order == 1` |
| 15 | CPU 单 batch 前向+反向耗时 < 120s | — |

### 6.1 四种 scope 组合的实测（每行都有独立可 `torch.load` 复核的产物）

`SMALL_CONFIG`（N=64 / y=4×4 / **H=0.15 / D=0.15** / batch=32 / seed=42），产物位于
`checkpoints/n3d_sphere/_verify/`，汇总记录见同目录 `smoke_scope_matrix.json`
（由 `run_smoke_matrix.py` 从矩阵的四条 scope 记录重建）：

| `input_scope` | `readout_scope` | `S_in` | `S_out` | loss（原始值） | 产物文件 | 退出码 |
|---|---|---|---|---|---|---|
| `any_isolated` | `any_isolated` | 55 | 53 | `2.1496479511260986` | `smoke.pt` | 0 |
| `any_isolated` | `all_isolated` | 55 | 16 | `2.236379623413086` | `smoke_N64_y4x4_H0.15_D0.15_plfcc_axz_isany_rsall_bs32_s42.pt` | 0 |
| `all_isolated` | `any_isolated` | 11 | 53 | `2.2899417877197266` | `smoke_N64_y4x4_H0.15_D0.15_plfcc_axz_isall_rsany_bs32_s42.pt` | 0 |
| `all_isolated` | `all_isolated` | 11 | 16 | `2.337554454803467` | `smoke_N64_y4x4_H0.15_D0.15_plfcc_axz_isall_rsall_bs32_s42.pt` | 0 |

四种组合均 **15/15 PASS、退出码 0**，且 **loss 互不相同** —— 这直接证明
`input_scope` 与 `readout_scope` 两个开关都真正生效（`S_out` 不同 → 严格 readout 下
`h` 不同 → logits/loss 不同）。

默认组合（`neuron3d` + `flow_axis=z` + 两个 `any_isolated`）写入历史文件名
`checkpoints/n3d_sphere/_verify/smoke.pt`；**其它任何组合**（含 `arch=mlp`、`flow_axis=x`、
任一 scope 取 `all_isolated`）写入 `smoke[_ar{arch}]_{完整指纹}.pt`。

> **默认路径必须是 `smoke.pt`**（历史缺陷：曾无条件传入指纹，使默认路径**永远不再写**
> `smoke.pt`，而文档与验证脚本仍指向它 —— 于是读到上一版代码留下的陈旧产物）。
> 现在 `run_smoke_test` 只在"默认组合"时退化为 `smoke.pt`，并在每次运行时**刷新**它。

默认产物 `smoke.pt` 的实测值（可由下方复核命令直接验证）：

```
python n3d_sphere/train.py --smoke-test          -> 退出码 0，15/15 PASS
loss = 2.1496479511260986（可 torch.load 复核 checkpoints/n3d_sphere/_verify/smoke.pt）
单 batch 前向+反向耗时 = 3.41s（随机器负载波动，不作契约）；可学习参数 = 43930
四个可学习参数的梯度范数（L2，**全精度、与产物逐位相等**）：
  W_in         1.5796021223068237
  edge_weight  0.2537662386894226
  neuron_bias  0.18929001688957214
  W_out        0.49458175897598267
E=106，平均出度 1.6562，最大出度 4，层数 7，S_in=55，S_out=53，双副本神经元 44
最近邻距 0.299999952（= 2H），R=0.663198 ∈ [R_min=0.663198, R_max=1.326395]
config 中不含已删除字段（无迭代轮数维度）
```

> 复核命令：`python -c "import torch; c=torch.load('checkpoints/n3d_sphere/_verify/smoke.pt',map_location='cpu',weights_only=False); print(c['loss'], c['grad_norms'])"`
> —— 该命令对应当前代码，输出必须与上表逐位一致。
>
> **数值随配置口径变化的说明（D≤H 改造）**：本轮把三个预设改为 `D = H` 后，SMALL 冒烟的
> 拓扑与 loss 全部改变（`E 181→106`、`|S_in| 13→55`、`|S_out| 17→53`、参数 `11077→43930`、
> `loss 2.326995849609375 → 2.1496479511260986`）；表格与 `doc_numbers.json` 均按新产物
> 整体刷新。**末位说明（F17）**：`W_in` 梯度范数的末位随实现中的**求和次序**可能微变，
> 故本表**一律以产物实测值（全精度）为准**，同一 seed 内逐位可复现；该值同时登记在
> `_verify/doc_numbers.json`，由 `verify_all.py` 现跑比对，任何偏移都会报 FAIL。
>
> 其余三种 scope 组合的梯度范数（同样来自各自产物，全精度）：
> `any/all`：W_in `1.0119041204452515` / edge_weight `0.1697416603565216` /
> neuron_bias `0.1355101466178894` / W_out `0.39588287472724915`；
> `all/any`：W_in `0.6916628479957581` / edge_weight `0.1618543416261673` /
> neuron_bias `0.2808758020401001` / W_out `0.32272979617118835`；
> `all/all`：W_in `0.2666252553462982` / edge_weight `0.0700279250741005` /
> neuron_bias `0.17869356274604797` / W_out `0.15432971715927124`。
>
> `--arch mlp` 冒烟的实测：退出码 0、6/6 PASS（MLP 基线只有 6 条适用判据），产物
> `smoke_armlp_N64_y4x4_H0.15_D0.15_plfcc_axz_isany_rsany_bs32_s42.pt`，
> loss = `2.2989346981048584`（`fc1.weight` `6.219344139099121` /
> `fc1.bias` `0.20135721564292908` / `fc2.weight` `4.08964204788208` /
> `fc2.bias` `0.18698155879974365`）。
> 注：判据 3 按 arch 解析输出层参数名（neuron3d 为 `W_out`，MLP 为 `fc2.*`），两种架构通用；
> MLP 基线不使用 `D`（`hidden_dim=2048` 全连接），故其 loss 与改造前逐位相同。

---

## 7. 实测拓扑统计（全部标注 seed，可由脚本复现）

`DEFAULT` 规模（N=256 / y=8×8 / **H=0.10 / D=0.10** / `flow_axis=z` /
`input_scope=readout_scope=any_isolated`，`space_radius` 取默认 `R_min=0.701840`）：

| seed | E | 平均出度 | 最大出度 | 平均入度 | 最大入度 | 层数 | S_in | S_out | 双副本 | 孤立输入突触 | 孤立输出突触 | params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 736 | 2.8750 | 4 | 2.8750 | 4 | 9 | 193 | 187 | 180 | 605 | 555 | 154864 |
| 7 | 736 | 2.8750 | 4 | 2.8750 | 4 | 9 | 192 | 196 | 178 | 573 | 602 | 154080 |
| 2024 | 740 | 2.8906 | 5 | 2.8906 | 5 | 9 | 194 | 185 | 178 | 537 | 563 | 155652 |
| 123 | 750 | 2.9297 | 5 | 2.9297 | 5 | 9 | 192 | 191 | 177 | 580 | 542 | 154094 |
| 0 | 738 | 2.8828 | 4 | 2.8828 | 5 | 9 | 197 | 189 | 182 | 580 | 570 | 158002 |
| 1 | 727 | 2.8398 | 5 | 2.8398 | 5 | 9 | 205 | 190 | 192 | 602 | 586 | 164263 |
| 2 | 738 | 2.8828 | 5 | 2.8828 | 5 | 9 | 195 | 190 | 180 | 593 | 568 | 156434 |
| 3 | 735 | 2.8711 | 4 | 2.8711 | 5 | 9 | 195 | 192 | 179 | 602 | 597 | 156431 |
| 4 | 729 | 2.8477 | 5 | 2.8477 | 5 | 9 | 187 | 203 | 173 | 543 | 622 | 150153 |
| 5 | 733 | 2.8633 | 4 | 2.8633 | 4 | 9 | 182 | 197 | 167 | 562 | 576 | 146237 |

* **E 的跨 seed 区间 = 727 ~ 750**（均值 ≈ 736.2），平均出度 2.8398 ~ 2.9297；
* 最大出度与最大入度在 10 个 seed 中均为 **4 或 5**（不再出现改造前的 6）；
* 层数恒为 9（流向轴 FCC 分层）；
* 可学习参数随 `|S_in|` 变化（**146237 ~ 164263**，取自 `_verify/topology_snapshot.json` 中
  **默认测点 `default_any_any_z`（N=256/y=8×8/H=0.10/D=0.10/z）的 10 个 seed** 记录），
  这是"按连接构建"的必然结果；`|S_in|` 因 `D=H` 而从改造前的 40~52 涨到 **182 ~ 205**，
  参数量随之从 35094~44489 涨到 146237~164263（约 ×3.3~3.7）；
* 复现脚本：`python checkpoints/n3d_sphere/_verify/verify_topology_snapshot.py`
  （60 条记录落盘 `_verify/topology_snapshot.json`；全部 6 个测点均为 `D = H`）。

**判据开关实测**（同 `DEFAULT` 规模，seed=42；`E` 与判据无关，恒为 736）：

| `input_scope` | `readout_scope` | `S_in` | `S_out` | 双副本神经元数 |
|---|---|---|---|---|
| `any_isolated` | `any_isolated` | 193 | 187 | 180 |
| `any_isolated` | `all_isolated` | 193 | 14 | 180 |
| `all_isolated` | `any_isolated` | 13 | 187 | **0** |
| `all_isolated` | `all_isolated` | 13 | 14 | **0** |

> **如实标注**：`input_scope=all_isolated` 时 `S_in` 只有 13 个神经元且其中没有神经元
> 拥有上游连接，故"双副本展开"在该取值下**不触发**（双副本计数 = 0）。这是判据选择的
> 自然结果，不是实现缺陷 —— 此时阶段 2 对 `a_in` 的依赖仍然存在：`d(sum a_up)/d(a_in)`
> 在 `all/any` 下于 **479/512** 个元素上非零（`any/any` 为 **439/512**），见
> `verify_scope_and_fingerprint.py` 的 S1-S3 即时输出（该脚本退出码 0、`[S1-S3] PASS`）。

### 7.1 跨模块回归（一期未被触碰）

```
python n3d_proto/train.py --smoke-test   -> 9/9 PASS、退出码 0、loss = 2.419689
```

本模块所有命令的产物均写入 `checkpoints/n3d_sphere/`，与一期 `checkpoints/` 物理隔离。

---

## 8. 产物纪律与命名

* 正式产物目录：`checkpoints/n3d_sphere/`；验证类运行（`--smoke-test` / `--max-batches > 0`）
  一律写入 `checkpoints/n3d_sphere/_verify/`，**绝不覆盖正式产物**；
* 冒烟产物命名：**完全默认组合**（neuron3d + SMALL_CONFIG 的 N/y/H/D/seed/batch_size +
  `flow_axis=z` + 两个 `any_isolated` + `space_radius=0`）退化为 `smoke.pt`；
  其它任何组合为 `smoke[_ar{arch}]_{指纹}.pt`，指纹为
  `N{N}_y{y_in}x{y_out}_H{H}_D{D}_pl{placement}_ax{axis}_is{scope}_rs{scope}_bs{batch_size}[_R{space_radius}]_s{seed}`
  —— **指纹维度必须与默认判定维度严格对齐**（历史缺陷：默认判定含 `batch_size` /
  `space_radius` 而指纹不含，导致 `--batch-size 64` 与 `--space-radius 0.9`
  落回同名文件、互相覆盖取证产物；`arch` 缺维度亦曾导致 `--arch mlp` 与主模型互覆）；
* 限批产物命名：
  `verify_<bpe>_N{N}_y{y_in}x{y_out}_H{H}_D{D}_pl{placement}_ax{axis}_is{scope}_rs{scope}_s{seed}[_tag].pt`；
* 全量产物的默认路径：与 `DEFAULT_CONFIG` 完全同配置时用 `model.pt`，其它配置使用
  `full_N{N}_y{..}_H{..}_D{..}_pl{..}_ax{..}_is{..}_rs{..}_s{seed}[_tag].pt`；
* **指纹维度**：`flow_axis` / `input_scope` / `readout_scope` / `placement` / `seed` /
  `H` / `D` / `N` / `y_in` / `y_out`（冒烟另含 `arch` / `batch_size` / `space_radius`；
  架构中无迭代轮数，故指纹不含该维度）—— 任一维度不同即不同
  文件名，互不覆盖（历史纠正：同名互覆曾导致筛选记录丢失）。注意 `seed` 影响突触采样，
  **不同 seed 即不同边集**，故必须进指纹；
* 覆盖已有 checkpoint 前默认生成 `<path>.bak` 备份（`--no-backup` 关闭）。

### 8.1 checkpoint 元数据

| 字段 | 内容 |
|---|---|
| `config` | 完整超参字典（含 `flow_axis` / `space_radius` / `placement` / 两个 scope） |
| `connection_stats` | 神经元级连接统计（`num_edges` / 度分布 / `num_layers` / `num_in_scope` / `num_out_scope` / 孤立突触数） |
| `topology_stats` | 几何指纹（`placement` / `flow_axis` / `space_radius` / `placement_radius` / `lattice_constant` / `nearest_neighbour_dist` / 神经元流向轴高度分布 / 判据编码 / `edge_dist_*` / `dual_copy_count`） |
| `dag_selfcheck` | `dag_acyclic` / `all_edges_uphill` / `topo_covers_all` / `topo_matches_axis_order` |
| `model_state_dict` 中的拓扑 buffer | `topo_index` / `edge_offset` / `edge_perm` / `edge_perm_in` / `neuron_in_edge_reach` / `edge_dst_in` / **`level_edge_reach` / `level_node_reach`** / `in_scope_mask` / `out_scope_mask` / 度分布 —— **全部 `register_buffer`**，故 `.to(device)` 与 `state_dict()` 都能搬运/持久化（F16 设备契约） |
| 顶层 | `placement` / `flow_axis` / `space_radius` / `effective_space_radius` / `min_space_radius` / `max_space_radius` / `input_scope` / `readout_scope` / 训练超参 |

### 8.2 字节确定性：SHA256 作为等价判据的前提（离朱第 11 轮 D1）

**`torch.save` 产物的字节并不只由数值决定，还取决于 pickle 的记忆化（memo），而 memo
依赖对象的身份（`id`）而非取值。** 因此"SHA256 相等"只有在**同一语义配置走同一代码路径**
时才等价于"数值相同"。

* 实测反例（D1）：`--smoke-test`（走 `SMALL_CONFIG` 单例）与
  `--smoke-test --input-scope any_isolated --readout-scope any_isolated`（scope 取值来自
  argparse 构造的**等值新字符串**）写出的 `smoke.pt` —— loss / `config` / `grad_norms` /
  `model_state_dict` 全部张量**逐位相等**，但 `data.pkl` 为 4097 vs 4117 字节、
  共 1520 字节不同：单例下两个 scope 字段指向同一个 `"any_isolated"` 字符串对象，
  第二次出现被写成 pickle memo 引用；显式传参下两处分别内联写出。
* **修复（根因侧）**：`train.py` 的 `apply_overrides` 在"显式覆盖值与基线**逐字段相同**"
  时**复用基线对象本身**，使"同一语义配置 ⇒ 同一字节"。修复后 `--smoke-test` / 显式传等值
  scope / `--space-radius 0.0` / `--preset default` / `--seed 0` 等路径写出的 `smoke.pt`
  **SHA256 全部相同**（本轮 `D = H` 改造后为 `4E11F12F…6A71`；改造前为 `1A9D68D8…43D7`，
  同一路径不同轮次的 SHA 不可跨轮比较）。
  **日志出现的精确范围（离朱第 12 轮 D2 澄清）**：`apply_overrides` 的短路分支会打印一行
  `显式覆盖参数与基线取值逐字段一致（值等价）：复用基线配置对象…`。因此**只有"显式给出
  覆盖参数且其值与基线相同"的形式**才会打印该行 —— 实测为 3 条：
  `--input-scope any_isolated`（或两个 scope 都显式传等值）与 `--space-radius 0.0`；
  而**裸跑**、`--preset default`、`--seed 0` 这三种形式在 `build_smoke_config` 的
  `if not explicit: return SMALL_CONFIG` 处就**提前返回**（它们按 CLI 约定本就不算"显式覆盖"），
  根本不会进入 `apply_overrides`，故不打印该行 —— 但**产物字节与前者完全一致**
  （都走 `SMALL_CONFIG` 单例）。反之，真正改变取值的覆盖仍打印
  `冒烟测试配置已被显式覆盖…` 告警且不打印值等价行，语义边界正确。
* **修复（判据侧）**：`run_smoke_matrix.py` 新增"**同一产物路径被多条组合写入时必须逐位
  一致**"的断言，并在每条记录中追加 `artifact_sha256_at_end` / `artifact_sha256_stable` /
  `artifact_shared_writers` 三个字段把该不变量固化 —— 实测 `smoke.pt` 被 3 条组合写入
  （`scope any/any seed42` / `seed 0` / `preset default`）、SHA 去重后 1 种，
  11/11 记录的 `artifact_sha256_at_end == artifact_sha256` 且与盘上文件实际 SHA 一致；
  "取证记录里的 SHA 在矩阵结束后已不存在于盘上"这类**证据不自洽**不会再出现。
* **结论**：引用产物 SHA256 时必须同时给出**配置与调用路径**；跨路径判断"数值等价"
  应以 `torch.load` 后的张量比对（`torch.equal`）为准，而不是文件 SHA。

---

## 9. 验证脚本

| 脚本 | 覆盖内容 |
| --- | --- |
| `_verify/verify_sphere_dag.py` | R1 几何（球内/半球/体积均匀）、R2 FCC（晶格常数、最近邻距=2H、seed 无关性）、R3 DAG（无环/严格上行/拓扑序）、R4 去重（神经元对数==E、代表连接为块内最近）、R5 双副本（零化 `a_in` 改变 `a_up`、`d(a_up)/d(a_in)` 非零）、**R5b 逐边数值正确性**（独立重建阶段 2 累加，核对每条边必须读到自己起点的激活；并含"错误槽位映射"反例以证明判据敏感）、R6 判据（包含关系/独立性/互斥）、R7 产物可 `torch.load` |
| `_verify/verify_dh_constraint.py` | **H1-H4 `D <= H` 硬约束与连通性下限（G1/G3 回归取证与负例）**：H1 `Config(D > H)` 必须抛 `ValueError`（含 `H=0.15/D=0.25`、`H=0.10/D=0.15` 两条负例，另含 `D == H` / `D < H` 的边界正例）；H2 三预设满足 `D = H` 且 `describe()` / `to_dict()` 往返一致；H3 三预设通过连通性下限并报出 `E / N / E÷N / 层数K / |S_in| / |S_out| / H / D`，D 过小必须被拦下；H4 用"临时停用下限校验"取出被拒配置的**真实拓扑量**（H=0.10 下 D=0.05/0.03/0.02 → E=146/17/1）证明拦下的确是退化图 |
| `_verify/verify_scope_and_fingerprint.py` | S1-S3 判据语义与双副本随 scope 的行为、S4-S5 产物指纹维度与冒烟命名规则 |
| `_verify/verify_config_contracts.py` | C1 半径公式、C2-C3 窗口校验与 FCC 容纳性、C4-C5 字段清理与取值域、C6 零残留断言 |
| `_verify/verify_topology_snapshot.py` | 固定实验点 × 10 seed 的拓扑量快照（落盘 JSON，供报告引用时标注 seed；6 个测点均为 `D = H`） |
| `_verify/verify_device_regression.py` | **D1-D7 设备契约（F16 回归取证）**：层拓扑量的类型/注册状态、`vars(model)` 通用扫描（不得存在搬不动的张量容器）、层切分等价性、`.to('meta')` 搬运实验 + 普通 Python list 搬不动的机制复现、两条守卫负例（必须抛 RuntimeError）、CUDA 实测（无 GPU 时自动 skip 并标注"静态取证"）、`state_dict` 往返逐位一致 |
| `_verify/run_smoke_matrix.py` | 以当前代码重跑 **11 种冒烟配置组合**，从日志解析**实际落盘路径**并与独立预测的产物名比对、回读 config 与期望逐字段比对、断言 `smoke.pt` 未被非默认组合污染且**同一路径多次写入逐位一致**；刷新 `smoke_matrix_f9.json` / `log_smoke_matrix_f9.txt`，并由四条 scope 记录重建 `smoke_scope_matrix.json` |
| `_verify/verify_all.py` | 一键依次执行上面全部命令（含一期回归）与**文档数字防线**（`doc_numbers.json` 现跑比对），汇总退出码 |
| `_verify/doc_numbers.json` | **文档数字登记表**：README / spec 中出现的实测值（本轮 **174 项** = 49 产物字段 + 111 全精度指标 + 14 文本计数），由 `verify_all.py` 现跑比对 |
| `_verify/legacy_dh_baseline.json` | **`D = H` 改造前的历史基线**（`D > H`，现已不可由 `Config` 构造）：登记改造前 `default_any_any_z` / `small_any_any_z` 的 E / 层数 / `S_in` / `S_out` / params，以及来源产物与快照的路径与 SHA256 |
| `_verify/sphere_dag_metrics.json` | `verify_sphere_dag.py` 每次运行落盘的**全精度**指标（R1-R7b，111 条），供 `doc_numbers.json` 现跑比对取用 |
| `_verify/smoke_matrix_f9.json` / `log_smoke_matrix_f9.txt` | 11 种冒烟组合的取证记录（产物名 / 退出码 / PASS / FAIL / loss / 梯度范数 / config 核对 / SHA256 一致性字段）与完整日志 |
| `_verify/log_verify_all_f21.txt` / `log_verify_all_g5.txt` / `log_verify_all_h3.txt` | 三轮（F21 / G5 / H3）`verify_all.py` 的完整验收输出：10 条命令的退出码与判据计数、文档数字防线结果与总结句 |

一键复现（也可直接 `verify_all.py` 一次跑完）：

```bash
python checkpoints/n3d_sphere/_verify/verify_all.py                              # 全部退出码 0、文档数字全部一致
python -m compileall -q n3d_sphere                                              # 退出码 0
python n3d_sphere/train.py --smoke-test                                          # 退出码 0，15/15 PASS
python checkpoints/n3d_sphere/_verify/verify_sphere_dag.py all                   # 退出码 0（R1-R7b）
python checkpoints/n3d_sphere/_verify/verify_dh_constraint.py                    # 退出码 0（H1-H4，D<=H 与连通性下限）
python checkpoints/n3d_sphere/_verify/verify_device_regression.py                # 退出码 0（D1-D7，D6 无 GPU 时 skip）
python checkpoints/n3d_sphere/_verify/verify_scope_and_fingerprint.py            # 退出码 0
python checkpoints/n3d_sphere/_verify/verify_config_contracts.py                 # 退出码 0
python checkpoints/n3d_sphere/_verify/verify_topology_snapshot.py                # 退出码 0
python checkpoints/n3d_sphere/_verify/run_smoke_matrix.py                        # 退出码 0（11/11 组合取证）
python n3d_proto/train.py --smoke-test                                           # 9/9 PASS、loss=2.419689
```

---

## 10. 已知边界与如实说明

1. **参数形状随随机几何变化**：`W_in` 的列数 = `|S_in|`、`edge_weight` 长度 = `E`，
   两者都随 `seed`（突触采样）变化，故可学习参数总数随 seed 变化。这是"按连接构建"的
   必然结果；同一 seed 内完全自洽可复现。
2. **`input_scope=all_isolated` 下双副本不触发**（第 7 节已量化），机制仍在、只是该判据
   选出的集合不满足"有上游连接"的条件。
3. **阶段 2 是单遍逐层递推**：架构中已无迭代轮数参数，感受野覆盖全部层（不依赖任何
   固定跳数）；参考实现逐节点重建的逐步数值对比由 R5b 固化（最大差异 < 1e-5）。
4. **`mlp` 对照基线**仍保留（`--arch mlp`），它与主模型共用完全相同的训练循环 / 优化器 /
   调度器 / 梯度裁剪 / 评估代码，连接类统计返回 0 占位；其冒烟产物与主模型**分文件**
   （见第 8 节 arch 指纹）。
5. **"坐标与 seed 无关"的边界**：仅神经元坐标如此（FCC 规则堆积 + 确定性字典序排序）。
   突触坐标、边集与可学习参数**都随 seed 变化**；报告引用任何拓扑数字时必须标注 seed。
6. **本机为 CPU-only 环境**（实测 `torch.cuda.is_available() == False`，`torch 2.14.0+cpu`）：
   所有实测数字都来自 CPU 运行；**CUDA 路径为静态取证**（`verify_device_regression.py`
   的 D4 `meta` 搬运实验 + D5 守卫负例），该脚本在具备 GPU 的机器上会自动追加真实
   `.cuda()` 前向与 CPU 结果比对。设备契约（所有索引张量必须注册为 buffer/parameter）
   由 `stage2_recurrence` 每次前向的 `_assert_index_device` 强制。
7. **`--seed 0` 按 CLI 约定表示"不覆盖"**（见 `--seed` 帮助与 `validate_overrides`），
   `--preset default` 在冒烟路径下的基线即 `SMALL_CONFIG` —— 故这两种命令行写法在冒烟
   矩阵中都与默认组合等价（产物同为 `smoke.pt` 且逐位相同）。矩阵记录逐条注明，
   真正的额外 seed 覆盖由 `--seed 7` / `--seed 2024` 承担（见 `smoke_matrix_f9.json`）。

### 10.1 第一/二轮修复的缺陷

> **阅读提示**：§10.1 / §10.2 的表格是**历史记录** —— 其中的数字是**该轮当时的实测值**，
> 后续轮次（尤其本轮的 `D = H` 改造）已把它们整体刷新，故不应与第 1～9 节的当前值混用；
> 当前值一律以第 1～9 节（及 `_verify/doc_numbers.json` 登记的现跑值）为准。

皋陶第一/二轮审查（1 error + 5 warning + info；2 error + 2 warning + 5 info）与离朱
独立测试（3 项失败）共同暴露的缺陷：

| 编号 | 缺陷 | 严重度 | 修复 |
|---|---|---|---|
| F1 | README / spec 中的冒烟实测值（`loss` 与四个梯度范数）来自**修复前的旧代码**，与当前产物不符 | **error** | 用当前代码重跑并逐一登记为可 `torch.load` 复核的真实值（见第 6.1 节）；同时补四种 scope 组合的独立产物 |
| F2 | 读出未按计划口径实现（曾用"非 `S_out` 取 `a_up + a_in` 合并版本"），`readout_scope` 之间**只有 `S_out` 规模不同、logits 不受影响** | warning | 改为严格口径 `h[n] = a_up[n]`（仅当 `n ∈ S_out`）否则 `0`；两取值下 logits 与 loss 均不同（R7b + 第 6.1 节四组合 loss 全不同） |
| F3 | 阶段 2 用固定轮数同步迭代（感受野被跳数截断） | warning | 改为按拓扑序**单遍逐层递推**；新增 R5c 取证"祖先覆盖全部层 + 第一层扰动可传至最深层" |
| M1 | 旧的 `stage2_propagate` 用 `repeat_interleave(arange(N), counts)` 生成源激活槽位，而边已按 `edge_perm` 重排 —— 每条边读到**错误源神经元**的激活（N=64 实测 174/181 条边错配，与规格映射相差约 0.5 个 logit，反向梯度同样沿错误边分配） | **高危** | 重写为"按目标分组的入边表 + 逐层向量化精确递推"（旧函数与旧名称均已移除）；新增 R5b 逐边数值回归（独立逐节点重建 + M1 形态反例） |
| M2 | `topo_index` 与流向轴升序不一致（`topo_matches_axis_order = 0`）：FCC 放置末段的轴排序是空操作，且 Kahn 就绪集按神经元编号排序 | 中 | `_build_fcc_positions` 末段改为显式字典序 key `(轴坐标, 壳层名次)`；Kahn 就绪集改为按 `(轴坐标, 索引)` 取最小（`heapq`），并断言"轴升序 == Kahn 结果" |
| M3 | 冒烟产物名不含 `arch`，`--arch mlp` 与主模型互相覆盖，使 `verify_sphere_dag.py` 的 R7 误报 FAIL | 低 | 冒烟指纹纳入 `arch` 与全部容量/几何维度 |
| 审查项 | 冒烟判据 1/2 为恒真断言、参数区间写法与实测不符、`space_radius` 语义未明确、残留扫描范围不足、`utils.py` 遗留算子未标注 | info | 判据 1 改为真实形状断言、判据 2 逐个核验"参数是否都有梯度"；参数区间改为 **35094 ~ 44489**；`space_radius` 标注"仅作窗口校验与元数据"；C6 扫描范围扩到 `utils.py`/`data.py`/`current_spec.md`/`module_definition.json`；`utils.py` 文件头如实标注调用现状 |

> 说明：M1 不影响任何"梯度是否存在 / 形状是否正确"的判据，当时的冒烟判据与全部验证
> 脚本都未能捕获，只有**逐边数值核对**能发现 —— 这正是 R5b 的由来。F2 的影响同样隐蔽：
> 旧的合并口径下 `readout_scope` 表面"生效"（`S_out` 规模变了），但 logits 完全不受影响；
> 改为严格口径后，两个开关才真正改变前向结果。

### 10.2 第三轮修复的缺陷（皋陶第三轮审查：1 error + 3 warning + 1 info；离朱第 11 轮独立测试：19/19 通过 + 1 项低severity 注记）

| 编号 | 缺陷 | 严重度 | 修复 |
|---|---|---|---|
| **F16** | 阶段 2 向量化（F11）**新引入的设备回归**：`level_edge_reach, level_nodes = self._build_level_groups(...)` 未 `register_buffer`，`level_nodes` 是普通 Python `list`（元素为 CPU int64 张量），却被 forward 当索引张量用（`index_select` / `index_copy`）。`nn.Module.to(device)` **不搬运普通 list 中的张量** → CUDA 前向必然抛 RuntimeError；同批新增的 `edge_dst_in` 已正确注册，说明意图就是注册 | **error** | ① 彻底去掉 `level_nodes` Python list：`_build_level_groups` 改为返回两张 `[K,2]` int64 张量（`level_edge_reach` / `level_node_reach`），二者均 `register_buffer`；层节点集合由 **`topo_index[s:e]` 张量切片**得到（随 `.to(device)` 自动搬运）。② 新增 `_assert_index_device`：每次前向校验全部索引张量**已注册且与激活同设备**，违反即抛 RuntimeError。③ 新增 `verify_device_regression.py`（D1-D7，含 `meta` 搬运实验与两条守卫负例）纳入 `verify_all`。④ CPU 冒烟 `loss` 与向量化前**逐位不变**（`2.326995849609375`） |
| F17 | README / spec 登记的 `W_in` 梯度范数是 F11 向量化**之前**的值（求和次序变化使末位改变，产物实测 `0.30047276616096497`） | warning | 重跑默认冒烟，把登记值刷新为**产物全精度实测值**（四个范数全部改为全精度），并在表下明确注明"末位随实现求和次序可能微变，**以产物为准**"；同时纳入 `doc_numbers.json` 由 `verify_all` 现跑比对 |
| F18 | `smoke_matrix_f9.json` / `log_smoke_matrix_f9.txt` 引用的 11 个产物名有 **8 个已不存在**（指纹后来加入 `_bs{batch_size}`，取证文件未随之重跑） | warning | 新增 `run_smoke_matrix.py`：用**当前代码**重跑 11 种组合（子进程强制 `PYTHONIOENCODING=utf-8` 以免 GBK 管道编码把中文日志变乱码而无法解析），从日志解析**实际落盘路径**并与脚本独立预测的产物名比对，逐个 `torch.load` 复核 config / loss / 梯度范数 / SHA256，并断言 `smoke.pt` 在全矩阵前后**逐位且 config 均未被污染**；实测 **11/11 通过**，两个取证文件已按实际产物名刷新 |
| F19 | README / spec 中 R5 / R5b / R5c 的实测值是 F3（单遍递推）之前的旧值，且**未覆盖 `flow_axis=x` 配置** | warning | 用**同一脚本的一次真实运行**整体刷新（R1-R7b 均在含 `flow_axis=x` 的 4 组配置上执行）：R5 `max\|Δa_up\|` = `0.941795`/`0.866614`/`0.918773`/`0.866614`，受影响下游非 `S_in` 神经元 = 51/197/204/197，`d(sum a_up)/d(a_in)` 非零 = 166/192、713/768、711/768、713/768；R5b 最大偏差 = `2.980e-08`/`5.960e-08`/`2.384e-07`/`5.960e-08`，反例错配 = 180/181、892/903、902/913；R5c = `0.004414`/`0.003379`/**`0.007765`**/`0.003379`。全部注明"**以脚本即时输出为准**"；`verify_sphere_dag.py` 新增落盘 `sphere_dag_metrics.json`（111 条全精度指标）供机器比对 |
| F20 | `verify_all.py` 汇总行只统计 `[PASS]`/`[FAIL]` 字面量，而 `verify_sphere_dag.py` 等脚本格式为 `[R1] PASS` → 汇总恒显示 `PASS=0 FAIL=0`，掩盖真实判据数量；且**缺少"文档数字与实现脱节"的机制性防线** | info | ① 计数改为正则同时匹配 `[PASS]` / `] PASS` / `：PASS`（FAIL 同理），修正"恒为 0"的假象（现实测：R1-R7b=10、D1-D7=6、E2=15、E2b=6、E5=9）；② 新增**文档数字防线**：`_verify/doc_numbers.json` 登记 README / spec 中出现的实测值（本轮 **93 项** = 32 产物字段 + 54 全精度指标 + 7 文本计数），`verify_all.py` 在同一轮用**本次现跑的产物 + `sphere_dag_metrics.json` + 命令标准输出**逐项比对，任何不一致直接判失败；③ `verify_all` 命令集扩充为 **9 条**（新增 D1-D7 设备契约），子进程统一强制 UTF-8 输出 |
| **F21** | 验收 | —— | `python checkpoints/n3d_sphere/_verify/verify_all.py`：9 条命令全部退出码 0、无 FAIL，文档数字 **93/93 一致、0 不一致、0 跳过**；CPU 冒烟 `loss` 与向量化前逐位不变；一期 `n3d_proto/train.py --smoke-test` 9/9 PASS、`loss=2.419689`、`git status --porcelain -- n3d_proto` 为空、一期三件产物 SHA256 未变 |
| **D1**（离朱第 11 轮） | 显式传"与默认相同的 scope 值"会写出**语义相同但字节不同**的 `smoke.pt`：裸跑走 `SMALL_CONFIG` 单例时两个 scope 字段指向同一个 `"any_isolated"` 字符串对象（pickle 第二次出现写成 memo 引用），显式传参时取值来自 argparse 的**等值新字符串**（两处分别内联写出），于是 `data.pkl` 4097 vs 4117 字节；后果是矩阵记录 [0] 登记的 SHA 在矩阵结束后**已被同轮后续写入覆盖**，取证文件不自洽 | 低 | ① **根因侧**：`train.py` 的 `apply_overrides` 在"显式覆盖值与基线**逐字段相同**"时**复用基线对象本身**（打印一行"值等价、复用基线对象"日志），使"同一语义配置 ⇒ 同一字节"——修复后 5 条等值路径写出的 `smoke.pt` SHA256 全为 `1A9D68D8…43D7`；② **判据侧**：`run_smoke_matrix.py` 新增"同一产物路径被多条组合写入时必须逐位一致"的断言，记录中追加 `artifact_sha256_at_end` / `artifact_sha256_stable` / `artifact_shared_writers` 字段（实测 `smoke.pt` 被 3 条组合写入、SHA 去重后 1 种）；③ 文档中明确 **SHA256 等价判据只在"同语义同代码路径"下成立**（见第 8.2 节） |

| **D2**（离朱第 12 轮） | 复测说明中把"值等价日志"写成"6 条等值命令形式均会打印"，实测只有进入 `apply_overrides` 的 3 条会打印（裸跑 / `--preset default` / `--seed 0` 在 `build_smoke_config` 的 `if not explicit: return SMALL_CONFIG` 处提前返回）—— 属**测试说明文字**与实现不一致，非代码缺陷（6 条形式的产物字节、loss、判据全部一致） | 低（文档） | 不改代码语义，只把"日志出现的精确范围"写进本文档第 8.2 节：明确列出会打印的 3 条形式与不打印的 3 条形式，并说明两者**产物字节一致**的原因（都走 `SMALL_CONFIG` 单例）；同时 D1 行中的通道口径同步更新为"3 条组合写入 `smoke.pt`、SHA 去重后 1 种" |

> 第三轮的核心教训与前两轮同源：**"凡是在 CPU 上静默、只在其它设备/其它配置下爆发的
> 不变量，必须写成可执行判据"**。F16 的普通 Python list 是典型例子 —— 所有 CPU 判据
> 全绿、15/15 PASS，缺陷只在 CUDA 上显形；F18/F19 则说明**取证文件与文档数字会随实现
> 演进而脱节**，因此本轮把"文档登记数字"本身变成一个由 `verify_all` 现跑比对的机器可读
> 契约（`doc_numbers.json`），而不是继续依赖人工核对。
>
> D1 补充了第三类教训：**"数值等价"与"字节等价"是两件事** —— `torch.save` 的字节受
> pickle memo（对象身份）影响。因此（a）产物的 SHA256 只有在同语义同代码路径下才是等价
> 判据，（b）凡是"证据文件里登记了某个文件的 SHA"的地方，都必须断言该 SHA 在同一轮运行
> 结束后**仍然存在于盘上**（本轮已把这条写成 `run_smoke_matrix.py` 的可执行判据）。

### 10.3 本轮变更：`D <= H` 约束 + 预设 `D = H` + 连通性下限（G1~G5）

| 编号 | 变更 | 落地内容 |
|---|---|---|
| **G1** | `D <= H` **硬校验** | `n3d_sphere/config.py::Config.__post_init__` 在 `D > H` 时抛 `ValueError`，消息含当前 `H`、`D`、`D/H` 与"**连接半径 D 不得超过接收/发送范围半径 H**"的说明；CLI `--d` 的帮助文本同步注明该硬约束与连通性下限。**负例取证**：`Config(H=0.10, D=0.15)` 与 `Config(H=0.15, D=0.25)` 均报错（`verify_dh_constraint.py` 的 H1 逐条打印异常消息） |
| **G2** | 三预设改为 **`D = H`** | `DEFAULT_CONFIG` D `0.15 → 0.10`、`HIGHACC_CONFIG` D `0.15 → 0.10`、`SMALL_CONFIG` D `0.25 → 0.15`；`Config` 数据类字段默认值 `D = H = 0.1`。三预设均能构造，`describe()` 含 H/D，`Config(**cfg.to_dict()) == cfg` 往返一致（H2 取证）。连带 `R_max` 降为改造前的一半（DEFAULT `1.403681`、SMALL `1.326395`），FCC 放置半径仍远小于它，C3 容纳性断言照旧成立 |
| **G3** | **连通性下限校验** | `ThreeDNeuronSpace.__init__` 构图完成后调用新增的 `check_connectivity_floor()`：要求 `E >= N`、层数 `K >= 2`、`|S_in| >= 1`、`|S_out| >= 1`，任一不满足即抛 `ValueError`（消息含实测 `E / N / (E÷N) / 层数K / |S_in| / |S_out| / H / D` 与具体违反项），并把通过时的指标挂在 `model.connectivity_floor` 供日志/脚本引用。**负例取证**：`Config(H=0.10, D=0.05)` 被拦下（实测 `E=146 < N=256`，`E/N=0.5703`），**不是**静默生成退化模型 |
| **G4** | 全量重跑刷新数字与取证 | 重跑冒烟矩阵（11 组合）、`verify_sphere_dag.py`（R1-R7b，4 组配置）、`verify_topology_snapshot.py`（6 测点 × 10 seed）、`verify_scope_and_fingerprint.py`、`verify_config_contracts.py`、`verify_device_regression.py`、`verify_all.py`；刷新 `smoke_scope_matrix.json` / `smoke_matrix_f9.json` / `log_smoke_matrix_f9.txt` / `sphere_dag_metrics.json` / `topology_snapshot.json` / `doc_numbers.json`（174 项）；README / `current_spec.md` / `module_definition.json` 中全部受影响的 E、`|S_in|`/`|S_out|`、层数、**参数区间（35094~44489 → 146237~164263）**、R5/R5b/R5c、冒烟 loss 与四个梯度范数、10-seed 拓扑表、产物命名示例按新实测整体刷新；改造前（`D > H`）的基线固化在 `legacy_dh_baseline.json` 并在第 1.4 节给出对照 |
| **G5** | 验收 | `python checkpoints/n3d_sphere/_verify/verify_all.py` 全绿（**10 条命令**全部退出码 0、无 FAIL），`doc_numbers.json` 现跑比对全部一致（174/174）；三预设均通过 `D <= H` 与连通性下限；两条负例（`D > H`、`D = 0.05` 过小）均按预期报错并有取证；CPU 冒烟 15/15 PASS，四种 scope 组合均通过且 loss 互异；一期 `n3d_proto` 零改动 |
| **G6**（离朱第 13 轮先行捕获） | **元数据更新破坏了 C6 零残留扫描**：C6 的扫描目标包含 `.module_agent/n3d_sphere/module_definition.json` **它自己**，而我在最后一次 `update_definition` 中把旧架构字段名（`tau_init` / `min_neuron_dist` / `max_sample_tries`）逐字写进了 `verify_config_contracts.py` 的**文件描述**，且**在该次元数据更新之后没有再跑 `verify_all`** → `[C6] 真实命中 1 行`、`[C6] FAIL`（离朱第 13 轮的先行探针 `r13_c6.txt` 捕获，我随后复现确认） | 低（流程/元数据） | ① 重写该文件描述，改为"以脚本内的 `REMOVED_FIELDS` 清单为准"而**不逐字列举被禁词**，使"扫描目标"与"扫描词表"互不污染；② 复跑 `verify_config_contracts.py` → `真实命中 0 行、上下文豁免 4 行`、`[C6] PASS`；③ **流程加固**：`update_definition` 会改动 C6 的扫描对象，故**元数据更新必须排在最终 `verify_all` 之前**（本轮已按此顺序重跑验证并留证据） |

### 10.4 收尾修复（皋陶第 4 轮遗留的 2 个小项，H1~H4）

| 编号 | 问题 | 严重度 | 修复 |
|---|---|---|---|
| **H1** | `n3d_sphere/config.py` 模块 docstring 的"参考值"仍写 `N=256, H=0.10, D=0.15 → R_min=0.701840, R_max=1.754601` —— 这是**已被本模块 `D <= H` 硬校验拒绝的非法组合**，且该 `R_max` 已不对应任何预设；"关键不变量"清单也只写了 `H > 0` / `D > 0`，缺 `D <= H` | warning（文档） | ① 参考值改为**合法组合** `N=256, H=0.10, D=0.10`（DEFAULT/HIGHACC 口径）→ `R_min = 0.701840`、`R_max = 1.403681`（`R_max/R_min = 2.0 = (H+D)/H`），并补一行 SMALL 口径 `N=64, H=0.15, D=0.15 → 0.663198 / 1.326395` 作对照；② "关键不变量"补入 **`D <= H`** 与连通性下限（`E >= N`、层数 `K >= 2`、`|S_in| >= 1`、`|S_out| >= 1`，由 `check_connectivity_floor()` 把关）；③ 新增一段"连接半径硬约束 `D <= H`"说明（含几何理由与负例脚本指向）；④ 全文复查后确认除三个预设的合法取值外，已无其它 `D=0.15` / `D=0.25` 非法示例值 |
| **H2** | `module_definition.json` 未登记本轮 G5 的验收证据日志 `checkpoints/n3d_sphere/_verify/log_verify_all_g5.txt`（上一轮同类文件 `log_verify_all_f21.txt` 已登记） | info | 补登记 `log_verify_all_g5.txt`，并同时登记 H3 的验收证据日志 `log_verify_all_h3.txt`；README §9 与 `current_spec.md` 的取证文件清单同步列出三轮验收日志（F21 / G5 / H3） |
| **H3** | 因 H2 改动了**被 C6 扫描的元数据**，必须按"先改元数据、后跑验收"的顺序重跑 | —— | `verify_all.py` 退出码 0：10 条命令全部退出码 0、无 FAIL，`doc_numbers` 一致 174 / 不一致 0 / 跳过 0，C6 真实命中 0 行（证据 `_verify/log_verify_all_h3.txt`）；一期回归：`git status --porcelain -- n3d_proto` 为空、proto 9/9 PASS、`loss=2.419689`、三件产物 SHA256 未变 |
| **H4** | 收尾 | —— | 写入执行总结并 `plan_complete`（**不**自行启动离朱，由风后统一启动） |
| **H5**（本轮自查发现） | 上一轮 G6 修复时我用 `update_spec(mode="add")` 追加"验证脚本清单"，导致 `current_spec.md` 中该 `###` 子标题**重复出现两次**（同一 heading 两个副本） | 低（文档结构） | 改为 `mode="set"` 重写整节，合并两份清单为一节；复核标题总数 25、**无重复 heading** |

> **第六类教训（来自 H1/H5）**：**"改一处语义，必须回头扫一遍同一语义的所有载体"**。
> `D <= H` 落到 `Config` 与 README 之后，`config.py` 的模块 docstring 仍是旧世界的描述；
> 而 `update_spec(mode="add")` 会**无脑追加**而不是替换，容易制造重复 heading。
> 可执行结论：**结构性改动后必须跑一次"载体一致性扫描"**（本项目已把它固化为
> C6（禁用词）、`doc_numbers.json`（现跑数字）、以及"标题唯一性"三项检查）。

> **本轮教训（第四类）**：**"几何约束必须在构造期闭环，而不是靠下游发现"**。
> `D > H` 此前在三个预设里长期存在却无人校验；更隐蔽的是"`D` 合法但过小"——
> `D = 0.05` 时 `E/N` 只有 `0.57`，模型仍能构造、能训练，只是**图已经退化**。
> 因此本轮把约束拆成两道：`Config` 期校验 `D <= H`（G1），`ThreeDNeuronSpace` 期校验
> 图连通性下限（G3），并为两者各写可执行负例（H1/H3/H4）。
>
> **本轮教训（第五类，来自 G6）**：**"当判据扫描的元数据本身可被判据之外的流程改写时，
> 写入顺序就是一种不变量"**。C6 会扫描 `module_definition.json`，而 `update_definition`
> 会改写该文件 —— 因此"先跑验收、再改元数据"必然留下**未被验证的状态**。
> 可执行结论：**任何元数据 / 文档更新之后，必须以一次完整的 `verify_all`（含 C6）收尾**，
> 不得复用更新前的结论。
