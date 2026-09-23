# 测试报告 —— n3d_sphere 审查后修复（W1 动态覆盖容器 / checkpoint 元数据 / gap 缓存 / 文档口径）

- 测试对象：`n3d_sphere/model.py`、`n3d_sphere/train.py`、`n3d_sphere/config.py`、`n3d_sphere/README.md`
- 工程根目录：`E:\neuron3d`；执行环境：Windows + Python 3.12.10 + torch 2.14.0+cpu；CPU
- 结论：**全部适用测试类型通过（FAIL=0）**；5 项硬门槛逐条达标；零数值行为改动得到逐位验证。

## 1. 测试概览

| 测试类型 | 用例数 | PASS | FAIL | 跳过 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 编译测试 | 1 | 1 | 0 | 0 | `python -m compileall -q n3d_sphere` 退出码 0 |
| 单元测试（W1 契约专项） | 56 | 56 | 0 | 0 | 自建脚本 `.lizhu_env/n3d_sphere_w1/test_w1_contract.py` |
| 逐位不变门槛（冒烟） | 9+3 | 12 | 0 | 0 | 二期 9/9 PASS + 与一期产物逐位比对 3 项 |
| 验证脚本门槛（proto/geom） | 2 | 2 | 0 | 0 | `verify_n3d_sphere_phase2.py proto|geom` 均退出码 0 |
| W1 专项回归脚本 | 2 产物 × 5 项 | 10 | 0 | 0 | `verify_w1_dynamic_coverage.py` 退出码 0 |
| 产物纪律（SHA256 / git） | 4 | 4 | 0 | 0 | 一期 SHA256 未变、`n3d_proto` 无改动 |
| 接口测试 | — | — | — | 全部跳过 | 本模块为纯 Python 训练库，**不对外暴露 HTTP 接口**；CLI 参数语义已由单元测试覆盖 |
| E2E 测试 | — | — | — | 全部跳过 | 本模块为纯 Python/PyTorch 项目，**无前台 UI/DOM**，无浏览器旅程可测（Playwright 已装 v1.63.0，但不适用） |
| **合计** | **84** | **84** | **0** | — | |

## 2. 测试点 1：动态 2a 覆盖容器（缺陷 W1 修复，核心）

### 2.1 正向（语义正确）

| # | 用例 | 结果 | 实测 |
| --- | --- | --- | --- |
| 1e | `_accumulate_round_coverage` 累加语义（sum/count/last） | PASS | 输入 0.25/0.5/1.0 → sum=1.75, count=3, last=1.0 |
| 1f | `mean == sum / count` 逐位相等 | PASS | 0.5833333333333334 == 0.5833333333333334 |
| 1h | 1 次前向后 rounds == T | PASS | count=2, T=2 |
| 1h-2 | 6 次前向后 rounds == 6×T | PASS | count=12 |
| 1h-3 | 再前向 4 次 rounds 精确 +4×T（线性无截断） | PASS | 12 → 20 |
| 1h-4 | 长跑后 mean == sum/count 仍成立 | PASS | mean=1.0 |
| 3f | 冒烟 checkpoint 轮数 == 1 batch × T = 2 | PASS | rounds=2.0 |

### 2.2 反向（异常/缺失接口）

| # | 用例 | 结果 | 实测 |
| --- | --- | --- | --- |
| 2b | `reset_dynamic_coverage(MLPBaseline)` 静默跳过 | PASS | 无异常抛出 |
| 2e | `dynamic_coverage_metadata(MLPBaseline)` 返回 `{}` | PASS | `{}` |
| 3'6 | `arch=mlp` 训练路径 `dynamic_coverage`/`topology_stats` 均为 `{}` | PASS | 训练与保存均正常，退出无异常 |
| 1d | 不存在 `_round_output_coverage` 无界列表属性 | PASS | `hasattr=False` |

### 2.3 边界与极限值

| # | 用例 | 结果 | 实测 |
| --- | --- | --- | --- |
| 1a | 从未前向：last/mean = NaN、rounds = 0.0 | PASS | `nan/nan/0.0` |
| 1b | reset 后 `sum==0.0`、`count==0`、`last=NaN` | PASS | 一致 |
| 1g | 容器元素个数恒为 3（任意多次前向后不增长） | PASS | attrs 恰为 `_round_coverage_sum/_count/_last` |
| 1i | 3 个字段全为 Python 标量（无 list/append 结构） | PASS | float/int/float |
| 1h-3 | 20 轮累计后容器仍为 3 个标量（O(1) 定长） | PASS | 元素个数不变 |
| 1c | `count` 为 `int` 非 `bool`、初值 0 | PASS | `0` |

### 2.4 状态机（重置/隔离/幂等）

| # | 用例 | 结果 | 实测 |
| --- | --- | --- | --- |
| 1k | `reset` 幂等（连续两次后 rounds=0、last/mean=NaN） | PASS | 一致 |
| 1'1 | `load_state_dict` 后新模型统计仍为初值（不随 state_dict 持久化） | PASS | m1.rounds=2.0 → m2.rounds=0.0 |
| 1'2 | `model.to(device)` 不重置/不污染容器 | PASS | rounds 仍为 0 |
| 2a | `reset_dynamic_coverage` 对支持接口的模型真正归零 | PASS | rounds=0 |
| 3'1 | 限批模式产物写入 `checkpoints/n3d_sphere/_verify/`，未覆盖正式产物 | PASS | 路径核对一致 |
| 3'3 | 阶段隔离：(训练 3 + 评估 3)×2epoch×T2 = 24，**build 期配置探针前向未混入** | PASS | rounds=24.0（该探针已被训练开始时的 reset 清零） |
| 3a/3b/3d | `get_topology_stats()` 三键与 `round_output_coverage()` 快照逐位一致 | PASS | 5b 同项亦 PASS |
| 5a | 从未前向时 `topology_stats` 的 last/mean=NaN、rounds=0.0 | PASS | 一致 |

## 3. 测试点 2：checkpoint 显式元数据（真跑 `run_smoke_test` 与 `_run_training_with_config`）

用 `torch.load(<产物>, weights_only=False)` 直接校验，未重建模型：

| # | 用例 | 结果 | 实测 |
| --- | --- | --- | --- |
| 3b | `topology_stats` 含 3 个动态覆盖键 | PASS | 20 键中含 `out_nonzero_coverage_last/_mean/_rounds` |
| 3c | 顶层新增 `dynamic_coverage` 键且为 3 键 dict | PASS | `{'last':1.0,'mean':1.0,'rounds':2.0}` |
| 3d | 与 `topology_stats` 同名键**逐位一致** | PASS | 三键全部 `==` |
| 3e | 末轮/均值有限（非 NaN）、轮数为正整数 | PASS | last=1.0, mean=1.0, rounds=2.0（整数） |
| 3'2 | 阶段 B 路径同样满足 3b–3e | PASS | last=1.0, mean=1.0, rounds=24.0 |
| 3g | 冒烟产物 loss 基线 | PASS | `loss=2.419689416885376` → `%.6f` = 2.419689 |
| 3h/3'5 | `edge_axis_gap` 不在 `model_state_dict` | PASS | 两路径均 False（键数 21） |
| 3i | state_dict 含且仅含 3 个二期增量键 | PASS | `edge_reverse_flag`/`neuron_component_id`/`connected_output_mask` |

## 4. 测试点 3：gap 缓存（次要优化）

覆盖 3 种配置（`cube/z`、`sphere/z`、`sphere/x`），逐项 `torch.equal`：

| # | 用例 | 结果 | 实测 |
| --- | --- | --- | --- |
| 4a | `edge_axis_gap` == 用 `edge_index` 现场重算 gap | PASS | 三配置 `max|diff| = 0.000e+00` |
| 4b | `edge_axis_gap` 不在 `state_dict()` | PASS | 三配置均 False（`persistent=False`） |
| 4c | `axis_gap_mean/min/max` 与重算逐位相等 | PASS | 三配置浮点值完全相同（见下） |
| 4d | `edge_reverse_flag == (gap < 0)` 口径自洽 | PASS | n_rev=1966 / 259 / 365 |
| 4e | `.to(cpu)` 后 gap 统计不变（buffer 语义） | PASS | mean 逐位不变 |
| 3h/3'5 | 不进入 torch.save 的 `model_state_dict` | PASS | 两产物均不含该键 |

实测值（逐位相等）：
- `cube/z`：mean=-0.0005199902225285769, min=-0.2413412630558014, max=0.24846258759498596
- `sphere/z`：mean=0.04500800743699074, min=-0.1463976949453354, max=0.14828145503997803
- `sphere/x`：mean=0.03746608644723892, min=-0.14682967960834503, max=0.1466180980205536

## 5. 测试点 4：文档口径一致性检查

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
| 半径口径四处一致（`model.py` 顶部 / 两个采样方法 docstring / `config.py` 顶部与 `Config.H` / `README` 2.1） | PASS | 均为「方向球面均匀 + `r = H·u^(1/3)`，u ~ U(0,1)，立方根 = 球内按体积均匀，非半径线性采样」 |
| gap 口径一致（`model.py` 顶部 / `get_topology_stats()` docstring / `README` 2.3） | PASS | 统一为 `gap = axis(output_syn_pos) − axis(input_syn_pos)` |
| 不得残留与代码不符的 gap 表述 | PASS（附说明） | `Δz/δ_out/δ_in` 仅出现在「**文档更正**：早期版本曾写成…该表述已删除」的更正注释中（`model.py:54-56`、`README.md:76-78`），语义为"已删除"，非有效口径 |
| `train.py` `--checkpoint` help 与 `run_full_training` docstring 路径口径 | PASS | 为 `checkpoints/n3d_sphere/model.pt` 与 `checkpoints/n3d_sphere/_verify/`；全目录 grep 无 `checkpoints/model.pt`、`checkpoints/_verify/` |
| `resolve_checkpoint_path` **行为**不变 | PASS | 抽取函数体（去注释/空行）与 git HEAD 逐字符比对：**40 行完全相同 = True**；docstring 变更为文案 |
| `README` 2.3 实测数字可复现 | PASS | 实测 cube 0.5009/0.4991、sphere 正向 0.6150/逆向 0.3850、sphere gap 均值 0.0163、cube gap 均值 0.0001 —— 与文档逐项吻合 |

## 6. 测试点 5：逐位不变门槛（全部真实执行，粘贴终端输出）

### 6.1 `python n3d_sphere/train.py --smoke-test` → 退出码 0，9/9 PASS，loss=2.419689

```
[N3D][INFO ] 阶段 A 结果：loss=2.419689，处理 batch 数=1，耗时=2.92s
[N3D][INFO ]   [PASS] 前向无 shape mismatch —— 所有 batch 前向成功返回 [B, output_dim]
[N3D][INFO ]   [PASS] 反向无错误 —— 收集到 7 个参数的梯度
[N3D][INFO ]   [PASS] 所有可学习参数梯度范数 > 0 —— 全部 > 0
[N3D][INFO ]   [PASS] loss 非 NaN/Inf —— loss=2.419689
[N3D][INFO ]   [PASS] 连接稀疏度（密度 E/(N*y_out*N*y_in)）< 0.1 —— sparsity=0.059601
[N3D][INFO ]   [PASS] tau > 0 —— tau=0.808689
[N3D][INFO ]   [PASS] CPU 单 batch 耗时 < 120s —— 实际 2.92s
[N3D][INFO ]   [PASS] 附加①：边级参数数 == E —— W_conn_sparse.numel()=3906，E=3906
[N3D][INFO ]   [PASS] 附加②：不存在 [N*y_out, N*y_in] 形状的权重张量 —— 命中 0 个
[N3D][INFO ] 阶段 A 冒烟测试结论：全部通过
[exit=0]
```

与一期 `python n3d_proto/train.py --smoke-test`（退出码 0、9/9 PASS、`loss=2.419689`）逐位相等：

```
loss : proto=2.419689416885376  sphere=2.419689416885376
两侧逐位相等：True    grad_norms 完全相同：True
```

### 6.2 `python -m compileall -q n3d_sphere` → 退出码 0

```
[exit=0]
```

### 6.3 `git status --porcelain -- n3d_proto` → 输出为空

```
<空输出>
```
同时 `git status --porcelain -- n3d_sphere n3d_proto` 仅列出二期 4 个改动文件：
` M n3d_sphere/README.md`、` M n3d_sphere/config.py`、` M n3d_sphere/model.py`、` M n3d_sphere/train.py`
（diffstat：261 insertions(+), 48 deletions(-)）。

### 6.4 `verify_n3d_sphere_phase2.py proto` → 退出码 0；state_dict 键集增量**恰为 3 个**

```
loss      : proto=2.419689416885376  sphere=2.419689416885376
state_dict 键集：一期 18 个；二期新增=['connected_output_mask', 'edge_reverse_flag', 'neuron_component_id']，缺失=[] -> 符合预期
共有 18 个键逐位相等：True
connection_stats 既有 5 键逐位相同：True
[R1] PASS    [R4] PASS    总体结论：全部 PASS
[exit=0]
```
`edge_axis_gap` **未出现在增量中**（增量集合恰为上述 3 键）。

### 6.5 `verify_n3d_sphere_phase2.py geom` → 退出码 0

```
cube SMALL / cube DEFAULT / sphere z / sphere x 四组几何断言全 PASS
体积均匀性 (|p|/R)^3：mean=0.4841（理论 0.5）、std=0.2791（理论 0.2887）
半球切分正确：True（输入 max=-5.1e-05 ≤ 0，输出 min=+1.7e-05 ≥ 0）
连通性自检：合成图 7/7；向量化 vs 独立并查集 1/1 划分匹配=1
[跨模块对照] 一期 n3d_proto vs 二期 cube 路径坐标/边集逐位相等：True
总体结论：全部 PASS
[exit=0]
```

### 6.6 一期既有产物 SHA256 未变化

| 产物 | 实测 SHA256 | 基线 | 结果 |
| --- | --- | --- | --- |
| `checkpoints/n3d_model_full.pt` | `888556B0913C9F46419A674117FD13A99F2C71BA6692367A839DB873A58D8924` | 同 | 未变化 |
| `checkpoints/n3d_model_highacc.pt` | `9F21AC34C91977FE60F623138F3DEE24428E44B473BD86DE60A8B577F11BE3F8` | 同 | 未变化 |
| `checkpoints/n3d_model_capacity.pt` | `0F7CF500C256BFE41408E3DC68CED9316C21EE4EE94527790F861C76E6C35011` | 同 | 未变化 |

### 6.7 附加：官方 W1 专项回归脚本

`python checkpoints/n3d_sphere/_verify/verify_w1_dynamic_coverage.py` → 退出码 0，两个冒烟产物 × C1–C5 全 PASS：

```
[smoke.pt] C1 遗留 List 属性存在=False，标量容器字段数=3/3
[smoke.pt] C1b 5 次前向后：累计轮数=10（期望 5*T=10），容器元素个数=3
[smoke.pt] C2 均值口径逐位相等=True；末轮一致=True
[smoke.pt] C3 topology_stats 与顶层 dynamic_coverage 逐位一致=True
[smoke.pt] C4 重置后：轮数=0，末轮/均值均为 NaN=True
[smoke.pt] C5 缓存 vs 重算逐位相等=True，缓存出现在 state_dict 中=False
（smoke_sphere_z.pt 同样全 PASS）  总体结论：全部 PASS
```

## 7. 失败用例分析

**无失败用例（FAIL=0）。** 测试过程中出现过 3 条自建用例的**预期值写错**，经定位为测试脚本自身缺陷、非被测代码缺陷，已修正并复跑通过，如实记录如下：

| 初版用例 | 现象 | 根因 | 处置 |
| --- | --- | --- | --- |
| `1h 6 次前向后 rounds == 6×T` | 报 count=15 ≠ 12 | 该实例在同一用例内先用 `_accumulate_round_coverage` 手动累加了 3 次（0.25/0.5/1.0），手动累加**按契约同样计入轮数**，故 3 + 6×2 = 15 才是正确值 | 将手动累加与 forward 计数**拆到不同模型实例**上测量；现 1e/1f 用手动累加实例（3/1.75/1.0），1h 用全新实例（1 次→2、6 次→12、10 次→20），全部 PASS |
| `3'3 rounds == epochs × max_batches × T = 12` | 报 rounds=24 | 算术错误：`2 epochs × 3 batches × T2 = 12` 只算了训练前向，漏算 `evaluate()` 的评估前向（每个 epoch 另跑 3 个 batch） | 修正期望为 `(3 训练 + 3 评估) × 2 × 2 = 24`，并加一条「每 epoch 前向次数 == 6」的独立口径断言 |
| `4b edge_axis_gap 不在 state_dict()/named_buffers()` | 报 `named_buffers` 含该键 | 断言超出规格：规格只要求「不在 `state_dict()` / `torch.save` 的 `model_state_dict` 中」；`named_buffers()` 按 PyTorch 语义**会**列出 `persistent=False` 的 buffer（这正是它不落盘的原因） | 断言收窄为规格原文的 `state_dict()` 口径，并在明细中标注 `named_buffers()` 含该键属预期 |

上述三条均已复跑确认，最终单元测试 **PASS=56 / FAIL=0（退出码 0）**。

另有 1 项**已定位为测试侧误报、非缺陷**的观察记录：在"同一模型实例上先手动累加、再连续前向"的混合操作序列中，我曾观察到计数与"手动次数 + 前向次数×T"不吻合（读数 15 vs 预期 12）。经 spy/栈追踪逐次核对，确认 15 恰为「3 次手动累加 + 6 次前向×T=12」，即容器行为完全符合契约；官方 `verify_w1_dynamic_coverage.py` 在干净序列上亦测得「5 次前向 → 10 轮（= 5×T）」精确成立。故**无代码缺陷，无需修复**。

## 8. 环境问题说明

无环境阻塞，全部测试均在本机实际执行完毕：

- Python 3.12.10 / torch 2.14.0+cpu / CPU 推理与训练；MNIST IDX 数据已就位（`data/mnist/MNIST/raw`），无需联网下载。
- 控制台为 GBK 时中文日志乱码，按测试说明预先设置 `$env:PYTHONIOENCODING='utf-8'` 规避。
- 已知**预先存在**问题（测试说明已豁免）：`python n3d_sphere/train.py --help` 在 GBK 控制台因 `--flow-axis` help 文本含 U+2212 抛 `UnicodeEncodeError`；本次未修改该字符串，不作为失败判据。已用 `--smoke-test` 等命令覆盖 CLI 主通路。
- 被跳过的测试类型及原因：
  - **接口测试**：模块为纯 Python 训练库，无 HTTP/RPC 接口，无适用对象。
  - **E2E 测试**：模块无前台 UI/DOM（`module_agent_testing(check_playwright)` 测得 Playwright v1.63.0 已安装，但本变更无浏览器可测面），无适用对象。

## 9. 产物与副作用说明

- 新增测试脚本：`.lizhu_env/n3d_sphere_w1/test_w1_contract.py`（56 条断言，退出码 0），另含调试脚本 `dbg_*.py`。
- 复跑副作用（均在规格允许范围内）：
  - 幂等刷新 `checkpoints/n3d_sphere/_verify/smoke.pt`（cube 冒烟产物，loss 值逐位不变）；
  - 新增两个限批验证产物：`_verify/verify_3_N64_y4x4_H0.15_D0.25_T2_topcube_axz_s42_lizhu_w1.pt`、`_verify/verify_2_..._lizhu_w1_mlp.pt`（文件名带配置指纹，互不覆盖）；
  - **未**创建/覆盖 `checkpoints/n3d_sphere/model.pt`（该目录下仅存既有的 `full_sphere_z_...pt` 全量产物），**未**触碰 `checkpoints/` 根下任何一期产物（SHA256 已复核）。

## 10. 修复建议

1. 被测代码**无需修复**：本次全部测试类型通过，逐位不变门槛与产物纪律均达标，零数值行为改动的目标达成。
2. 可选（非阻塞、纯口径澄清）：
   - `round_output_coverage()` / `get_topology_stats()` 的 docstring 中「轮数 = 前向次数 × T」建议补一句：**该计数覆盖训练与评估阶段的全部前向**，因此阶段 B 产物中 `rounds = (每 epoch 训练 batch 数 + 评估 batch 数) × epochs × T`（实测 `(3+3)×2×2 = 24`），避免读者按"仅训练前向"误算。
   - 若后续希望"仅训练轮次"口径，可在 `train_one_epoch` 与 `evaluate` 之间做阶段性快照，现有 `reset_round_output_coverage()` 接口已足以支撑，无需改结构。
3. 文档口径已一致，无需再改；`Δz/δ_out/δ_in` 仅以"已删除的更正说明"形式保留，符合规格"不得再作为有效表述"的要求。
