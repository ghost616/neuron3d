# 离朱测试报告 —— n3d_shape 第 23 轮：产物不再落盘突触信息 + 产物名 `_nosyn` 段 + 文档修订

**测试时间**：2026-09-29（本副本）
**被测模块**：`n3d_shape`
**结论**：**功能实现通过**。共 67 项自动断言 + 8 条 CLI 命令 + 2 个浏览器 E2E 用例；其中 **代码层面 0 缺陷**；**4 项断言失败全部落在文档一致性上（2 个根因，均为 info 级）**。

---

## 一、被测版本与环境

| 项 | 值 |
| --- | --- |
| 解释器 | `C:\Users\wb3094\AppData\Local\Programs\Python\Python313\python.exe`（3.13.15） |
| torch | `2.14.0+cpu` |
| Node / Playwright | node v24.21.0；`@playwright/test` 1.63.0（浏览器下载失败，改用系统 Google Chrome 150.0.7871.187，`channel: chrome`） |
| 被测文件（SHA256） | `n3d_shape/model.py` `dbede6396f1b49680cdd8e522d68a01b305573ba7b35e5f98e2807665d640394`（110,299 B）<br>`n3d_shape/train.py` `d33a1fa8a8fb5c97c5afeafbfc06a342ce085d21c88bca53b0b3866c36d06885`（99,578 B）<br>`n3d_shape/README.md` `e1e774f5e9752eabb53c3d0dd4d151a43c77c81402b3b2c8c75a738acd4f7dd5`（140,229 B）<br>`.module_agent/n3d_shape/current_spec.md` `0f8743a43fa5715280107b4c2e5963be7ae663f41a72be00ad4bb68e70153b75`（58,583 B） |
| 工作区变更范围（`git status`） | `M n3d_shape/{model.py,train.py,README.md}`；`M n3d_sphere/{model.py,train.py,README.md}`（**同期另一任务，按 spec 不计入本轮**）；`n3d_proto` / `n3d_viz` **未改动**（一期零改动口径成立） |

**受保护产物完整性**：`checkpoints/n3d_shape/full_*.pt`、`checkpoints/n3d_shape/_verify/verify_2_*.pt`、`checkpoints/n3d_viz/` 既有三件套 —— 测试前后 SHA256 **逐位不变（UNCHANGED × 8）**，旧格式产物仅被读取。

---

## 二、测试概览

| 测试类型 | 用例/断言数 | PASS | FAIL | 说明 |
| --- | --- | --- | --- | --- |
| 单元测试（结构/数值/命名/守卫/断言存活） | 27 | 27 | 0 | 6 个脚本之一：`test_unit_nosyn.py` |
| 变异测试（断言是否活代码） | 8 | 8 | 0 | `test_mutation_asserts.py`：8 个变异全部被对应断言捕获 |
| 改动前基线逐位比对（5 配置） | 6 | 6 | 0 | `test_baseline_before_after.py` |
| 产物往返与兼容性边界 | 6 | 6 | 0 | `test_artifacts.py` |
| `n3d_viz` 契约回归 | 5 | 5 | 0 | `test_viz_contract.py` |
| 文档一致性（README + current_spec） | 15 | 11 | **4** | `test_docs_consistency.py`（2 个根因，见 §四） |
| **自动断言合计** | **67** | **63** | **4** | |
| 接口测试（CLI 命令） | 8 条命令 | 8/8 符合预期 | 0 | 冒烟 ×3、负例 ×2、compileall、verify_shape、台账类 ×3 |
| E2E（浏览器，Playwright + 系统 Chrome） | 2 | 2 | 0 | FC 产物 HTML、fc0 产物 HTML |

---

## 三、各测试类型详细结果

### 3.1 单元测试（27/27 PASS）

**A 结构（persistent 口径）**
- A1 `_non_persistent_buffers_set` 在 `fc_dim=0 / -1 / 8 / cube / cylinder λ=2` 五种路径下**恰好**等于这 8 个：`syn_dist`/`input_syn_pos`/`output_syn_pos`/`representative_syn_out`/`representative_syn_input`/`input_isolated_mask`/`output_isolated_mask`/`neuron_conn_mask`。
- A2 `state_dict() == 参数 ∪ (buffer − 8 键)`，**恰少这 8 键、不多不少**（fc0 20 键 / fc-1 26 键）。
- A3 `named_buffers()` 键集合与形状/dtype 不变：fc0 组 **24**、fc-1 组 **25**（与 current_spec §产物持久性记录一致）；`syn_dist [256,256] f32`、`representative_* [E] int64`、掩码 int64 等逐项核对通过。
- A4 `edge_dist` 与全部索引拓扑量（`topo_index`/`edge_offset`/`edge_perm`/`edge_perm_in`/`neuron_in_edge_reach`/`edge_dst_in`/`level_edge_reach`/`level_node_reach`/`in_scope_mask`/`out_scope_mask`/`in_degree`/`out_degree`/`edge_src`/`edge_dst`/`neuron_pos`，以及 fc≠0 时的 `out_scope_index`）**仍 persistent=True 且进入 state_dict**。
- A5 `n3d_viz.core.REQUIRED_KEYS` 12 键在新口径 state_dict 下**无缺失**（`missing_required_keys() == []`）。

**B 数值零变化**
- B1 同 config+seed 两次构造：全部 buffer/参数逐位相同（确定性）。
- B2 旧格式产物（`verify_2_..._fc-1_s42.pt`，含 8 键）与 `config+seed` 重算结果：共享 **17 个 buffer** 逐位相同；8 个目标张量**逐位相同**（`torch.equal` 全 True，含 shape/dtype）；键差集恰为这 8 键。
- B3 forward 零变化：用产物内 8 张量 vs 用重算张量（同参数、同输入 `[4,784]`）→ `torch.equal` 成立；两次全新构造 forward 亦逐位相同。
- B4 8 张量可复算自证：`syn_dist == cdist(output_syn_pos, input_syn_pos)` 逐位；`representative_syn_out // y_out == edge_src`、`representative_syn_input // y_in == edge_dst`；`neuron_conn_mask.sum() == E`。

**C `fc_dim` 族零改动**
- C1 `fc_dim=0`：`W_in`/`W_out` 为 Parameter，无 `fc_*` 参数、无 `out_scope_index`。
- C2 `fc_dim=-1/-8`：`fc_in_weight`/`proj_weight`/`fc_out_weight`/`head_weight` 仍为 Parameter（`proj_weight.shape == (|S_in|, H)`），`out_scope_index` 仍为 int64 持久 buffer，`W_in`/`W_out` 不创建。

**D `_nosyn` 命名口径（6 种配置 × 3 处函数）**
- D1 三处（`smoke_fingerprint`/`config_fingerprint`/`full_checkpoint_name`）**恒定含且仅含 1 个** `_nosyn`；位置在 `_s{seed}` 之前；`fc_dim != 0` 时字面包含 `_fc{n}_nosyn`（`-1` → `_fc-1_nosyn`）。
- D2 `fc_dim=0` 三处名字**均不含** `_fc`；`--tag` 仍追加在**末尾**；`bad tag`/中文/`a/b`/`x.y` 四种非法 tag 仍抛 `ValueError`。
- D3 默认组合 → `checkpoints/n3d_shape/_verify/smoke_nosyn.pt`；非默认组合 → `smoke_{指纹}.pt` 含 `_nosyn`；`--arch mlp` → 名含 `_armlp`。
- D4 `verify_*`（限批）与 `full_*`（正式）产物名均含 `_nosyn`。
- D5 实测 `smoke_fingerprint(SMALL_CONFIG) == shapesphere_N64_y4x4_H0.15_D0.15_plfcc_axz_isany_rsany_bs32_nosyn_s42`；spec 点名的两份新格式产物均存在。

**E 断言存活（AST + 注入 + 负例 + 变异，见 §3.5）**
- E1/E2/E3 AST 核查：`_build_neuron_edges` 3 处契约断言、2H 两道防线（`< 1e-9`、`1e-6*scale + 3e-6`）、连通性下限 4 条判据均仍在且未被常量假条件包裹。
- E4 打桩注入 `_nearest_neighbour_distance` 返回 `+1e-8` → **float64 防线**按 `[契约失败] (float64 格点)` 拦截。
- E5 第一道放行、第二道收到 `+1e-5` → **float32 防线**按 `[契约失败] (float32 坐标)` 拦截。
- E6 负例配置 `N=256,y=8x8,H=0.1,D=0.03` → `ValueError [连通性下限校验失败]`（E=17 < N=256）。
- E7 就地打桩指标，4 条下限分支运行时逐一触发（`E<N`、`K<2`、`|S_in|<1`、`|S_out|<1`），报文逐条核对。
- E8 非法 Config（`shape="sphere "`、`cylinder λ=0`、`cube + λ=2`、`fc_dim=-2`）仍被拒绝。

**F `is_default_smoke` 维度对齐守卫**
- F1 AST：`is_default_smoke` 布尔块内**不含** `_nosyn`（格式常量不入判定，判定维度与改动前相同）；守卫表达式 `smoke_fingerprint(config) != smoke_fingerprint(small)` 存在。
- F2 **运行时注入**：打桩 `smoke_fingerprint` 使两次调用不一致（默认判定为真）→ 如期抛错
  `ValueError: is_default_smoke 与 smoke_fingerprint 的维度不对齐：… 继续执行会**静默覆盖默认冒烟产物** _verify/smoke_nosyn.pt…`（守卫是活代码，不会静默覆盖）。
- F3 正向：默认组合指纹与 `Config(**SMALL_CONFIG.to_dict())` 等价；7 个可变维度（N/y_in/y_out/H/D/seed/batch_size/space_radius）+ 4 个判据维度（flow_axis/scope×2/shape）**全部进入指纹**（`placement` 仅有 `fcc` 一种取值，非可变维度）。

### 3.2 接口（CLI）测试 — 8 条命令全部符合预期

| 命令 | 期望 | 实测 | 关键证据 |
| --- | --- | --- | --- |
| `python n3d_shape/train.py --smoke-test` | 16/16、退码 0 | **16 PASS / 0 FAIL，退码 0** | 产物 `_verify/smoke_nosyn.pt`；日志含 `[产物格式] nosyn：8 个突触类张量…persistent=False、不进入 state_dict；edge_dist 与全部索引拓扑量仍持久化（n3d_viz 契约）` |
| `--smoke-test --fc-dim -1` | 16/16、退码 0 | **16 PASS / 0 FAIL，退码 0** | 产物 `_verify/smoke_shapesphere_N64_y4x4_H0.15_D0.15_plfcc_axz_isany_rsany_bs32_fc-1_nosyn_s42.pt`（256,437 B） |
| `--smoke-test`（复跑，验字节稳定） | 同名同 SHA | **退码 0、16/16、SHA 不变** | 前后均 `a1f4772f797487eadf37ce777ad367fbda02af22d5540bd41286c6c83eaeb6fc`，193,551 B |
| `python -m compileall -q n3d_shape` | 退码 0 | **退码 0** | 无输出 |
| `--smoke-test --shape cube --cyl-aspect 1.5` | 退码 2 | **退码 2** | `参数校验失败：--cyl-aspect 仅在 --shape cylinder 时生效…（拒绝静默无效参数）` |
| `--smoke-test --shape cylinder --cyl-aspect 5.0` | 退码 2 | **退码 2** | `参数校验失败：形状专属尺寸窗口为空：R_max=0.260127 < R_min=0.338810` |
| `python n3d_shape/verify_shape.py --quick` | 78/78、退码 0 | **断言总数 78；通过 78；失败 0，退码 0** | 含 `S11 指纹可区分性`、`S12b 全部张量 torch.equal [SMALL/sphere] —— 比对 27 个张量，不一致=无` |
| `python -m n3d_viz -c <去 8 键临时产物> -d <临时目录> -o <临时 HTML>` | 退码 0、FC 识别、三件套 | **退码 0** | `[FC] 两端全连接包裹：fc_dim=-1 H=256 [13,256]×[256,14]`；产物 HTML(160,615 B)/PLY/OBJ 齐备 |

> 另：`--tag` 非法字符 → `ValueError --tag 仅允许字母/数字/下划线/连字符`（单测 D2 覆盖）。

### 3.3 编译测试

`python -m compileall -q n3d_shape` → **退码 0**（无语法/字节码编译错误）。

### 3.4 E2E 测试

**(a) CLI 级端到端链路**（无浏览器）：`train.py --smoke-test` 落盘 → 产物 `strict=True` 往返加载（missing=[] unexpected=[]）→ `python -m n3d_viz` 渲染三件套 → 退出码 0。全链路成功，覆盖“训练产物 → 复用 → 可视化”的核心用户旅程。

**(b) 浏览器级 E2E（Playwright + 系统 Chrome，2/2 PASS）**：对 nosyn 格式产物渲染出的自包含 HTML 做真实浏览器验证（`file://` 加载）：

| 用例 | 断言 | 结果 |
| --- | --- | --- |
| FC 产物（去 8 键的 full 产物） | 页面加载无 pageerror/console.error；内联 JS 渲染 `#stats`；`#view` 画布布局非 0 尺寸；负载含 `"fcDim":-1`/`"hasFc":true`/`proj_weight`；阈值滑块 0.30→0.9→0.3 使保留边数 **372 → 13 → 372**；连线复选框切换生效；刷新后状态复位（无残留） | **PASS（2.3s）** |
| fc0 新格式产物 `smoke_nosyn.pt` | 同上渲染与交互；负载**不含**任何 FC 段（无 `fcDim`/`hasFc`/`proj_weight`）；阈值 0.30→0.9 使保留边 **43 → 0** | **PASS（2.1s）** |

`#stats` 实测文本（FC 用例）：`checkpoint: viz_sim_nosyn_fc-1.pt seed=42 N=256 E=736 K=9 S_in=13 S_out=14 密度=1.12e-2 测试准确率=0.9727 层规模: 13/24/37/35/39/34/37/24/13 当前保留边: 372`。
> 浏览器获取方式的环境说明：`npx playwright install chromium` **下载失败**（网络/TLS 超时，`Error: Failed to download Chrome for Testing 153.0.8010.12 … Download failure, code=1`，退码 1，日志 `.lizhu_env/pw/pw_install2.log`）；改用系统已安装 Chrome（`channel: chrome`）完成 E2E，未降级为跳过。

### 3.5 变异测试：断言「未被移除」且是活代码（8/8 PASS）

对 `n3d_shape` 包做**整包副本 + 单点文本变异**（不触碰被测源码），在子进程中构造 `SMALL_CONFIG` 模型，要求对应断言抛出：

| 变异 | 目标断言 | 结果 |
| --- | --- | --- |
| M1 `best_dist <= D + 1e-6` → `D - 1.0` | 代表连接间距 ≤ D | 捕获：`[契约失败] 代表连接间距超过 D` |
| M2 `pair_min…` → `+ 1.0` | 代表连接与 pair_min 逐位一致 | 捕获：`[契约失败] 代表连接间距与 pair_min 不一致（块内索引错位）` |
| M3 `<` → `>`（轴高比较） | 无反向边 | 捕获：`[契约失败] 构图出现 z_A >= z_B 的反向边` |
| M4 float64 容差 1e-9 → 负值 | 2H 第一道防线 | 捕获：`[契约失败] (float64 格点) … 实测 0.2999999999999999` |
| M5 `tol_f32` → `-1.0` | 2H 第二道防线 | 捕获：`[契约失败] (float32 坐标) …（尺度感知容差 -1.000000e+00 …）` |
| M6/M7/M8 下限阈值 `e<n` / `layers<2` / `s_in<1` → `10**9` | 连通性下限 3 条分支 | 均捕获：`[连通性下限校验失败] … E=106 < N=64 / 层数 K=7 < 2 / |S_in|=55 < 1` |

结论：缺陷轮次反复出现的「断言被删除/空转」风险在本轮**未复现**；8 个变异 0 逃逸。

### 3.6 产物往返与兼容性边界（6/6 PASS）

- G1 新格式 `smoke_nosyn.pt`：`strict=True` 往返 **missing=[] unexpected=[]**，重建模型 state_dict 键集合与产物**逐字相同**（20 键），逐键 `torch.equal`。
- G2 新格式 fc-1 产物（26 键）：同上，且 FC 四键齐全。
- G3 旧格式 `verify_2_..._fc-1_s42.pt`：`strict=True` **必报**，报文首行 `Error(s) in loading state_dict for ThreeDNeuronSpace:`、列出 `Unexpected key(s) in state_dict:` 且**这 8 个键逐一命中**，无 missing。
- G4 同产物 `strict=False` 可复用：missing=[]，unexpected 恰为这 8 键；重算的 8 张量与产物内旧值**逐位相同**。
- G5 正式旧产物 `full_*.pt`：键集合恰比现模型多这 8 键（不多不少）。
- G0 受保护产物哈希测试前后一致（只读约束成立）。

### 3.7 改动前基线逐位比对（5 配置 × 6 项，全 PASS）

数据源：`checkpoints/n3d_shape/_verify/nosyn_probe/baseline.json`（改动前落盘基线）。**先验证基线确为改动前**（其 `state_dict_keys` 仍含这 8 键且 = 当前 + 8），再独立复算比对：

| 断言 | 结果 |
| --- | --- |
| 全部 buffer 字节级 SHA256（24/24/25/25/24 个 × `small_sphere_fc0`、`small_sphere_fc-1`、`default_sphere_allall_fc-1`、`default_cube_fc0`、`default_cyl2_fc0`） | **全部逐位相同**（含 8 个突触类张量） |
| 全部 Parameter SHA256 | **逐位相同**（参数量 43930/58036/211690/156409/160333 不变） |
| `forward` 的 `a_in`/`a_up`/`h`/`logits` 字节 SHA256 | **全部逐位相同** |
| `state_dict` 相对基线 | **恰少这 8 键，无新增键** |
| buffer 形状 / dtype | 不变 |
| `persistent` 标记 | 仅这 8 个由 True → False，其余全为 True |
| 实现方探针 `probe_snapshot.py after` 复跑 | **PASS、退码 0**（独立复核一致） |

### 3.8 文档一致性（11/15 PASS；4 项失败见 §四）

通过项：§5.1 与 §19（19.0–19.5）齐备；§0/§7.1/§8.2/§8.4/§9/§11/§13.0/§14.3/§15.4/§16.2/§17.1/§18.1/§18.4 均已同步本轮口径（`_nosyn` 或“第 19 轮修订/失效/改名”标注）；硬约束第 1 条已改写为「一期零改动 + 二期同批同步改造」并**写明理由**（跨模块格式口径/同批同步/数值逐位不变）；`_nosyn` 与 `_fc{n}` 相对位置已写明；8 个 buffer 现为 `persistent=False`、不进 `state_dict()`、复核回到 `config+seed` 已写明；兼容性边界（`strict=True` → `Unexpected key(s) in state_dict`，复用须 `strict=False`）已写明；名字类记录已标注「已随本轮改名」且既有实测数字（`62 个`、`78/78`、`145/145`、`16/16`、三处 SHA256 等）保留；§19.5 五条披露齐备；current_spec「产物持久性（nosyn 口径）」节齐备。

---

## 四、失败用例分析（4 项失败 → 2 个根因，均为**文档**问题，info 级）

### D1（info，文档数字自洽性）★ 新发现

**位置**：`n3d_shape/README.md` §19.2 表格（第 1673 行）、`.module_agent/n3d_shape/current_spec.md`「本轮冒烟产物」表（第 609 行）。

**现象**：两处均记 `smoke_nosyn.pt` 为 **192,755 B**，但**同一行**记录的 SHA256 `a1f4772f797487eadf37ce777ad367fbda02af22d5540bd41286c6c83eaeb6fc` 对应的文件实测为 **193,551 B**（本副本 3 次复跑均同 SHA、同 193,551 B，`os.path.getsize` 与 `Get-Item` 双口径一致）。

**佐证**：192,755 B 正是 README 第 1693–1695 行为**历史名 `smoke.pt`**（SHA `e9c83f61…`）记录的字节数 —— 即表格“字节”列沿用了旧文件的大小，未随改名/重跑更新；产物改名必然改 SHA/字节（§19.2 自身已披露 zip 条目名带基名前缀），故该列与同行 SHA 自相矛盾。

**影响**：无功能影响；但违反 §19.2/§19.4 自身“该产物确由最终源码产出且可复现”的可复核承诺（读者按 SHA 核对会发现字节数对不上）。

**修复建议**：把两处 `192,755` 更正为 `193,551`（SHA 列不动）。相关断言：I12（README）、I14（current_spec）。

### D2（info，失效表述未就地标注）

**位置**：`n3d_shape/README.md` 第 607–608 行（§12.2「需风后决策的事项」末句）：
> …则须**同时**修改二期 `n3d_sphere` 并重跑二期全部取证（与"二期零改动"冲突，需显式豁免）。

**现象**：该句仍以已失效的「二期零改动」硬约束为前提，且**未加**“第 19 轮修订/该历史轮次实测记录”标注；同章其它位置（§13.0 第 644 行、§14.3 第 772 行、§15.4 第 863 行、§0/§8.4/§9/§18.1/§18.4）**均已就地标注**，说明本轮同步漏了这一处。它位于“## 12 离朱实测缺陷与处置（测试后修订记录）”历史章节内，章节标题有一定覆盖，但按 spec 第 6 项“历史轮次记录允许保留，但必须就地标注”的标准仍属残留。

**影响**：无功能影响；读者可能误以为该“需显式豁免”的决策项仍然有效（事实是二期本轮已同批同步改造，§19.3）。

**修复建议**：在该行后追加就地标注，例如「（**第 19 轮修订**：本句为该历史轮次的决策记录；该硬约束第 1 条已改写为“一期零改动 + 二期同批同步改造”，见 §19.3）」。相关断言：I3、I3b。

### 观察项（非缺陷，仅留档）

- **O1**：`checkpoints/n3d_shape/_verify/nosyn_probe/baseline.json` 的 `buffer_meta[*].persistent` 字段**全为 `false`**（探针首版极性写反，见 `probe_snapshot.py` 注释“探针首版写反，已修”）。该字段不可用作前后参照（本报告的持久性结论来自 `_non_persistent_buffers_set` 的**直接复算**，不受影响）；文件内的哈希、键集合、forward 哈希等**有效**，已用于 §3.7。
- **O2**：仓库根目录遗留临时探针文件 `_tmp_inspect_smoke.py`、`_tmp_nosyn_size.py`、`_tmp_nosyn_verify.py`、`_tmp_*_out.txt`、`_tmp_smoke_out.txt`、`_verify_tmp/`（8 个 `.pt`，约 38 MB）—— 其内容指向 **`checkpoints/n3d_sphere/`**，属**同期另一任务**的产物，**不计入本轮缺陷**；但 `.gitignore` 未忽略它们（`git status` 显示为 `??`），提交前建议清理或补忽略规则。
- **O3**（测试方自身说明，已整改）：本次为安装 Playwright 浏览器，曾误在仓库根执行 `npm install`，生成了根目录 `node_modules/`、`package.json`、`package-lock.json`；已**全部删除并复核干净**（后续安装均正确限定在 `.lizhu_env/pw/` 内）。当前工作区未因该操作留下残留。

---

## 五、已知不可复现项（按 spec 声明，**不记为缺陷**）

`checkpoints/n3d_shape/_verify/` 的历史台账在本副本不存在（目录被 `.gitignore` 忽略），故：

| 命令 | 实测 | 原因 |
| --- | --- | --- |
| `python n3d_shape/verify_full_runs.py` | 退码 1 | `[FAIL] 账本不存在：…\_verify\full_runs_shape.json` |
| `python n3d_shape/verify_ladder.py` | 退码 1 | `[FAIL] 台账不存在：…\_verify\ladder_runs.json` |
| `python n3d_shape/verify_fc_alignment.py` | 退码 1 | `[FAIL] 台账不存在：…\_verify\fc_alignment_runs.json` |

三者**均只因台账缺失而失败**，与代码无关；其冻结 SHA256 承重断言在本副本不可复现。

---

## 六、环境问题与修复建议

1. **Playwright 浏览器无法下载**（唯一环境受限项）：
   - 命令（在 `.lizhu_env/pw/` 内执行）：`npx playwright install chromium` → 退码 1
   - 错误：`Failed to install browsers / Error: Failed to download Chrome for Testing 153.0.8010.12 (playwright chromium v1243), caused by Error: Download failure, code=1`（TLS/网络超时）
   - 处置：改用系统已安装 **Google Chrome 150.0.7871.187**（`channel: 'chrome'`）完成浏览器 E2E，**未跳过**。
   - 修复建议：放开 `cdn.playwright.dev`/`storage.googleapis.com` 出网，或预置 `PLAYWRIGHT_BROWSERS_PATH` 缓存。
2. 其余测试**无环境阻塞**（torch、数据、产物齐备）。

---

## 七、复现命令清单

```powershell
$py = "C:\Users\wb3094\AppData\Local\Programs\Python\Python313\python.exe"
# 单元 / 变异 / 基线 / 产物 / viz / 文档
& $py .lizhu_env\n3d_shape_r23\test_unit_nosyn.py            # 27/27 PASS，退码 0
& $py .lizhu_env\n3d_shape_r23\test_mutation_asserts.py      # 8/8  PASS，退码 0
& $py .lizhu_env\n3d_shape_r23\test_baseline_before_after.py # 6/6  PASS，退码 0
& $py .lizhu_env\n3d_shape_r23\test_artifacts.py             # 6/6  PASS，退码 0
& $py .lizhu_env\n3d_shape_r23\test_viz_contract.py          # 5/5  PASS，退码 0
& $py .lizhu_env\n3d_shape_r23\test_docs_consistency.py      # 11/15（3 项文档问题）
# CLI
& $py n3d_shape\train.py --smoke-test                        # 16/16，退码 0
& $py n3d_shape\train.py --smoke-test --fc-dim -1            # 16/16，退码 0
& $py -m compileall -q n3d_shape                             # 退码 0
& $py n3d_shape\verify_shape.py --quick                      # 78/78，退码 0
# 浏览器 E2E（工作目录 .lizhu_env\pw）
$env:VIZ_HTML="D:\git\neuron3d\.lizhu_env\n3d_shape_r23\viz_tmp\out_fc-1\viz_sim_nosyn_fc-1.html"; $env:E2E_EXPECT_FC="1"
& npx.cmd playwright test viz_e2e.spec.js --config=pw.config.js   # 1 passed
```

**测试资产**：`.lizhu_env/n3d_shape_r23/`（6 个测试脚本 + 机读结果 `results_*.json` + 全部日志）、`.lizhu_env/pw/`（Playwright 工程与 E2E 日志）。注：受保护产物哈希基线 `protected_hashes_before.txt` 为 UTF-8 **带 BOM**（读取需 `utf-8-sig`）。

---

## 八、总评

- 本轮改动**只改 buffer 持久性 + 产物名格式段 + 文档**：5 个配置的 24/25 个 buffer、全部参数、`forward` 四个中间量的字节级 SHA256 与改动前基线**逐位相同**；`state_dict` **恰少这 8 键、不多不少**；`named_buffers()` 键集合/形状/dtype 未变；`edge_dist` 与全部索引拓扑量保持持久化，`n3d_viz` 契约（`REQUIRED_KEYS`、FC 分支、三件套）回归通过。
- `_nosyn` 命名三处一致、默认冒烟产物名与文档一致、`is_default_smoke` 维度对齐守卫为**活代码**（运行时注入已证）。
- 断言体系（3 处图契约 + 2H 两道防线 + 连通性下限）**全部存活**：8 个变异 0 逃逸、2 处打桩注入均被拦截、4 条下限分支运行时全部触发。
- 仅存 **2 个文档问题（info 级）**：产物字节数与同行 SHA 自相矛盾（README + current_spec 各一处）；§12.2 一处失效表述缺就地标注。**无代码缺陷，无需返工**；建议一并修正上述两处文档后再收口。
