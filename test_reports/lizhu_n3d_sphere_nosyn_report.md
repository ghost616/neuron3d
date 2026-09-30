# 离朱测试报告：n3d_sphere 训练产物不再落盘突触信息（state_dict 瘦身 + `_nosyn` 格式段）

- **被测范围**：`n3d_sphere/model.py`（8 个突触类 buffer 改 `persistent=False`）、`n3d_sphere/train.py`（`NOSYN_FORMAT_SEGMENT="nosyn"`、`CHECKPOINT_PATH`/`smoke_fingerprint`/`config_fingerprint`/`full_checkpoint_name` 插 `_nosyn`、默认冒烟产物 `_verify/smoke_nosyn.pt`）、`n3d_sphere/README.md` 与 `.module_agent/n3d_sphere/current_spec.md` 文档披露。
- **结论**：**全部通过**。独立测试 **70/70 PASS**（退出码 0）；冒烟真跑 **15/15 PASS**（退出码 0）；一期回归 **9/9 PASS**（退出码 0）；编译检查退出码 0。**未发现功能缺陷。**
- 测试类型：单元测试（含数值/契约等价性）、编译测试、真跑冒烟/回归、CLI 集成实跑。**E2E（Playwright）不适用**（`check_playwright` → `installed=false`；被测对象为 Python 训练库，无浏览器 UI）。

---

## 一、测试概览

| 测试类型 | 用例数 | 通过 | 失败 | 跳过 | 说明 |
|---|---:|---:|---:|---:|---|
| 编译测试 | 4 模块 | 4 | 0 | 0 | `python -m compileall -q n3d_sphere n3d_proto n3d_viz n3d_shape` → 退出码 0 |
| 单元 / 契约单测（独立脚本，70 项） | 70 | 70 | 0 | 0 | `.lizhu_env/lizhu_nosyn_test.py`，退出码 0 |
| 真跑：n3d_sphere 冒烟 | 15 | 15 | 0 | 0 | `train.py --smoke-test`，退出码 0，产物 `_verify/smoke_nosyn.pt` |
| 真跑：n3d_proto 回归冒烟 | 9 | 9 | 0 | 0 | `n3d_proto/train.py --smoke-test`，退出码 0，`loss=2.419689` |
| 集成实跑：n3d_viz CLI（新产物） | 1 | 1 | 0 | 0 | `python -m n3d_viz -c …/smoke_nosyn.pt`，退出码 0，三件套生成 |
| 接口测试（HTTP API） | — | — | — | — | 不适用：本模块无 HTTP 接口 |
| E2E（Playwright） | — | — | — | 1（跳过） | Playwright 未安装（`installed=false`），且无浏览器 UI 可测 |
| **整链验收（`_verify/verify_all.py` 等）** | — | — | — | **不可执行** | `checkpoints/n3d_sphere/_verify/` 既有脚本与台账不在本副本（详见第五节） |

| 环境 | 值 |
|---|---|
| 解释器 | `C:\Users\wb3094\AppData\Local\Programs\Python\Python313\python.exe`（**不在 PATH**，须显式调用） |
| 版本 | Python 3.13.15 / torch 2.14.0+cpu / `torch.cuda.is_available()=False`（无 CUDA） |
| 编码 | 所有命令设 `PYTHONIOENCODING=utf-8` |
| 独立基线来源 | **`git show HEAD:n3d_sphere/model.py` 与 `train.py`**（HEAD = `05bb849`），落地为 `.lizhu_env/ref/{model_old.py,train_old.py}`，按文件路径动态加载为参照包 —— 对比有**真实参照物**，非自证 |

---

## 二、测试说明逐条对照（功能点 1–7）

### 功能点 1：持久性契约 —— **PASS**
| 判据 | DEFAULT/seed42 | SMALL/seed42 |
|---|---|---|
| `state_dict()` 相对改前**恰好少这 8 个键、不多不少** | **PASS**：少 8（`syn_dist`/`input_syn_pos`/`output_syn_pos`/`representative_syn_out`/`representative_syn_input`/`input_isolated_mask`/`output_isolated_mask`/`neuron_conn_mask`）、多 **0**；**28 键 → 20 键** | **PASS**：同（28 → 20） |
| `named_buffers()` 键集合与改前**逐字相同** | **PASS**：改前 24 项 / 改后 24 项，对称差 `[]` | **PASS**：同 |
| `edge_dist` 与 15 个索引/拓扑量仍在 `state_dict()` | **PASS**：16/16 在册 | **PASS**：16/16 |
| 补充：8 张量**形状/dtype**与改前一致（"只改持久性"） | **PASS**：8/8 | **PASS**：8/8 |
| 补充：`.to(dtype)` 后 24 个 buffer（含 8 个非持久化）仍随模块搬运 | **PASS** | **PASS** |
| 补充：逐条解析源码确认 8 条 `register_buffer` 均为 `persistent=False`（基线均为 `persistent=True`） | **PASS** 8/8 | — |

### 功能点 2：数值不变 —— **PASS**
| 判据 | DEFAULT/seed42 | SMALL/seed42 |
|---|---|---|
| 20 个共同键（参数 + 保留 buffer）改前/改后逐位相同 | **PASS** | **PASS** |
| 同 config/seed/输入下前向输出 `torch.equal` | **PASS=True**，`max|Δ|=0.000e+00`，logits sha 改前=改后=`3fd4f98e…` | **PASS=True**，`max|Δ|=0`，sha=`04d68829…` |
| 8 个非持久化张量的**重算值**与改前**存储值**逐位相同（shape+dtype+value） | **PASS 8/8** | **PASS 8/8** |

> 结论：**不落盘 ≠ 不可还原**成立；"仅改持久性"在数值上零影响。

### 功能点 3：往返可用 —— **PASS**
- `torch.save` 新 `state_dict` → 新类 `load_state_dict(strict=True)`：**无 missing / unexpected**（DEFAULT 与 SMALL 均 PASS）；载入后前向与源模型逐位相同。
- **阴性对照（证明判据非恒真）**：删 `neuron_bias` → `RuntimeError: Missing key(s)…"neuron_bias"`；多加 `bogus_key` → `Unexpected key(s)…"bogus_key"`。
- **格式隔离取证**：改动前的旧格式 `state_dict`（含 8 个突触键）用新类 `strict=True` 装载**被拒**（`Unexpected key(s): syn_dist, input_syn_pos, …`）→ 证明 `_nosyn` 分段留痕确有必要（否则新旧产物互相覆盖且装载语义混淆）。

### 功能点 4：断言保留 —— **PASS**
| 判据 | 结果 |
|---|---|
| `D > H` 仍抛 `ValueError` | **PASS**（`D=0.16, H=0.15`：`Config.D 不得超过 Config.H…`） |
| 退化配置仍抛 `ValueError` | **PASS**（`D=0.03` 小规模：`E=0` 拒绝构造；`N=512, D=0.03`：命中 `[连通性下限校验失败] E=1 < N=512（E/N=0.0020 < 1）` 分支） |
| 契约断言文本未被删减 | **PASS**：9/9 保留（最近邻距 `==2H`、代表间距 `<=D`、代表与 `pair_min` 一致、无反向边、Kahn 无环、拓扑序==轴升序、入边表单调、分层区间覆盖、边分组总数） |
| `__init__` 计算与断言**零改动** | **PASS**：对 HEAD 取全量 diff，`model.py` 的改动**仅有** docstring、注释与 8 条 `register_buffer` 语句；几何计算、参数初始化、断言、`check_connectivity_floor` 等**无一行改动** |
| 实测最近邻距 == 2H | **PASS**：`0.299999952` vs `2H=0.300000000`（偏差 4.77e-08） |

### 功能点 5：产物命名 —— **PASS（10/10）**
| 判据 | 实测 |
|---|---|
| `NOSYN_FORMAT_SEGMENT == "nosyn"` | `'nosyn'` |
| 默认全量 `CHECKPOINT_PATH` | `checkpoints/n3d_sphere/model_nosyn.pt` |
| 默认冒烟 | `checkpoints/n3d_sphere/_verify/smoke_nosyn.pt` |
| 非默认冒烟 | `smoke_N64_y4x4_H0.15_D0.15_plfcc_axz_isany_rsany_bs32_nosyn_s42.pt` |
| `smoke_fingerprint` | 含 `_nosyn_s42`（`_nosyn` 在 `_s42` **之前**） |
| `full_checkpoint_name(DEFAULT)` | `full_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_nosyn_s42.pt` |
| 限批 `config_fingerprint(DEFAULT,200)` | `verify_200_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_nosyn_s42.pt` |
| 带 `--tag` | `…_nosyn_s42_tagX.pt`（全量 + 限批均 `_nosyn` 在 tag 之前） |
| `resolve_checkpoint_path("",0)` / `("",200)` | `model_nosyn.pt` / `verify_200_…_nosyn_s42.pt`（限批重定位语义未变，并打印重定位日志） |
| 非默认配置不污染默认名 | HIGHACC 全量名为 `full_…_nosyn_s42.pt` ≠ `model_nosyn.pt` |
| `is_default_smoke` 判定维度不变 | 源码核对：仍比对 arch / N / y_in / y_out / H / D / seed / flow_axis / placement / 两 scope / space_radius / batch_size（**未新增也未减少任何维度**） |

### 功能点 6：冒烟判据 —— **PASS**
- `python n3d_sphere/train.py --smoke-test` → **15/15 PASS、退出码 0**（重新真跑，非引用既有记录）。
- `loss=2.149648`（产物内 `2.1496481895446777`，与登记值 `2.1496479511260986` 逐位一致）；`E=106`、`|S_in|=55`、`|S_out|=53`、层数 7、可学习参数 43930、单 batch 2.38s（< 120s）。
- 四个梯度范数与 README 登记值逐位一致：`W_in=1.5796021`、`edge_weight=0.2537662`、`neuron_bias=0.1892900`、`W_out=0.4945818`。
- 产物：`_verify/smoke_nosyn.pt` = **192,719 B**，SHA256 = **`175ff12c94f540888d54a96e12211294892d7e858a61ce2d8e491f31821794cc`**（与力牧登记值**完全一致**，且为本次独立重跑所得 → 产物生成**逐字节可复现**）；`model_state_dict` **20 键**（改前 28 键），8 个突触键 8/8 不在产物内，16 个持久化量 16/16 在产物内；`stage=smoke`；可用新类 `strict=True` 装载。

### 功能点 7：一期与可视化未受影响 —— **PASS（且超出说明范围补做了实跑）**
- `n3d_proto`、`n3d_viz` 对 HEAD 的 `git diff --stat` **均为空**（零改动）；`n3d_proto/train.py --smoke-test` 真跑 **9/9 PASS、退出码 0、`loss=2.419689`**（与既有登记一致）；产物仍写 `checkpoints/_verify/smoke.pt`（与 `n3d_sphere/_verify/smoke_nosyn.pt` 物理隔离，无污染）。
- `n3d_viz`：`REQUIRED_KEYS` 12 项**含 `edge_dist`**；`_syn_dist_bytes` 使用 `sd.get("syn_dist")`（缺失不报错）。
- **补做（README 8.0.2 第 3 条原标记"本轮未做"）**：`python -m n3d_viz -c checkpoints/n3d_sphere/_verify/smoke_nosyn.pt` → **退出码 0**，生成 HTML 35,453 B / PLY 1,245 B / OBJ 106 条 l 行；HTML 自包含（无 `http://`、无 `syn_dist`）。**已观测的预期行为**：日志 `[数据] syn_dist 体积 = 0 字节` —— 新产物不含 `syn_dist`，故该展示量由原来的字节数降为 0（**不报错、不影响渲染**，符合"产物不再自证突触几何"的设计口径）。此为**文档未明写的可观测差异**，建议在 README 8.0 节补一句说明。

---

## 三、边界值 / 反向覆盖补充

| 用例 | 结果 |
|---|---|
| 空/缺失键（`neuron_bias` 删除） | `RuntimeError: Missing key(s)` ✔ 正确拒绝 |
| 多余键（`bogus_key`） | `RuntimeError: Unexpected key(s)` ✔ 正确拒绝 |
| 旧格式产物（多 8 键）→ 新类 strict | `RuntimeError: Unexpected key(s)` ✔ 正确拒绝 |
| 退化规模下限（`E=0`） | `ValueError` ✔ |
| 连通性下限分支（`E=1 < N=512`） | `ValueError [连通性下限校验失败]` ✔ |
| 硬约束越界（`D=0.16 > H=0.15`） | `ValueError` ✔ |
| 8 张量在 `state_dict()` 缺失但 `named_buffers()` 仍在 | ✔ 无 AttributeError，重算路径可达 |
| 超长/超大输入数据量 | 未做压力测试（本改动不涉及 O(N·y) 计算路径的复杂度变化；`syn_dist` 仍照常在 `__init__` 构造） |

---

## 四、失败用例分析

**无失败用例**（70/70、15/15、9/9 全 PASS）。

首次运行时出现 3 条 FAIL，经复核**全部为测试脚本自身缺陷**，已修正后重跑通过，特此留痕以示判据非恒真、且已真跑验证：

1. `B0b`：断言写法把基线源码里的**局部变量名**（`rep_syn_out` / `rep_syn_input`）误当作 buffer 名（`representative_syn_out`）比对 → 改为正则解析 `register_buffer("name", <任意变量>, persistent=…)` 后 PASS。
2. `(4b)`：断言要求异常消息含"连通性"，但 `D=0.03` 在小规模下命中的是更早的 `E=0` 分支（消息为"没有任何神经元级连接"）→ 拆分判据：`E=0` 分支 + 新增 `(4b2)` 显式命中"连通性下限校验失败"分支。
3. `(6f)`：`loss` 与 6 位小数常量 `2.149648` 用 `1e-9` 容差比对（差值 1.9e-07）→ 容差与口径对齐后 PASS。

---

## 五、环境问题与方法学限制（如实声明，不声称通过）

1. **整链验收不可执行**：`checkpoints/n3d_sphere/_verify/` 下的既有脚本与台账 —— `verify_all.py`、`verify_sphere_dag.py`（R7）、`verify_scope_and_fingerprint.py`（S4-S5）、`run_smoke_matrix.py`（11 组合落盘名比对与"默认产物未被污染"断言）、`verify_full_runs.py`、`doc_numbers.json`（252 项数字防线）、`artifacts/` 冻结快照、`full_runs.json`、历史 `log_*.txt` —— **在本工作副本中不存在**（`checkpoints/` 被 `.gitignore` 忽略、未随仓库分发，实测该目录下仅有 `smoke_nosyn.pt` 一个文件）。故这些验收项 **本轮未执行、不可复现，本报告不声称其通过**。
   **修复建议**：在具备该目录的副本上重跑；重跑前须先把脚本内按旧名预测的 `smoke.pt` 同步为 `smoke_nosyn.pt`（README 8.0.2 第 1 条已提示）。
2. **无 CUDA**：`torch.cuda.is_available()=False`，GPU 设备路径与 F16 的 CUDA 搬运实验未覆盖（仅以 `.to(dtype)` / 设备一致性探针间接佐证 buffer 仍随模块搬运）。
3. **解释器不在 PATH**：`Get-Command python` 为空，所有命令均以绝对路径调用；若在不设 `PYTHONIOENCODING=utf-8` 的 GBK 管道下运行，中文日志乱码但不影响退出码。
4. **E2E 跳过**：`module_agent_testing(check_playwright)` 返回 `installed=false`；本改动无浏览器 UI，E2E 无适用对象。
5. **方法学透明**：本次"改前"基线取自 `git show HEAD:…`（HEAD=`05bb849`）而非力牧的 `module_agent_backup` 备份，来源独立；`.lizhu_env` 为测试环境目录，未修改仓库内任何被测源码。

---

## 六、修复建议（均为可选改进，非缺陷）

1. **低**：`n3d_viz` 对**新产物**的 `syn_dist 体积 = 0 字节` 属预期行为，建议在 `n3d_sphere/README.md` 第 8.0 节（或 `n3d_viz/README.md`）补一句口径说明，避免后续被误读为"可视化读取失败"。
2. **低**：`.module_agent/n3d_sphere/current_spec.md` 文件**末尾缺少换行**（diff 提示 `\ No newline at end of file`），建议补上。
3. **中（需在具备 `_verify/` 的副本上做）**：`run_smoke_matrix.py` / `verify_sphere_dag.py` / `verify_scope_and_fingerprint.py` / `verify_all.py` 的产物名预测须与 `_nosyn` 对齐后重跑，否则这些脚本会因按旧名预测而误判失败。
4. **信息**：`n3d_shape/{model.py,train.py,README.md}` 在本工作树中同样带有 `_nosyn` 类改动痕迹（`git status` 显示已修改），但**不在本次测试说明范围内**，未做验证；若该模块也已同步实施同一改动，建议单独下一份测试说明进行验证。

---

## 七、可复现命令清单

```powershell
$py = "C:\Users\wb3094\AppData\Local\Programs\Python\Python313\python.exe"
$env:PYTHONIOENCODING = "utf-8"

# 编译检查（4 模块，退出码 0）
& $py -m compileall -q n3d_sphere n3d_proto n3d_viz n3d_shape

# 独立单测 / 契约测试（70/70 PASS，退出码 0）
& $py .lizhu_env/lizhu_nosyn_test.py

# 真跑冒烟（15/15 PASS，退出码 0；产物 _verify/smoke_nosyn.pt）
& $py n3d_sphere/train.py --smoke-test

# 一期回归（9/9 PASS，退出码 0，loss=2.419689）
& $py n3d_proto/train.py --smoke-test

# 新产物在可视化路径上的实跑（退出码 0）
& $py -m n3d_viz -c checkpoints/n3d_sphere/_verify/smoke_nosyn.pt -d .lizhu_env/out/viz_nosyn

# 基线提取（改动前参照物）
cmd /c "git show HEAD:n3d_sphere/model.py > .lizhu_env\ref\model_old.py"
cmd /c "git show HEAD:n3d_sphere/train.py > .lizhu_env\ref\train_old.py"
```

**测试脚本**：`.lizhu_env/lizhu_nosyn_test.py`（70 项判据）
**原始输出**：`.lizhu_env/lizhu_nosyn_test_out.txt`、`.lizhu_env/smoke_sphere_nosyn.log`、`.lizhu_env/smoke_proto.log`
