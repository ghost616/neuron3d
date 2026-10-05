# n3d_qa_learn 步骤 2（文本数据集匹配 + 分项验收）测试报告

- 测试智能体：离朱（R40）
- 测试时间：2026-10-05（本地 UTC+8）
- 工作目录：`E:\neuron3d`；解释器：`.venv\Scripts\python.exe`（Python 3.12.10，torch 2.14.1+cpu，numpy 2.5.3）
- 被测对象：`n3d_qa_learn/step2.py`、`n3d_qa_learn/step2_run.py`、`n3d_qa_learn/README_step2.md`
- 本轮新增测试脚本：`lizhu_r40_scripts/verify_step2.py`（产物级 A1–A6/A8/A9）、`lizhu_r40_scripts/verify_step2_model.py`（模型级 M0/A7/M2/M3）
- 明细 JSON：`lizhu_r40_scripts/_tmp/artifact_assertions.json`、`lizhu_r40_scripts/_tmp/model_assertions.json`

## 一、测试概览

| 项目 | 结果 |
| --- | --- |
| `python -m compileall -q n3d_qa_learn` | **通过**（exit 0） |
| `step2_run probe` | **通过**（exit 0，与 README 4.1 逐项一致） |
| `step2_run guard` | **通过**（exit 0，`all_tampered_rejected=True`） |
| `step2_run drill` | **失败（exit 1）**：`AssertionError: 单条演练要求全部可学习参数梯度非零；实测零梯度 = ['backbone.W_in','backbone.edge_weight','backbone.neuron_bias','backbone.W_out']` |
| 8 项真实产物驱动独立复算（A1–A6、A8、A9） | **8/8 通过** |
| 模型级断言（M0/A7/M2/M3） | **4/4 通过**（M0 记录产物不可加载、M2 记录骨干不连通） |
| `step2_run eval` | **未重跑**（规格允许：约 25 分钟长跑，仅在需复现报告数字时执行；改为对既有报告独立复算） |
| 上游零改动 | **通过** |

**结论：9 条验收判据 7 通过 / 2 不成立（drill 判据失败），另有 2 项「报告/产物与代码不匹配」的真实缺陷；8 项独立复算断言全部成立。判定：不通过，需返工。**

## 二、分类型结果

### 2.1 编译测试（通过）
```
.venv\Scripts\python.exe -m compileall -q n3d_qa_learn   -> exit 0
```

### 2.2 CLI 入口测试
| 入口 | 命令 | 退码 | 关键实测 |
| --- | --- | --- | --- |
| probe | `--product-dir checkpoints/qa_learn/dataset --out-dir checkpoints/qa_learn/_verify/step2_lz` | 0 | 文本行 2665；库/查询 1999/666，交集 0，并集 2665，复现==行索引表 True；特征最大偏差 5.066e-07、超容差 0 行；答案表 298 类；分层留出 judge 2→1、choice 4→1、blank 153→31、solve 78→16、triviaqa 100→20，并集 68 → C=230；合并记录 12727（train 4576/3908、test 2245/1998） |
| guard | `--out-dir checkpoints/qa_learn/step2 --verify-dir .../step2_lz` | 0 | 合法产物 [2665,192]；篡改键表→拒绝；篡改口径指纹→拒绝；`all_tampered_rejected=True` |
| drill | `--product-dir checkpoints/qa_learn/dataset --out-dir .../step2_lz` | **1** | 训练完成（256 样本/2 batch/0.11s），`n_parameters_checked=6`、**零梯度 4 个** → `nonzero=False` → 断言失败 |
| eval | 未重跑 | — | 既有报告由 A8/A9 与第 5 节逐位复算 |

### 2.3 单元/集成测试（独立复算，真实产物驱动）
见第四节；全部通过。

### 2.4 E2E 测试（不适用）
本模块为离线数据/训练管线，无前台 UI 页面，Playwright 不适用（未调用 `check_playwright`）。

## 三、失败用例分析（真实缺陷）

### 缺陷 1（阻断验收）：`drill` 硬门禁必然失败 —— N3D 骨干与计算图断开
- 现象：`drill` 退码 1，`gradient.nonzero=False`，零梯度恒为 4 个骨干参数。判据「drill 退码 0 且 `gradient.nonzero==True`」不成立。
- 根因（现网取证，非推测）：`step2.build_model()` 只传 `dim/output_mode/label_smoothing`，未传 `head_input_mode`；`heads.py` 的 `N3DQAConfig.head_input_mode` 默认值现为 **`"raw"`**（heads.py L106），而 `N3DQA.query()` 的 `raw` 分支为 `q = q_head(F.normalize(features))`，**不调用 `self.adapter.features(...)`**（heads.py L261–270）→ 骨干既不参与前向也无梯度。
- 独立证据（M2，现网复算；按 `train_model` 口径先做质心初始化再反传）：

  | head_input_mode | 骨干梯度绝对和 | 骨干一步位移 | 结论 |
  | --- | --- | --- | --- |
  | `raw`（当前默认） | 全部 0.0 | 全部 0.0 | 骨干断开 |
  | `concat` | W_in 20.73 / edge 4.10 / neuron 13.51 / W_out 190.87 | ≈0.01 | 连通 |
  | `n3d` | W_in 35.38 / edge 5.56 / neuron 22.16 / W_out 317.86 | 非零 | 连通 |

- 同源影响：`step2.train_model()` 仍把骨干参数放进优化器并 `requires_grad(True)`，但在 `raw` 下永不更新——「N3D 后端参与步骤 2」在当前默认配置下不成立。
- 修复方向：`step2.build_model` 显式固定 `head_input_mode`；或把断言改为「**参与前向的参数**零梯度为空」并显式列出豁免（`N3DQA.zero_grad_parameters()` 已提供该接口，当前未使用）。二者必居其一。

### 缺陷 2（阻断复现）：冻结模型产物与当前代码架构不兼容
- 现象：7 个 `qa_step2_model_*_D192_C230_s*.pt.zip` 的 `state_dict` 中 **`q_head.weight` 形状 [192, 384]**；而当前**任何**代码路径构造出的 `q_head` 均为 **[192, 192]**（heads.py L186 恒为 `nn.Linear(dim, dim)`；`raw`/`concat`/`n3d` 三种模式皆然）。
- 后果：`torch.load` 加载全部抛 `RuntimeError: size mismatch for q_head.weight: copying a param with shape torch.Size([192, 384]) ... current model is torch.Size([192, 192])`（`concat` 另报 `Missing key(s): "mix_logit"`）。7 个产物 SHA256 与 README 第 5 节表格逐位一致 → 产物未被篡改，是代码侧口径变更导致产物失效；产物 meta 未记录 `head_input_mode`，无法自证口径。
- 推断（有据）：产物训练时的实现为「`concat` = 原始特征 ∥ N3D 读出拼接进 q 头（输入宽 2D=384）」；`heads.py`（mtime 14:29:02）已改写为「`concat` = 凸混合、输入宽 D」，旧形状再也无法构造。
- 后果：报告 4.4–4.7 的 `Recall@1/@5`、双模式一致率、①②③④⑤ 训练后量**无法由产物复算**，只能重训；而按当前默认重训得到的是骨干不参与的另一种模型。

### 缺陷 3（文档与实现不一致）：README 的 drill 取证与「concat 已应用」陈述不再成立
- README 4.2 记「检查 6 个可学习参数、零梯度参数 0 个（`nonzero=True`）」，依据 `_verify/step2/drill.json`（`created_utc=2026-10-05T06:11:30Z`，mtime 14:11:32）中 `backbone.W_out=36.76` 等非零值——这在 `raw` 口径下不可能出现；`heads.py` 于 14:29 改写后现场重跑 drill 已复现为失败。
- README 6.9 称「本文档全部数字来自 13:44 之后、应用适配后的同一棵树」，与产物 `q_head=[192,384]` 矛盾。README 4.2 与 6.9（含 `metrics_sha256`）须在重跑后重写。

## 四、8 项独立复算断言（全部通过）

| # | 断言 | 结果 | 关键实测 |
| --- | --- | --- | --- |
| 1 | 库/查询划分可复核 | PASS | `default_rng(20261005).permutation(2665)`：库 1999、查询 666、交集 0、并集 2665；复现查询集合 == `doclines_row_index.jsonl` 中 `positives>=1` 的 666 行；`index` 字段与行序一致；与 meta 声明一致 |
| 2 | 特征实现与产物等价（全部 2665 行） | PASS | 自配对口径 vs 产物 `feature[:192]`：最大绝对偏差 **5.066394805908203e-07**（README 5.066e-07 同值）、超容差 0 行；反面对照换成 `buckets_per_order=32` 偏差升到 **0.894**，证明比对由产物驱动 |
| 3 | 行向量口径 | PASS | `dim==192==64*3`；`spec_hash==8f2523e4…9fae` 与产物一致；重复编码逐位一致；**不消耗 numpy/torch 全局 RNG**；空文本全零无 NaN |
| 4 | 冻结键表守卫 | PASS | 现场对 float32 字节重算 `key_table_sha256=121f27a2…4f21`、`line_ids_sha256`、`spec_hash` 均与 meta 一致；**自行构造 4 件篡改产物**（改张量/改口径指纹/改 meta 键表指纹/改维数）**全部抛 ValueError**；既有 `guard.json` 复核通过 |
| 5 | 开集协议可复算 | PASS | judge 2→1、choice 4→1、blank 153→31、solve 78→16、triviaqa 100→20（seed 20261010+i）；并集 **68**、C=**230**；清单 SHA256 `2944e155…9605` 与报告+probe **双侧一致**；留出类均在答案表且不在候选空间，且都是其任务 `label==1` 的真实候选 |
| 6 | 自检索命中定义 + rank 方向 | PASS | 池 2665、库序==键表行 id 序；666 查询金标**全部在池中**；人工 3 行样例：文本一致→rank1，换成他行文本→`Recall@1=0.0`（rank 3，**方向正确**）；金标缺池抛错；报告池/查询口径 2665/666 |
| 8 | 跨 seed 聚合可复算 | PASS | 由 `per_seed` 独立重算 5 任务 × 7 字段的均值/极差/最小/最大/n，与 `cross_seed` **最大绝对偏差 0.0**（容差 1e-12） |
| 9 | 训练后量取值域 + 如实登记（不设门槛） | PASS | 全部训练后量 ∈ [0,1]；④ 精确率/召回在三 seed 五任务**恒为 0.0**（如实登记，未当作通过项）；② 逐任务同值且 `step2_recall_at_1_is_task_independent=True`；`gain` 复算 == 报告 `-0.0930930930930931`（<0，README 6.2 已登记） |

**模型级补充**

| # | 断言 | 结果 | 关键实测 |
| --- | --- | --- | --- |
| 7 | 双模式结构性取证 | PASS | 同一模型以 `index`(2D matmul)/`pointer`(bmm) 对同一批 **666 行**打分：`raw` 模型 top-1 一致率 **1.0**（666/666）、logits 最大偏差 **6.56e-07**；`concat` 一致率 **1.0**、偏差 **1.49e-07**；均 **< 1e-5**；`logits_bitwise_equal=False`（与 README 4.6 一致）；`index_key_table_requires_grad=False` |
| M0 | 产物↔代码兼容性 | 记录 | 见缺陷 2 |
| M2 | 骨干连通性 | PASS | 见缺陷 1 表 |
| M3 | probe/drill/report 外部事实 | PASS | probe 2665/192/68/230/合并 12727；报告消费产物 **15 个、缺失 0**；`dim=192`、`backend=n3d_shape`、`modes=[index,pointer]`、`seeds=[42,43,44]`、12 epoch |

## 五、报告内部一致性与复算边界
- `metrics_sha256` 现场重算 = **`ca7e2c2c4577d000`**，与 README 第 5 节一致。
- README 第 5 节 7 个产物的「字节数 + SHA256 前 16 位」**全部现场复算一致**：键表 `5c812e44696827ab`；index s42 `7010b86069b8016f`、pointer s42 `0c7002a7039a2c05`、index s43 `dd0b21c0f8b175a6`、pointer s43 `3a52399a5ed7b79d`、index s44 `0692ce089afda779`、pointer s44 `3784d28bbf3ebdec`。
- `probe`/`drill`/`guard` 既有 JSON 与 README 4.1/4.2/4.3 逐项一致。
- **不能复算的部分（如实登记）**：4.4–4.7 的 Recall 与分项属训练后量，本轮未重跑 eval，且因缺陷 2 **无法由产物复算**；A8/A9 只证明报告内部自洽与取值域合理，**不构成对训练后量的独立复现**。规格禁止对训练后量断言门槛，本报告遵守。

## 六、上游零改动核验（通过）
| 核验项 | 结果 |
| --- | --- |
| `git status --porcelain -- n3d_shape n3d_sphere n3d_proto` | 空（无跟踪改动） |
| `git status --porcelain -- n3d_qa` | `M README.md`、`M __init__.py`、`?? adapters.py / build_qa.py / probe_zh.py / verify_qa.py / zh_features.py / tools/` |
| 上游改动归属 | 上述文件 mtime **14:02–14:03**，早于步骤 2 产物与报告（14:11–14:36）；属**步骤 1**（n3d_qa 数据集构建）在制改动，非步骤 2 引入 |
| 报告消费的 15 个 n3d_qa 冻结产物 | 现场逐个重算 SHA256/字节数，与 `product_files` **全部一致**（缺失 0） |

## 七、环境说明与修复建议
- 未发生环境阻断：`.venv` 依赖齐备，全部命令可直接执行；`drill` 失败为**真实断言失败**，非环境问题。
- 本轮新增文件仅在 `lizhu_r40_scripts/`；`probe`/`guard` 验证输出写独立目录 `checkpoints/qa_learn/_verify/step2_lz/`，未覆盖既有产物。
- 注：本工作区 `write` 工具创建文件报 `EISDIR`（硬链接失败），改用 PowerShell here-string + `Set-Content -Encoding utf8` 写文件。

**修复建议（按优先级）**
1. （阻断验收）修 `drill` 梯度门禁：`step2.build_model` 显式固定 `head_input_mode` 使骨干参与前向；若确实不用骨干（README 记 `concat` 显著降准确率），则断言须改为「参与前向的参数零梯度为空」并显式列出结构性豁免。
2. （阻断复现）恢复产物↔代码口径一致：让 `concat` 重新支持拼接形态，或**重跑 eval 重生成 7 个模型产物**，并在产物 meta 记录 `head_input_mode` / `normalize_query`。
3. （文档）按重跑实测重写 `README_step2.md` 4.2 与 6.9（含 `metrics_sha256`）。
4. （增强）补「加载产物 → 复算自检索/分项」校验入口，否则模型侧「产物驱动复算」永远不可执行。
