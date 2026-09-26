# n3d_viz 修复轮（皋陶 1 warning + 2 info）测试报告 —— 离朱 r4

- **被测改动**：`n3d_viz/verify_viz.py`、`n3d_viz/core.py`、`n3d_viz/README.md`（未改任何产物）
- **测试轮次**：W1（[2d] 两层承重断言）+ I1（撞色回退分支覆盖）+ I2（相对路径字节级复现口径）
- **环境**：Windows / Python 3.12.10 / Node v25.2.1 / Playwright 1.63.0（Chromium 153.0.8010.12，npm 已装）
- **项目根**：`E:\neuron3d`；全部临时产物位于 `.lizhu_env/r4/`（测试后已清理大件）

## 一、测试概览

| # | 测试类型 | 用例/断言数 | 通过 | 失败 | 跳过 | 退出码 |
|---|---|---|---|---|---|---|
| 1 | 编译测试（`compileall` + 2×`node --check`） | 3 | 3 | 0 | 0 | 0 |
| 2 | 单元/断言测试（`verify_viz.py` 全量自检） | 170 | 170 | 0 | 0 | 0 |
| 3 | 单元测试——I1 撞色回退分支专项 | 12 | 12 | 0 | 0 | 0 |
| 4 | 单元测试——I2 相对/绝对路径字节级专项 | 18 组 × 5 判据 | 90 | 0 | 0 | 0 |
| 5 | 单元测试——既有契约回归专项 | 5 类 | 5 | 0 | 0 | 0 |
| 6 | 单元测试——README 构成/序号/引用纪律 | 4 类 | 4 | 0 | 0 | 0 |
| 7 | **承重性拒绝证明**（W1 注入—恢复闭环） | 8 项判据 | 8 | 0 | 0 | 0 |
| 8 | E2E 测试（Playwright + 真实 Chromium） | 28 | 28 | 0 | 0 | 0 |
| 9 | 接口测试（CLI 契约） | 5 | 5 | 0 | 0 | 0 |
| **合计** | | **343** | **343** | **0** | **0** | 全 0 |

> 无跳过项、无环境阻塞。所有数字均来自真实执行输出（留档见第十一节）。

## 二、被测功能说明中的「必须确认未被破坏的既有契约」逐条核对

| 契约要求 | 实测 | 结论 |
|---|---|---|
| `python -m compileall -q n3d_viz` → exit 0 | exit 0 | ✅ |
| 两个 `node --check` → exit 0 | `viewer.js` exit 0；`viewer_smoke.js` exit 0 | ✅ |
| `verify_viz.py --report …` → 通过 170 / 失败 0 / 跳过 0，exit 0 | 170 / 0 / 0，共 170 行，exit 0 | ✅ |
| `[2a]` 原有 4 项仍全部通过 | 第 10–13 号断言全 PASS（K∈[1,64] 去重数==K；K≤9 逐字节；HTML-PLY 同源；`hex_to_rgb` 契约） | ✅ |
| `[2b]` 42 项全部通过 | 42 / 42 PASS | ✅ |
| `[2c]` 32 项全部通过 | 32 / 32 PASS | ✅ |
| `[2d]` 共 14 项（磁盘 6 + 重渲 6 + 清理 2） | 14 / 14 PASS（解析实测：`[2d][磁盘]` 6、`[2d][重渲]` 6） | ✅ |
| 零回归锚点 SHA256/字节数逐位不变 | `15A80EBB…`/88,521；`9A097D16…`/4,120；`1F594ECF…`/15,695，三项 SHA 与字节数均 == 常量 | ✅ |
| 5 套异构交付件 15 件 SHA256 逐位不变 | `git status --porcelain -- checkpoints` 输出为空（工作区零改动），且 15 件 SHA 已逐件留档（`regression_snapshot.json`） | ✅ |
| README 项数构成 77 + 81 + 12 = 170，`[2a]`5 + `[2b]`42 + `[2c]`32 + `[2d]`14 | 报告实测分组项数恰为 5 / 42 / 32 / 14，合计 93；README 声明文字齐全 | ✅ |
| README 断言清单序号 14 / 138 / 90-91 / 136-137 / 164-165 可被真实报告复现 | 序号 14=`[2a] _hue_palette_rgb 撞色回退分支（必撞色构造）`；138=`内联渲染器可执行且投影正确`；90/91=`[2b] 合成组…清理/残留 0 字节`；136/137=`[2d] 重渲…清理/残留 0 字节`；164/165=`负例…清理/残留 0 字节` —— **全部逐字对上** | ✅ |
| README 不出现「第 N 行」形式引用 | 正则 `第\s*\d+\s*行` 命中 0 处 | ✅ |
| `requirements.txt` 无新增依赖 | 与 `git show HEAD:requirements.txt` **逐字节相同**（SHA256 `F69450D5…`），仅 torch/torchvision/numpy | ✅ |
| `n3d_viz` 顶层 import 仅 `torch`/`numpy`/标准库 | 顶层名集合 23 个，第三方仅 `torch`、`numpy`；黑名单（matplotlib/plotly/pyvista/tkinterdnd2/scipy/PIL/pandas）命中 0 | ✅ |
| `^\s*(from\|import) n3d_(sphere\|proto\|shape)` 仍 0 命中 | 0 命中 | ✅ |
| `n3d_sphere`/`n3d_proto`/`n3d_shape` 源码与产物零改动 | `git status --porcelain -- n3d_sphere n3d_proto n3d_shape checkpoints data requirements.txt` 输出为空 | ✅ |
| 不得引入任何「形状类型」字段或分支 | 运行时代码（`core/export_geometry/render_html/gui/__main__/__init__`）命中 0；HTML 负载 `meta` 的 22 个键名均不含 shape/geom 字段 | ✅ |

## 三、W1（warning）—— `[2d]` 两层承重断言：拒绝证明实测

复现方式（离朱自建闭环脚本 `.lizhu_env/r4/w1_inject.py`）：
把 `core.LEVEL_PALETTE_BASE` 前两项文本对调（一处替换）→ 子进程跑与基线**同一命令形式**的
`python n3d_viz/verify_viz.py` → `finally` 无条件把原始字节写回 → 断言 SHA256 还原。

| 观测项 | 注入前 | 注入后 | 恢复后 |
|---|---|---|---|
| `core.py` SHA256 | `2770BFCA…`（41,041 B） | `21B5CCBC…` | `2770BFCA…`（41,041 B，**逐字节相同**） |
| `verify_viz.py` 汇总 | 170/0/0，exit 0 | 168/2/0，**exit 1** | 170/0/0，exit 0 |
| `[2a]` 5 项 | 5/5 PASS | **5/5 PASS** | 5/5 PASS |
| `[2b]` 42 项 | 42/42 PASS | 42/42 PASS | 42/42 PASS |
| `[2c]` 32 项 | 32/32 PASS | 32/32 PASS | 32/32 PASS |
| `[2d][磁盘]` 6 项 | 6/6 PASS | **6/6 PASS** | 6/6 PASS |
| `[2d][重渲] html` SHA256 | PASS `15A80EBB…` | **FAIL**（实测 `4E147DA5…`） | PASS |
| `[2d][重渲] ply` SHA256 | PASS `9A097D16…` | **FAIL**（实测 `B7449AB4…`） | PASS |
| `[2d][重渲] obj` SHA256 / 三件字节数 | PASS | **PASS**（OBJ 无色，字节数不变） | PASS |
| `[2d]` 重渲临时产物清理 2 项 | PASS | PASS（失败路径同样清理，残留 0 文件 / 0 字节） | PASS |

**结论：W1 承重性成立**。注入后 `[2a]` 与 `[2d][磁盘]` 对缺陷**完全无感**（正是 W1 修复的动机），
承重完全由 `[2d][重渲]` 的 html / ply 两条 SHA256 断言提供，并使 `verify_viz.py` 退出码非 0。
恢复后 `core.py` 与注入前逐字节相同（SHA256 一致），全量自检回到 170/0/0。
另核实：注入态下重渲目标目录 `_anchor_rerender/` 内 0 个文件，且交付锚点
（`checkpoints/n3d_viz/viz_model.*`）在整轮注入—恢复期间 SHA256/字节数始终未变。

## 四、I1（info）—— `_hue_palette_rgb` 撞色回退分支：实测值与承重性

### 4.1 实测（`K = COLLISION_PROBE_K = 1536`）

| 指标 | 实测值 | 说明 |
|---|---|---|
| 纯色相扩展撞色个数 | **636** | 前提成立（>0），回退分支真的可达 |
| 纯色相扩展去重数 | 900 | 1536 − 636 |
| 候选调用次数（白盒计数钩子） | **2,172** | == K + 636，> K 即证明回退执行 |
| 回退次数 | **636** | 与撞色个数一致（每个撞色恰好回退 1 次） |
| 最终去重数 `_hue_palette_rgb(K)` | **1,536 == K** | 回退结果正确 |
| 公开入口 `layer_palette_hex(K)` 去重数 | **1,536 == K** | 入口一致性成立 |
| 钩子恢复 | `core._hsv_to_rgb_bytes is 原函数` → **true**；断言后 `layer_palette_hex(33)` 去重数 == 33 → true | 钩子未泄漏到后续断言 |
| 官方断言返回串 | `K=1536：纯色相扩展撞色 636 个（前提成立）；候选调用 2172 次、回退 636 次；最终去重数 == 1536` | 与 README §5 表逐字一致 |

### 4.2 承重性（拒绝证明，全部用内存内猴子补丁，不动磁盘源码）

| 反事实构造 | 期望 | 实测 | 结论 |
|---|---|---|---|
| A：把回退判据 `candidate not in used` 改为恒真（永不回退） | 候选调用恰 == K、去重数 < K | 调用 **1,536 == K**、去重 **900 < 1536** | 断言项 2「调用次数 > K」承重 ✅ |
| B：撞色前提不成立（K=64）时断言必须报错 | 报错 | K=64 实测撞色 **0** 个，前提断言抛错 | 断言项 1「前提成立」承重 ✅ |
| C：恒撞色 + 极小 `_PALETTE_SEARCH_LIMIT=3` | 抛 `ValueError` | 抛 `ValueError: 无法为第 1 个层生成未占用的颜色（K=2）` | 搜索上限分支可终止 ✅ |

补丁后均显式断言常量与函数已恢复（`_PALETTE_SEARCH_LIMIT`、`_hsv_to_rgb_bytes`）。

## 五、I2（info）—— 字节级复现必须用相对路径：实测复核

对 6 个 checkpoint（二期基准 + 5 套异构）各做三种调用形式对照，逐字节 diff 定位差异来源：

| checkpoint（相对路径） | 相对 HTML | 绝对 HTML | 差 | 内嵌 `meta.checkpoint` JSON 字节差 | 逐字节 diff 操作数 | PLY/OBJ |
|---|---|---|---|---|---|---|
| `checkpoints/n3d_sphere/model.pt` | 88,521 | 88,535 | **+14** | 14 | 1 处插入 | 逐字节相同 |
| `…/full_shapesphere_…_s42.pt` | 88,634 | 88,648 | **+14** | 14 | 1 处插入 | 逐字节相同 |
| `…/full_shapecube_…_s42.pt` | 87,755 | 87,769 | **+14** | 14 | 1 处插入 | 逐字节相同 |
| `…/full_shapecylinder_a0.5_…_s42.pt` | 85,507 | 85,521 | **+14** | 14 | 1 处插入 | 逐字节相同 |
| `…/full_shapecylinder_a1_…_s42.pt` | 87,173 | 87,187 | **+14** | 14 | 1 处插入 | 逐字节相同 |
| `…/full_shapecylinder_a2_…_s42.pt` | 88,429 | 88,443 | **+14** | 14 | 1 处插入 | 逐字节相同 |

补充结论（比测试说明更强的证据）：

1. **+14 的机理被定位到字节级**：对 `viz_model.html` 做 `SequenceMatcher` 逐字节 diff，全文件
   **只有一处差异** —— 相对路径版 `rel[2977:2977]`（空）→ 绝对路径版 `abs[2977:2991]` 插入
   `b'E:\\\\neuron3d\\\\'`，插入长度 **14 字节**。其余 88,521 字节完全相同。
2. **HTML 字节差恒等于内嵌 `meta.checkpoint` 字段的 JSON 字节差**（18/18 组成立）；
   HTML 负载除 `meta.checkpoint` 外逐字段完全相同（其余 `meta`/`layers`/`neurons`/`edges`/`labels` 全等）。
3. **PLY / OBJ 逐字节相同**（18/18 组 SHA256 一致）。
4. **README §2 实测表 6 行数字与实际逐项一致**（`variant_A_matches_readme = true` × 6）。
5. **相对路径口径可复现锚点**：重渲 `viz_model.html` 的 SHA256 == `15A80EBB…`、字节数 88,521，
   与磁盘交付件**逐字节相同**；由 `python -m n3d_viz -c checkpoints/n3d_sphere/model.pt` 直接
   产出的三件套 SHA256 也与锚点常量**完全一致**。

### 5.1 需提请开发者注意的一处口径细节（非缺陷，建议补一句 README 说明）

在 Windows 上，`Path()` 会把 checkpoint 路径字符串的 `/` **归一化为 `\`**，而 `meta.checkpoint`
记录的是**归一化后**的字符串；JSON 中每个 `\` 转义为 2 字节。因此「相对 vs 绝对」的 HTML 字节差
取决于**两个入参是否需要归一化**：

| 入参形式（Windows 实测） | 相对路径入参 | 绝对路径入参 | 内嵌路径字符差 | HTML 字节差 |
|---|---|---|---|---|
| 基准/交付件口径（相对用 `/`、绝对用 `\`） | `checkpoints/n3d_sphere/model.pt` | `E:\neuron3d\checkpoints\n3d_sphere\model.pt` | 13 | **+14** ✅ |
| 纯 POSIX 口径（两者都用 `/`） | `checkpoints/n3d_sphere/model.pt` | `E:/neuron3d/checkpoints/n3d_sphere/model.pt` | 14 | **+14** ✅ |
| 两者都用 `\`（相对路径自带 `\`） | `checkpoints\n3d_sphere\model.pt` | `E:\neuron3d\checkpoints\n3d_sphere\model.pt` | 13 | **+18**（13 个 `\` 全转义：13×2−13=+13，再计 `E:` 前缀 2 字节） |

**判定**：README 与 `verify_viz.py` 注释里的「**+14**」在基准/交付件实际使用的两种调用形式下
**实测成立**（上表前两行），结论与口径**不需要修正**；仅当「相对路径入参自身也写成反斜杠」时
字节差会变成 +18。若希望口径完全无歧义，建议在 README §2「字节级复现口径」补一句：
「Windows 下 `<path>` 会被归一化为 `\` 并被 JSON 转义，故请以**正斜杠相对路径**形式给出
`--checkpoint`（与交付件、锚点一致），此时相对/绝对差恒为 +14 字节」。
（此为 **info 级建议**，不影响本次验收；未改动任何交付件。）

## 六、E2E 测试（Playwright + 真实 Chromium，28/28 通过）

`check_playwright` 检测结果：`installed=true, source=npm, version=1.63.0`；Chromium 153.0.8010.12。
脚本 `.lizhu_env/r4/e2e_viewer_playwright.js` 以 `file://` 直接打开**自包含 HTML**（无需服务器），
在页面脚本执行前挂钩 `HTMLCanvasElement.prototype.getContext`，统计真实绘制调用。

| 场景 | 断言 | 实测 |
|---|---|---|
| **核心用户旅程**：打开 → 首屏渲染 | 页面加载成功 / 首屏绘制神经元圆 == N+S_in+S_out / 边数 ≥ 阈值内边 / 画布真实像素 / 图例 K+3 / 保留计数 / 层平面初始态 | status=200；**636 == 256+193+187**；lineTo=406 ≥ moveTo=388 ≥ 379；1280×800；12 == 9+3；`保留 379 / 736 条边`；checked=true==meta |
| 核心旅程：耗时 | 加载 + 渲染 < 10s | **97 ms** |
| 零外部依赖 | 全程无非 `file://` 请求；无 `http/https/协议相对` | **0 个非 file:// 请求** |
| 页面健康 | 无未捕获 JS 异常、无 console error | **0 条** |
| 边界值 | 阈值 0 → 保留全部边 & 单帧绘制 **== E** | keep=`736 / 736`；开关差值隔离单帧 **moveTo == 736 == E** |
| 边界值 | 阈值 0.9 → 保留 0 边 & 单帧不画边 | keep=`0 / 736`；单帧边绘制 **0** |
| 交互-阈值 | 标签随滑块更新 | `0.00` / `0.90` |
| 交互-开关 | 关闭「连接」不画边、重开恢复 | stroke 8,876 → 10,124 |
| 交互-拖拽/重置 | 左键拖拽旋转后持续重绘、重置视角仍正常 | stroke 10,124 → 18,060 → 19,052 |
| 交互-悬停 | **真实鼠标**网格扫描命中神经元并弹详情 | 命中 `神经元 #156 层 L3（z=-2.828e-1）入度 1/出度 1 S_in:是 S_out:是` |
| 跨页面状态 | K=9 锚点产物：层色去重 == K、图例 K+3、首屏绘制 | dedup=9，K=9，12 == 9+3，arc=636 |
| 跨页面状态 | K=15 异构产物（cylinder λ=2）：层色去重 == K、图例 K+3 | **dedup=15，K=15，18 == 15+3**，arc=655 |
| 异常路径 | 打开不存在的 HTML：不渲染、无未捕获异常 | body 文本长度 0；errors 0；`net::ERR_FILE_NOT_FOUND` 由导航捕获 |
| 资源极限 | 100 次阈值变更 + 50 次鼠标移动 | 页面存活，arc=64,236，耗时 2,007 ms，**无 JS 异常** |

## 七、接口测试（CLI 契约）

| 用例 | 命令 | 期望 | 实测 |
|---|---|---|---|
| 合法调用（相对路径） | `python -m n3d_viz -c checkpoints/n3d_sphere/model.pt --out-dir checkpoints/n3d_viz/_verify/_lizhu_r4` | exit 0，三件套字节级复现锚点 | **exit 0**；HTML 88,521 `15A80EBB…` / PLY 4,120 `9A097D16…` / OBJ 15,695 `1F594ECF…` —— 三项 SHA256 **与锚点常量完全一致** |
| 一期产物（缺 6 键） | `--checkpoint checkpoints/n3d_model_full.pt` | 退出码非 0、含 6 个缺失键名、无 traceback | 自检内 8 条断言全 PASS（退出码 3） |
| 路径不存在 | `--checkpoint checkpoints/__not_exist__.pt` | 退出码非 0、报错可读含「不存在」 | PASS（退出码 3） |
| 越界索引负例（4 例） | `topo_index`/`edge_src`/`edge_dst` 置非法值 | 退出码 3、报错含 `key[i]=v`、无 traceback、不产生产物 | 16 条断言全 PASS，且 `_bad_index/` 残留 0 文件 / 0 字节 |
| 布尔开关真实生效 | `--no-plan-planes` / `--with-ply-edges` | 产物可观测差异 | PASS（`showPlanes:false`；PLY 含 736 条 `element edge`） |

## 八、失败用例分析

**本轮无失败用例**（343/343 通过）。

测试过程中出现并已定位/修正的 **2 处「测试脚本自身」缺陷**（非被测代码缺陷，记录以备追溯）：

1. `I2` 首版脚本误用**原始入参字符串长度差**（Windows 上为 12）去校验 HTML 字节差（14），
   产生 6 条假失败。定位方式：对两份 HTML 做逐字节 `SequenceMatcher`，确认唯一差异是
   `meta.checkpoint` 中 `E:\\neuron3d\\` 前缀（14 字节）。修正为「HTML 字节差 == 内嵌字段 JSON
   字节差」，并补充分隔符口径三变体对照。
2. `E2E-2` 首版用**累计** `moveTo` 计数跨阈值比较（滑块的 `fill` + `input` 会触发两次绘制），
   产生 1 条假失败。修正为「开关差值隔离单帧绘制边数」，实测阈值 0 时单帧恰好 736 == E。

## 九、环境问题说明

**无环境阻塞**。要点：

- 测试环境无需构建：所有脚本只依赖仓库现有 `torch`/`numpy` 与 Python 标准库；
  `verify_viz.py` 的渲染器冒烟只需系统已装的 `node`；Playwright 用**已安装**的 npm 版
  1.63.0 与已缓存的 Chromium 153（`%LOCALAPPDATA%\ms-playwright\chromium-1243`），
  **未下载任何新依赖**（符合「零新增依赖」契约）。
- 按目录纪律，所有测试脚本与临时渲染产物均落在 `.lizhu_env/r4/` 下；
  未在 `.lizhu_env/` 之外执行任何环境构建命令。
- 运行期自检脚本 `verify_viz.py` 自身的产物纪律也被核实：`_anchor_rerender/` 0 文件、
  `_bad_index/` 0 文件 0 字节、`_synthetic/` 无残留。
- 测试结束后已清理大件临时产物（`.lizhu_env/r4/_i2_out` 等 111 个文件 / 约 3.98 MB、
  `checkpoints/n3d_viz/_verify/_lizhu_r4`），交付件与锚点未被触碰（SHA256 复核一致）。

## 十、修复建议（全部为 info 级，不影响验收）

1. **【建议，非必须】README §2「字节级复现口径」补一句 Windows 归一化说明**（见 §5.1）：
   明确「请用正斜杠相对路径给出 `--checkpoint`，此时相对/绝对差恒为 +14 字节」，
   以免后续有人用反斜杠相对路径复核时得到 +18 而误判为口径失效。
2. **【可选】`verify_viz.py` 的 `[2d][重渲]` 可再补一条「HTML 负载除 `meta.checkpoint` 外全等」的
   断言**（把 §5 第 2 条的复核口径固化进自检），这样「仅路径字段不同」这一点也由脚本把守。
3. **【可选】`[2a]` 撞色回退断言可加「回退次数 >= 撞色个数」的不变量断言**（本轮实测 636 == 636），
   成本极低，可防止回退次数与撞色个数脱钩的改动。
4. 三处待交付的改动本身（`verify_viz.py` / `core.py` / `README.md`）**建议原样通过**：
   W1 的承重性、I1 的分支覆盖、I2 的口径与数字均已实测坐实，且全部既有契约零回归。

## 十一、证据留档（`.lizhu_env/r4/artifacts/`）

| 文件 | 内容 |
|---|---|
| `final_verify.txt` | 清理后最终 `verify_viz.py` 全量输出（170/0/0，exit 0） |
| `w1_out.txt` | W1 注入—恢复闭环：三段 SHA256、注入块、退出码 1、分组统计 |
| `_w1_out/verify_report_injected.md`、`_w1_out/verify_stdout_injected.txt` | 注入态 170 行报告（含 2 条 FAIL 原文与实测哈希） |
| `i1_out.txt` | I1 撞色实测 + 三项承重性反事实结果 |
| `i2_out.txt` | I2 18 组字节级对照明细（含逐字节 diff 操作数） |
| `e2e_out.json`、`e2e_stdout.txt` | Playwright 28/28 明细 |
| `regression_out.txt`、`regression_snapshot.json` | 依赖/导入/零改动/锚点/15 件异构 SHA/无形状类型字段 |
| `readme_out.txt`、`parse_out.txt` | README 构成、分组项数、序号引用、实测数字核对 |
| `core_before.json`、`core_after.json` | `core.py` 注入前/恢复后 SHA256 逐字节一致证据 |
