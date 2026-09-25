# n3d_viz 测试报告（离朱）

- 被测模块：`n3d_viz`（新建模块，二期拓扑三维可视化）
- 测试依据：`read_test_specs` 待测试功能说明 + `n3d_viz/README.md`
- 被测代码基线：`n3d_viz/` 全部文件（core / export_geometry / render_html / gui / __main__ / verify_viz / assets）
- 真实基准产物：`checkpoints/n3d_sphere/model.pt`（seed=42, N=256, E=736, K=9）；异常路径基准：`checkpoints/n3d_model_full.pt`（一期产物）
- 测试时间：2026-09-25

## 一、测试概览

| 测试类型 | 用例数 | 通过 | 失败 | 跳过 | 退出码 |
|---|---|---|---|---|---|
| 编译 / 静态检查 | 4 | 4 | 0 | 0 | 0 |
| 既有验证入口 `verify_viz.py` | 50 | 50 | 0 | 0 | 0 |
| 单元测试 `lizhu_n3d_viz_unit_tests.py` | 85 | 84 | 1 | 0 | 1 |
| 接口测试（CLI）`lizhu_n3d_viz_cli_tests.py` | 28 | 25 | 3 | 0 | 1 |
| GUI 功能测试 `lizhu_n3d_viz_gui_tests.py` | 12 | 11 | 1 | 0 | 1 |
| E2E 测试（Playwright + Chromium）`lizhu_n3d_viz_e2e.js` | 33 | 31 | 2 | 0 | 1 |
| 硬约束核验（哈希比对） | 1 | 1 | 0 | 0 | 0 |
| **合计** | **213** | **206** | **7** | **0** | — |

7 条失败用例全部归因于 **2 个真实缺陷**（详见第三节），均属「参数/控件存在但功能空操作」，不涉及崩溃或数据错误。

## 二、各测试类型详细结果

### 1. 编译与静态检查（4/4 通过）

| 检查 | 命令 | 结果 |
|---|---|---|
| Python 语法/字节码编译 | `python -m compileall -q n3d_viz` | exit 0 |
| 渲染器 JS 语法 | `node --check n3d_viz/assets/viewer.js` | exit 0 |
| 冒烟脚本 JS 语法 | `node --check n3d_viz/assets/viewer_smoke.js` | exit 0 |
| 运行时依赖洁净 | 新进程 `import n3d_viz / gui / __main__ / verify_viz` 后检查 `sys.modules` | 无 `n3d_sphere` / `n3d_proto` / 第三方绘图库 |

### 2. 既有验证入口（50/50 通过）

`python n3d_viz/verify_viz.py --report .lizhu_env/lizhu_viz/verify_viz_report.md` → **通过 50 / 失败 0 / 跳过 0，exit 0**（日志 `verify_viz_run.txt`）。
关键实测值复现：N=256、E=736、K=9、层规模 13/24/37/35/39/34/37/24/13、S_in=193、S_out=187、阈值 0.30 保留 379/736、PLY/OBJ/HTML 坐标与 `neuron_pos` 逐位一致（≤1e-6）、HTML 88292 字节且无外部引用。

### 3. 单元测试（85 项：84 通过 / 1 失败）

文件 `.lizhu_env/lizhu_viz/lizhu_n3d_viz_unit_tests.py`（日志 `unit_run.txt`），8 个测试类，全部使用**独立断言**（不复用 `verify_viz.py` 的口径）：

| 测试类 | 覆盖内容 |
|---|---|
| `TestCheckpointLoading` | 路径不存在 / 路径是目录 / 空文件 / 垃圾字节 / 顶层非字典 → 三类异常可区分且信息可读；缺键列表顺序；`get_state_dict` 回退与类型校验；异常继承链 |
| `TestExtractTopology` | 分层切片与层 z 均值、度直方图、S_in/S_out 计数、连接密度、`layer_of_neuron` 越界返回 -1、阈值统计键格式与边界（|w| 恰等于阈值保留）、`weight_extremes` 奇偶/空集、空层 z=0、int 掩码兼容；**反向**：edge_dst/edge_weight/mask/degree 长度不符、neuron_pos 维度与数量不符、level_node_reach 形状/越界/负起点/逆序区间 → 均抛 `CheckpointSchemaError` |
| `TestNamingAndPaths` | `viz_<ckpt名>.html/.ply/.obj` 派生、含点号文件名不截断、不同 ckpt 不撞名、`out` 覆盖时 PLY/OBJ 同目录同名、裸文件名 `out` |
| `TestExportGeometry` | PLY binary float32 回读逐位一致、PLY ascii 回读 ≤1e-6、`with_edges` 附加 edge 元素且不影响顶点解析、逐层 RGB 与 `LAYER_COLORS` 一致、头部类型声明；OBJ `v` 行在 `l` 行之前、索引从 1 开始（`l 1 4`）、无 CRLF、`with_colors` 不影响行数、字节数与报告一致；**反向**：PLY 缺 `end_header` / 不支持 format / 体长不足、OBJ 空文件；边排序颜色单调、越界层 id 不崩 |
| `TestRenderHtml` | 自包含断言 ok（无 http/https、无协议相对、无 syn_dist、<2MB）；载荷字段完整；**反向**：注入 `http://`、`src="//"`、`syn_dist`、超限体积 → 均 `AssertionError`；模板缺占位标记 → `ValueError`；模板缺失 → `FileNotFoundError`；`</script>` 注入防御（config 值承载 `</script><script>alert(1)</script>` 时页面脚本标签数仍为 2）；二次写盘 `existed=True` 且日志提示不静默覆盖 |
| `TestWriteOutputs` | 首写/重写 `existed` 语义与 3 条「已存在将被覆盖」提示、`out` 覆盖命名、`--ply-ascii` 传递、嵌套输出目录自动创建 |
| `TestHardConstraints` | 源码无 `n3d_sphere`/`n3d_proto` import、无白名单外顶层 import、`requirements.txt` 无新增依赖、新解释器运行时不加载被禁模块 |
| `TestRealCheckpoint` | 真实二期产物 N/E/K/层规模/S_in/S_out/阈值统计复现；坐标与原始 `torch.load` 逐分量 ≤1e-6；层分组恰好划分 0..255；真实数据 HTML 自包含且不含 syn_dist |

**唯一失败**：`test_include_planes_flag_changes_html` —— `core.build_html(include_planes=True/False)` 生成的 HTML **完全相同**（缺陷 LZ-VIZ-01）。

### 4. 接口测试（CLI，28 项：25 通过 / 3 失败）

文件 `.lizhu_env/lizhu_viz/lizhu_n3d_viz_cli_tests.py`（日志 `cli_run.txt`），子进程真实调用 `python -m n3d_viz` 与直跑脚本。

正向（全部通过）：默认生成三件套 exit 0 + `[完成]` 摘要；二次运行提示覆盖；`--quiet` 抑制 `[数据]` 行；嵌套 `--out-dir` 自动创建；`--out custom.html` 派生 `custom.ply/.obj`；`--threshold 0.5` 写入 HTML（`"threshold":0.5`）；`--ply-ascii` 生效（`format ascii 1.0`）；真实产物 CLI（N=256/E=736/K=9/S_in=193/S_out=187，HTML < 2MB）；`--help` 列出全部 8 个参数。

反向（全部通过）：一期产物 → **exit 3** 且错误信息逐一列出 6 个缺失键、且不产生任何产物；路径不存在 → exit 3 含「不存在」；目录当 checkpoint → exit 3 含「不是文件」；损坏文件 → exit 3 含「损坏」；未知参数 → **exit 2**；`--threshold abc` → exit 2；只给 `--out-dir` 不给 `--checkpoint` → exit 2 且提示 `--checkpoint`；直跑 `export_geometry.py` 缺 `--checkpoint` → exit 2、坏 checkpoint → exit 3；直跑 `render_html.py` 坏 checkpoint → exit 3、缺参 → exit 2；无参启动 `python -m n3d_viz` 驻留 GUI mainloop 未崩溃退出。

**3 条失败**：`test_with_ply_edges_flag`（LZ-VIZ-02）、`test_no_plan_planes_actually_disables_planes` 与 `test_render_html_script_no_plan_planes`（LZ-VIZ-01）。

### 5. GUI 功能测试（12 项：11 通过 / 1 失败）

文件 `.lizhu_env/lizhu_viz/lizhu_n3d_viz_gui_tests.py`（日志 `gui_run.txt`）：实例化**真实 `VizApp`**（withdraw 状态），调用真实按钮回调，替换 `messagebox`/`filedialog`/`webbrowser` 避免阻塞。

通过项：默认控件状态（默认 ckpt/输出目录/阈值 0.30/层平面勾选/两个「打开」按钮 disabled）；阈值滑块与标签联动（0.55 → "0.55"）；`浏览…` 按钮回填 `StringVar`；checkpoint 不存在时 `start()` 不建线程、写日志并置状态为「失败」；真实生成走「后台线程 → queue → after 轮询」链路：`[开始]/[数据]/[PLY]/[OBJ]/[HTML]/[完成]` 全部落到日志区、三件套落盘、`start` 按钮恢复、两个「打开」按钮启用、状态栏「完成…」、`last_html` 正确；生成前点击「打开」弹出提示且不打开任何目标；生成后「打开 HTML」以 `file://` 打开产物、「打开输出文件夹」调用文件管理器；任务运行中再次点击「开始生成」被拒并弹「已有生成任务」；`清空日志` 生效；后台线程内一期产物异常 → 主线程弹错误框、状态「失败」、按钮恢复且「打开」按钮保持 disabled；`gui.py` 无 `tkinterdnd2` import（不做拖拽）。

**唯一失败**：`test_gui_planes_checkbox_affects_html` —— GUI「默认显示层平面」复选框勾选/取消产生的 HTML 字节相同（缺陷 LZ-VIZ-01 的第三处表现）。
### 6. E2E 测试（Playwright + 真实 Chromium，33 项：31 通过 / 2 失败）

文件 `.lizhu_env/lizhu_viz/lizhu_n3d_viz_e2e.js`（日志 `e2e_run.txt`，截图 `e2e_viewer.png`）。以 `file://` 打开真实产物 HTML，视口 1280×800，`getImageData` 全画布统计（`ink` = 亮像素，`nonbg` = 偏离深色背景的像素）。

| 分组 | 断言 |
|---|---|
| 核心旅程：打开 | 加载零 JS 错误（pageerror + console.error）；**零外部网络请求**（仅 1 个 `file://` 请求）；canvas 铺满视口（1280×800）；画布有实际绘制（ink=37280）；HUD 显示 `checkpoint: model.pt / seed=42 N=256 E=736 K=9 / S_in=193 S_out=187 / 密度 1.12e-2 / 测试准确率 0.9759 / 层规模 13/24/37/35/39/34/37/24/13`；`保留 379 / 736 条边（|w| >= 0.30）`；图例 12 个色块（K+3） |
| 阈值滑块 | 阈值 0 → 736 条边且亮像素 23096→29393（弱边被绘出）；阈值 0.50 → 140 条边且亮像素降到 15686（与 checkpoint 统计一致）；标签同步 0.50；回到 0.30 后介于两者之间 |
| 图层开关 | 关闭「连接」亮像素 23096→9534、关闭「神经元」23096→18728，重开均精确恢复；层平面关闭后 `nonbg` 194876→26269（平面半透明层被移除） |
| 交互 | 左键拖拽旋转、滚轮缩放、右键拖拽平移均改变画面哈希；「重置视角」后画布哈希**精确回到**初始值（1646810397） |
| 悬停详情 | 在 (590,260) 命中神经元 #234，浮层显示「层 L7（z = 0.2828）入度 2 / 出度 1 S_in: 是 / S_out: 是」；移开后浮层隐藏 |
| 注入防御 | 载荷含 `</script><script>window.__XSS=1</script>` 时：`window.__XSS` 未定义、无 JS 错误、`<script>` 标签数仍为 2、无法解析的注入串以字面量保留在 `N3D_DATA.meta.placement`、渲染器继续正常绘制 |
| `--no-plan-planes` 产物 | **失败 2 项**：产物打开后 `#cb-planes` 仍为勾选（层平面照样显示）；`on.html` 与 `off.html` 字节完全相同（SHA256 一致，均 88292B） |

### 7. 硬约束核验（全部通过）

| 硬约束 | 核验方式 | 结果 |
|---|---|---|
| 零新依赖 | 全量测试前后 `requirements.txt` 与 `n3d_sphere`/`n3d_proto` 源码 + `checkpoints/n3d_sphere/*.pt` 共 20 项 SHA256 比对；`git status` 无已跟踪文件被改动 | **完全一致** |
| 不 import 业务模块 | 源码正则扫描 + 新解释器运行时 `sys.modules` 检查 | 无 `n3d_sphere`/`n3d_proto` |
| 不改动既有模块 | 上述哈希比对 + `git status`（仅新增 `n3d_viz/`） | 零改动 |
| 产物落盘位置 | CLI 默认写入 `checkpoints/n3d_viz/`；验证类产物写入 `checkpoints/n3d_viz/_verify/` | 符合约定 |
| 无环境构建 | 未执行 `npm install` / `pip install`；仅复用已安装的 Playwright 与 Chromium | 符合 |

## 三、缺陷清单

### LZ-VIZ-01（中）`--no-plan-planes` / `include_planes` 全链路空操作
- **现象**：`python -m n3d_viz --no-plan-planes`、`python n3d_viz/render_html.py --no-plan-planes`、`core.build_html(include_planes=False)`、GUI「默认显示层平面」复选框取消勾选 —— 四条路径产出的 HTML 与默认路径**字节完全相同**；浏览器打开后 `#cb-planes` 仍为勾选，层参考平面照常绘制。README 第 35/37/53 行、`--help` 与 `core/render_html` 文档均宣称该开关有效。
- **证据**：
  - 单测 `test_include_planes_flag_changes_html`：`build_html(True) == build_html(False)`；
  - CLI `on.html` 与 `off.html` SHA256 均为 `8FFB67B6…4E3D00`；
  - E2E：`--no-plan-planes` 产物 `cb-planes.checked=true`（默认应为 false）；
  - GUI `test_gui_planes_checkbox_affects_html`：两种勾选状态产物字节相同。
- **根因（三处叠加，任一处单独存在都不会生效）**：
  1. `core.py:526-527` `build_html_payload(..., include_planes=True)` 接收该参数但**函数体内从未使用**，`meta` 中没有 `showPlanes` 键；
  2. `core.py:731-732` 用 `html.replace('"showPlanes": true', '"showPlanes": false')` 做兜底，但负载里根本没有这个键；且 `json.dumps(..., separators=(",", ":"))`（`core.py:658`）输出的是无空格形式 `"showPlanes":true`，即使补上键，当前带空格的匹配串也**永不匹配**（双重失效）；
  3. `assets/viewer.js:86` 硬编码 `showPlanes: true`，`assets/viewer.js:381/387` 的 `bindCheckbox` 又把 `el.checked` 同步为该值，因此模板 `viewer.html:42` 刻意不写 `checked` 属性的意图也被覆盖。
- **修复建议**：
  1. 在 `build_html_payload` 的 `meta` 中写入 `"showPlanes": bool(include_planes)`；
  2. `viewer.js:86` 改为 `showPlanes: (DATA.meta && DATA.meta.showPlanes !== undefined) ? DATA.meta.showPlanes : true`（`bindCheckbox` 会自动同步复选框初始态）；
  3. 删除 `core.py:731-732` 的字符串替换（脆弱且不必要）；
  4. 补一条断言：`build_html(include_planes=True) != build_html(include_planes=False)`，并断言 CLI/E2E 产物中 `#cb-planes` 初始勾选态与参数一致。

### LZ-VIZ-02（中）`python -m n3d_viz --with-ply-edges` 未透传，PLY 不写 edge 元素
- **现象**：`python -m n3d_viz -c … --with-ply-edges` 正常退出（exit 0）但不产生 `element edge`，PLY 与不加该参数时一致；而直跑 `python n3d_viz/export_geometry.py … --with-ply-edges` **生效**（头部出现 `element edge 2`）。README 第 37 行宣称该参数在 CLI 生效。
- **证据**：CLI 单测 `test_with_ply_edges_flag` 失败（头部无 `element edge 2`）；对照实验日志：直跑入口 `element edge 2`，模块入口只有 `element vertex 4`。
- **根因**：`__main__.py:59-62` 定义并解析了 `--with-ply-edges`，但 `__main__.py:112-120` 调用 `core.write_outputs(...)` 时**未传该参数**；`core.py:673` 的 `write_outputs` 签名中也没有 `with_edges`，`core.py:710` 固定调用 `write_ply(data, target, binary=ply_binary)`。
- **修复建议**：`write_outputs` 增加 `with_edges: bool = False` 并透传 `write_ply(..., with_edges=with_edges)`；`__main__.py` 传 `with_edges=args.with_ply_edges`。

### 过程建议（非缺陷）
- `verify_viz.py` 的 50 项断言全部通过，但**没有一项覆盖「开关类参数是否真的改变了产物」**——这正是两个死参数同时漏检的原因。建议为 CLI/GUI 的每个布尔开关补一条「产物可观测差异」断言。
- `assets/viewer_smoke.js:98` 把投影质心断言硬编码为画布中心 `(450, 310)`，与桩尺寸 900×620 / `devicePixelRatio=1` 绑定；若调整桩尺寸需同步修改（提示级）。
- 未发现崩溃、数据错误或安全性问题；渲染器对注入字符串的 `</` 转义经真实浏览器验证有效。
## 四、失败用例索引

| # | 测试类型 | 用例 | 归因 |
|---|---|---|---|
| 1 | 单元测试 | `TestWriteOutputs.test_include_planes_flag_changes_html` | LZ-VIZ-01 |
| 2 | 接口测试 | `CliInterfaceTest.test_no_plan_planes_actually_disables_planes` | LZ-VIZ-01 |
| 3 | 接口测试 | `CliInterfaceTest.test_render_html_script_no_plan_planes` | LZ-VIZ-01 |
| 4 | 接口测试 | `CliInterfaceTest.test_with_ply_edges_flag` | LZ-VIZ-02 |
| 5 | GUI 功能 | `GuiFunctionalTest.test_gui_planes_checkbox_affects_html` | LZ-VIZ-01 |
| 6 | E2E | `--no-plan-planes 产物打开后层平面默认关闭` | LZ-VIZ-01 |
| 7 | E2E | `on.html 与 off.html 内容不同` | LZ-VIZ-01 |

## 五、测试资产与复现方式

| 资产 | 路径 |
|---|---|
| 单元测试（85 项） | `.lizhu_env/lizhu_viz/lizhu_n3d_viz_unit_tests.py` |
| 接口测试（28 项） | `.lizhu_env/lizhu_viz/lizhu_n3d_viz_cli_tests.py` |
| GUI 功能测试（12 项） | `.lizhu_env/lizhu_viz/lizhu_n3d_viz_gui_tests.py` |
| E2E 测试（33 项） | `.lizhu_env/lizhu_viz/lizhu_n3d_viz_e2e.js` |
| 运行日志 | `.lizhu_env/lizhu_viz/{unit_run,cli_run,gui_run,e2e_run,verify_viz_run}.txt` |
| E2E 截图 / 汇总 | `.lizhu_env/lizhu_viz/e2e_viewer.png`、`e2e_summary.json` |
| 硬约束哈希 | `.lizhu_env/lizhu_viz/{baseline,after}_hashes.txt`（20 项一致） |

复现命令：

```bash
python n3d_viz/verify_viz.py --report checkpoints/n3d_viz/_verify/verify_report.md
python .lizhu_env/lizhu_viz/lizhu_n3d_viz_unit_tests.py
python .lizhu_env/lizhu_viz/lizhu_n3d_viz_cli_tests.py
python .lizhu_env/lizhu_viz/lizhu_n3d_viz_gui_tests.py
NODE_PATH=%LOCALAPPDATA%\npm-cache\_npx\e41f203b7505f1fb\node_modules node .lizhu_env/lizhu_viz/lizhu_n3d_viz_e2e.js
```

## 六、环境说明

- **环境构建**：全程**未安装任何依赖**（无 `npm install` / `pip install`）。Playwright 1.63.0 由 npm 的 `npx` 缓存在本机已存在，Chromium 浏览器二进制已安装于 `%LOCALAPPDATA%\ms-playwright`，通过 `NODE_PATH` 指向该缓存目录直接使用；因此不存在「因环境缺失而跳过」的测试类型。
- **文件系统限制**：工作区位于 exFAT 分区的 `E:` 盘，DSH 文件写入工具的原子写（硬链接）在此返回 `EISDIR`，测试脚本改用 PowerShell/.NET 以 UTF-8（无 BOM）写入；这是工具链限制，与被测代码无关，也不影响结论。
- **GUI 测试方式**：tkinter 无第三方 UI 自动化库可用（受「零新依赖」约束），故采用「真实 `VizApp` 实例 + 真实回调 + 真实后台线程/queue/after 链路」的功能测试（12 项），并以 `python -m n3d_viz` 无参启动驻留 mainloop 作为进程级验证；未做像素级窗口截图比对。
- 无图形环境相关的跳过项（GUI 测试在带显示环境的 Windows 上实际执行通过）。

## 七、结论

1. **功能主体正确**：checkpoint 三类错误区分、拓扑抽取（含异常/边界形状校验）、产物命名派生与不静默覆盖、PLY/OBJ 写出与独立回读、单文件自包含 HTML（含注入防御）、三级交互渲染器（旋转/缩放/平移/悬停/分层着色/S_in-S_out 高亮/阈值与图层开关）、GUI 后台线程不阻塞、CLI 退出码 0/2/3 约定 —— 均由真实执行验证通过（213 项断言中 206 项通过）。
2. **存在 2 个中等缺陷**（LZ-VIZ-01、LZ-VIZ-02）：两个对外承诺的参数/控件为静默空操作，建议修复后回归本报告第 4 节索引的 7 条用例。
3. **硬约束全部满足**：零新依赖、不 import 业务模块、既有模块源码与产物零改动（20 项哈希一致）、产物落盘位置符合约定。
4. 未发现崩溃、数据错误、越权或安全隐患。