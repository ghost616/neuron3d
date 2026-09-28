/* N3D 两端全连接包裹渲染器（Canvas 2D 叠加层，零依赖，无外部 URL 引用）。
 *
 * ============================ 为什么是叠加层 ============================
 * 硬约束：**无 FC 产物（config 无 fc_dim 或 fc_dim == 0）的 HTML 必须逐字节零回归**。
 * 而本模块的 HTML 是把 assets/viewer.js 逐字内联进模板的，因此哪怕只改它一行，
 * 无 FC 产物的字节数也会变化。故这里把 FC 相关渲染**完全隔离在本文件**：
 * core.py 只在 DATA.fc 存在时才把它内联进 HTML 追加的一段脚本块。
 * 代价是基础渲染器的内部函数（viewMatrix / project / draw）在本 IIFE 作用域内
 * 不可见，需要按**同一套相机口径**重算投影 —— 下面的 fcViewMatrix / fcProject
 * 与 viewer.js 的 viewMatrix / project 等价（同一组相机参数、同一旋转顺序、
 * 同一透视公式），并把结果画到一块透明的叠加 canvas 上；指针事件全部透传给
 * 底层画布（pointer-events:none），因此交互仍由基础渲染器处理。
 *
 * ====================== 相机：单一事实来源（window.__n3d_cam） ======================
 * **本文件不自建相机**。基础渲染器 `viewer.js` 把它的 cam 对象**原样引用**挂到
 * `window.__n3d_cam`（那里是唯一写入方），本文件每帧**只读**该对象，
 * yaw / pitch / dist / panX / panY / focal / tx,ty,tz 全部直读，因此：
 *   * **左键旋转、右键平移、滚轮缩放、点击重置**四种交互与基础层天然同步；
 *   * 无正反馈风险 —— 不再需要「用上一帧 dist 反推用户缩放倍率」这类补丁
 *     （旧实现正是这么做的：既自建相机、又额外监听 wheel/重置维护 `userZoom`，
 *     结果旋转与平移都不同步 —— 这是被修复的真实缺陷，真实浏览器实测叠加层
 *     非透明像素 bbox 在拖拽前后完全不变）；
 *   * 取景（自动缩放）也交给基础层：它的 dist 已包含「神经元云最大半径 × 3.6」，
 *     而 FC 面板与边界块本就位于云的两端外侧、处于同一取景范围内。
 * 约定（必须遵守）：**写入方只有 viewer.js**；本文件不得写 `__n3d_cam` 的任何字段。
 * 若取不到该对象（例如本脚本被单独加载），**不绘制**并与控制台 + 页面内给出一次
 * 可读告警，绝不画出一张与基础层不同步的错图。
 *
 * 绘制内容（与 README「几何与不重叠判据」一节对应）：
 *   * 面板单元：垂直于流向轴的两片平面，按 ceil(sqrt(H)) 列网格排布的小方块，
 *     颜色按该单元权重范数映射（与 Python 侧 export_geometry.weight_color_rgb 同源）；
 *   * 边界块：输入 784 / 输出 10 的半透明线框方盒 + 标签；
 *   * 聚合箭头：边界块 -> 面板；
 *   * 抽样连线：每神经元 top-k 的细虚线，**非全部连接**（声明文本同时显示在
 *     图例与统计面板）。
 */
"use strict";

(function () {
  var DATA = window.N3D_DATA;
  if (!DATA || !DATA.fc) { return; }   // 无 FC：本文件不执行任何操作

  var FC = DATA.fc;
  var META = DATA.meta || {};
  var base = document.getElementById("view");
  if (!base) { return; }

  // ---------------------------------------------------------------- 叠加画布
  // 透明覆盖层，与基础画布同尺寸同位置；pointer-events:none 保证交互仍归底层画布。
  var canvas = document.createElement("canvas");
  canvas.id = "view-fc";
  canvas.style.position = "absolute";
  canvas.style.left = "0";
  canvas.style.top = "0";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.style.pointerEvents = "none";
  // z-index 必须**严格大于**基础画布（`#view` 是 `position:absolute; inset:0` 且
  // z-index 为 auto，参与同一个层叠上下文）。实测 z-index:5 时会**排在基础画布之后**
  // 被它完全盖住：因为基础画布的堆叠顺序由文档位置决定（它在 HTML 中更靠前），
  // 而带 z-index 的定位元素只在同 z-index 内按文档序比较——结果 FC 叠加层
  // 一个像素都看不见（用「改动 FC 绘制颜色后截图字节数完全不变」定位到该缺陷）。
  canvas.style.zIndex = "20";
  var ctx = canvas.getContext("2d");
  var host = base.parentNode || document.body;
  if (host && host.insertBefore) { host.insertBefore(canvas, base.nextSibling || null); }

  // ---------------------------------------------------------------- 相机（共享）
  // **不再自建相机**。基础渲染器 `viewer.js` 把它的 cam 对象原样挂在
  // `window.__n3d_cam` 上（同一对象引用），本文件每帧**只读**该对象，
  // 因此 yaw / pitch / dist / panX / panY / focal / tx,ty,tz 与基础画布
  // 永远一致 —— 左键旋转、右键平移、滚轮缩放、重置四种交互天然同步。
  //
  // 为什么不能自建一份等价相机（旧实现的缺陷）：那样旋转/平移只会作用于
  // 基础层，叠加层恒停在初始视角，表现为「旋转神经元时全连接层不跟随」
  // （真实浏览器实测：叠加层非透明像素 bbox 在拖拽前后完全不变）。
  //
  // 约定：**写入方只有 `viewer.js`**，本文件不得写 `__n3d_cam` 的任何字段。
  /** 每帧读取共享相机（**惰性、实时**）。
   *
   *  为什么不在这里一次性捕获到闭包变量：`__n3d_cam` 由 `viewer.js` 在脚本执行时
   *  定义；若本脚本被执行顺序变化（例如被单独加载、或两者顺序被调换），一次性捕获
   *  会让叠加层**永久**拿不到相机。做成每帧直读后，只要基础渲染器开始运行就立刻
   *  恢复同步，同时也便于「共享相机尚未就绪」时给出一次告警。
   */
  function sharedCam() {
    var c = window.__n3d_cam;
    return (c && typeof c === "object") ? c : null;
  }
  /** 「相机不可用」的连续帧计数与告警去抖阈值。
   *
   *  为什么要去抖：本脚本与 `viewer.js` 在同一页面内是**先后执行的两段**内联脚本，
   *  正常产物已把本块排在 `viewer.js` **之后**（见 core.build_html），但若执行顺序
   *  被改动（或本文件被单独加载），首帧会读不到相机。此时若立刻弹告警面板，就会
   *  在「只是晚一帧就绪」的场景下留下一个永久误报（E2E 抓到过该误报）。
   *  故：先只重试不告警，连续若干帧仍取不到才告警；**相机一旦出现就撤掉告警**。
   */
  var camMissFrames = 0;
  var CAM_WARN_AFTER_FRAMES = 30;
  /** 已注入的告警面板（存在时用于撤销）。 */
  var camWarnEl = null;

  /** 注入 / 撤销「相机不可用」告警（控制台只发一次）。 */
  function setCamWarning(on) {
    if (on) {
      if (camWarnEl) { return; }
      var warnMsg = "[n3d_viz] FC 叠加渲染器未找到共享相机 window.__n3d_cam，" +
        "为避免绘制与基础层不同步的错误图形，本次不绘制全连接层。" +
        "请确认 assets/viewer.js 已在本脚本之前执行（正常产物的内联顺序满足该条件）。";
      if (window.console && console.warn) { console.warn(warnMsg); }
      if (document.body && document.body.appendChild) {
        var warn = document.createElement("div");
        warn.id = "fc-cam-warning";
        warn.style.position = "absolute";
        warn.style.left = "12px";
        warn.style.bottom = "12px";
        warn.style.zIndex = "30";
        warn.style.background = "rgba(90,20,20,0.92)";
        warn.style.color = "#ffd9d9";
        warn.style.border = "1px solid #c66";
        warn.style.borderRadius = "6px";
        warn.style.padding = "8px 10px";
        warn.style.fontSize = "12px";
        warn.style.maxWidth = "46vw";
        warn.textContent = warnMsg;
        document.body.appendChild(warn);
        camWarnEl = warn;
      }
      return;
    }
    if (camWarnEl && camWarnEl.parentNode && camWarnEl.parentNode.removeChild) {
      camWarnEl.parentNode.removeChild(camWarnEl);
    }
    camWarnEl = null;
  }

  var neurons = DATA.neurons || [];
  var centroid = { x: 0, y: 0, z: 0 };
  var radius = 1;
  var i;
  for (i = 0; i < neurons.length; i++) {
    centroid.x += neurons[i].x; centroid.y += neurons[i].y; centroid.z += neurons[i].z;
  }
  if (neurons.length) {
    centroid.x /= neurons.length; centroid.y /= neurons.length; centroid.z /= neurons.length;
  }
  for (i = 0; i < neurons.length; i++) {
    var dx = neurons[i].x - centroid.x, dy = neurons[i].y - centroid.y, dz = neurons[i].z - centroid.z;
    radius = Math.max(radius, Math.sqrt(dx * dx + dy * dy + dz * dz));
  }
  // 含 FC 节点的半径：**仅用于告警信息里的诊断输出**（取景由共享相机的 dist 决定，
  // 不再由本文件自行推导）。
  var fcNodes = [];
  var pi, uu;
  for (pi = 0; pi < (FC.panels || []).length; pi++) {
    var punits = FC.panels[pi].units || [];
    for (uu = 0; uu < punits.length; uu++) { fcNodes.push(punits[uu]); }
  }
  var pblocks = FC.blocks || [];
  for (pi = 0; pi < pblocks.length; pi++) { fcNodes.push(pblocks[pi]); }
  var radiusWithFc = radius;
  for (i = 0; i < fcNodes.length; i++) {
    var fx = fcNodes[i].x - centroid.x, fy = fcNodes[i].y - centroid.y,
      fz = fcNodes[i].z - centroid.z;
    radiusWithFc = Math.max(radiusWithFc, Math.sqrt(fx * fx + fy * fy + fz * fz));
  }

  /** 行主序矩阵乘法 C[r][c] = sum_k A[r][k]*B[k][c]（与 viewer.js 的 mat4Mul 同式）。 */
  function mat4Mul(a, b) {
    var out = new Array(16);
    for (var r = 0; r < 4; r++) {
      for (var c = 0; c < 4; c++) {
        out[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] +
          a[r * 4 + 2] * b[8 + c] + a[r * 4 + 3] * b[12 + c];
      }
    }
    return out;
  }

  /** 绕 X 轴旋转（与 viewer.js 同式）。 */
  function mat4RotX(a) {
    var c = Math.cos(a), s = Math.sin(a);
    return [1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1];
  }

  /** 绕 Y 轴旋转（与 viewer.js 同式）。 */
  function mat4RotY(a) {
    var c = Math.cos(a), s = Math.sin(a);
    return [c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1];
  }

  /** 视图矩阵：p_cam = R*(p - center) + (0,0,-dist)，R = RotX(pitch)*RotY(yaw)。
   *
   *  与 viewer.js 的 viewMatrix() 等价（同一旋转顺序、同一平移列构造）。
   *  相机由参数传入（来自共享对象 `window.__n3d_cam`），**函数内不写入**它。
   *
   *  @param {Object} camera 共享相机对象（yaw/pitch/dist/panX/panY/tx/ty/tz）。
   */
  function fcViewMatrix(camera) {
    var rot = mat4Mul(mat4RotX(camera.pitch), mat4RotY(camera.yaw));
    var m = rot.slice();
    m[12] = -(rot[0] * camera.tx + rot[4] * camera.ty + rot[8] * camera.tz) + camera.panX;
    m[13] = -(rot[1] * camera.tx + rot[5] * camera.ty + rot[9] * camera.tz) + camera.panY;
    m[14] = -(rot[2] * camera.tx + rot[6] * camera.ty + rot[10] * camera.tz) - camera.dist;
    return m;
  }

  /** 世界坐标 -> 屏幕坐标 + 深度（depth = -zCam，越大越靠近相机）。
   *
   *  与 viewer.js 的 project() 等价：同样的 cz > -0.001 剔除、同样的屏幕中心
   *  偏移与 y 轴翻转、同样的 dpr 缩放；焦距也取自共享相机（不再自算）。
   */
  function fcProject(m, camera, x, y, z) {
    var cz = m[2] * x + m[6] * y + m[10] * z + m[14];
    if (cz > -0.001) { return null; }
    var cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    var cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    var dpr = window.devicePixelRatio || 1;
    var k = camera.focal / (-cz);
    return {
      x: canvas.width * 0.5 + cx * k * dpr,
      y: canvas.height * 0.5 - cy * k * dpr,
      depth: -cz, scale: k
    };
  }

  // -------------------------------------------------------- 配色（与 Python 同源）
  /** 权重范数 -> CSS 颜色。与 export_geometry.weight_color_rgb 逐项同式：
   *  t = (v - lo) / (hi - lo)，r = 0.20+0.75t、g = 0.60-0.45t、b = 0.95-0.85t。
   *  本文件不持有第二份层色表（层色仍由基础渲染器的 DATA.layers 提供）。
   */
  function normToRgb(value, lo, hi) {
    var span = (hi - lo) || 1;
    var t = (value - lo) / span;
    function q(x) { return Math.max(0, Math.min(255, Math.round(x * 255))); }
    return "rgb(" + q(0.20 + 0.75 * t) + "," + q(0.60 - 0.45 * t) + "," + q(0.95 - 0.85 * t) + ")";
  }

  var norms = [];
  for (pi = 0; pi < (FC.panels || []).length; pi++) {
    var pu = FC.panels[pi].units || [];
    for (uu = 0; uu < pu.length; uu++) { norms.push(pu[uu].norm); }
  }
  var normLo = norms.length ? Math.min.apply(null, norms) : 0;
  var normHi = norms.length ? Math.max.apply(null, norms) : 1;

  var fcEdges = FC.edges || [];
  var wMax = 0.0001;
  for (i = 0; i < fcEdges.length; i++) {
    wMax = Math.max(wMax, Math.abs(fcEdges[i].w));
  }

  function dprOf() { return window.devicePixelRatio || 1; }

  // ---------------------------------------------------------------- 显隐开关
  var showFc = true;

  /** 动态注入「全连接层」开关、图例项与统计小节。
   *
   *  为什么注入而不写进 viewer.html / viewer.js：两者都会被逐字内联进每一份
   *  HTML，改动它们会让无 FC 产物的字节数变化（破坏零回归锚点）。无 FC 时
   *  本文件整体不执行，因此那些产物里既没有该开关、也没有 FC 图例与统计小节。
   */
  function injectControls() {
    var toolbar = document.getElementById("toolbar");
    if (toolbar && !document.getElementById("cb-fc") && toolbar.appendChild) {
      var label = document.createElement("label");
      var box = document.createElement("input");
      box.type = "checkbox";
      box.id = "cb-fc";
      box.checked = showFc;
      if (label.appendChild) {
        label.appendChild(box);
        label.appendChild(document.createTextNode("全连接层（抽样）"));
        toolbar.appendChild(label);
      }
      if (box.addEventListener) {
        box.addEventListener("change", function () { showFc = box.checked; scheduleDraw(); });
      }
    }
    var legend = document.getElementById("legend");
    if (legend && legend.getAttribute && legend.getAttribute("data-fc") !== "1") {
      legend.setAttribute("data-fc", "1");
      legend.innerHTML +=
        '<span class="sw" style="background:linear-gradient(90deg,#33a5f2,#ff7838)"></span>' +
        '全连接层 H=' + FC.fcWidth + '（色 = 权重范数）' +
        '<span class="sw" style="background:repeating-linear-gradient(90deg,#78e6ff 0 3px,transparent 3px 6px)"></span>' +
        'FC 抽样连线 ' + (META.fcSampleEdges || 0) + ' 条（非全部连接）';
    }
    // FC 统计信息放进**独立面板**（而不是往共享的 #stats 里追加文本）：
    // 基础渲染器每次重绘都会整体重写 #stats.textContent，追加的文本会被冲掉
    // （实测：滑块一动作，追加的 FC 小节即消失）。独立面板彻底避免互相覆盖，
    // 也让「有 FC / 无 FC」的页面结构差异完全可控。
    if (!document.getElementById("fc-stats") && document.body && document.body.appendChild) {
      var panel = document.createElement("div");
      panel.id = "fc-stats";
      panel.style.position = "absolute";
      panel.style.right = "12px";
      panel.style.top = "150px";
      panel.style.background = "rgba(12,18,34,0.86)";
      panel.style.border = "1px solid rgba(120,160,240,0.28)";
      panel.style.borderRadius = "8px";
      panel.style.padding = "10px 12px";
      panel.style.fontSize = "11.5px";
      panel.style.lineHeight = "1.65";
      panel.style.whiteSpace = "pre";
      panel.style.zIndex = "8";
      panel.textContent =
        "FC 两端全连接包裹（抽样）" +
        "\nfc_dim=" + META.fcDim + "  H=" + META.fcWidth + "  top-k=" + META.fcTopK +
        "\n面板点=" + (2 * META.fcWidth) + "（2×H）" +
        "\nproj_weight=" + FC.projWeightShape.join("×") + "  " + FC.projCount + " 条" +
        "\nfc_out_weight=" + FC.fcOutWeightShape.join("×") + "  " + FC.fcOutCount + " 条" +
        "\n抽样连线=" + fcEdges.length + " 条（" + META.fcDeclaration + "）";
      document.body.appendChild(panel);
    }
    var stat = document.getElementById("stats");
    if (stat && stat.getAttribute && stat.getAttribute("data-fc") !== "1") {
      stat.setAttribute("data-fc", "1");
    }
  }
  // ---------------------------------------------------------------- 绘制
  /** 把面板 / 边界块 / 聚合箭头 / 抽样连线投影后按深度排序绘制到叠加层。
   *
   *  深度口径与基础渲染器一致（线段取两端均值、方块取四角均值），因此叠加层
   *  内部的前后关系正确；与底层元素之间的遮挡由「叠加层整体画在最上面」决定
   *  —— 这也正是「全连接层是包裹在核心两端的外层结构」的直观表达。
   *
   *  相机**全部来自共享对象** `window.__n3d_cam`（基础渲染器是唯一写入方）：
   *  yaw / pitch / dist / panX / panY / focal / tx,ty,tz 一律直读，
   *  本函数**不写入**其中任何字段。因此旋转 / 平移 / 缩放 / 重置四种交互
   *  与基础画布天然同步（旧实现自建相机时旋转与平移都不同步）。
   */
  function draw() {
    if (!ctx) { return; }
    var dpr = dprOf();
    if (ctx.setTransform) { ctx.setTransform(1, 0, 0, 1, 0, 0); }
    if (ctx.clearRect) { ctx.clearRect(0, 0, canvas.width, canvas.height); }

    // ---- 共享相机不可用：去抖后告警、**不画错图** -------------------------
    // 触发场景：`viewer_fc.js` 被单独加载、或内联脚本执行顺序被改动。此时若沿用
    // 「自建相机兜底」就会画出与基础层不同步的图（正是被修复的缺陷形态），
    // 故明确拒绝绘制；告警按 `CAM_WARN_AFTER_FRAMES` 去抖，且相机一旦就绪即撤销。
    var cam = sharedCam();
    if (!cam) {
      camMissFrames++;
      if (camMissFrames >= CAM_WARN_AFTER_FRAMES) { setCamWarning(true); }
      return;
    }
    camMissFrames = 0;
    setCamWarning(false);
    if (!showFc) { return; }

    var m = fcViewMatrix(cam);
    var items = [];
    var j, k, sc;

    // 1) 面板单元（小方块，颜色 = 权重范数）
    for (j = 0; j < (FC.panels || []).length; j++) {
      var panel = FC.panels[j];
      var units = panel.units || [];
      var half = (panel.cellSize || 0.1) * 0.5;
      for (k = 0; k < units.length; k++) {
        var un = units[k];
        var qs = [
          fcProject(m, cam, un.x - half, un.y - half, un.z),
          fcProject(m, cam, un.x + half, un.y - half, un.z),
          fcProject(m, cam, un.x + half, un.y + half, un.z),
          fcProject(m, cam, un.x - half, un.y + half, un.z)
        ];
        if (!qs[0] || !qs[1] || !qs[2] || !qs[3]) { continue; }
        items.push({
          kind: "unit", d: (qs[0].depth + qs[1].depth + qs[2].depth + qs[3].depth) / 4,
          pts: qs, c: normToRgb(un.norm, normLo, normHi)
        });
      }
    }

    // 2) 边界块（线框方盒 + 淡填充）
    var blocks = FC.blocks || [];
    for (j = 0; j < blocks.length; j++) {
      var bk = blocks[j];
      var hx = (bk.sx || 0) / 2, hy = (bk.sy || 0) / 2, hz = (bk.sz || 0) / 2;
      var corners = [
        [bk.x - hx, bk.y - hy, bk.z - hz], [bk.x + hx, bk.y - hy, bk.z - hz],
        [bk.x + hx, bk.y + hy, bk.z - hz], [bk.x - hx, bk.y + hy, bk.z - hz],
        [bk.x - hx, bk.y - hy, bk.z + hz], [bk.x + hx, bk.y - hy, bk.z + hz],
        [bk.x + hx, bk.y + hy, bk.z + hz], [bk.x - hx, bk.y + hy, bk.z + hz]
      ];
      var proj = [], ok = true, acc = 0;
      for (k = 0; k < 8; k++) {
        sc = fcProject(m, cam, corners[k][0], corners[k][1], corners[k][2]);
        if (!sc) { ok = false; break; }
        proj.push(sc); acc += sc.depth;
      }
      if (ok) { items.push({ kind: "block", d: acc / 8, box: proj, label: bk.label }); }
    }

    // 3) 聚合箭头：边界块中心 -> 面板中心
    var centers = {};
    for (j = 0; j < (FC.panels || []).length; j++) {
      var pl2 = FC.panels[j], us2 = pl2.units || [];
      if (!us2.length) { continue; }
      var sx = 0, sy = 0, sz = 0;
      for (k = 0; k < us2.length; k++) { sx += us2[k].x; sy += us2[k].y; sz += us2[k].z; }
      centers[pl2.name] = { x: sx / us2.length, y: sy / us2.length, z: sz / us2.length };
    }
    for (j = 0; j < blocks.length; j++) {
      var b2 = blocks[j], ctr = centers[b2.name];
      if (!ctr) { continue; }
      var pa = fcProject(m, cam, b2.x, b2.y, b2.z), pb = fcProject(m, cam, ctr.x, ctr.y, ctr.z);
      if (!pa || !pb) { continue; }
      items.push({ kind: "arrow", d: (pa.depth + pb.depth) / 2, a: pa, b: pb });
    }

    // 4) 抽样连线（每神经元 top-k；虚线 = 「非全部连接」的视觉提示）
    var unitCache = {};
    for (j = 0; j < fcEdges.length; j++) {
      var fe = fcEdges[j];
      if (!unitCache[fe.side]) {
        var found = null, t;
        for (t = 0; t < (FC.panels || []).length; t++) {
          if (FC.panels[t].name === fe.side) { found = FC.panels[t]; break; }
        }
        unitCache[fe.side] = found;
      }
      var pnl = unitCache[fe.side];
      if (!pnl) { continue; }
      var unit = (pnl.units || [])[fe.unit];
      var nn = neurons[fe.neuron];
      if (!unit || !nn) { continue; }
      var p1 = fcProject(m, cam, unit.x, unit.y, unit.z);
      var p2 = fcProject(m, cam, nn.x, nn.y, nn.z);
      if (!p1 || !p2) { continue; }
      items.push({
        kind: "sample", d: (p1.depth + p2.depth) / 2, a: p1, b: p2,
        w: 0.55 + 1.3 * Math.min(1, Math.abs(fe.w) / wMax)
      });
    }

    // 画家算法：远（depth 小）先画
    items.sort(function (x, y) { return x.d - y.d; });

    var labels = [];
    for (j = 0; j < items.length; j++) {
      var it = items[j];
      if (it.kind === "unit") {
        // 面板单元：**不透明**填充 + 亮边框。为什么不透明：基础渲染器会绘制
        // 半透明的层参考平面（尺寸约为云的 2.3 倍、覆盖整个视野），
        // FC 面板位于云的两端外侧、在其之后，若再半透明就会与平面糊成一片、
        // 完全读不出「两端包裹」的结构（实测截图确认）。
        // 单元按面板单元间距的 0.9 倍画，留出细缝让网格结构可辨
        // （实测单元与间距等大时 29×29 会糊成一块实心色板）。
        var q0 = it.pts[0], q2 = it.pts[2];
        var ux = (q2.x - q0.x), uy = (q2.y - q0.y);
        var shrink = 0.45 * 0.88;   // 相对半对角线的收缩比
        var mx = (q0.x + q2.x) / 2, my = (q0.y + q2.y) / 2;
        ctx.beginPath();
        ctx.fillStyle = it.c;
        ctx.strokeStyle = "rgba(255,255,255,0.50)";
        ctx.lineWidth = 0.6 * dpr;
        for (var vi = 0; vi < 4; vi++) {
          var vx = mx + (it.pts[vi].x - mx) * 2 * shrink;
          var vy = my + (it.pts[vi].y - my) * 2 * shrink;
          if (vi === 0) { ctx.moveTo(vx, vy); } else { ctx.lineTo(vx, vy); }
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      } else if (it.kind === "block") {
        var spans = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
          [0, 4], [1, 5], [2, 6], [3, 7]];
        ctx.strokeStyle = "rgba(190,205,235,0.80)";
        ctx.lineWidth = 1.0 * dpr;
        ctx.beginPath();
        for (k = 0; k < spans.length; k++) {
          ctx.moveTo(it.box[spans[k][0]].x, it.box[spans[k][0]].y);
          ctx.lineTo(it.box[spans[k][1]].x, it.box[spans[k][1]].y);
        }
        ctx.stroke();
        ctx.fillStyle = "rgba(150,175,225,0.10)";
        ctx.beginPath();
        ctx.moveTo(it.box[0].x, it.box[0].y);
        ctx.lineTo(it.box[1].x, it.box[1].y);
        ctx.lineTo(it.box[2].x, it.box[2].y);
        ctx.lineTo(it.box[3].x, it.box[3].y);
        ctx.closePath();
        ctx.fill();
        labels.push({ x: it.box[6].x, y: it.box[6].y, t: it.label });
      } else if (it.kind === "arrow") {
        ctx.beginPath();
        ctx.strokeStyle = "rgba(215,225,250,0.85)";
        ctx.lineWidth = 1.6 * dpr;
        ctx.moveTo(it.a.x, it.a.y);
        ctx.lineTo(it.b.x, it.b.y);
        ctx.stroke();
        var ang = Math.atan2(it.b.y - it.a.y, it.b.x - it.a.x);
        var hl = 9 * dpr;
        ctx.beginPath();
        ctx.moveTo(it.b.x, it.b.y);
        ctx.lineTo(it.b.x - hl * Math.cos(ang - 0.4), it.b.y - hl * Math.sin(ang - 0.4));
        ctx.moveTo(it.b.x, it.b.y);
        ctx.lineTo(it.b.x - hl * Math.cos(ang + 0.4), it.b.y - hl * Math.sin(ang + 0.4));
        ctx.stroke();
      } else {
        ctx.beginPath();
        if (ctx.setLineDash) { ctx.setLineDash([4 * dpr, 4 * dpr]); }
        // 抽样连线必须**弱于**面板单元：3,510 条连线在面板上大面积交叉
        // （每个神经元各连到网格上不同位置的 top-k 单元），若画得太亮会形成
        // 一片密不透光的网把面板本身盖住（实测：alpha 0.95 时输出侧面板
        // 完全糊成一片青蓝雾，网格结构不可辨）。细虚线 + 低透明度让
        // 「连线存在」可读，同时保留面板的网格结构。
        ctx.strokeStyle = "rgba(150,240,255,0.16)";
        ctx.lineWidth = 0.6 * dpr;
        ctx.moveTo(it.a.x, it.a.y);
        ctx.lineTo(it.b.x, it.b.y);
        ctx.stroke();
        if (ctx.setLineDash) { ctx.setLineDash([]); }
      }
    }

    // 标签最后画，避免被方块遮住
    ctx.fillStyle = "rgba(215,230,255,0.92)";
    ctx.font = Math.round(11 * dpr) + "px sans-serif";
    for (j = 0; j < (FC.panels || []).length; j++) {
      var pnl3 = FC.panels[j], us3 = pnl3.units || [];
      if (!us3.length) { continue; }
      var last = us3[us3.length - 1];
      var ls = fcProject(m, cam, last.x, last.y, last.z);
      if (ls) {
        ctx.fillText(
          (pnl3.name === "input" ? "输入侧 H=" : "输出侧 H=") + us3.length +
          "（" + pnl3.cols + "×" + pnl3.rows + "）",
          ls.x + 6 * dpr, ls.y - 6 * dpr
        );
      }
    }
    for (j = 0; j < labels.length; j++) {
      ctx.fillText(labels[j].t, labels[j].x + 6 * dpr, labels[j].y - 6 * dpr);
    }
  }
  // ---------------------------------------------------- 与基础渲染器同步刷新
  // 基础渲染器在 mousedown / mousemove / wheel / resize 时重绘自己的画布，
  // 但不会通知我们。这里用 requestAnimationFrame 持续把叠加层与底层对齐：
  // 尺寸变化时同步重设画布像素尺寸（等价于 viewer.js 的 resize()），然后重画。
  // 每帧只做「读尺寸 + 一次 Canvas 绘制」，开销与基础渲染器自身的一帧相当，
  // 且不需要改动 viewer.js 的任何一个字节。
  function syncSize() {
    var dpr = dprOf();
    var w = Math.round((base.clientWidth || 800) * dpr);
    var h = Math.round((base.clientHeight || 600) * dpr);
    if (w !== canvas.width || h !== canvas.height) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  var pending = false;
  function raf(fn) {
    if (window.requestAnimationFrame) { return window.requestAnimationFrame(fn); }
    return setTimeout(fn, 16);
  }
  function scheduleDraw() {
    if (pending) { return; }
    pending = true;
    raf(function () { pending = false; syncSize(); draw(); });
  }

  // 防「同步 requestAnimationFrame」递归：真实浏览器里 RAF 必然异步（每帧一次），
  // 但最小 DOM 桩环境可能把它实现为同步调用，此时 loop -> raf(loop) 会同栈无限
  // 递归（实测 node 桩下表现为 RangeError / 堆内存耗尽，使整个冒烟失败）。
  // 这里限制同栈嵌套深度：桩环境下完成首帧绘制后即停止，真实浏览器下深度恒为 1、
  // 持续刷新不受影响。
  var loopDepth = 0;
  var LOOP_DEPTH_LIMIT = 4;
  function loop() {
    syncSize();
    draw();
    if (loopDepth >= LOOP_DEPTH_LIMIT) { return; }
    loopDepth++;
    try {
      raf(loop);
    } finally {
      loopDepth--;
    }
  }

  if (window.addEventListener) { window.addEventListener("resize", scheduleDraw); }
  // 旋转 / 平移 / 缩放 / 重置**都不需要在这里打补丁**：两层的取景参数来自同一个
  // `window.__n3d_cam` 对象，基础渲染器（唯一写入方）改动它，叠加层下一帧即读到新值。
  // 旧实现为此额外监听 wheel 与重置按钮维护自有的 userZoom —— 已随自建相机一并删除。
  injectControls();
  loop();
})();