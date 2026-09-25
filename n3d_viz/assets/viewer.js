/* N3D 二期拓扑三维渲染器（Canvas 2D，零依赖，无任何外部 URL 引用）。
 *
 * 渲染原理（画家算法）：
 *   1) 视图变换：世界坐标 -> 相机坐标。相机绕世界中心做球面轨道运动，
 *      yaw 绕世界 Y 轴（屏幕竖直轴），pitch 绕相机右轴；平移量 pan 直接加在
 *      视图矩阵的平移列上（等价于在相机平面内平移观察目标）。
 *   2) 透视投影：屏幕坐标 = 焦距 * (x/-z, y/-z) + 画布中心；相机空间中
 *      z 越小（越靠近相机）投影尺度越大。深度用 -z 记录，越大越靠近相机。
 *   3) 深度排序：把神经元、边、层平面统一按深度升序（远 -> 近）绘制，
 *      后画的覆盖先画的，从而在二维画布上得到正确的遮挡关系。
 *
 * 交互：左键拖拽旋转、滚轮缩放、右键拖拽平移、悬停查看神经元详情。
 * 说明：全部数据由构建期内联进 N3D_DATA，运行时不再请求任何外部资源。
 */
"use strict";

(function () {
  var DATA = window.N3D_DATA || { meta: {}, layers: [], neurons: [], edges: [] };
  var canvas = document.getElementById("view");
  var ctx = canvas.getContext("2d");
  var labelEl = document.getElementById("hover-label");
  var statEl = document.getElementById("stats");
  var countEl = document.getElementById("keep-count");

  // ---------------------------------------------------------------- 数学工具
  // 约定：矩阵用 16 个数的平铺数组表示， index = row * 4 + col（行主序）。
  // 这里只使用纯旋转矩阵（R），位移单独用向量表示，
  // 避免混合“矩阵乘法顺序”与“列主序/行主序存储”两个约定
  // 而导致整个点云被投影到相机背后。

  /** 单位矩阵（行主序）。 */
  function mat4Identity() {
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  }

  /** 绕 X 轴旋转：R[1][1]=cos, R[1][2]=-sin, R[2][1]=sin, R[2][2]=cos。 */
  function mat4RotX(a) {
    var c = Math.cos(a), s = Math.sin(a);
    return [1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1];
  }

  /** 绕 Y 轴旋转：R[0][0]=cos, R[0][2]=sin, R[2][0]=-sin, R[2][2]=cos。 */
  function mat4RotY(a) {
    var c = Math.cos(a), s = Math.sin(a);
    return [c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1];
  }

  /** 矩阵乘法 A*B（行主序）：C[r][c] = Σ_k A[r][k] * B[k][c]。
   *
   * 注意：写成 a[k*4+r] 会得到 (A*B) 的转置，在此相机约定下
   * 等价于把整个点云翻到相机背后（画面只剩背景）。
   */
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

  /** 用 4x4 矩阵变换一个点（w 分量视为 1）。 */
  function mat4Apply(m, p) {
    return {
      x: m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
      y: m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
      z: m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]
    };
  }

  // ---------------------------------------------------------------- 相机状态
  var cam = {
    yaw: -0.62,         // 绕世界 Y 轴（屏幕竖直轴）旋转
    pitch: 0.42,        // 俯仰，限幅避免翻转
    dist: 3.6,          // 相机到目标的距离（世界单位）
    panX: 0, panY: 0,   // 屏幕平面内的平移
    focal: 900,         // 焦距（像素），决定透视强度
    tx: 0, ty: 0, tz: 0 // 观察目标（取点云质心）
  };

  // 初始开关一律以负载为准（构建时写入的 meta），
  // 避免在这里硬编码导致命令行开关成为空操作。
  var META = DATA.meta || {};
  var view = {
    showNeurons: true,
    showEdges: true,
    showPlanes: (META.showPlanes === undefined) ? true : Boolean(META.showPlanes),
    threshold: (META.threshold !== undefined) ? META.threshold : 0.3
  };

  // ------------------------------------------------------------ 数据预处理
  var neurons = DATA.neurons || [];
  var edges = DATA.edges || [];
  var layers = DATA.layers || [];
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
  cam.tx = centroid.x; cam.ty = centroid.y; cam.tz = centroid.z;
  cam.dist = radius * 3.6;

  for (i = 0; i < edges.length; i++) {
    edges[i].aw = Math.abs(edges[i].w);
  }
  var wMax = 0.0001;
  for (i = 0; i < edges.length; i++) { wMax = Math.max(wMax, edges[i].aw); }

  var buf = document.createElement("canvas");
  var bctx = buf.getContext("2d");

  // ---------------------------------------------------------------- 尺寸
  function resize() {
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth || 800;
    var h = canvas.clientHeight || 600;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    buf.width = canvas.width;
    buf.height = canvas.height;
    cam.focal = Math.max(canvas.height * 0.9, 300);
    draw();
  }

  /** 生成视图矩阵。
   *
   * 原理（标准轨道相机）：相机本体位于目标点的 +Z 方向 dist 处，
   * 朝 -Z 方向观察。世界点 p 到相机坐标的变换为：
   *     p_cam = R * (p - center) + (0, 0, -dist)
   * 其中 R = RotX(pitch) * RotY(yaw) 是相机的旋转，最后的 -dist 把目标点放到
   * 相机前方 dist 处（即 z_cam = -dist < 0，满足透视投影的可见条件）。
   * 展开成一个 4x4 矩阵后，旋转部分为 R，平移列为 -R*center + (0,0,-dist)。
   */
  function viewMatrix() {
    var rot = mat4Mul(mat4RotX(cam.pitch), mat4RotY(cam.yaw));
    var m = rot.slice();
    m[12] = -(rot[0] * cam.tx + rot[4] * cam.ty + rot[8] * cam.tz) + cam.panX;
    m[13] = -(rot[1] * cam.tx + rot[5] * cam.ty + rot[9] * cam.tz) + cam.panY;
    m[14] = -(rot[2] * cam.tx + rot[6] * cam.ty + rot[10] * cam.tz) - cam.dist;
    return m;
  }

  /** 世界坐标 -> 屏幕坐标 + 深度。depth = -zCam，越大越靠近相机。 */
  function project(m, x, y, z) {
    var cz = m[2] * x + m[6] * y + m[10] * z + m[14];
    if (cz > -0.001) { return null; } // 位于相机之后，剔除
    var cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    var cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    var dpr = window.devicePixelRatio || 1;
    var k = cam.focal / (-cz);
    return {
      x: canvas.width * 0.5 + cx * k * dpr,
      y: canvas.height * 0.5 - cy * k * dpr,
      depth: -cz,
      scale: k
    };
  }

  function layerColor(lv) {
    return (layers[lv] && layers[lv].color) ? layers[lv].color : "#cccccc";
  }

  /** 观感：弱边青蓝细线，强边橙红粗线（粗细按 |w| 归一化）。 */
  function edgeStyle(aw) {
    var t = Math.min(1, aw / wMax);
    var width = 0.6 + 2.6 * t;
    var r = Math.round(60 + 195 * t);
    var g = Math.round(190 - 120 * t);
    var b = Math.round(235 - 195 * t);
    return { width: width, color: "rgba(" + r + "," + g + "," + b + ",0.85)" };
  }

  // -------------------------------------------------- 各图层「可绘制项」构建
  /** 构建神经元绘制项：投影 + 深度，颜色由层决定，S_in / S_out 覆写。 */
  function collectNeurons(m) {
    var items = [];
    for (var j = 0; j < neurons.length; j++) {
      var n = neurons[j];
      var sc = project(m, n.x, n.y, n.z);
      if (!sc) { continue; }
      // 基础圆用层色（保留 z 分层信息）；
      // S_in / S_out 以“同心小圆”叠加高亮，而不直接覆盖层色，
      // 这样即便高亮也能看出其所属层。
      items.push({
        d: sc.depth, x: sc.x, y: sc.y, r: 3.6, c: layerColor(n.layer), id: n.id,
        sIn: Boolean(n.s_in), sOut: Boolean(n.s_out)
      });
    }
    return items;
  }

  /** 构建边绘制项：按阈值过滤后投影，深度取两端点均值。 */
  function collectEdges(m, threshold) {
    var items = [];
    var kept = 0;
    for (var j = 0; j < edges.length; j++) {
      var e = edges[j];
      if (e.aw < threshold) { continue; }
      kept++;
      var a = neurons[e.src], b = neurons[e.dst];
      if (!a || !b) { continue; }
      var pa = project(m, a.x, a.y, a.z);
      var pb = project(m, b.x, b.y, b.z);
      if (!pa || !pb) { continue; }
      var st = edgeStyle(e.aw);
      items.push({ d: (pa.depth + pb.depth) * 0.5, x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, c: st.color, w: st.width });
    }
    return { items: items, kept: kept };
  }

  /** 构建层参考平面：每层一张水平方片，边长覆盖点云包围盒。 */
  function collectPlanes(m) {
    var items = [];
    var half = radius * 1.15;
    for (var j = 0; j < layers.length; j++) {
      var lv = layers[j];
      if (lv.count <= 0) { continue; }
      var corners = [
        { x: centroid.x - half, y: centroid.y - half, z: lv.z }, { x: centroid.x + half, y: centroid.y - half, z: lv.z },
        { x: centroid.x + half, y: centroid.y + half, z: lv.z }, { x: centroid.x - half, y: centroid.y + half, z: lv.z }
      ];
      var pts = [], ok = true, depth = 0;
      for (var k = 0; k < 4; k++) {
        var sc = project(m, corners[k].x, corners[k].y, corners[k].z);
        if (!sc) { ok = false; break; }
        pts.push(sc); depth += sc.depth;
      }
      if (!ok) { continue; }
      items.push({ d: depth / 4, pts: pts, c: lv.color, level: lv.level });
    }
    return items;
  }

  // ---------------------------------------------------------------- 主绘制
  var lastProjected = [];

  function hexToRgba(hex, a) {
    var h = hex.replace("#", "");
    if (h.length === 3) { h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2]; }
    var num = parseInt(h, 16);
    return "rgba(" + ((num >> 16) & 255) + "," + ((num >> 8) & 255) + "," + (num & 255) + "," + a + ")";
  }

  function drawHud(kept) {
    if (!statEl) { return; }
    var meta = DATA.meta || {};
    var lines = [
      "checkpoint: " + (meta.ckpt_stem || "?") + ".pt",
      "seed=" + meta.seed + "  N=" + meta.N + "  E=" + meta.E + "  K=" + meta.n_layers,
      "S_in=" + meta.n_s_in + "  S_out=" + meta.n_s_out + "  密度=" + (meta.conn_density || 0).toExponential(2),
      "测试准确率=" + meta.test_acc,
      "层规模: " + (meta.layer_counts || []).join("/"),
      "当前保留边: " + kept
    ];
    statEl.textContent = lines.join("\n");
  }

  function draw() {
    var m = viewMatrix();
    var keep = { items: [], kept: 0 };
    if (view.showEdges) {
      keep = collectEdges(m, view.threshold);
    } else {
      for (var q = 0; q < edges.length; q++) { if (edges[q].aw >= view.threshold) { keep.kept++; } }
    }
    var nItems = view.showNeurons ? collectNeurons(m) : [];
    var pItems = view.showPlanes ? collectPlanes(m) : [];
    lastProjected = nItems;

    // 背景：竖直渐变
    var g = ctx.createLinearGradient(0, 0, 0, canvas.height);
    g.addColorStop(0, "#0b1020");
    g.addColorStop(1, "#04060d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "rgba(0,0,0,0.45)";  // 重置状态，避免渐变对象泄露到后续图元

    var all = [];
    var j;
    for (j = 0; j < pItems.length; j++) { all.push({ kind: "plane", d: pItems[j].d, it: pItems[j] }); }
    for (j = 0; j < keep.items.length; j++) { all.push({ kind: "edge", d: keep.items[j].d, it: keep.items[j] }); }
    for (j = 0; j < nItems.length; j++) { all.push({ kind: "node", d: nItems[j].d, it: nItems[j] }); }
    // 画家算法：远（depth 小）先画，近的后画覆盖
    all.sort(function (a, b) { return a.d - b.d; });

    var dpr = window.devicePixelRatio || 1;
    for (j = 0; j < all.length; j++) {
      var rec = all[j], it = rec.it;
      if (rec.kind === "plane") {
        ctx.beginPath();
        ctx.moveTo(it.pts[0].x, it.pts[0].y);
        for (var k = 1; k < 4; k++) { ctx.lineTo(it.pts[k].x, it.pts[k].y); }
        ctx.closePath();
        ctx.fillStyle = hexToRgba(it.c, 0.055);
        ctx.fill();
        ctx.strokeStyle = hexToRgba(it.c, 0.35);
        ctx.lineWidth = 1.0;
        ctx.stroke();
        ctx.fillStyle = hexToRgba(it.c, 0.75);
        ctx.font = Math.round(11 * dpr) + "px sans-serif";
        ctx.fillText("L" + (it.level + 1), it.pts[3].x + 4, it.pts[3].y - 4);
      } else if (rec.kind === "edge") {
        ctx.beginPath();
        ctx.moveTo(it.x1, it.y1);
        ctx.lineTo(it.x2, it.y2);
        ctx.strokeStyle = it.c;
        ctx.lineWidth = it.w * dpr;
        ctx.stroke();
      } else {
        // 神经元：层色实心圆 + S_in/S_out 同心高亮。
        // 每个圆都显式重置 fillStyle / strokeStyle，
        // 不依赖上一个图元遗留的状态。
        ctx.beginPath();
        ctx.fillStyle = it.c;
        ctx.strokeStyle = "rgba(0,0,0,0.45)";
        ctx.lineWidth = 0.7 * dpr;
        ctx.arc(it.x, it.y, it.r * dpr, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        if (it.sIn) {
          ctx.beginPath();
          ctx.fillStyle = "#ff5ec7";   // 接入输入层
          ctx.arc(it.x, it.y, it.r * 0.55 * dpr, 0, Math.PI * 2);
          ctx.fill();
        }
        if (it.sOut) {
          ctx.beginPath();
          ctx.fillStyle = "#ffd84d";   // 接出读出头
          ctx.arc(it.x, it.y, it.r * (it.sIn ? 0.26 : 0.55) * dpr, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    if (countEl) {
      countEl.textContent = "保留 " + keep.kept + " / " + edges.length + " 条边（|w| >= " + view.threshold.toFixed(2) + "）";
    }
    drawHud(keep.kept);
  }

  // ---------------------------------------------------------------- 交互
  var drag = null;

  function pos(ev) {
    var rect = canvas.getBoundingClientRect();
    return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
  }

  /** 悬停命中：在投影后的神经元里找最近的一个（阈值 12 像素）。 */
  function hover(p2) {
    if (!labelEl) { return; }
    var dpr = window.devicePixelRatio || 1;
    var best = null, bestD = 1e9;
    for (var j = 0; j < lastProjected.length; j++) {
      var it = lastProjected[j];
      var ddx = it.x / dpr - p2.x, ddy = it.y / dpr - p2.y;
      var d2 = ddx * ddx + ddy * ddy;
      if (d2 < bestD) { bestD = d2; best = it; }
    }
    if (!best || bestD > 144) { labelEl.style.display = "none"; return; }
    var n = neurons[best.id];
    labelEl.style.display = "block";
    labelEl.style.left = (p2.x + 14) + "px";
    labelEl.style.top = (p2.y + 12) + "px";
    labelEl.innerHTML = "<b>神经元 #" + n.id + "</b><br>层 L" + (n.layer + 1) +
      "（z = " + n.z.toFixed(4) + "）<br>入度 " + n.in_degree + " / 出度 " + n.out_degree +
      "<br>S_in: " + (n.s_in ? "是" : "否") + " / S_out: " + (n.s_out ? "是" : "否");
  }

  // ---------------------------------------------------------------- 控件绑定
  function bindCheckbox(id, key) {
    var el = document.getElementById(id);
    if (!el) { return; }
    el.checked = view[key];
    el.addEventListener("change", function () { view[key] = el.checked; draw(); });
  }
  bindCheckbox("cb-neurons", "showNeurons");
  bindCheckbox("cb-edges", "showEdges");
  bindCheckbox("cb-planes", "showPlanes");

  var slider = document.getElementById("threshold");
  if (slider) {
    slider.value = String(view.threshold);
    slider.addEventListener("input", function () {
      view.threshold = parseFloat(slider.value);
      var tEl = document.getElementById("threshold-label");
      if (tEl) { tEl.textContent = view.threshold.toFixed(2); }
      draw();
    });
  }

  var legend = document.getElementById("legend");
  if (legend) {
    var html = [];
    for (var j = 0; j < layers.length; j++) {
      html.push('<span class="sw" style="background:' + layers[j].color + '"></span>L' +
        (layers[j].level + 1) + " (" + layers[j].count + ")");
    }
    html.push('<span class="sw" style="background:#ff5ec7"></span>S_in ' + (DATA.meta.n_s_in || 0));
    html.push('<span class="sw" style="background:#ffd84d"></span>S_out ' + (DATA.meta.n_s_out || 0));
    html.push('<span class="sw" style="background:linear-gradient(90deg,#3cbEEB,#ff7838)"></span>边粗细/颜色 = |w|');
    legend.innerHTML = html.join(" ");
  }

  var resetBtn = document.getElementById("btn-reset");
  if (resetBtn) {
    resetBtn.addEventListener("click", function () {
      cam.yaw = -0.62; cam.pitch = 0.42; cam.dist = radius * 3.6;
      cam.panX = 0; cam.panY = 0;
      draw();
    });
  }

  canvas.addEventListener("contextmenu", function (ev) { ev.preventDefault(); });
  canvas.addEventListener("mousedown", function (ev) {
    var p2 = pos(ev);
    drag = { x: p2.x, y: p2.y, button: ev.button };
    ev.preventDefault();
  });
  window.addEventListener("mouseup", function () { drag = null; });
  window.addEventListener("mousemove", function (ev) {
    if (drag) {
      var p2 = pos(ev);
      var dx = p2.x - drag.x, dy = p2.y - drag.y;
      drag.x = p2.x; drag.y = p2.y;
      if (drag.button === 2) {
        cam.panX += dx * 0.0022 * cam.dist;
        cam.panY -= dy * 0.0022 * cam.dist;
      } else {
        cam.yaw += dx * 0.008;
        cam.pitch += dy * 0.008;
        var lim = Math.PI / 2 - 0.02;
        cam.pitch = Math.max(-lim, Math.min(lim, cam.pitch));
      }
      draw();
      return;
    }
    if (ev.target !== canvas) { if (labelEl) { labelEl.style.display = "none"; } return; }
    hover(pos(ev));
  });
  canvas.addEventListener("wheel", function (ev) {
    ev.preventDefault();
    cam.dist *= Math.exp(ev.deltaY * 0.0012);
    cam.dist = Math.max(radius * 0.35, Math.min(radius * 40, cam.dist));
    draw();
  }, { passive: false });

  window.addEventListener("resize", resize);
  resize();
})();
