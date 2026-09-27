/* N3D 渲染器逻辑冒烟（零依赖：仅 Node + 最小 DOM/Canvas 桩）。
 *
 * 用途：在不安装任何第三方浏览器自动化依赖的前提下，真实执行 HTML 里内联的
 * viewer.js（以及两端全连接包裹的叠加渲染器），检查：
 *   1) 视图矩阵正确（世界中心必须投影到画布中心）；
 *   2) 相机变换把神经元与边投影到画布范围内（bbox 有限且在画布附近）；
 *   3) 阈值滑块联动逻辑生效（阈值 0 -> 保留全部边）；
 *   4) 悬停命中与图例渲染可执行；
 *   5) 有 FC 负载时（第三个参数给出叠加渲染器源码）：
 *      叠加画布被创建、面板单元方块数 == 2×H、抽样连线数 == meta.fcSampleEdges、
 *      每个面板的流向轴坐标区间与神经元云区间不相交、图例多出 FC 条目。
 *
 * 用法：node n3d_viz/assets/viewer_smoke.js <viewer.js 路径> <内联数据 JSON 路径> [<FC 叠加渲染器路径>]
 * 退出码：0 全部通过，1 存在失败，2 参数缺失。
 */
"use strict";

const fs = require("fs");

const jsPath = process.argv[2];
const dataPath = process.argv[3];
const fcJsPath = process.argv[4] || null;
if (!jsPath || !dataPath) {
  console.error("用法: node viewer_smoke.js <viewer.js> <data.json> [<viewer_fc.js>]");
  process.exit(2);
}
const DATA = JSON.parse(fs.readFileSync(dataPath, "utf8"));
const JS = fs.readFileSync(jsPath, "utf8");
const FC_JS = fcJsPath ? fs.readFileSync(fcJsPath, "utf8") : null;

const stats = { arc: 0, lineTo: 0, moveTo: 0, fill: 0, stroke: 0, fillText: 0, fillRect: 0 };
const arcs = [];
const lines = [];
const texts = [];

function ctxStub() {
  const c = {};
  ["beginPath", "closePath", "setTransform", "save", "restore", "clearRect",
    "translate", "rotate", "scale", "setLineDash"].forEach((k) => { c[k] = () => {}; });
  c.arc = (x, y, r) => { stats.arc++; arcs.push([x, y, r, c.fillStyle]); };
  c.moveTo = (x, y) => { stats.moveTo++; lines.push([x, y]); };
  c.lineTo = (x, y) => { stats.lineTo++; lines.push([x, y]); };
  c.fill = () => { stats.fill++; };
  c.stroke = () => { stats.stroke++; };
  c.fillText = (t, x, y) => { stats.fillText++; texts.push([String(t), x, y]); };
  c.fillRect = () => { stats.fillRect++; };
  c.strokeRect = () => {};
  c.createLinearGradient = () => ({ addColorStop: () => {} });
  return c;
}

const els = {};
let tmpSeq = 0;
/** 模拟 document.getElementById：**已存在才返回，否则返回 null**。
 *
 *  必须严格模拟真实 DOM 语义：若像早期版本那样「取不到就顺手创建一个」，
 *  渲染器里 `if (!document.getElementById("x"))` 这类**存在性判断**会永远为假，
 *  导致被守卫的代码块被静默跳过（实测：FC 统计面板因此从未被创建，而其他断言
 *  仍然全绿 —— 典型的「桩比实现宽松」假绿灯）。
 */
function lookup(id) {
  return Object.prototype.hasOwnProperty.call(els, id) ? els[id] : null;
}

/** 取（必要时创建）桩元素，仅用于**断言读取**页面元素，不模拟 DOM API。 */
function el(id) {
  if (!els[id]) {
    const o = {
      style: {}, textContent: "", innerHTML: "", value: "0", checked: true,
      width: 0, height: 0, clientWidth: 900, clientHeight: 620,
      children: [], setAttribute: function (k, v) { this[k] = v; },
      getAttribute: function (k) { return this[k] === undefined ? null : this[k]; },
      appendChild: function (child) { this.children.push(child); return child; },
      insertBefore: function (child) { this.children.push(child); return child; },
      addEventListener: function (ev, fn) { this.__h = this.__h || {}; this.__h[ev] = fn; },
      getContext: () => ctxStub(),
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 900, height: 620 })
    };
    // id 用访问器：渲染器会给 createElement 出来的元素赋 id（叠加画布的 "view-fc"、
    // FC 统计面板的 "fc-stats" 等），桩必须让 els[新 id] 指向同一个对象，否则
    // 后续 getElementById 取不到它。真实 DOM 里元素只有**最后一个** id 生效，
    // 因此赋值时把旧键删掉再写新键。
    let _id = id;
    Object.defineProperty(o, "id", {
      get: function () { return _id; },
      set: function (v) {
        if (_id === v) { return; }
        if (els[_id] === this) { delete els[_id]; }
        _id = v;
        els[v] = this;
      },
      configurable: true,
      enumerable: true
    });
    els[id] = o;
  }
  return els[id];
}

/** 模拟 document.createElement：返回**新的**桩元素（渲染器随后会给它赋自己的 id）。
 *
 *  注意必须返回对象本身的引用，而不是再按 id 去 els 里查一次：渲染器赋新 id 时
 *  访问器会把注册键迁移掉，按旧键查会得到 undefined（实测该 bug 会让叠加画布
 *  变成一个空引用，后续所有绘制调用都不落桩）。
 */
function newTmpElement() {
  const o = el("tmp" + (++tmpSeq));
  o.__tmp = true;
  return o;
}

const listeners = {};
// RAF 预算：桩里同步调用回调（真实浏览器必然异步）。预算见底后不再回调，
// 避免「同步 RAF」把渲染器的刷新循环变成同栈无限递归。
let rafBudget = 5;
global.window = {
  N3D_DATA: DATA, devicePixelRatio: 1,
  addEventListener: (ev, fn) => { listeners[ev] = fn; },
  requestAnimationFrame: (fn) => { if (rafBudget-- > 0) { fn(); } return 0; }
};
global.document = {
  getElementById: lookup,
  // 每次创建都返回**新的**桩元素（渲染器随后会给它赋自己的 id）。
  createElement: () => newTmpElement(),
  // 文本节点桩：叠加渲染器用它给「全连接层」开关补说明文字。
  createTextNode: (text) => ({ nodeType: 3, textContent: String(text) }),
  // 真实浏览器里 document.body 一定存在；叠加渲染器用 body.appendChild 挂 FC 面板。
  body: el("body")
};
// 预先把 <canvas id="view"> 与两个统计/图例容器登记好（真实 HTML 里它们本来就在）。
el("view");
el("hover-label");
el("stats");
el("keep-count");
el("threshold");
el("legend");
el("toolbar");

const checks = [];
function check(name, cond, detail) {
  checks.push({ name: name, ok: Boolean(cond), detail: detail === undefined ? "" : String(detail) });
  console.log((cond ? "  [PASS] " : "  [FAIL] ") + name + (detail === undefined ? "" : ": " + detail));
}

// 基础渲染器（viewer.js）
eval(JS);
// 叠加渲染器（viewer_fc.js）：无 FC 负载时该源码自身会立即 return，不做任何事
if (FC_JS) { eval(FC_JS); }

// 1) 世界中心必须投影到画布中心：验证视图矩阵与透视投影方向正确
const cx = (arcs.reduce((s, p) => s + p[0], 0) / (arcs.length || 1));
const cy = (arcs.reduce((s, p) => s + p[1], 0) / (arcs.length || 1));
check("圆形总数 == N + S_in + S_out",
  arcs.length === DATA.neurons.length + DATA.meta.n_s_in + DATA.meta.n_s_out,
  arcs.length + " 个圆 / 期望 " + (DATA.neurons.length + DATA.meta.n_s_in + DATA.meta.n_s_out) +
  " (= N " + DATA.neurons.length + " + S_in " + DATA.meta.n_s_in + " + S_out " + DATA.meta.n_s_out + ")");
const magenta = arcs.filter((a) => a[3] === "#ff5ec7").length;
const gold = arcs.filter((a) => a[3] === "#ffd84d").length;
check("S_in 高亮圈数 == meta.n_s_in", magenta === DATA.meta.n_s_in, magenta + " / " + DATA.meta.n_s_in);
check("S_out 高亮圈数 == meta.n_s_out", gold === DATA.meta.n_s_out, gold + " / " + DATA.meta.n_s_out);
const thresholdKey = (DATA.meta.threshold || 0.3).toFixed(2);
check("渲染了阈值内的边（moveTo == 保留边数）",
  stats.moveTo >= (DATA.meta.threshold_counts[thresholdKey] || 0),
  stats.moveTo + " 段 / 阈值 " + thresholdKey + " 期望 >= " + DATA.meta.threshold_counts[thresholdKey]);

const base = arcs.slice(0, DATA.neurons.length);
const xs = base.map((p) => p[0]);
const ys = base.map((p) => p[1]);
const bbox = [Math.min.apply(null, xs), Math.max.apply(null, xs), Math.min.apply(null, ys), Math.max.apply(null, ys)];
check("投影 bbox 有限（无 NaN/Infinity）", bbox.every((v) => isFinite(v)), bbox.map((v) => v.toFixed(1)).join(","));
check("投影 bbox 落在画布附近（不过分越界）",
  bbox[0] > -2500 && bbox[1] < 3400 && bbox[2] > -2500 && bbox[3] < 3400,
  "x[" + bbox[0].toFixed(0) + "," + bbox[1].toFixed(0) + "] y[" + bbox[2].toFixed(0) + "," + bbox[3].toFixed(0) + "]");
// 画布中心由桩尺寸推导（而非硬编码），改桩尺寸断言仍然有效
const ccx = el("view").clientWidth / 2;
const ccy = el("view").clientHeight / 2;
check("投影质心接近画布中心",
  Math.abs(cx - ccx) < el("view").clientWidth * 0.15 && Math.abs(cy - ccy) < el("view").clientHeight * 0.2,
  "centroid=(" + cx.toFixed(1) + "," + cy.toFixed(1) + ") center=(" + ccx + "," + ccy + ")");

// 1b) 层平面初始开关必须与负载一致（否则命令行开关是空操作）
const planeBox = el("cb-planes");
const wantPlanes = DATA.meta.showPlanes === undefined ? true : Boolean(DATA.meta.showPlanes);
check("复选框初始态 == meta.showPlanes", planeBox.checked === wantPlanes,
  "checked=" + planeBox.checked + " meta.showPlanes=" + DATA.meta.showPlanes);

// 2) 阈值滑块联动
const slider = el("threshold");
let afterZero = null;
if (slider.__h && slider.__h.input) {
  slider.value = "0";
  slider.__h.input();
  afterZero = el("keep-count").textContent;
}
check("阈值滑块已绑定 input 事件", Boolean(slider.__h && slider.__h.input));
check("阈值 0 时保留全部边", afterZero !== null && afterZero.indexOf(String(DATA.edges.length)) >= 0,
  afterZero === null ? "未绑定" : afterZero);

// 3) 悬停命中
if (listeners.mousemove) {
  listeners.mousemove({ clientX: cx, clientY: cy, target: el("view") });
}
check("悬停命中神经元并填充详情", el("hover-label").style.display === "block",
  el("hover-label").innerHTML.replace(/<[^>]+>/g, " ").trim());
// 图例色块数：K 个层 + S_in + S_out + 边 = K+3；有 FC 时渲染器会再追加 2 个条目
const wantSwatches = DATA.layers.length + 3 + (DATA.fc ? 2 : 0);
check("图例渲染了 K+3(+FC 2) 个色块",
  (el("legend").innerHTML.match(/class="sw"/g) || []).length === wantSwatches,
  (el("legend").innerHTML.match(/class="sw"/g) || []).length + " 块 / 期望 " + wantSwatches);

// 4) 两端全连接包裹（仅当负载含 fc 段）
if (DATA.fc) {
  const F = DATA.fc;
  // 4a) 叠加画布已创建，且尺寸与基础画布一致（投影口径必须完全同步）
  const fcCanvas = els["view-fc"];
  check("FC 叠加画布已创建且尺寸同步",
    Boolean(fcCanvas) && fcCanvas.width === el("view").clientWidth && fcCanvas.height === el("view").clientHeight,
    fcCanvas ? (fcCanvas.width + "×" + fcCanvas.height) : "未创建");
  // 4b) 面板单元方块数 == 2×H：直接按负载核对（渲染器逐单元绘制，不依赖统计量）
  const unitCount = (F.panels || []).reduce((s, p) => s + (p.units || []).length, 0);
  check("面板单元点 == 2×H", unitCount === 2 * F.fcWidth,
    unitCount + " / 期望 " + (2 * F.fcWidth) + "（H=" + F.fcWidth + "）");
  // 4c) 抽样连线条数 == meta.fcSampleEdges == (|S_in|+|S_out|)×k
  check("抽样连线数 == meta.fcSampleEdges",
    (F.edges || []).length === DATA.meta.fcSampleEdges && F.edges.length === F.sampleEdgesExpected,
    F.edges.length + " == meta " + DATA.meta.fcSampleEdges + " == 期望 " + F.sampleEdgesExpected);
  // 4d) 面板流向轴区间与神经元云流向轴区间不相交（几何硬约束在渲染侧的独立复核）
  const axisIdx = { x: 0, y: 1, z: 2 }[F.flowAxis || "z"];
  const cloudLo = Math.min.apply(null, DATA.neurons.map((n) => [n.x, n.y, n.z][axisIdx]));
  const cloudHi = Math.max.apply(null, DATA.neurons.map((n) => [n.x, n.y, n.z][axisIdx]));
  let disjoint = true;
  let panelDetail = [];
  (F.panels || []).forEach((p) => {
    const us = p.units || [];
    const lo = Math.min.apply(null, us.map((u) => [u.x, u.y, u.z][axisIdx]));
    const hi = Math.max.apply(null, us.map((u) => [u.x, u.y, u.z][axisIdx]));
    if (!(hi < cloudLo || lo > cloudHi)) { disjoint = false; }
    panelDetail.push(p.name + "[" + lo.toFixed(4) + "," + hi.toFixed(4) + "]");
  });
  check("面板流向轴区间与神经元云区间不相交", disjoint,
    "云[" + cloudLo.toFixed(4) + "," + cloudHi.toFixed(4) + "] vs " + panelDetail.join(" "));
  // 4e) 声明文本必须出现（「非全部连接」不得缺失）
  check("meta 声明含『非全部连接』",
    typeof DATA.meta.fcDeclaration === "string" && DATA.meta.fcDeclaration.indexOf("非全部连接") >= 0,
    DATA.meta.fcDeclaration);
  // FC 统计面板必须是**独立面板**：共享的 #stats 会被基础渲染器每次重绘整体重写，
  // 因此 FC 小节若追加进 #stats 会被冲掉（这正是本断言要防的回归）。
  const fcStats = el("fc-stats");
  check("FC 统计面板独立存在且含抽样声明与矩阵形状",
    Boolean(fcStats) &&
    fcStats.textContent.indexOf("抽样连线") >= 0 &&
    fcStats.textContent.indexOf("非全部连接") >= 0 &&
    fcStats.textContent.indexOf(F.projWeightShape.join("×")) >= 0 &&
    fcStats.textContent.indexOf(F.fcOutWeightShape.join("×")) >= 0,
    fcStats ? fcStats.textContent.replace(/\n/g, " | ") : "fc-stats 未创建");
  // 4f) 叠加渲染器真的画了东西：面板方块 + 抽样虚线都要有 moveTo/lineTo 调用
  check("FC 叠加层实际产生绘制调用（方块 + 采样线）",
    stats.moveTo >= (F.edges || []).length + unitCount && stats.lineTo >= (F.edges || []).length,
    "moveTo=" + stats.moveTo + " lineTo=" + stats.lineTo +
    "（期望 >= 抽样 " + F.edges.length + " + 单元 " + unitCount + "）");
  check("FC 面板/边界块标签已写出",
    texts.some((t) => t[0].indexOf("输入侧 H=") === 0) &&
    texts.some((t) => t[0].indexOf("输出侧 H=") === 0),
    texts.map((t) => t[0]).filter((t) => t.indexOf("H=") >= 0).join(" | "));
} else {
  check("无 FC 负载时叠加渲染器不做任何事（cb-fc 未注入）", !els["cb-fc"],
    "viewer_fc.js 在无 fc 段时立即 return");
}

const failed = checks.filter((c) => !c.ok).length;
console.log("汇总：通过 " + (checks.length - failed) + " / 失败 " + failed + "（共 " + checks.length + "）");
process.exit(failed === 0 ? 0 : 1);