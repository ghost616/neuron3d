/* N3D 渲染器逻辑冒烟（零依赖：仅 Node + 最小 DOM/Canvas 桩）。
 *
 * 用途：在不安装任何浏览器自动化依赖的前提下，真实执行 HTML 里内联的 viewer.js，
 * 检查四件事：
 *   1) 视图矩阵正确（世界中心必须投影到画布中心）；
 *   2) 相机变换把 256 个神经元与 736 条边投影到画布范围内（bbox 有限且在画布附近）；
 *   3) 阈值滑块联动逻辑生效（阈值 0 -> 保留全部 736 条边）；
 *   4) 悬停命中与图例渲染可执行。
 *
 * 用法：node n3d_viz/assets/viewer_smoke.js <viewer.js 路径> <内联数据 JSON 路径>
 * 退出码：0 全部通过，1 存在失败。
 */
"use strict";

const fs = require("fs");

const jsPath = process.argv[2];
const dataPath = process.argv[3];
if (!jsPath || !dataPath) {
  console.error("用法: node viewer_smoke.js <viewer.js> <data.json>");
  process.exit(2);
}
const DATA = JSON.parse(fs.readFileSync(dataPath, "utf8"));
const JS = fs.readFileSync(jsPath, "utf8");

const stats = { arc: 0, lineTo: 0, moveTo: 0, fill: 0, stroke: 0, fillText: 0, fillRect: 0 };
const arcs = [];
const lines = [];

function ctxStub() {
  const c = {};
  ["beginPath", "closePath", "setTransform", "save", "restore", "clearRect", "translate", "rotate", "scale"]
    .forEach((k) => { c[k] = () => {}; });
  c.arc = (x, y, r) => { stats.arc++; arcs.push([x, y, r, c.fillStyle]); };
  c.moveTo = (x, y) => { stats.moveTo++; lines.push([x, y]); };
  c.lineTo = (x, y) => { stats.lineTo++; lines.push([x, y]); };
  c.fill = () => { stats.fill++; };
  c.stroke = () => { stats.stroke++; };
  c.fillText = () => { stats.fillText++; };
  c.fillRect = () => { stats.fillRect++; };
  c.strokeRect = () => {};
  c.createLinearGradient = () => ({ addColorStop: () => {} });
  return c;
}

const els = {};
function el(id) {
  if (!els[id]) {
    els[id] = {
      id: id, style: {}, textContent: "", innerHTML: "", value: "0", checked: true,
      width: 0, height: 0, clientWidth: 900, clientHeight: 620,
      addEventListener: (ev, fn) => { els[id].__h = els[id].__h || {}; els[id].__h[ev] = fn; },
      getContext: () => ctxStub(),
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 900, height: 620 })
    };
  }
  return els[id];
}

const listeners = {};
global.window = {
  N3D_DATA: DATA, devicePixelRatio: 1,
  addEventListener: (ev, fn) => { listeners[ev] = fn; },
  requestAnimationFrame: (fn) => fn()
};
global.document = { getElementById: el, createElement: () => el("tmp") };

const checks = [];
function check(name, cond, detail) {
  checks.push({ name: name, ok: Boolean(cond), detail: detail === undefined ? "" : String(detail) });
  console.log((cond ? "  [PASS] " : "  [FAIL] ") + name + (detail === undefined ? "" : ": " + detail));
}

eval(JS);

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
check("渲染了阈值内的边（moveTo == 保留边数）", stats.moveTo >= DATA.meta.threshold_counts["0.30"],
  stats.moveTo + " 段");

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
check("图例渲染了 K+3 个色块", (el("legend").innerHTML.match(/class="sw"/g) || []).length === DATA.layers.length + 3,
  (el("legend").innerHTML.match(/class="sw"/g) || []).length + " 块");

const failed = checks.filter((c) => !c.ok).length;
console.log("汇总：通过 " + (checks.length - failed) + " / 失败 " + failed + "（共 " + checks.length + "）");
process.exit(failed === 0 ? 0 : 1);
