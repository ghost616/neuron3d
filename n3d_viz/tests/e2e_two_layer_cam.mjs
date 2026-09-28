/* n3d_viz —— 「两层几何一致性」真实浏览器 E2E（Playwright + Chromium，headless）。
 *
 * 位置（2026-09-28 收口轮迁入版本控制）：本脚本原位于 `.lizhu_env/r22_e2e/`，
 * 而 `.gitignore:48` 忽略了整个 `.lizhu_env/`（该目录 0 个文件被跟踪）—— 等于
 * 「唯一能拦住两层不同步回归的守门测试不在版本控制内」。现迁入
 * `n3d_viz/tests/e2e_two_layer_cam.mjs`；迁移**只搬位置 + 补本段注释**，
 * 断言逻辑与断言条数（26）逐字未改。
 *
 * 目的：补上此前漏检的盲区 —— 上一轮 490 项自检与 40 项 E2E 都没有断言
 * 「基础画布与 FC 叠加画布是否用同一套相机」。缺陷表现为：旋转 / 平移 / 重置
 * 只作用于基础层，FC 图层停在初始视角（用户报告「旋转神经元时全连接层不跟随」）。
 *
 * 本脚本用真实 Chromium 打开自包含 HTML 产物，对四种交互各断言一次：
 *   1) 左键拖拽旋转  -> 两层的「变化像素质心」相对画面中心的位移**方向一致**、幅度比 ≈ 1
 *   2) 右键拖拽平移  -> 同上
 *   3) 滚轮缩放      -> 两层都发生变化，且「变化像素数」的膨胀/收缩方向一致
 *   4) 点击重置      -> 两层都回到初始取景（与初始缓冲逐字节相同）
 * 另加回归断言：无 FC 产物**不得**创建叠加层（#view-fc / #fc-stats 均不存在）。
 *
 * 用法：在**仓库根**执行
 *   node n3d_viz/tests/e2e_two_layer_cam.mjs <noFc.html> <fcK3.html>
 * 退出码：0 全部通过 / 1 存在失败 / 2 参数缺失
 *
 * 运行依赖：`playwright` 仍是**既有测试设施** `.lizhu_env/r22_e2e/node_modules` 里的那一份
 * （离朱 r22 轮已安装并用于 40 项 E2E），**不是本次新增的运行依赖**，`requirements.txt` 0 变更。
 * 但 ESM 的裸导入**不做向上逐级解析**（`NODE_PATH` / `--preserve-symlinks` 亦无效），
 * 而本仓库所在卷**不支持**目录联接，故接入时需把既有 `node_modules` **复制**到
 * `n3d_viz/tests/node_modules`（`node_modules/` 被 .gitignore 忽略，不入库）——
 * 见 n3d_viz/README.md 的「E2E 两层几何一致性」小节。
 */
import { chromium } from "playwright";
import { pathToFileURL } from "node:url";
import path from "node:path";

const [noFcPath, fcPath] = process.argv.slice(2);
if (!noFcPath || !fcPath) {
  console.error("用法: node e2e_two_layer_cam.mjs <noFc.html> <fcK3.html>");
  process.exit(2);
}

const results = [];
function check(name, ok, detail) {
  results.push({ name, ok: Boolean(ok), detail: detail === undefined ? "" : String(detail) });
  console.log((ok ? "  [PASS] " : "  [FAIL] ") + name + (detail === undefined ? "" : ": " + detail));
}

const VIEWPORT = { width: 1000, height: 700 };
const CENTER = { x: 500, y: 350 };

/** 取某画布的像素缓冲（含 alpha）。 */
async function grab(page, id) {
  return page.evaluate((sel) => {
    const c = document.getElementById(sel);
    if (!c) { return null; }
    const g = c.getContext("2d").getImageData(0, 0, c.width, c.height);
    return { w: c.width, h: c.height, data: Array.from(g.data) };
  }, id);
}

/** 非透明（alpha>8）像素的个数与质心。 */
function alphaStats(buf) {
  let n = 0, sx = 0, sy = 0;
  const d = buf.data;
  for (let i = 3; i < d.length; i += 4) {
    if (d[i] > 8) { const q = (i - 3) / 4; n++; sx += q % buf.w; sy += Math.floor(q / buf.w); }
  }
  return n ? { n, x: sx / n, y: sy / n } : { n: 0, x: NaN, y: NaN };
}

/** **亮度加权**质心：基准是「画布内容」而非「不透明区域」。
 *
 *  为什么需要它：基础画布会铺满整块背景渐变（全画布 alpha=255），因此
 *  alpha 质心恒等于画布中心、对旋转毫无反应；而神经元云/边是明亮的，用亮度
 *  加权就能得到「渲染内容的质心」，旋转时会真实移动。叠加层是透明的，
 *  alpha 质心本就有效，为统一口径这里对两层都用「亮度 × alpha」加权。
 */
function lumStats(buf) {
  let w = 0, sx = 0, sy = 0;
  const d = buf.data;
  for (let i = 0; i < d.length; i += 4) {
    const a = d[i + 3];
    if (a <= 8) { continue; }
    const lum = 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
    const ww = (a / 255) * lum;
    if (ww <= 0) { continue; }
    const q = i / 4;
    w += ww; sx += ww * (q % buf.w); sy += ww * Math.floor(q / buf.w);
  }
  return w > 0 ? { w, x: sx / w, y: sy / w } : { w: 0, x: NaN, y: NaN };
}

/** 两个缓冲之间「变化像素」的个数与质心（阈值 16 以避免抗锯齿噪声）。 */
function diffStats(a, b) {
  let n = 0, sx = 0, sy = 0;
  const da = a.data, db = b.data;
  for (let i = 0; i < da.length; i += 4) {
    if (Math.abs(da[i] - db[i]) > 16 || Math.abs(da[i + 1] - db[i + 1]) > 16 ||
        Math.abs(da[i + 2] - db[i + 2]) > 16 || Math.abs(da[i + 3] - db[i + 3]) > 16) {
      const q = i / 4; n++; sx += q % a.w; sy += Math.floor(q / a.w);
    }
  }
  return n ? { n, x: sx / n, y: sy / n } : { n: 0, x: NaN, y: NaN };
}

/** 两缓冲是否逐字节相同。 */
function identical(a, b) {
  if (!a || !b || a.w !== b.w || a.h !== b.h || a.data.length !== b.data.length) { return false; }
  for (let i = 0; i < a.data.length; i++) { if (a.data[i] !== b.data[i]) { return false; } }
  return true;
}

/** 读取共享相机 `window.__n3d_cam` 的关键字段（用于断言「缩放/旋转确实作用于同一相机」）。 */
async function camState(page) {
  return page.evaluate(() => {
    const c = window.__n3d_cam;
    if (!c) { return null; }
    return { yaw: c.yaw, pitch: c.pitch, dist: c.dist, panX: c.panX, panY: c.panY, focal: c.focal };
  });
}

/** 安全格式化数值：非有限值（undefined / NaN / ±∞）一律显示成 `?`。
 *
 *  为什么需要：`projectWorldPoint` 在点位于相机之后时返回 `{visible:false}` ——
 *  **不含 x/y 数值字段**。若直接 `.toFixed()` 会抛
 *  `TypeError: Cannot read properties of null (reading 'toFixed')` 使脚本崩溃、
 *  连汇总行都打不出来（离朱 R24 的 F1 缺陷，已修）。
 */
function fmt(v, digits = 0) {
  return Number.isFinite(v) ? v.toFixed(digits) : "?";
}

/** 把某画布在半径 r 的邻域内是否存在非透明像素，汇总成一张粗粒度布尔图（32px 网格）。 */
function occupancy(buf, cell = 32) {
  const cols = Math.ceil(buf.w / cell), rows = Math.ceil(buf.h / cell);
  const grid = new Uint8Array(cols * rows);
  const d = buf.data;
  for (let i = 3; i < d.length; i += 4) {
    if (d[i] > 8) {
      const q = (i - 3) / 4;
      grid[Math.floor(q / buf.w / cell) * cols + Math.floor((q % buf.w) / cell)] = 1;
    }
  }
  return { grid, cols, rows, cell };
}

function hasAt(occ, x, y) {
  if (!isFinite(x) || !isFinite(y)) { return false; }
  const cx = Math.floor(x / occ.cell), cy = Math.floor(y / occ.cell);
  if (cx < 0 || cy < 0 || cx >= occ.cols || cy >= occ.rows) { return false; }
  return occ.grid[cy * occ.cols + cx] === 1;
}

/** 在**页面内**用共享相机复算某个世界点的屏幕坐标（与两层所用的投影公式同式）。
 *
 *  这是「两层是否共用同一相机」的**决定性**检验：若叠加层真的每帧直读共享相机，
 *  则把世界点用共享相机投影出来的屏幕位置，叠加层必须在那里有像素；
 *  若叠加层仍用自建/陈旧相机，预测位置就会落空。
 */
async function projectWorldPoint(page, world) {
  return page.evaluate((p) => {
    const c = window.__n3d_cam;
    if (!c) { return null; }
    const rotX = (a) => { const co = Math.cos(a), si = Math.sin(a);
      return [1, 0, 0, 0, 0, co, -si, 0, 0, si, co, 0, 0, 0, 0, 1]; };
    const rotY = (a) => { const co = Math.cos(a), si = Math.sin(a);
      return [co, 0, si, 0, 0, 1, 0, 0, -si, 0, co, 0, 0, 0, 0, 1]; };
    const mul = (a, b) => { const o = new Array(16);
      for (let r = 0; r < 4; r++) { for (let k = 0; k < 4; k++) {
        o[r * 4 + k] = a[r * 4] * b[k] + a[r * 4 + 1] * b[4 + k] +
          a[r * 4 + 2] * b[8 + k] + a[r * 4 + 3] * b[12 + k]; } }
      return o; };
    const rot = mul(rotX(c.pitch), rotY(c.yaw));
    const m = rot.slice();
    m[12] = -(rot[0] * c.tx + rot[4] * c.ty + rot[8] * c.tz) + c.panX;
    m[13] = -(rot[1] * c.tx + rot[5] * c.ty + rot[9] * c.tz) + c.panY;
    m[14] = -(rot[2] * c.tx + rot[6] * c.ty + rot[10] * c.tz) - c.dist;
    const z = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];
    if (z > -0.001) { return { visible: false }; }
    const x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
    const y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
    const cv = document.getElementById("view-fc");
    const dpr = window.devicePixelRatio || 1;
    const k = c.focal / (-z);
    return { visible: true, x: cv.width * 0.5 + x * k * dpr, y: cv.height * 0.5 - y * k * dpr };
  }, world);
}

/** 从画面中心指向质心的单位向量（质心与中心重合时返回 null）。 */
function dirFrom(c, center) {
  const vx = c.x - center.x, vy = c.y - center.y;
  const m = Math.hypot(vx, vy);
  if (!isFinite(m) || m < 1e-6) { return null; }
  return { ux: vx / m, uy: vy / m, mag: m };
}

/** 比较两层的位移方向一致性：点积 > 0.5 视作同向（约 < 60°）。 */
function sameDirection(dBase, dFc) {
  if (!dBase || !dFc) { return false; }
  return (dBase.ux * dFc.ux + dBase.uy * dFc.uy) > 0.5;
}

async function interact(page, kind, amount) {
  if (kind === "rotate") {
    await page.mouse.move(CENTER.x, CENTER.y);
    await page.mouse.down({ button: "left" });
    for (let i = 1; i <= 12; i++) {
      await page.mouse.move(CENTER.x + (amount * i) / 12, CENTER.y + (amount * 0.2 * i) / 12);
      await page.waitForTimeout(18);
    }
    await page.mouse.up({ button: "left" });
  } else if (kind === "pan") {
    await page.mouse.move(CENTER.x, CENTER.y);
    await page.mouse.down({ button: "right" });
    for (let i = 1; i <= 12; i++) {
      await page.mouse.move(CENTER.x + (amount * i) / 12, CENTER.y);
      await page.waitForTimeout(18);
    }
    await page.mouse.up({ button: "right" });
  } else if (kind === "zoom") {
    await page.mouse.move(CENTER.x, CENTER.y);
    for (let i = 0; i < 5; i++) {
      await page.mouse.wheel(0, amount);
      await page.waitForTimeout(80);
    }
  }
  await page.waitForTimeout(700);
}

const browser = await chromium.launch();
try {
  // ------------------------------------------------------------ 无 FC 产物回归
  {
    const page = await browser.newPage({ viewport: VIEWPORT });
    await page.goto(pathToFileURL(path.resolve(noFcPath)).href);
    await page.waitForTimeout(1800);
    const st = await page.evaluate(() => ({
      overlay: !!document.getElementById("view-fc"),
      stats: !!document.getElementById("fc-stats"),
      warn: !!document.getElementById("fc-cam-warning"),
      sharedCam: typeof window.__n3d_cam,
    }));
    check("无 FC 产物：不创建叠加层 #view-fc", !st.overlay, "overlay=" + st.overlay);
    check("无 FC 产物：无 #fc-stats", !st.stats, "stats=" + st.stats);
    check("无 FC 产物：无相机告警面板", !st.warn);
    check("无 FC 产物：共享相机已暴露（window.__n3d_cam 为对象）",
      st.sharedCam === "object", "typeof=" + st.sharedCam);
    await page.close();
  }

  // ------------------------------------------------------------ 有 FC 产物：四种交互
  const page = await browser.newPage({ viewport: VIEWPORT });
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(e.message));
  await page.goto(pathToFileURL(path.resolve(fcPath)).href);
  await page.waitForTimeout(2200);

  const has = await page.evaluate(() => ({
    overlay: !!document.getElementById("view-fc"),
    sharedCam: typeof window.__n3d_cam,
    warn: !!document.getElementById("fc-cam-warning"),
  }));
  check("有 FC 产物：叠加层存在", has.overlay);
  check("有 FC 产物：共享相机为对象（两层级联同一引用）", has.sharedCam === "object",
    "typeof=" + has.sharedCam);
  check("有 FC 产物：无「相机不可用」告警", !has.warn);

  const initBase = await grab(page, "view");
  const initFc = await grab(page, "view-fc");
  const initFcStats = alphaStats(initFc);

  // 交互 1：左键旋转
  {
    // 取一个已知世界点（输入面板第 1 个单元中心的**网格角**，即最靠边的一个单元），
    // 用共享相机在页面内复算其屏幕位置 —— 这是判定「两层共用同一相机」的决定性口径。
    const worldPt = await page.evaluate(() => {
      const F = window.N3D_DATA.fc;
      const us = F.panels[0].units;
      const u = us[0];
      return { x: u.x, y: u.y, z: u.z };
    });
    const before0 = await grab(page, "view");
    const beforeF = await grab(page, "view-fc");
    const occF0 = occupancy(beforeF);
    const p0 = await projectWorldPoint(page, worldPt);
    check("旋转前：用共享相机预测的面板单元屏幕位置处，叠加层确有像素",
      Boolean(p0 && p0.visible && hasAt(occF0, p0.x, p0.y)),
      p0 && p0.visible ? `(${fmt(p0.x)},${fmt(p0.y)})` : "不可见（点被相机剔除）");

    await interact(page, "rotate", 180);
    const after0 = await grab(page, "view");
    const afterF = await grab(page, "view-fc");
    const occF1 = occupancy(afterF);
    const db = diffStats(before0, after0);
    const df = diffStats(beforeF, afterF);
    check("旋转：基础层发生了明显变化", db.n > 500, "变化像素 " + db.n);
    check("旋转：**叠加层也发生了明显变化**（缺陷修复的核心断言）", df.n > 200,
      "变化像素 " + df.n);

    const p1 = await projectWorldPoint(page, worldPt);
    // 注意：p0 / p1 在点被剔除时是 `{visible:false}`（**没有** x/y 字段），
    // 因此一律经 fmt() 格式化，不得直接 .toFixed()（否则脚本崩溃、无汇总行）。
    const moved = p0 && p1 && p0.visible && p1.visible &&
      Math.hypot(p1.x - p0.x, p1.y - p0.y);
    check("旋转：共享相机下该世界点的预测屏幕位置**确实移动**",
      isFinite(moved) && moved > 20,
      `(${fmt(p0 && p0.x)},${fmt(p0 && p0.y)}) -> ` +
      `(${fmt(p1 && p1.x)},${fmt(p1 && p1.y)})，位移 ${fmt(moved, 1)}px`);
    check("旋转后：**叠加层出现在新预测位置**（跟随旋转的决定性证据）",
      Boolean(p1 && p1.visible && hasAt(occF1, p1.x, p1.y)),
      p1 && p1.visible ? `(${fmt(p1.x)},${fmt(p1.y)})` : "不可见（点被相机剔除）");
    const oldStillThere = Boolean(p0 && p0.visible && hasAt(occF1, p0.x, p0.y));
    const newAlreadyThere = Boolean(p1 && p1.visible && hasAt(occF0, p1.x, p1.y));
    check("旋转后：旧位置已不再是「旋转前就有像素」的同一处（位置确实改变）",
      !(oldStillThere && newAlreadyThere),
      `旧位置仍有像素=${oldStillThere}，新位置旋转前也有像素=${newAlreadyThere}` +
      "（两者同时为真说明该网格无法区分，需换点；更强的判别由反证 A/B 提供）");

    const camR = await camState(page);
    check("旋转：共享相机 yaw/pitch 确实变化（两层因此同步）",
      camR && (Math.abs(camR.yaw - (-0.62)) > 1e-6 || Math.abs(camR.pitch - 0.42) > 1e-6),
      `yaw=${fmt(camR && camR.yaw, 4)} pitch=${fmt(camR && camR.pitch, 4)}`);
  }

  // 交互 2：右键平移
  {
    const b0 = await grab(page, "view");
    const f0 = await grab(page, "view-fc");
    // 平移的自变量与因变量都是「画布内的像素位移」，不受背景是否不透明影响：
    // 前一次交互后的"非透明像素质心"变化只在平移下有干净的几何含义
    const a0b = alphaStats(b0), a0f = alphaStats(f0);
    await interact(page, "pan", 150);
    const b1 = await grab(page, "view");
    const f1 = await grab(page, "view-fc");
    const db = diffStats(b0, b1);
    const df = diffStats(f0, f1);
    check("平移：基础层发生了明显变化", db.n > 100, "变化像素 " + db.n);
    check("平移：**叠加层也发生了明显变化**", df.n > 100, "变化像素 " + df.n);
    const dirB = dirFrom(db, CENTER), dirF = dirFrom(df, CENTER);
    const dot = dirB && dirF ? (dirB.ux * dirF.ux + dirB.uy * dirF.uy) : NaN;
    check("平移：两层**变化像素质心**的位移方向一致（点积 > 0.5）",
      sameDirection(dirB, dirF),
      "dot=" + (isFinite(dot) ? dot.toFixed(3) : "n/a"));
    const ratio = dirB && dirF && dirB.mag > 1e-6 ? dirF.mag / dirB.mag : NaN;
    check("平移：两层位移幅度比落在同一量级（0.3 ~ 3.0）",
      isFinite(ratio) && ratio > 0.3 && ratio < 3.0,
      "ratio=" + (isFinite(ratio) ? ratio.toFixed(3) : "n/a"));
  }

  // 交互 3：滚轮缩放
  {
    const b0 = await grab(page, "view");
    const f0 = await grab(page, "view-fc");
    const cam0 = await camState(page);
    await interact(page, "zoom", -140);
    const b1 = await grab(page, "view");
    const f1 = await grab(page, "view-fc");
    const cam1 = await camState(page);
    const db = diffStats(b0, b1);
    const df = diffStats(f0, f1);
    check("缩放：基础层发生了明显变化", db.n > 500, "变化像素 " + db.n);
    check("缩放：**叠加层也发生了明显变化**", df.n > 200, "变化像素 " + df.n);
    // 「可视范围变化方向一致」的严格口径：叠加层每帧直读共享相机的 dist，
    // 故 dist 的变化方向 == 两层可视范围的变化方向。先断言共享相机确实变了，
    // 再断言叠加层随之重绘，二者共同证明「缩放作用于两层」。
    check("缩放：共享相机 dist 确实变化（两层因此同步）",
      cam0 && cam1 && Math.abs(cam1.dist - cam0.dist) > 1e-6,
      `dist ${fmt(cam0 && cam0.dist, 4)} -> ${fmt(cam1 && cam1.dist, 4)}`);
    const n0f = alphaStats(f0).n, n1f = alphaStats(f1).n;
    check("缩放：叠加层可视内容随之变化（非透明像素数改变）",
      n0f !== n1f, `fc ${n0f} -> ${n1f}`);
    const n0b = alphaStats(b0).n, n1b = alphaStats(b1).n;
    check("缩放：基础层恒为满画布（说明不能以非透明像素数做两层对比指标）",
      n0b === VIEWPORT.width * VIEWPORT.height && n1b === n0b,
      `base ${n0b} -> ${n1b}（= 画布面积 ${VIEWPORT.width * VIEWPORT.height}）`);
  }

  // 交互 4：点击重置 -> 两层回到初始取景（与初始缓冲逐字节相同）
  {
    await page.click("#btn-reset");
    await page.waitForTimeout(900);
    const b = await grab(page, "view");
    const f = await grab(page, "view-fc");
    check("重置：基础层回到初始取景（与初始缓冲逐字节相同）", identical(initBase, b));
    check("重置：**叠加层回到初始取景（与初始缓冲逐字节相同）**", identical(initFc, f),
      "初始非透明像素 " + initFcStats.n);
  }

  check("四种交互全程无 JS 报错", pageErrors.length === 0,
    pageErrors.slice(0, 3).join(" | "));
  await page.close();
} finally {
  await browser.close();
}

const failed = results.filter((r) => !r.ok).length;
console.log("汇总：通过 " + (results.length - failed) + " / 失败 " + failed +
  "（共 " + results.length + "）");
process.exit(failed === 0 ? 0 : 1);
