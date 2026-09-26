"""tkinter 图形界面（薄封装，全部渲染逻辑复用 core / export_geometry / render_html）。

设计要点：

* 长任务在**后台线程**执行，通过 ``queue.Queue`` + ``root.after()`` 把日志回传到
  主线程刷进日志区，因此生成期间窗口不卡死（Tk 只在主线程被调用）。
* 浏览按钮使用 ``filedialog.askopenfilename`` / ``askdirectory``。
* 一键打开产物：HTML 用 ``webbrowser`` 交给默认浏览器；输出目录用
  ``os.startfile``（Windows）/ ``xdg-open``（Linux）/ ``open``（macOS）。
* **不做拖拽**：拖拽需要第三方 ``tkinterdnd2``，与「零新依赖」冲突。
"""

from __future__ import annotations

import os
import pathlib
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable

# 允许脚本直跑：此时 __package__ 为空，相对导入会失败。
if __package__ in (None, ""):  # pragma: no cover - 仅脚本直跑时命中
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from n3d_viz import core
else:
    from . import core

_WINDOW_TITLE = "N3D 二期拓扑三维可视化"
_DEFAULT_CKPT = "checkpoints/n3d_sphere/model.pt"


def open_in_file_manager(path: str | Path) -> None:
    """用系统文件管理器打开目录（跨平台，零第三方依赖）。"""
    target = str(path)
    if sys.platform.startswith("win"):
        os.startfile(target)  # type: ignore[attr-defined]  # 仅 Windows 存在
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


class VizApp:
    """N3D 可视化生成器的 Tk 界面。

    用法::

        app = VizApp()
        app.run()          # 进入 mainloop
        app.build_once()   # 无界面地构造一次窗口（供冒烟测试）

    Attributes:
        root: ``tk.Tk`` 主窗口。
        log_queue: 后台线程 -> 主线程的日志队列。
        worker: 当前后台线程（空闲时为 None）。
        last_html: 最近一次成功生成的 HTML 路径。
        last_out_dir: 最近一次使用的输出目录。
    """

    def __init__(self, master: tk.Misc | None = None) -> None:
        self.root: tk.Tk = master if isinstance(master, tk.Tk) else tk.Tk(master)
        self.root.title(_WINDOW_TITLE)
        self.root.geometry("820x620")
        self.root.minsize(680, 520)

        self.log_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.last_html: Path | None = None
        self.last_out_dir: Path | None = None

        self.ckpt_var = tk.StringVar(value=_DEFAULT_CKPT)
        self.outdir_var = tk.StringVar(value=core.DEFAULT_OUT_DIR)
        self.status_var = tk.StringVar(value="就绪：选择 .pt 产物与输出文件夹后点击「开始生成」。")
        self.threshold_var = tk.DoubleVar(value=core.DEFAULT_THRESHOLD)
        self.planes_var = tk.BooleanVar(value=True)

        self._build_widgets()
        self.root.after(100, self._pump)

    # ------------------------------------------------------------- 界面构建
    def _build_widgets(self) -> None:
        """构建窗口控件（输入框 / 浏览按钮 / 开始按钮 / 日志区 / 快捷按钮）。"""
        pad = {"padx": 8, "pady": 6}
        root = self.root

        row1 = ttk.Frame(root)
        row1.pack(fill="x", **pad)
        ttk.Label(row1, text="checkpoint (.pt)：").pack(side="left")
        ttk.Entry(row1, textvariable=self.ckpt_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row1, text="浏览…", command=self.browse_ckpt).pack(side="left", padx=(6, 0))

        row2 = ttk.Frame(root)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="输出文件夹：").pack(side="left")
        ttk.Entry(row2, textvariable=self.outdir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row2, text="浏览…", command=self.browse_outdir).pack(side="left", padx=(6, 0))

        row3 = ttk.Frame(root)
        row3.pack(fill="x", **pad)
        ttk.Label(row3, text="边权重阈值 |w| >= ").pack(side="left")
        ttk.Scale(
            row3, from_=0.0, to=0.9, variable=self.threshold_var, orient="horizontal",
            command=lambda _v: self._sync_threshold_label(),
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.threshold_label = ttk.Label(row3, text=f"{core.DEFAULT_THRESHOLD:.2f}", width=5)
        self.threshold_label.pack(side="left")
        ttk.Checkbutton(row3, text="默认显示层平面", variable=self.planes_var).pack(side="left", padx=(10, 0))

        row4 = ttk.Frame(root)
        row4.pack(fill="x", **pad)
        self.start_btn = ttk.Button(row4, text="开始生成", command=self.start)
        self.start_btn.pack(side="left")
        self.open_html_btn = ttk.Button(row4, text="打开 HTML", command=self.open_html, state="disabled")
        self.open_html_btn.pack(side="left", padx=(6, 0))
        self.open_dir_btn = ttk.Button(row4, text="打开输出文件夹", command=self.open_out_dir, state="disabled")
        self.open_dir_btn.pack(side="left", padx=(6, 0))
        ttk.Button(row4, text="清空日志", command=self.clear_log).pack(side="left", padx=(6, 0))

        status = ttk.Label(root, textvariable=self.status_var, foreground="#1a4f9c", wraplength=780)
        status.pack(fill="x", padx=10, pady=(0, 4))

        self.log_box = scrolledtext.ScrolledText(root, height=22, wrap="word",
                                                 font=("Consolas", 10))
        self.log_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_box.configure(state="disabled")

        hint = ttk.Label(
            root,
            text="提示：不实现拖拽（拖拽需要第三方 tkinterdnd2，与零依赖约束冲突）。",
            foreground="#777777",
        )
        hint.pack(fill="x", padx=10, pady=(0, 8))

    def _sync_threshold_label(self) -> None:
        """阈值滑块联动显示（保留两位小数）。"""
        self.threshold_label.configure(text=f"{self.threshold_var.get():.2f}")

    # ------------------------------------------------------------- 浏览按钮
    def browse_ckpt(self) -> None:
        """选择 checkpoint 文件（``filedialog.askopenfilename``）。"""
        path = filedialog.askopenfilename(
            title="选择二期 checkpoint",
            filetypes=[("PyTorch 产物", "*.pt"), ("全部文件", "*.*")],
            initialdir=str(Path(self.ckpt_var.get()).parent),
        )
        if path:
            self.ckpt_var.set(path)

    def browse_outdir(self) -> None:
        """选择输出文件夹（``filedialog.askdirectory``）。"""
        path = filedialog.askdirectory(
            title="选择输出文件夹",
            initialdir=self.outdir_var.get(),
        )
        if path:
            self.outdir_var.set(path)

    # ------------------------------------------------------------- 日志管道
    def log(self, message: str) -> None:
        """线程安全地投递一条日志（可从后台线程调用）。"""
        self.log_queue.put(("log", message))

    def set_status(self, message: str) -> None:
        """线程安全地更新状态栏。"""
        self.log_queue.put(("status", message))

    def _pump(self) -> None:
        """主线程定时排空队列并刷新界面（``after`` 轮询，避免跨线程操作 Tk）。"""
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "done":
                    self._on_done(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._pump)

    def _append_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def clear_log(self) -> None:
        """清空日志区。"""
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------- 生成任务
    def start(self) -> None:
        """点击「开始生成」：校验输入后启动后台线程。"""
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo(_WINDOW_TITLE, "已有生成任务在运行，请等待其完成。")
            return
        ckpt = Path(self.ckpt_var.get().strip())
        out_dir = Path(self.outdir_var.get().strip() or core.DEFAULT_OUT_DIR)
        if not ckpt.exists():
            self._append_log(f"[错误] checkpoint 路径不存在：{ckpt}")
            self.status_var.set("失败：checkpoint 路径不存在。")
            return
        self.start_btn.configure(state="disabled")
        self.open_html_btn.configure(state="disabled")
        self.open_dir_btn.configure(state="disabled")
        self.set_status(f"生成中…（{ckpt.name}）")
        self.log(f"[开始] checkpoint = {ckpt}")
        self.log(f"[开始] 输出目录 = {out_dir}")
        self.worker = threading.Thread(
            target=self._run_job,
            args=(ckpt, out_dir, float(self.threshold_var.get()), bool(self.planes_var.get())),
            daemon=True,
        )
        self.worker.start()

    def _run_job(self, ckpt: Path, out_dir: Path, threshold: float, planes: bool) -> None:
        """后台线程体：加载 -> 抽取 -> 写三件套；结果通过队列回传主线程。"""
        try:
            data = core.load_topology(ckpt)
            self.log(
                f"[数据] N={data.n_neurons} E={data.n_edges} K={data.n_layers} "
                f"S_in={data.n_s_in} S_out={data.n_s_out} seed={data.config.get('seed')} "
                f"test_acc={data.test_acc}"
            )
            self.log(f"[数据] 层规模 = {'/'.join(str(c) for c in data.layer_counts)}")
            self.log(f"[数据] 阈值过滤统计 = {data.edge_threshold_stats((0.05, 0.10, 0.20, 0.30, 0.50))}")
            stats = data.edge_threshold_stats((threshold,))
            self.log(f"[数据] 当前阈值 {threshold:.2f} 保留 {list(stats.values())[0]} / {data.n_edges} 条边")
            reports = core.write_outputs(
                data, out_dir=out_dir, threshold=threshold,
                include_planes=planes, log=self.log,
            )
            self.log_queue.put(("done", {"ok": True, "reports": reports, "out_dir": str(out_dir)}))
        except core.CheckpointError as exc:
            self.log(f"[错误] {exc}")
            self.log_queue.put(("done", {"ok": False, "error": str(exc), "out_dir": str(out_dir)}))
        except Exception as exc:  # noqa: BLE001 - 兜底：任何异常都要回到主线程提示
            self.log(f"[错误] 未预期异常：{type(exc).__name__}: {exc}")
            self.log_queue.put(
                ("done", {"ok": False, "error": f"{type(exc).__name__}: {exc}", "out_dir": str(out_dir)})
            )

    def _on_done(self, payload: dict[str, Any]) -> None:
        """主线程收尾：恢复按钮、更新状态、启用「打开」按钮。"""
        self.start_btn.configure(state="normal")
        if payload.get("ok"):
            reports = payload["reports"]
            self.last_html = Path(reports["html"]["path"])
            self.last_out_dir = Path(payload["out_dir"])
            self.open_html_btn.configure(state="normal")
            self.open_dir_btn.configure(state="normal")
            # 同名覆盖必须同时进日志与状态栏（产物纪律：不得静默覆盖）。
            # 若只写日志，随后的状态栏设置会把覆盖信息冲掉。
            existed = [k for k, v in reports.items() if v.get("existed")]
            if existed:
                self.log(f"[提示] 以下同名产物此前已存在并被覆盖：{existed}")
            prefix = (
                f"完成（已覆盖同名产物：{', '.join(sorted(existed))}）："
                if existed else "完成："
            )
            self.status_var.set(
                f"{prefix}{self.last_html.name}（{reports['html']['bytes']} 字节）"
                f"，PLY {reports['ply']['vertices']} 项点，OBJ {reports['obj']['lines']} 条 l 行。"
            )
            self.log("[完成] 三件套已写出。")
        else:
            self.status_var.set(f"失败：{payload.get('error', '未知错误')}")
            messagebox.showerror(_WINDOW_TITLE, str(payload.get("error", "生成失败")))

    # ------------------------------------------------------------- 快捷操作
    def open_html(self) -> None:
        """用系统默认浏览器打开最近生成的 HTML。"""
        if self.last_html and self.last_html.exists():
            webbrowser.open(self.last_html.resolve().as_uri())
            self._append_log(f"[打开] {self.last_html}")
        else:
            messagebox.showinfo(_WINDOW_TITLE, "还没有生成 HTML 产物。")

    def open_out_dir(self) -> None:
        """用系统文件管理器打开输出目录。"""
        if self.last_out_dir and self.last_out_dir.exists():
            open_in_file_manager(self.last_out_dir)
            self._append_log(f"[打开] {self.last_out_dir}")
        else:
            messagebox.showinfo(_WINDOW_TITLE, "还没有可打开的输出目录。")

    # ------------------------------------------------------------- 运行/冒烟
    def run(self) -> None:
        """进入 Tk 主循环。"""
        self.root.mainloop()

    def destroy(self) -> None:
        """销毁窗口（供冒烟测试使用）。"""
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def build_smoke_window() -> VizApp:
    """在 ``withdraw()`` 状态下构造一次窗口并返回（GUI 冒烟断言用）。

    不进入 ``mainloop``，只验证控件树可构造、变量可访问；调用方负责销毁。

    Returns:
        构造好的 :class:`VizApp`。
    """
    app = VizApp()
    app.root.withdraw()
    app.root.update_idletasks()
    return app


def main(argv: list[str] | None = None) -> int:
    """脚本直跑入口：``python n3d_viz/gui.py``。

    Returns:
        进程退出码（0 正常关闭）。
    """
    app = VizApp()
    app.run()
    return 0


if __name__ == "__main__":  # pragma: no cover - 脚本直跑分支
    raise SystemExit(main())
