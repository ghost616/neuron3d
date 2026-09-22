"""N3D 一期原型训练入口。

两种运行方式
------------
* 阶段 A（冒烟测试，默认路径）：``python n3d_proto/train.py --smoke-test``
  使用 `SMALL_CONFIG`（N=64, y_in=y_out=4, T=2, batch=32），仅跑 1 个 batch 的
  前向 + 反向，逐项核对验收标准：
  前向无 shape mismatch / 反向无错误 / 全部可学习参数梯度范数 > 0 /
  loss 非 NaN/Inf / 连接稀疏度（密度 E/(N*y_out*N*y_in)）< 0.1 / tau > 0，
  并附加校验「边级参数数 == E」与「不存在 [N*y_out, N*y_in] 形状的权重张量」。
  该命令退出码为 0 表示阶段 A 通过（无需任何额外开关）。
* 阶段 B（正式训练）：``python n3d_proto/train.py``
  使用 `DEFAULT_CONFIG`（N=256, y_in=y_out=8, T=3, batch=64），训练 `epochs` 轮，
  逐 epoch 打印 loss 与 test_acc，结束后保存 checkpoints/model.pt。

命令行参数
----------
--smoke-test        仅执行阶段 A（冒烟测试）
--epochs N          覆盖正式训练的 epoch 数
--device DEV        覆盖设备（cpu / cuda / auto）
--max-batches N     正式训练每轮最多处理多少个 batch（0 表示不限制，便于 CPU 快速验证）
--checkpoint PATH   checkpoint 保存路径（默认 checkpoints/model.pt）
--criterion-conflict
                    兼容别名，当前无需使用（连接稀疏度已回归规格口径）

产物保护
--------
冒烟测试（`--smoke-test`）与限批验证跑（`--max-batches > 0`）**不会覆盖正式
checkpoint**，其产物写入 `checkpoints/_verify/` 并在日志中明确说明写入位置。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 兼容"以脚本方式运行"（python n3d_proto/train.py）与"作为包导入"两种情形
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .config import Config, DEFAULT_CONFIG, SMALL_CONFIG
    from .data import get_mnist_loaders
    from .model import ThreeDNeuronSpace
    from .utils import (
        count_parameters,
        get_device,
        log_error,
        log_info,
        set_seed,
        tensor_grad_norms,
    )
except ImportError:  # pragma: no cover
    from config import Config, DEFAULT_CONFIG, SMALL_CONFIG  # type: ignore
    from data import get_mnist_loaders  # type: ignore
    from model import ThreeDNeuronSpace  # type: ignore
    from utils import (  # type: ignore
        count_parameters,
        get_device,
        log_error,
        log_info,
        set_seed,
        tensor_grad_norms,
    )

# 阶段 A 验收阈值
SPARSITY_MAX: float = 0.1      # 连接稀疏度（规格口径 = 密度 E/(N*y_out*N*y_in)）必须 < 0.1
GRAD_NORM_MIN: float = 0.0     # 所有可学习参数梯度范数必须 > 0
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CHECKPOINT_DIR: str = os.path.join(PROJECT_ROOT, "checkpoints")
CHECKPOINT_PATH: str = os.path.join(CHECKPOINT_DIR, "model.pt")
# 验证类运行（冒烟测试 / 限批短跑）的独立写入目录，避免覆盖正式产物
VERIFY_CHECKPOINT_DIR: str = os.path.join(CHECKPOINT_DIR, "_verify")
VERIFY_CHECKPOINT_PATH: str = os.path.join(VERIFY_CHECKPOINT_DIR, "smoke.pt")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数。

    参数
    ----
    argv : Optional[List[str]]
        参数列表（None 表示使用 sys.argv）。

    返回
    ----
    argparse.Namespace
        含 smoke_test / epochs / device / max_batches / checkpoint /
        criterion_conflict 六个字段。
    """
    parser = argparse.ArgumentParser(
        description="N3D 一期原型（三维神经元空间架构）训练入口"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="仅执行阶段 A 冒烟测试（SMALL_CONFIG，1 个 batch 的前向+反向验收）",
    )
    parser.add_argument(
        "--epochs", type=int, default=0, help="覆盖正式训练的 epoch 数（默认取 Config.epochs）"
    )
    parser.add_argument(
        "--device", type=str, default="", help="覆盖设备：cpu / cuda / auto（默认取 Config.device）"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="正式训练每轮最多处理的 batch 数（0 = 不限制，便于 CPU 上快速验证）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help=(
            "checkpoint 保存路径（默认 checkpoints/model.pt）。"
            "冒烟测试与 --max-batches 限批模式不会写入该路径，"
            "而是写入 checkpoints/_verify/ 下，避免覆盖正式产物。"
        ),
    )
    parser.add_argument(
        "--backup",
        dest="backup",
        action="store_true",
        default=True,
        help="覆盖已有 checkpoint 前先备份为 <path>.bak（默认开启）",
    )
    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="关闭覆盖前备份（不生成 .bak）",
    )
    parser.add_argument(
        "--criterion-conflict",
        action="store_true",
        help=(
            "**兼容别名，默认无需使用，且仅对 --smoke-test 生效**。历史上的验收条款曾按"
            "『零元素占比』口径解释『连接稀疏度 < 0.1』，与本设计冲突；第 2 轮已回归规格"
            "原义（连接稀疏度 = 密度 E/(N*y_out*N*y_in) < 0.1）。保留该开关仅为兼容旧命令行，"
            "传入后不改变任何判定结果；在非 --smoke-test 的正式训练路径下被忽略。"
        ),
    )
    return parser.parse_args(argv)

def build_model_and_data(
    config: Config,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[ThreeDNeuronSpace, DataLoader, DataLoader]:
    """按配置构建模型与数据加载器（统一的装配入口，保证两阶段行为一致）。

    参数
    ----
    config : Config
        超参配置。
    device : torch.device
        目标设备。
    max_batches : int
        仅用于日志提示（0 表示使用全量数据）。

    返回
    ----
    Tuple[ThreeDNeuronSpace, DataLoader, DataLoader]
        (model, train_loader, test_loader)。

    实现说明（设备探针的代价）
    --------------------------
    装配末尾会用 `next(iter(train_loader))` 取一个 batch 作为"输入与模型设备一致性"的
    探针。由于训练集 DataLoader 开启了 `shuffle=True`，这一次取数会**从打乱流中先取走
    一个 batch**（DEFAULT 配置下 batch_size=64，约占 60000 样本的 **0.11%**）。
    因此单个 epoch 实际参与训练的 batch 数为 **937 而非 938**（`len(train_loader)`）。
    这是有意的取舍：用 0.11% 的样本换"设备错误在命令行入口即报出"。
    **本说明仅为消除数目歧义，取数逻辑本身不做任何改动。**

    异常
    ------
    RuntimeError
        模型参数/buffer 与目标设备不一致，或首个 batch 的输入与模型设备不一致时抛出
        （把设备问题暴露在命令行入口附近，而不是等到反向传播深处）。
    """
    set_seed(config.seed)
    model = ThreeDNeuronSpace(config).to(device)
    train_loader, test_loader = get_mnist_loaders(
        batch_size=config.batch_size,
        data_root=config.data_root,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    # ---- 设备一致性断言（装配后立即校验，尽早暴露设备错误） ----
    for name, p in model.named_parameters():
        if p.device != device:
            raise RuntimeError(
                f"设备不一致：参数 {name} 在 {p.device}，期望 {device}"
            )
    for name, b in model.named_buffers():
        if b.device != device:
            raise RuntimeError(
                f"设备不一致：buffer {name} 在 {b.device}，期望 {device}"
            )
    probe_x, probe_y = next(iter(train_loader))
    if probe_x.device != device or probe_y.device != device:
        raise RuntimeError(
            f"设备不一致：首个 batch 数据在 {probe_x.device}/{probe_y.device}，期望 {device}"
        )
    log_info(
        f"设备一致性探针已通过（已校验 参数/buffer/首个 batch 均在 {device}）；"
        f"注意：探针从 shuffle 流中取走了 1 个 batch，故每 epoch 实际训练 "
        f"{len(train_loader) - 1} 个 batch（len(train_loader)={len(train_loader)}）"
    )

    stats = model.get_connection_stats()
    log_info(
        f"模型装配完成：N={config.N}, y_in={config.y_in}, y_out={config.y_out}, T={config.T}, "
        f"E={int(stats['num_edges'])}, 连接密度(sparsity)={stats['sparsity']:.6f}, "
        f"零元素占比(zero_ratio)={stats['zero_ratio']:.6f}, "
        f"avg_out_degree={stats['avg_out_degree']:.2f}, tau={stats['tau']:.6f}, "
        f"可学习参数={model.count_parameters()}, 设备={device}"
    )
    if max_batches > 0:
        log_info(f"本轮每 epoch 仅处理前 {max_batches} 个 batch（--max-batches 限制）")
    return model, train_loader, test_loader


def evaluate(
    model: ThreeDNeuronSpace,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> float:
    """在给定数据集上计算分类准确率。

    参数
    ----
    model : ThreeDNeuronSpace
        待评估模型。
    loader : DataLoader
        测试集加载器。
    device : torch.device
        计算设备。
    max_batches : int
        最多评估的 batch 数（0 表示全量）。

    返回
    ----
    float
        准确率，取值 [0, 1]。
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    model.train()
    if total == 0:
        raise RuntimeError("评估数据集为空，无法计算准确率")
    return correct / total


def train_one_epoch(
    model: ThreeDNeuronSpace,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_batches: int = 0,
    collect_grads: bool = False,
) -> Tuple[float, Dict[str, float], int]:
    """训练一个 epoch。

    参数
    ----
    model : ThreeDNeuronSpace
        待训练模型。
    loader : DataLoader
        训练集加载器。
    optimizer : torch.optim.Optimizer
        优化器。
    device : torch.device
        计算设备。
    epoch : int
        当前 epoch 序号（用于日志，从 1 开始）。
    max_batches : int
        本 epoch 最多处理的 batch 数（0 表示全量）。
    collect_grads : bool
        是否在最后一个 batch 反向之后收集各参数梯度范数（阶段 A 验收用）。

    返回
    ----
    Tuple[float, Dict[str, float], int]
        (平均 loss, 梯度范数字典, 实际处理 batch 数)。未收集梯度时字典为空。

    异常
    ------
    RuntimeError
        loss 为 NaN/Inf 时抛出（附带 epoch/batch 上下文）。
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    grad_norms: Dict[str, float] = {}
    for i, (x, y) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)                      # 前向：输入编码 -> T 轮四步闭环 -> 输出汇总
        loss = F.cross_entropy(logits, y)       # 分类损失
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"loss 出现 NaN/Inf：epoch={epoch}, batch={i}, loss={loss.item()}"
            )
        loss.backward()                         # 反向：梯度沿四步闭环回传
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        if collect_grads and (max_batches > 0 and i == max_batches - 1):
            # 阶段 A：在最后一个已反向的 batch 上收集全部可学习参数的梯度范数
            grad_norms = tensor_grad_norms(model)
        if (i + 1) % 100 == 0:
            log_info(f"epoch {epoch} | batch {i + 1} | running_loss={total_loss / n_batches:.4f}")
    if n_batches == 0:
        raise RuntimeError("训练集为空：没有任何 batch 被处理")
    return total_loss / n_batches, grad_norms, n_batches

def run_smoke_test(
    device_override: str = "",
    criterion_conflict: bool = False,
) -> bool:
    """阶段 A 冒烟测试：验证前向/反向跑通并逐条核对验收标准。

    参数
    ----
    device_override : str
        设备覆盖（空串表示使用 SMALL_CONFIG.device）。
    criterion_conflict : bool
        **兼容别名，当前不再需要**。第 2 轮已把"连接稀疏度"回归规格原义
        （密度 E/(N*y_out*N*y_in) < 0.1），该项现已正常 PASS。传入该开关不改变任何
        判定结果，仅为兼容旧命令行而保留。

    返回
    ----
    bool
        通过返回 True；有失败项打印 [FAIL] 并返回 False。

    产物保护
    --------
    冒烟测试只跑 1 个 batch，其 checkpoint 写入 `checkpoints/_verify/smoke.pt`，
    **不会覆盖 `checkpoints/model.pt` 等正式产物**。
    """
    config: Config = SMALL_CONFIG
    if device_override:
        # 直接覆盖副本的 device 字段（避免 Config(**{...}) 与位置参数冲突）
        config = Config(**{**config.to_dict(), "device": device_override})
    # 设备解析：auto -> 有 CUDA 用 CUDA，否则 CPU
    device = get_device(config.device)

    log_info("=" * 78)
    log_info("阶段 A 冒烟测试开始（SMALL_CONFIG）")
    log_info(f"配置：{config.describe()}")
    log_info(f"实际设备：{device}（torch_scatter 可用性见 utils.HAS_TORCH_SCATTER）")
    log_info("=" * 78)

    t0 = time.perf_counter()
    model, train_loader, _ = build_model_and_data(config, device, max_batches=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # ---- 阶段 A 约束：仅跑 1 个 batch 的前向 + 反向 ----
    avg_loss, grad_norms, n_batches = train_one_epoch(
        model, train_loader, optimizer, device, epoch=1, max_batches=1, collect_grads=True
    )

    stats = model.get_connection_stats()
    param_total = model.count_parameters()
    elapsed = time.perf_counter() - t0

    # ---- 打印各可学习参数的梯度范数 ----
    log_info("-" * 78)
    log_info(f"阶段 A 结果：loss={avg_loss:.6f}，处理 batch 数={n_batches}，耗时={elapsed:.2f}s")
    log_info("各可学习参数梯度范数（L2）：")
    if not grad_norms:
        log_error("未能收集到任何梯度，反向传播可能未执行")
    for name, gnorm in grad_norms.items():
        flag = "OK " if gnorm > GRAD_NORM_MIN else "BAD"
        log_info(f"  [{flag}] {name:<45s} grad_norm={gnorm:.6e}")
    log_info(f"可学习参数总数：{param_total}")
    log_info(
        f"连接统计：E={int(stats['num_edges'])}, "
        f"连接密度(sparsity)={stats['sparsity']:.6f}, "
        f"零元素占比(zero_ratio)={stats['zero_ratio']:.6f}, "
        f"avg_out_degree={stats['avg_out_degree']:.3f}, tau={stats['tau']:.6f}"
    )
    # ---- dense 对照诊断：证明"未 materialize dense [N*y_out, N*y_in] 权重矩阵" ----
    total_pairs = model.n_out_syn * model.n_in_syn
    conn_param = int(model.W_conn_sparse.numel())
    dense_weight_tensors = model.count_dense_weight_tensors()
    log_info(
        f"稀疏性实测：E={int(stats['num_edges'])} / 可能连接对 {total_pairs} "
        f"(连接密度 {100.0 * stats['sparsity']:.4f}%，零元素占比 "
        f"{100.0 * stats['zero_ratio']:.4f}%)，"
        f"稠密矩阵元素数={total_pairs}，边级参数数={conn_param}"
    )
    log_info(
        f"  -> 若把连接权重 materialize 成稠密矩阵，参数将从 {param_total} 膨胀到 "
        f"{param_total - conn_param + total_pairs}（{total_pairs / max(conn_param, 1):.0f}x）"
    )
    log_info("=" * 78)

    # ---- 逐条核对阶段 A 验收标准 ----
    checks: List[Tuple[str, bool, str]] = []
    checks.append(("前向无 shape mismatch", True, "所有 batch 前向成功返回 [B, output_dim]"))
    checks.append(("反向无错误", len(grad_norms) > 0, f"收集到 {len(grad_norms)} 个参数的梯度"))
    zero_grads = [n for n, v in grad_norms.items() if not (v > GRAD_NORM_MIN)]
    checks.append(
        (
            "所有可学习参数梯度范数 > 0",
            len(zero_grads) == 0,
            "全部 > 0" if not zero_grads else f"以下参数梯度范数为 0 或缺失：{zero_grads}",
        )
    )
    finite_loss = bool(avg_loss == avg_loss) and abs(avg_loss) != float("inf")
    checks.append(("loss 非 NaN/Inf", finite_loss, f"loss={avg_loss:.6f}"))
    # 规格口径：连接稀疏度 = E/(N*y_out*N*y_in)（密度定义），要求 < 0.1
    checks.append(
        (
            "连接稀疏度（密度 E/(N*y_out*N*y_in)）< 0.1",
            stats["sparsity"] < SPARSITY_MAX,
            f"sparsity={stats['sparsity']:.6f}（阈值 < {SPARSITY_MAX}）",
        )
    )
    checks.append(("tau > 0", stats["tau"] > 0.0, f"tau={stats['tau']:.6f}"))
    checks.append(("CPU 单 batch 耗时 < 120s", elapsed < 120.0, f"实际 {elapsed:.2f}s"))
    # ---- 附加断言：语义等价的"未 materialize dense"判据（皋陶建议） ----
    checks.append(
        (
            "附加①：边级参数数 == E",
            conn_param == int(stats["num_edges"]),
            f"W_conn_sparse.numel()={conn_param}，E={int(stats['num_edges'])}",
        )
    )
    checks.append(
        (
            "附加②：不存在 [N*y_out, N*y_in] 形状的权重张量",
            dense_weight_tensors == 0,
            f"命中 {dense_weight_tensors} 个（应为 0；dist/mask 属拓扑 buffer，不计入）",
        )
    )

    log_info("阶段 A 验收标准逐条核对：")
    all_ok = True
    for name, ok, detail in checks:
        log_info(f"  [{'PASS' if ok else 'FAIL'}] {name} —— {detail}")
        all_ok = all_ok and ok

    # ---- 冒烟测试产物写入独立目录，避免覆盖正式 checkpoint ----
    os.makedirs(VERIFY_CHECKPOINT_DIR, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "connection_stats": stats,
            "stage": "smoke",
            "loss": float(avg_loss),
            "grad_norms": grad_norms,
        },
        VERIFY_CHECKPOINT_PATH,
    )
    log_info(
        f"[产物保护] 冒烟测试 checkpoint 写入独立路径（不覆盖正式产物）：{VERIFY_CHECKPOINT_PATH}"
    )
    log_info(
        f"[产物保护] 正式 checkpoint 默认路径未被写入：{CHECKPOINT_PATH}"
        f"（该路径仅在正式全量训练时写入）"
    )
    if criterion_conflict:
        log_info(
            "[NOTE] --criterion-conflict 为兼容别名：连接稀疏度已回归规格口径，"
            "该开关不再影响判定结果。"
        )
    log_info(f"阶段 A 冒烟测试结论：{'全部通过' if all_ok else '存在失败项'}")
    log_info("=" * 78)
    return all_ok


def resolve_checkpoint_path(checkpoint_override: str, max_batches: int) -> str:
    """决定本次训练实际写入的 checkpoint 路径（产物保护核心逻辑）。

    规则
    ----
    1. `max_batches > 0`（限批短跑 / 验证跑）-> 一律写入 `checkpoints/_verify/`，
       文件名沿用用户给定名字，未给定则按限批量生成 `verify_<max_batches>.pt`
       （与冒烟的 `smoke.pt` 区分，避免相互覆盖），**绝不覆盖正式产物**；
       发生重定位时会打印「重定位前路径 -> 重定位后路径」，避免用户误以为写到了
       自己用 `--checkpoint` 指定的子目录；
    2. `checkpoint_override` 非空 -> 使用用户显式指定的路径（相对路径按工程根目录解析）；
    3. 否则 -> 默认 `checkpoints/model.pt`。

    参数
    ----
    checkpoint_override : str
        用户通过 `--checkpoint` 指定的路径（可为空）。
    max_batches : int
        每个 epoch 的最大 batch 数（> 0 表示限批模式）。

    返回
    ----
    str
        实际写入的 checkpoint 绝对路径。
    """
    if max_batches > 0:
        # 限批模式：隔离到 _verify/；默认文件名按限批量区分，避免与冒烟产物互相覆盖
        name = (
            os.path.basename(checkpoint_override)
            if checkpoint_override
            else f"verify_{max_batches}.pt"
        )
        if not name.endswith(".pt"):
            name = f"{name}.pt"
        requested = checkpoint_override or CHECKPOINT_PATH
        relocated = os.path.join(VERIFY_CHECKPOINT_DIR, name)
        # 显式打印「重定位前 -> 重定位后」，避免用户误以为写到了自己指定的子目录
        log_info(
            f"[产物保护] 限批模式（--max-batches={max_batches}）触发 checkpoint 重定位："
            f"{requested} -> {relocated}"
        )
        return relocated
    if checkpoint_override:
        path = checkpoint_override
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        return os.path.abspath(path)
    return CHECKPOINT_PATH


def backup_existing_checkpoint(path: str, enabled: bool = True) -> Optional[str]:
    """在覆盖已有 checkpoint 之前生成 `.bak` 备份（默认开启）。

    `checkpoints/` 目录被 `.gitignore` 忽略，正式产物一旦被覆盖便无法按位恢复；
    本函数在写入前把既有文件复制为 `<path>.bak`，为意外覆盖提供一层兜底。

    参数
    ----
    path : str
        即将写入的 checkpoint 绝对路径。
    enabled : bool
        是否启用备份（对应 CLI 的 `--backup/--no-backup`）。默认 True。

    返回
    ----
    Optional[str]
        备份文件路径；未备份（禁用或目标不存在）时返回 None。
    """
    if not enabled or not os.path.isfile(path):
        return None
    backup_path = f"{path}.bak"
    shutil.copyfile(path, backup_path)
    return backup_path


def run_full_training(
    device_override: str = "",
    epochs_override: int = 0,
    max_batches: int = 0,
    checkpoint_path: str = "",
    backup: bool = True,
) -> Dict[str, float]:
    """阶段 B 正式训练：使用 DEFAULT_CONFIG 训练若干 epoch 并保存 checkpoint。

    参数
    ----
    device_override : str
        设备覆盖（空串表示使用 DEFAULT_CONFIG.device）。
    epochs_override : int
        覆盖 epoch 数（0 表示使用 DEFAULT_CONFIG.epochs）。
    max_batches : int
        每个 epoch 最多处理的 batch 数（0 表示全量）。> 0 时 checkpoint 会写入
        `checkpoints/_verify/`，避免覆盖正式产物。
    checkpoint_path : str
        checkpoint 保存路径（空串表示使用默认 `checkpoints/model.pt`）。
    backup : bool
        覆盖已有 checkpoint 前是否先备份为 `<path>.bak`。默认 True。

    返回
    ----
    Dict[str, float]
        最终统计：{"test_acc": ..., "loss": ..., "sparsity": ..., "zero_ratio": ...,
        "tau": ..., "num_edges": ..., "params": ...}。

    说明
    ----
    训练结束后保存 checkpoint（含 model_state_dict、config、连接统计、test_acc、
    epochs），并打印最终参数统计。写入路径由 `resolve_checkpoint_path` 决定。
    """
    config: Config = DEFAULT_CONFIG
    overrides = config.to_dict()
    if device_override:
        overrides["device"] = device_override
    if epochs_override > 0:
        overrides["epochs"] = epochs_override
    config = Config(**overrides)
    # 设备解析：auto -> 有 CUDA 用 CUDA，否则 CPU
    device = get_device(config.device)

    log_info("=" * 78)
    log_info("阶段 B 正式训练开始（DEFAULT_CONFIG）")
    log_info(f"配置：{config.describe()}")
    log_info(f"实际设备：{device}")
    log_info("=" * 78)

    t0 = time.perf_counter()
    model, train_loader, test_loader = build_model_and_data(config, device, max_batches)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    last_loss = float("nan")
    test_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        last_loss, _, n_batches = train_one_epoch(
            model, train_loader, optimizer, device, epoch, max_batches=max_batches
        )
        test_acc = evaluate(model, test_loader, device, max_batches=max_batches)
        log_info(
            f"[epoch {epoch}/{config.epochs}] loss={last_loss:.4f} "
            f"test_acc={test_acc * 100:.2f}% (batches={n_batches})"
        )

    # ---- 保存 checkpoint（限批模式自动隔离到 _verify/，保护正式产物） ----
    save_path = resolve_checkpoint_path(checkpoint_path, max_batches)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 覆盖前先备份既有产物（checkpoints/ 未被 git 跟踪，覆盖后无法按位恢复）
    backup_path = backup_existing_checkpoint(save_path, enabled=backup)
    if backup_path:
        log_info(f"[产物保护] 覆盖前已备份既有 checkpoint：{backup_path}")
    stats = model.get_connection_stats()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "connection_stats": stats,
            "test_acc": test_acc,
            "epochs": config.epochs,
            "batches_per_epoch": (max_batches if max_batches > 0 else None),
        },
        save_path,
    )
    elapsed = time.perf_counter() - t0
    log_info(f"checkpoint 已保存：{save_path}（总耗时 {elapsed:.1f}s）")
    if max_batches > 0:
        log_info(
            f"[产物保护] 本次为限批验证跑（--max-batches={max_batches}），"
            f"已写入 {VERIFY_CHECKPOINT_DIR}；正式产物默认路径 {CHECKPOINT_PATH} 未被写入"
        )

    # ---- 最终参数统计 ----
    log_info("-" * 78)
    log_info("最终参数统计：")
    log_info(f"  可学习参数总数       ：{model.count_parameters()}")
    log_info(f"  连接稀疏度(密度)     ：{stats['sparsity']:.6f}")
    log_info(f"  零元素占比(zero_ratio)：{stats['zero_ratio']:.6f}")
    log_info(f"  边数 E               ：{int(stats['num_edges'])}")
    log_info(f"  平均出度             ：{stats['avg_out_degree']:.3f}")
    log_info(f"  tau                  ：{stats['tau']:.6f}")
    log_info(f"  最终 test_acc        ：{test_acc * 100:.2f}%")
    log_info(f"  checkpoint           ：{save_path}")
    log_info("-" * 78)

    return {
        "test_acc": float(test_acc),
        "loss": float(last_loss),
        "sparsity": float(stats["sparsity"]),
        "zero_ratio": float(stats["zero_ratio"]),
        "tau": float(stats["tau"]),
        "num_edges": float(stats["num_edges"]),
        "params": float(model.count_parameters()),
    }


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口：--smoke-test 只跑阶段 A，否则跑阶段 B 正式训练。

    参数
    ----
    argv : Optional[List[str]]
        命令行参数（None 表示使用 sys.argv）。

    返回
    ----
    int
        进程退出码：0 表示成功，1 表示冒烟测试验收失败。
    """
    args = parse_args(argv)
    if args.smoke_test:
        ok = run_smoke_test(
            device_override=args.device,
            criterion_conflict=args.criterion_conflict,
        )
        return 0 if ok else 1
    run_full_training(
        device_override=args.device,
        epochs_override=args.epochs,
        max_batches=args.max_batches,
        checkpoint_path=args.checkpoint,
        backup=args.backup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())