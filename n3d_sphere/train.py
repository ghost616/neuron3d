"""N3D 二期架构变体（球形有向拓扑）训练入口。

本文件由一期 `n3d_proto/train.py` 拷贝而来，新增 `--topology / --flow-axis /
--space-radius` 三个 CLI、二期产物指纹维度与 `checkpoints/n3d_sphere/` 独立产物目录。
**训练循环、数据管线、优化器/调度器/梯度裁剪、判据与日志格式均与一期保持一致**，
且默认（cube）路径的数值结果与一期**逐位相同**（`loss=2.419689`）。

两种运行方式
------------
* 阶段 A（冒烟测试，默认路径）：``python n3d_sphere/train.py --smoke-test``
  使用 `SMALL_CONFIG`（N=64, y_in=y_out=4, T=2, batch=32），仅跑 1 个 batch 的
  前向 + 反向，逐项核对验收标准：
  前向无 shape mismatch / 反向无错误 / 全部可学习参数梯度范数 > 0 /
  loss 非 NaN/Inf / 连接稀疏度（密度 E/(N*y_out*N*y_in)）< 0.1 / tau > 0，
  并附加校验「边级参数数 == E」与「不存在 [N*y_out, N*y_in] 形状的权重张量」。
  该命令退出码为 0 表示阶段 A 通过（无需任何额外开关）。
* 阶段 B（正式训练）：``python n3d_sphere/train.py``
  使用 `DEFAULT_CONFIG`（N=256, y_in=y_out=8, T=3, batch=64），训练 `epochs` 轮，
  逐 epoch 打印 loss 与 test_acc，结束后保存到本模块独立产物目录。

命令行参数
----------
--smoke-test        仅执行阶段 A（冒烟测试）
--topology {cube,sphere}
                    拓扑几何：cube（默认，逐位兼容一期）/ sphere（二期球形有向拓扑）
--flow-axis {x,y,z}
                    全局流向轴（仅 sphere 生效）：输入突触取负半球、输出突触取正半球
--space-radius R    球形拓扑下神经元球的半径（0/缺省 = 与立方体等体积的默认半径）
--epochs N          覆盖正式训练的 epoch 数
--device DEV        覆盖设备（cpu / cuda / auto）
--max-batches N     正式训练每轮最多处理多少个 batch（0 表示不限制，便于 CPU 快速验证）
--checkpoint PATH   checkpoint 保存路径（缺省时按配置指纹自动命名，避免不同拓扑互覆）
--criterion-conflict
                    兼容别名，当前无需使用（连接稀疏度已回归规格口径）

产物保护
--------
* 本模块产物目录为 `checkpoints/n3d_sphere/`，与一期 `checkpoints/` **物理隔离**，
  一期既有产物 `n3d_model_full.pt` / `n3d_model_highacc.pt` / `n3d_model_capacity.pt`
  不会被本模块的任何命令写入；
* 冒烟测试的产物文件名**含拓扑指纹**（`smoke_<top>_<axis>.pt`），避免 sphere 冒烟
  覆盖 cube 冒烟产物；
* 限批验证跑（`--max-batches > 0`）的产物文件名**含 topology / flow_axis 等配置指纹**，
  不同拓扑/流向/容量配置互不覆盖（历史纠正：同名互覆导致筛选记录丢失）；
* 正式全量训练未显式指定 `--checkpoint` 时，同样按配置指纹自动命名
  （`full_<指纹>.pt`），保证不同拓扑的全量结果各自留痕。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 兼容"以脚本方式运行"（python n3d_sphere/train.py）与"作为包导入"两种情形
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .config import Config, DEFAULT_CONFIG, HIGHACC_CONFIG, SMALL_CONFIG
    from .data import get_mnist_loaders
    from .model import MLPBaseline, ThreeDNeuronSpace
    from .utils import (
        count_parameters,
        get_device,
        log_error,
        log_info,
        log_warn,
        set_seed,
        tensor_grad_norms,
    )
except ImportError:  # pragma: no cover
    from config import (  # type: ignore
        Config,
        DEFAULT_CONFIG,
        HIGHACC_CONFIG,
        SMALL_CONFIG,
    )
    from data import get_mnist_loaders  # type: ignore
    from model import MLPBaseline, ThreeDNeuronSpace  # type: ignore
    from utils import (  # type: ignore
        count_parameters,
        get_device,
        log_error,
        log_info,
        log_warn,
        set_seed,
        tensor_grad_norms,
    )

# 预设名 -> 配置对象（--preset 使用；缺省为 default，保持既有默认行为）
PRESETS: Dict[str, Config] = {
    "small": SMALL_CONFIG,
    "default": DEFAULT_CONFIG,
    "highacc": HIGHACC_CONFIG,
}

# 阶段 A 验收阈值
SPARSITY_MAX: float = 0.1      # 连接稀疏度（规格口径 = 密度 E/(N*y_out*N*y_in)）必须 < 0.1
GRAD_NORM_MIN: float = 0.0     # 所有可学习参数梯度范数必须 > 0
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# 二期：产物目录与一期物理隔离（一期产物 n3d_model_*.pt 位于 checkpoints/ 根）
MODULE_CHECKPOINT_DIR_NAME: str = "n3d_sphere"
CHECKPOINT_DIR: str = os.path.join(PROJECT_ROOT, "checkpoints", MODULE_CHECKPOINT_DIR_NAME)
CHECKPOINT_PATH: str = os.path.join(CHECKPOINT_DIR, "model.pt")
# 验证类运行（冒烟测试 / 限批短跑）的独立写入目录，避免覆盖正式产物
VERIFY_CHECKPOINT_DIR: str = os.path.join(CHECKPOINT_DIR, "_verify")
# 冒烟测试默认产物路径（cube 拓扑，保持与一期相同的文件名语义）；
# sphere 等其它拓扑使用 `smoke_checkpoint_path(topology, flow_axis)` 生成带指纹的文件名，
# 避免不同拓扑的冒烟产物互相覆盖。
VERIFY_CHECKPOINT_PATH: str = os.path.join(VERIFY_CHECKPOINT_DIR, "smoke.pt")


def smoke_checkpoint_path(topology: str, flow_axis: str) -> str:
    """返回冒烟测试产物的路径（文件名含拓扑指纹，防止不同拓扑互覆）。

    命名规则：`smoke_<topology>_<flow_axis>.pt`（如 `smoke_cube_z.pt` /
    `smoke_sphere_z.pt`）。**cube + z（即默认配置）退化为 `smoke.pt`**，
    与一期以及历史验证产物的命名保持一致。

    参数
    ----
    topology : str
        拓扑类型（`cube` / `sphere`）。
    flow_axis : str
        全局流向轴（`x` / `y` / `z`）。

    返回
    ----
    str
        冒烟测试 checkpoint 的绝对路径。
    """
    if topology == "cube" and flow_axis == "z":
        return VERIFY_CHECKPOINT_PATH
    return os.path.join(VERIFY_CHECKPOINT_DIR, f"smoke_{topology}_{flow_axis}.pt")



def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数。

    参数
    ----
    argv : Optional[List[str]]
        参数列表（None 表示使用 sys.argv）。

    返回
    ----
    argparse.Namespace
        含 smoke_test / preset / epochs / device / max_batches / checkpoint /
        backup / criterion_conflict / threads / lr / weight_decay / dropout /
        batch_size / readout_bias 等字段。
    """
    parser = argparse.ArgumentParser(
        description="N3D 二期架构变体（球形有向拓扑）训练入口"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="仅执行阶段 A 冒烟测试（SMALL_CONFIG，1 个 batch 的前向+反向验收）",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="neuron3d",
        choices=["neuron3d", "mlp"],
        help=(
            "架构：neuron3d（默认，四步闭环主模型）/ mlp（对照基线 784->hidden->10）。"
            "mlp 走**完全相同**的训练循环、优化器、调度器、梯度裁剪与评估代码，"
            "仅替换模型构造，用于判定瓶颈在架构还是在数据。"
        ),
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=sorted(PRESETS.keys()),
        help=(
            "配置预设：small / default / highacc（缺省 default）。"
            "smoke / default 预设保持既有默认行为不变；highacc 为高精度冲刺配置。"
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "CPU 线程数：0（默认）= 不干预，保持 torch 默认线程数；"
            "正数则调用 torch.set_num_threads(N) 并打印生效线程数与 os.cpu_count()"
        ),
    )
    parser.add_argument(
        "--lr", type=float, default=0.0, help="覆盖学习率（默认取预设值）"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=-1.0,
        help="覆盖权重衰减（> 0 时优化器切换为 AdamW；默认取预设值）",
    )
    parser.add_argument(
        "--dropout", type=float, default=-1.0, help="覆盖输出端 dropout 概率（默认取预设值）"
    )
    parser.add_argument(
        "--batch-size", type=int, default=0, help="覆盖批大小（默认取预设值）"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="覆盖神经元数量 N（默认取预设值；用于快速筛选更大容量的配置）",
    )
    parser.add_argument(
        "--t-steps",
        "--t",
        dest="t_override",
        type=int,
        default=0,
        help=(
            "覆盖四步闭环迭代轮数 T（默认取预设值；T 必须 >= 2）。"
            "规范名为 --t-steps，--t 为同义别名（argparse 精确匹配优先，"
            "因此 --t 不会被前缀匹配到 --threads）。"
        ),
    )
    parser.add_argument(
        "--y-in",
        type=int,
        default=0,
        help="覆盖每个神经元的输入突触数 y_in（> 0；默认取预设值）",
    )
    parser.add_argument(
        "--y-out",
        type=int,
        default=0,
        help="覆盖每个神经元的输出突触数 y_out（> 0；默认取预设值）",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=0.0,
        help="覆盖突触分布半径 H（> 0 且 < L；默认取预设值）",
    )
    parser.add_argument(
        "--d",
        type=float,
        default=0.0,
        help="覆盖连接距离阈值 D（> 0；越大多连接越多、边数 E 越大；默认取预设值）",
    )
    # ---- 二期新增：拓扑几何 CLI ----
    parser.add_argument(
        "--topology",
        type=str,
        default="",
        help=(
            "拓扑几何：cube（默认，一期立方体几何，逐位不变）/ sphere（二期球形有向拓扑）。"
            "缺省（空串）表示沿用预设值（cube），**不计入冒烟测试的显式覆盖判定**。"
        ),
    )
    parser.add_argument(
        "--flow-axis",
        type=str,
        default="",
        help=(
            "全局流向轴 x / y / z（默认取预设值 z）。仅 topology=sphere 生效："
            "输入突触取负半球（−axis）、输出突触取正半球（+axis）。"
            "缺省（空串）表示沿用预设值，不计入显式覆盖判定。"
        ),
    )
    parser.add_argument(
        "--space-radius",
        type=float,
        default=-1.0,
        help=(
            "球形拓扑下神经元所在球体的半径（哨兵 -1 = 未提供，沿用预设值 0）。"
            "0 表示使用与立方体**等体积**的默认半径 L·(3/4π)^(1/3)（L=1 时约 0.620）。"
            "显式指定时必须 > H，否则 Config 构造期报错。"
        ),
    )
    parser.add_argument(
        "--seed",
        dest="seed_override",
        type=int,
        default=0,
        help=(
            "覆盖随机种子（默认沿用预设的 seed=42）。种子决定三维坐标采样与参数初始化，"
            "因此**不同种子意味着不同拓扑**（边数 E 与 test_acc 都会变）；负数报错、0 表示不覆盖。"
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help=(
            "追加到限批验证产物文件名末尾的标签（仅允许字母/数字/下划线/连字符），"
            "便于同名配置的多次运行各自留痕"
        ),
    )
    parser.add_argument(
        "--readout-bias",
        dest="readout_bias",
        action="store_true",
        default=None,
        help="为输出层 W_out 启用 bias（默认取预设值）",
    )
    parser.add_argument(
        "--no-readout-bias",
        dest="readout_bias",
        action="store_false",
        help="关闭输出层 bias（默认取预设值）",
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
    arch: str = "neuron3d",
) -> Tuple[nn.Module, DataLoader, DataLoader]:
    """按配置构建模型与数据加载器（统一的装配入口，保证两阶段行为一致）。

    参数
    ----
    config : Config
        超参配置。
    device : torch.device
        目标设备。
    max_batches : int
        仅用于日志提示（0 表示使用全量数据）。
    arch : str
        架构选择：`"neuron3d"`（默认，四步闭环主模型）或 `"mlp"`（对照基线）。
        **这是全流程中唯一按 arch 分支的地方**——训练循环、优化器、调度器、
        梯度裁剪与评估代码在两条路径下完全共用同一份实现。

    返回
    ----
    Tuple[nn.Module, DataLoader, DataLoader]
        (model, train_loader, test_loader)。

    实现说明（设备探针的代价）
    --------------------------
    装配末尾会用 `next(iter(train_loader))` 取一个 batch 作为"输入与模型设备一致性"的
    探针。**实测结论：该探针不会减少后续 epoch 的 batch 数。**

    原因：PyTorch 的 `DataLoader.__iter__` 在数据集开启了 `shuffle=True` 时，每次迭代都会
    基于 `generator` 的**当前状态**新建一个 `RandomSampler`；探针消耗的第一个 batch 只
    推进了生成器状态，训练循环随后新建的迭代器拿到的是**同一打乱流的剩余排列**，
    因此单 epoch 仍然处理 `len(train_loader)` 个 batch（例如 DEFAULT 配置下
    batch_size=64 时为 938，HIGHACC 配置下 batch_size=128 时为 469），
    **没有样本被"提前取走"**。

    换句话说：探针的代价只是"多算了一个 batch 的前向"（约一次前向的时间），
    不损失任何训练样本。**本说明仅为消除 batch 数目歧义，取数逻辑本身不做任何改动。**

    异常
    ------
    RuntimeError
        模型参数/buffer 与目标设备不一致，或首个 batch 的输入与模型设备不一致时抛出
        （把设备问题暴露在命令行入口附近，而不是等到反向传播深处）。
    """
    set_seed(config.seed)
    # ---- 唯一按 arch 分支的地方：模型构造（训练循环/优化器/评估代码完全共用） ----
    if arch == "mlp":
        model: nn.Module = MLPBaseline(config).to(device)
    else:
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
        f"探针只多算一个 batch 的前向，不减少训练样本："
        f"单 epoch batch 数 = len(train_loader) = {len(train_loader)}"
    )

    stats = model.get_connection_stats()
    if arch == "mlp":
        log_info(
            f"模型装配完成（arch=mlp 对照基线）：input_dim={config.input_dim}, "
            f"hidden_dim={config.hidden_dim}, output_dim={config.output_dim}, "
            f"dropout={config.dropout}, 可学习参数={model.count_parameters()}, 设备={device}"
        )
    else:
        topo_stats = model.get_topology_stats()
        log_info(
            f"模型装配完成：N={config.N}, y_in={config.y_in}, y_out={config.y_out}, T={config.T}, "
            f"topology={config.topology}, flow_axis={config.flow_axis}, "
            f"space_radius={config.effective_space_radius:.6f}, "
            f"E={int(stats['num_edges'])}, 连接密度(sparsity)={stats['sparsity']:.6f}, "
            f"零元素占比(zero_ratio)={stats['zero_ratio']:.6f}, "
            f"avg_out_degree={stats['avg_out_degree']:.2f}, tau={stats['tau']:.6f}, "
            f"可学习参数={model.count_parameters()}, 设备={device}"
        )
        # 二期拓扑指纹：逆向边占比 / 流向轴高度分布 / 2a 覆盖 / 连通性
        log_info(
            f"拓扑统计（拓扑指纹，全部在 __init__ 预计算）："
            f"逆向边占比={topo_stats['reverse_edge_ratio']:.6f}"
            f"（{int(topo_stats['reverse_edges'])}/{int(topo_stats['num_edges'])}），"
            f"流向轴 gap 均值={topo_stats['axis_gap_mean']:.4f}"
            f"（min={topo_stats['axis_gap_min']:.4f}, max={topo_stats['axis_gap_max']:.4f}），"
            f"神经元流向轴高度 mean={topo_stats['neuron_axis_mean']:.4f} / "
            f"min={topo_stats['neuron_axis_min']:.4f} / max={topo_stats['neuron_axis_max']:.4f} / "
            f"std={topo_stats['neuron_axis_std']:.4f}，"
            f"静态 2a 覆盖={topo_stats['connected_output_ratio']:.6f}，"
            f"弱连通分量数={int(topo_stats['weak_components'])}，"
            f"最大分量占比={topo_stats['largest_component_ratio']:.6f}"
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
    grad_clip: float = 0.0,
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
    grad_clip : float
        梯度范数裁剪阈值；> 0 时在 backward() 与 step() 之间执行
        `clip_grad_norm_`。默认 0.0（不裁剪，保持既有行为逐位不变）。

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
        if grad_clip > 0.0:
            # 梯度裁剪：在 backward 与 step 之间执行（默认关闭，不影响既有行为）
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        if collect_grads and (max_batches > 0 and i == max_batches - 1):
            # 阶段 A：在最后一个已反向的 batch 上收集全部可学习参数的梯度范数
            # 注意：若启用梯度裁剪，此处收集到的是**裁剪前**的原始梯度范数；
            # 阶段 A 默认不使用裁剪（HIGHACC 才用），故不影响验收语义。
            grad_norms = tensor_grad_norms(model)
        if (i + 1) % 100 == 0:
            log_info(f"epoch {epoch} | batch {i + 1} | running_loss={total_loss / n_batches:.4f}")
    if n_batches == 0:
        raise RuntimeError("训练集为空：没有任何 batch 被处理")
    return total_loss / n_batches, grad_norms, n_batches

def validate_override_args(args: argparse.Namespace) -> None:
    """校验全部"覆盖类"命令行参数的取值范围（**唯一校验入口**）。

    语义（必须严格遵守，`0` = 不覆盖）
    ---------------------------------
    * 取值范围型参数（`--n / --t-steps(--t) / --y-in / --y-out / --h / --d / --seed`）：
      **负数一律 `raise ValueError`**（由 `main()` 捕获后以非 0 退出码结束）；
      `0` / `0.0` 表示"未提供、不覆盖"（保持既有哨兵语义，不得改变）；
    * `--epochs` 同理（负数报错、0 表示不覆盖）；
    * `--lr`（负数报错、0 表示不覆盖）、`--weight-decay` 与 `--dropout`（**负值报错**、
      0 表示覆盖为 0，与既有语义一致）、`--batch-size`（负数报错、0 表示不覆盖）、
      `--threads`（负数报错、0 表示不干预）；
    * 二期新增 `--space-radius`（哨兵 `-1.0` = 未提供；`< -1.0` 报错）、
      `--topology`（空串 = 未提供；取值域 cube/sphere）、
      `--flow-axis`（空串 = 未提供；取值域 x/y/z）。

    历史缺陷（皋陶实测复现）
    ------------------------
    此前这些参数统一写成 `if x > 0: 覆盖`，负值会落入"未提供"分支被**静默忽略**
    （例如 `--y-in -1` 退出码 0 且沿用预设 y_in=8）。现在先判 `< 0` 报错、再判 `0` 跳过。

    参数
    ----
    args : argparse.Namespace
        `parse_args()` 的返回值。

    返回
    ----
    None

    异常
    ------
    ValueError
        任一覆盖参数为负数时抛出，消息含具体选项名与数值。
    """
    # 取值范围型参数（哨兵值 0/0.0 = 未提供）：负数一律报错
    rules: List[Tuple[str, float]] = [
        ("--n", args.n),
        ("--t-steps/--t", args.t_override),
        ("--y-in", args.y_in),
        ("--y-out", args.y_out),
        ("--h", args.h),
        ("--d", args.d),
        ("--seed", args.seed_override),
        ("--epochs", args.epochs),
        ("--lr", args.lr),
        ("--batch-size", args.batch_size),
        ("--threads", args.threads),
    ]
    for name, value in rules:
        if value < 0:
            raise ValueError(
                f"{name} 不能为负（0 表示不覆盖/不干预），当前 {name}={value}"
            )
    # --weight-decay / --dropout 的哨兵是 -1.0（表示"未提供"），
    # 因此只拒绝**小于** -1.0 的取值；-1.0 本身合法（= 不覆盖），0 是合法覆盖值。
    for name, value in (("--weight-decay", args.weight_decay), ("--dropout", args.dropout)):
        if value < -1.0:
            raise ValueError(
                f"{name} 不能为负（-1 表示不覆盖、0 表示覆盖为 0），当前 {name}={value}"
            )
    if args.dropout >= 1.0:
        raise ValueError(f"--dropout 必须落在 [0, 1)，当前 --dropout={args.dropout}")
    # ---- 二期新增：拓扑几何 CLI 的取值域校验（空串 = 未提供，不校验） ----
    if args.topology and args.topology not in ("cube", "sphere"):
        raise ValueError(
            f"--topology 仅允许 'cube' 或 'sphere'，当前 --topology={args.topology!r}"
        )
    if args.flow_axis and args.flow_axis not in ("x", "y", "z"):
        raise ValueError(
            f"--flow-axis 仅允许 'x' / 'y' / 'z'，当前 --flow-axis={args.flow_axis!r}"
        )
    if args.space_radius < -1.0:
        raise ValueError(
            f"--space-radius 必须 >= 0（-1 表示未提供、0 表示使用等体积球默认半径），"
            f"当前 --space-radius={args.space_radius}"
        )


def apply_overrides(
    base: Config,
    args: argparse.Namespace,
    warn_message: Optional[str] = None,
) -> Config:
    """把命令行覆盖应用到基线配置上（**唯一覆盖实现**）。

    参数
    ----
    base : Config
        基线配置（冒烟路径传 `SMALL_CONFIG`，正式训练传 `PRESETS[args.preset]`）。
    args : argparse.Namespace
        `parse_args()` 的返回值（调用前应已通过 `validate_override_args`）。
    warn_message : Optional[str]
        非 None 时，在**确实发生覆盖**时打印该告警（用于冒烟路径提示基线漂移）。

    返回
    ----
    Config
        覆盖后的配置对象；无任何覆盖时返回 `base` 本身（保证逐位不变）。

    说明
    ----
    `--threads` **不计入"显式覆盖"判定**：它只影响 CPU 并行度，不改变任何超参与数值结果。
    """
    overrides = base.to_dict()
    changed = False
    if args.device:
        overrides["device"] = args.device
        changed = True
    if args.epochs > 0:
        overrides["epochs"] = args.epochs
        changed = True
    if args.lr > 0.0:
        overrides["lr"] = args.lr
        changed = True
    # --weight-decay / --dropout 的哨兵是 -1.0（表示不覆盖），0 是合法覆盖值
    if args.weight_decay >= 0.0:
        overrides["weight_decay"] = args.weight_decay
        changed = True
    if args.dropout >= 0.0:
        overrides["dropout"] = args.dropout
        changed = True
    if args.batch_size > 0:
        overrides["batch_size"] = args.batch_size
        changed = True
    if args.n > 0:
        overrides["N"] = args.n
        changed = True
    if args.t_override > 0:
        log_info(f"T 覆盖：{overrides['T']} -> {args.t_override}")
        overrides["T"] = args.t_override
        changed = True
    if args.y_in > 0:
        overrides["y_in"] = args.y_in
        changed = True
    if args.y_out > 0:
        overrides["y_out"] = args.y_out
        changed = True
    if args.h > 0.0:
        overrides["H"] = args.h
        changed = True
    if args.d > 0.0:
        overrides["D"] = args.d
        changed = True
    # ---- 二期新增：拓扑几何覆盖（哨兵：空串 / -1.0 = 未提供） ----
    if args.topology:
        overrides["topology"] = args.topology
        changed = True
    if args.flow_axis:
        overrides["flow_axis"] = args.flow_axis
        changed = True
    if args.space_radius >= 0.0:
        overrides["space_radius"] = args.space_radius
        changed = True
    if args.seed_override > 0:
        log_info(f"seed 覆盖：{overrides['seed']} -> {args.seed_override}")
        overrides["seed"] = args.seed_override
        changed = True
    if args.readout_bias is not None:
        overrides["readout_bias"] = bool(args.readout_bias)
        changed = True
    if not changed:
        return base
    if warn_message:
        log_warn(warn_message)
    return Config(**overrides)


def build_smoke_config(args: argparse.Namespace) -> Config:
    """构造冒烟测试（阶段 A）使用的配置。

    默认契约（**必须逐位保持**）
    --------------------------
    不加任何覆盖参数（`--preset` 缺省且无 `--n/--t/--lr/...`）时，返回的配置与
    `SMALL_CONFIG` **逐字段相同**（同一对象），从而保证 `--smoke-test` 的
    `loss=2.419689` 不变。

    覆盖语义
    --------
    仅当用户**显式**给出 `--preset` / `--n` / `--t-steps(--t)` / `--y-in` / `--y-out` /
    `--h` / `--d` / `--seed` / `--lr` / `--weight-decay` / `--dropout` / `--batch-size` /
    `--readout-bias` / `--device` / **`--topology` / `--flow-axis` / `--space-radius`**
    时才覆盖，并通过日志提示"冒烟测试配置已被覆盖"。

    **二期注意**：新增的三个拓扑参数必须纳入下方 `explicit` 判定，否则
    `--smoke-test --topology sphere` 会被误判为"未显式覆盖"而静默返回
    `SMALL_CONFIG`（sphere 几何被丢弃、跑出 2.419689 的 cube 结果）。

    **基线规则**：冒烟路径固定以 `SMALL_CONFIG` 为基线，只有当显式指定**非 default**
    预设时才换基线。否则"只想改一个字段"（如 `--device cpu`）会把整个配置静默放大到
    `DEFAULT_CONFIG` 规模（N=256/T=3/batch=64），破坏冒烟测试的轻量语义。

    参数
    ----
    args : argparse.Namespace
        `parse_args()` 的返回值。

    返回
    ----
    Config
        冒烟测试最终生效的配置。
    """
    base = PRESETS[args.preset] if args.preset != "default" else SMALL_CONFIG
    if base is not SMALL_CONFIG:
        # --preset 的作用体现在"换基线"而非字段覆盖，故 apply_overrides 的字段级告警不会触发。
        # 这里显式提示，避免用户在 --smoke-test 下静默跑到 DEFAULT/HIGHACC 规模上。
        log_warn(
            f"冒烟测试基线已被 --preset {args.preset} 替换："
            f"N={base.N}, T={base.T}, batch_size={base.batch_size}（非 SMALL_CONFIG 规模）；"
            "阶段 A 的 `loss=2.419689` 基线仅在默认 SMALL_CONFIG 下成立"
        )
    explicit = (
        args.preset != "default"
        or args.n > 0
        or args.t_override > 0
        or args.y_in > 0
        or args.y_out > 0
        or args.h > 0.0
        or args.d > 0.0
        or args.seed_override > 0
        or args.lr > 0.0
        or args.weight_decay >= 0.0
        or args.dropout >= 0.0
        or args.batch_size > 0
        or args.readout_bias is not None
        or bool(args.device)
        # 二期新增拓扑参数：必须计入 explicit，否则 --topology sphere 会被静默丢弃
        or bool(args.topology)
        or bool(args.flow_axis)
        or args.space_radius >= 0.0
    )
    if not explicit:
        return SMALL_CONFIG
    return apply_overrides(
        base,
        args,
        warn_message=(
            "冒烟测试配置已被显式覆盖（--preset/--n/--t/--y-in/--seed/--lr/--topology/... 之一）："
            f"预设={args.preset}；阶段 A 的 `loss=2.419689` 基线仅在默认 SMALL_CONFIG 下成立"
        ),
    )


def run_smoke_test(
    device_override: str = "",
    criterion_conflict: bool = False,
    config_override: Optional[Config] = None,
    arch: str = "neuron3d",
) -> bool:
    """阶段 A 冒烟测试：验证前向/反向跑通并逐条核对验收标准。

    参数
    ----
    device_override : str
        设备覆盖（空串表示使用 SMALL_CONFIG.device；当 `config_override` 非空时忽略）。
    criterion_conflict : bool
        **兼容别名，当前不再需要**。第 2 轮已把"连接稀疏度"回归规格原义
        （密度 E/(N*y_out*N*y_in) < 0.1），该项现已正常 PASS。传入该开关不改变任何
        判定结果，仅为兼容旧命令行而保留。
    config_override : Optional[Config]
        完整的配置覆盖（通常来自 `build_smoke_config`）。为 None 时使用 `SMALL_CONFIG`
        （默认路径，保证 `loss=2.419689` 逐位不变）。

    返回
    ----
    bool
        通过返回 True；有失败项打印 [FAIL] 并返回 False。

    产物保护
    --------
    冒烟测试只跑 1 个 batch，其 checkpoint 写入
    `checkpoints/n3d_sphere/_verify/smoke[_<topology>_<flow_axis>].pt`
    （cube+z 退化为 `smoke.pt`），**不会覆盖 `checkpoints/n3d_sphere/model.pt`
    等正式产物，也绝不触碰一期 `checkpoints/` 下的任何文件**。
    """
    if config_override is not None:
        config: Config = config_override
    else:
        config = SMALL_CONFIG
        if device_override:
            # 直接覆盖副本的 device 字段（避免 Config(**{...}) 与位置参数冲突）
            config = Config(**{**config.to_dict(), "device": device_override})
    # 设备解析：auto -> 有 CUDA 用 CUDA，否则 CPU
    device = get_device(config.device)

    log_info("=" * 78)
    # 日志如实打印**实际生效**的规模标签，避免"写着 SMALL_CONFIG、跑着 DEFAULT 规模"的误导
    log_info(
        "阶段 A 冒烟测试开始"
        f"（{'SMALL_CONFIG' if config is SMALL_CONFIG else '已覆盖/已换预设的配置'}）"
    )
    log_info(f"配置：{config.describe()}")
    log_info(f"实际设备：{device}（torch_scatter 可用性见 utils.HAS_TORCH_SCATTER）")
    log_info("=" * 78)

    t0 = time.perf_counter()
    model, train_loader, _ = build_model_and_data(config, device, max_batches=1, arch=arch)
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
    # ---- 二期拓扑统计（含 forward 记录的动态 2a 覆盖，故在训练之后才打印） ----
    topo_stats: Dict[str, float] = (
        model.get_topology_stats() if arch == "neuron3d" else {}
    )
    if topo_stats:
        log_info(
            f"拓扑统计（topology={config.topology}, flow_axis={config.flow_axis}）："
            f"逆向边占比={topo_stats['reverse_edge_ratio']:.6f}"
            f"（{int(topo_stats['reverse_edges'])}/{int(topo_stats['num_edges'])}），"
            f"流向轴 gap mean/min/max={topo_stats['axis_gap_mean']:.4f}/"
            f"{topo_stats['axis_gap_min']:.4f}/{topo_stats['axis_gap_max']:.4f}，"
            f"神经元流向轴高度 mean/min/max/std={topo_stats['neuron_axis_mean']:.4f}/"
            f"{topo_stats['neuron_axis_min']:.4f}/{topo_stats['neuron_axis_max']:.4f}/"
            f"{topo_stats['neuron_axis_std']:.4f}，"
            f"静态 2a 覆盖={topo_stats['connected_output_ratio']:.6f}，"
            f"动态 2a 覆盖（末轮）={topo_stats['out_nonzero_coverage_last']:.6f}，"
            f"弱连通分量数={int(topo_stats['weak_components'])}，"
            f"最大分量占比={topo_stats['largest_component_ratio']:.6f}"
        )
    # ---- dense 对照诊断：证明"未 materialize dense [N*y_out, N*y_in] 权重矩阵" ----
    # arch 专属统计量：MLP 无稀疏连接概念，用 0 占位并跳过相应判据
    is_neuron3d = arch == "neuron3d"
    if is_neuron3d:
        total_pairs = model.n_out_syn * model.n_in_syn
        conn_param = int(model.W_conn_sparse.numel())
    else:
        total_pairs = 0
        conn_param = 0
    dense_weight_tensors = model.count_dense_weight_tensors()
    if is_neuron3d:
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
    else:
        log_info(
            f"arch=mlp 对照基线：无稀疏连接概念（输入 {config.input_dim} -> "
            f"hidden {config.hidden_dim} -> 输出 {config.output_dim}），"
            f"可学习参数 {param_total}"
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
    if is_neuron3d:
        # 规格口径：连接稀疏度 = E/(N*y_out*N*y_in)（密度定义），要求 < 0.1
        checks.append(
            (
                "连接稀疏度（密度 E/(N*y_out*N*y_in)）< 0.1",
                stats["sparsity"] < SPARSITY_MAX,
                f"sparsity={stats['sparsity']:.6f}（阈值 < {SPARSITY_MAX}）",
            )
        )
        checks.append(("tau > 0", stats["tau"] > 0.0, f"tau={stats['tau']:.6f}"))
    else:
        # MLP 对照基线：连接类判据不适用，改为结构性判据
        checks.append(
            (
                "arch=mlp 结构正确（784 -> hidden -> 10）",
                int(config.input_dim) == 784 and int(config.output_dim) == 10,
                f"input_dim={config.input_dim}, hidden_dim={config.hidden_dim}, "
                f"output_dim={config.output_dim}",
            )
        )
    checks.append(("CPU 单 batch 耗时 < 120s", elapsed < 120.0, f"实际 {elapsed:.2f}s"))
    if is_neuron3d:
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
    # 二期：文件名含拓扑指纹（cube+z 退化为 smoke.pt），防止不同拓扑的冒烟产物互相覆盖
    smoke_path = smoke_checkpoint_path(config.topology, config.flow_axis)
    os.makedirs(VERIFY_CHECKPOINT_DIR, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "connection_stats": stats,
            "topology_stats": topo_stats,
            "stage": "smoke",
            "arch": arch,
            "loss": float(avg_loss),
            "grad_norms": grad_norms,
        },
        smoke_path,
    )
    log_info(
        f"[产物保护] 冒烟测试 checkpoint 写入独立路径（不覆盖正式产物）：{smoke_path}"
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


def config_fingerprint(config: Config, max_batches: int, tag: str = "") -> str:
    """生成限批验证产物的配置指纹文件名（保证不同配置互不覆盖）。

    命名格式：
    `verify_<bpe>_N{N}_y{y_in}x{y_out}_H{H}_D{D}_T{T}_top{topology}_ax{flow_axis}_s{seed}[_<tag>].pt`

    `_s{seed}` 是第 6 轮补上的指纹维度：种子决定三维坐标采样，**不同 seed 即不同拓扑**，
    若指纹不含 seed，则只改 `--seed` 的限批跑会写入同名文件并互相覆盖（离朱第 6 轮
    实测报告该缺口）。

    `_top{topology}` / `_ax{flow_axis}` 是二期补上的维度：`--topology sphere` 与
    `--topology cube` 即使容量维度完全相同也是**两套不同几何**，若指纹不含拓扑，
    两个实验点会写入同名文件并互相覆盖（与第 6 轮 seed 缺口同类的产物纪律问题）。

    参数
    ----
    config : Config
        本次运行实际生效的配置（用于提取容量维度、拓扑与 seed）。
    max_batches : int
        每 epoch 的最大 batch 数（`bpe` = batches per epoch）。
    tag : str
        可选的用户标签，仅允许字母/数字/下划线/连字符。

    返回
    ----
    str
        产物文件名（不含目录）。

    异常
    ------
    ValueError
        `tag` 含非法字符时抛出。

    说明
    ----
    历史缺陷：早先所有限批跑统一写 `verify_<bpe>.pt`，不同配置互相覆盖，
    导致筛选记录丢失、数字无法复核。加入配置指纹后，每个容量点都有独立产物，
    可随时用 `torch.load` 复核。
    """
    if tag and not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError(
            f"--tag 仅允许字母/数字/下划线/连字符，当前 tag={tag!r}"
        )
    name = (
        f"verify_{max_batches}"
        f"_N{config.N}"
        f"_y{config.y_in}x{config.y_out}"
        f"_H{config.H:g}"
        f"_D{config.D:g}"
        f"_T{config.T}"
        # 二期新增：拓扑指纹（topology + flow_axis），防止不同几何互相覆盖
        f"_top{config.topology}"
        f"_ax{config.flow_axis}"
        f"_s{config.seed}"
    )
    if tag:
        name = f"{name}_{tag}"
    return f"{name}.pt"


def resolve_checkpoint_path(
    checkpoint_override: str,
    max_batches: int,
    config: Optional[Config] = None,
    tag: str = "",
) -> str:
    """决定本次训练实际写入的 checkpoint 路径（产物保护核心逻辑）。

    规则
    ----
    1. `max_batches > 0`（限批短跑 / 验证跑）-> 一律写入
       `checkpoints/n3d_sphere/_verify/`：用户显式给了 `--checkpoint` 则沿用其文件名，
       否则使用**带配置指纹**的
       `verify_<bpe>_N{N}_y{y_in}x{y_out}_H{H}_D{D}_T{T}_top{topology}_ax{flow_axis}_s{seed}[_tag].pt`
       （`config` 为 None 时退化为旧的 `verify_<bpe>.pt`），**绝不覆盖正式产物**；
       发生重定位时会打印「重定位前路径 -> 重定位后路径」；
    2. `checkpoint_override` 非空 -> 使用用户显式指定的路径（相对路径按工程根目录解析）；
    3. 否则 -> 正式全量默认路径。**cube + 默认设置（N=256/y=8x8/H=0.1/D=0.15/T=3/
       seed=42）退化为一期同名的 `checkpoints/n3d_sphere/model.pt`**；
       其它几何/容量配置使用 `full_<指纹>.pt`，保证"球形有向拓扑"等新实验点
       与 cube 基线各自留痕、互不覆盖。

    参数
    ----
    checkpoint_override : str
        用户通过 `--checkpoint` 指定的路径（可为空）。
    max_batches : int
        每个 epoch 的最大 batch 数（> 0 表示限批模式）。
    config : Optional[Config]
        本次生效的配置（用于生成配置指纹文件名）。
    tag : str
        可选标签，追加到指纹文件名末尾。

    返回
    ----
    str
        实际写入的 checkpoint 绝对路径。
    """
    if max_batches > 0:
        # 限批模式：隔离到 _verify/；文件名带配置指纹，保证不同配置不互相覆盖
        if checkpoint_override:
            name = os.path.basename(checkpoint_override)
            if not name.endswith(".pt"):
                name = f"{name}.pt"
        elif config is not None:
            name = config_fingerprint(config, max_batches, tag)
        else:
            name = f"verify_{max_batches}.pt"
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
    # 正式全量：默认路径按配置指纹区分，防止不同拓扑的全量结果互相覆盖。
    # cube 默认配置保持一期同名语义（checkpoints/n3d_sphere/model.pt）。
    if config is not None and not is_default_geometry(config):
        # 注意：这里**不**复用 `config_fingerprint`——其前缀 `verify_<bpe>` 是限批模式语义，
        # bpe=0 会产出 `full_verify_0_...` 这种误导性名字（离朱第 2 轮 D4）。
        # 全量产物用独立的指纹构造，去掉该前缀。
        name = (
            f"full_N{config.N}"
            f"_y{config.y_in}x{config.y_out}"
            f"_H{config.H:g}"
            f"_D{config.D:g}"
            f"_T{config.T}"
            f"_top{config.topology}"
            f"_ax{config.flow_axis}"
            f"_s{config.seed}"
        )
        if tag:
            name = f"{name}_{tag}"
        path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
        log_info(
            f"[产物保护] 正式全量训练使用配置指纹产物名（不同拓扑/容量互不覆盖）：{path}"
        )
        return path
    return CHECKPOINT_PATH


def is_default_geometry(config: Config) -> bool:
    """判断配置是否为"一期默认几何/容量"（cube + 默认超参）。

    该判定只影响**默认产物文件名**（保持 `model.pt` 语义），不影响任何数值计算。

    参数
    ----
    config : Config
        本次生效的配置。

    返回
    ----
    bool
        `True` 表示与一期默认配置（DEFAULT_CONFIG）的几何/容量维度一致。
    """
    return (
        config.topology == "cube"
        and config.flow_axis == "z"
        and int(config.N) == 256
        and int(config.y_in) == 8
        and int(config.y_out) == 8
        and float(config.H) == 0.1
        and float(config.D) == 0.15
        and int(config.T) == 3
        and int(config.seed) == 42
    )


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


def apply_threads(threads: int) -> None:
    """按 `--threads` 设置 CPU 线程数并打印生效信息。

    参数
    ----
    threads : int
        `0`（默认）= 不干预，保持 torch 当前线程数；正数则调用
        `torch.set_num_threads(threads)`。

    返回
    ----
    None

    异常
    ------
    ValueError
        `threads < 0` 时抛出。
    """
    if threads < 0:
        raise ValueError(f"--threads 不能为负，当前 {threads}")
    cpu_count = os.cpu_count()
    if threads == 0:
        log_info(
            f"线程设置：未干预（--threads 0），保持 torch 默认 "
            f"torch.get_num_threads()={torch.get_num_threads()}；os.cpu_count()={cpu_count}"
        )
        return
    torch.set_num_threads(int(threads))
    log_info(
        f"线程设置：torch.set_num_threads({threads}) 已生效，"
        f"当前 torch.get_num_threads()={torch.get_num_threads()}；os.cpu_count()={cpu_count}"
    )


def run_full_training(
    config: Optional[Config] = None,
    device_override: str = "",
    epochs_override: int = 0,
    max_batches: int = 0,
    checkpoint_path: str = "",
    backup: bool = True,
    preset: str = "default",
    lr_override: float = 0.0,
    weight_decay_override: float = -1.0,
    dropout_override: float = -1.0,
    batch_size_override: int = 0,
    readout_bias_override: Optional[bool] = None,
    n_override: int = 0,
    t_override: int = 0,
    y_in_override: int = 0,
    y_out_override: int = 0,
    h_override: float = 0.0,
    d_override: float = 0.0,
    topology_override: str = "",
    flow_axis_override: str = "",
    space_radius_override: float = -1.0,
    tag: str = "",
    arch: str = "neuron3d",
) -> Dict[str, float]:
    """阶段 B 正式训练：按预设配置训练若干 epoch 并保存 checkpoint。

    参数
    ----
    config : Optional[Config]
        **CLI 路径应传入已构造好的配置**（由 `validate_override_args` +
        `apply_overrides` 生成），本函数不再自行处理覆盖。
    device_override : str
        设备覆盖（空串表示使用预设的 device）。
    epochs_override : int
        覆盖 epoch 数（0 表示使用预设的 epochs）。
    max_batches : int
        每个 epoch 最多处理的 batch 数（0 表示全量）。> 0 时 checkpoint 会写入
        `checkpoints/_verify/`，避免覆盖正式产物。
    checkpoint_path : str
        checkpoint 保存路径（空串表示使用默认 `checkpoints/model.pt`）。
    backup : bool
        覆盖已有 checkpoint 前是否先备份为 `<path>.bak`。默认 True。
    preset : str
        预设名（small / default / highacc）。默认 "default"。
    lr_override : float
        覆盖学习率（<= 0 表示不覆盖）。
    weight_decay_override : float
        覆盖权重衰减（< 0 表示不覆盖；> 0 时优化器切换为 AdamW）。
    dropout_override : float
        覆盖输出端 dropout（< 0 表示不覆盖）。
    batch_size_override : int
        覆盖批大小（<= 0 表示不覆盖）。
    readout_bias_override : Optional[bool]
        覆盖输出层 bias（None 表示不覆盖）。
    n_override : int
        覆盖神经元数量 N（<= 0 表示不覆盖）。
    t_override : int
        覆盖四步闭环迭代轮数 T（<= 0 表示不覆盖）。
    y_in_override : int
        覆盖每个神经元的输入突触数 y_in（<= 0 表示不覆盖）。
    y_out_override : int
        覆盖每个神经元的输出突触数 y_out（<= 0 表示不覆盖）。
    h_override : float
        覆盖突触分布半径 H（<= 0 表示不覆盖）。
    d_override : float
        覆盖连接距离阈值 D（<= 0 表示不覆盖）。
    topology_override : str
        **二期新增**：覆盖拓扑类型 `cube` / `sphere`（空串表示不覆盖）。
    flow_axis_override : str
        **二期新增**：覆盖全局流向轴 `x` / `y` / `z`（空串表示不覆盖）。
    space_radius_override : float
        **二期新增**：覆盖球形拓扑的神经元球半径（< 0 表示不覆盖；0 表示使用
        与立方体等体积的默认半径）。
    tag : str
        追加到限批验证产物文件名末尾的标签（仅限批模式生效）。

    返回
    ----
    Dict[str, float]
        最终统计：{"test_acc": ..., "loss": ..., "sparsity": ..., "zero_ratio": ...,
        "tau": ..., "num_edges": ..., "params": ..., "checkpoint": ..., "elapsed_s": ...}。

    说明
    ----
    **覆盖与校验的唯一入口是 `apply_overrides` / `validate_override_args`**
    （`main()` 传入的 `config`）。本函数不再自行处理覆盖，避免出现两套并存实现。
    当 `config is None` 时，退回到"由各 `*_override` 参数拼装"的兼容模式，
    以便程序化调用（非 CLI 路径）仍可使用。
    优化器：`weight_decay > 0` 用 AdamW，否则用 Adam（默认路径保持 Adam 不变）。
    """
    if config is not None:
        # CLI 路径：配置已由 validate_override_args + apply_overrides 构造完毕
        return _run_training_with_config(
            config, preset, max_batches, checkpoint_path, backup, tag, arch
        )
    # 兼容模式：程序化调用时由各 *_override 参数拼装（哨兵值语义与 CLI 一致）
    base = PRESETS[preset]
    overrides = base.to_dict()
    if device_override:
        overrides["device"] = device_override
    if epochs_override > 0:
        overrides["epochs"] = epochs_override
    if lr_override > 0.0:
        overrides["lr"] = lr_override
    if weight_decay_override >= 0.0:
        overrides["weight_decay"] = weight_decay_override
    if dropout_override >= 0.0:
        overrides["dropout"] = dropout_override
    if batch_size_override > 0:
        overrides["batch_size"] = batch_size_override
    if readout_bias_override is not None:
        overrides["readout_bias"] = bool(readout_bias_override)
    if n_override > 0:
        overrides["N"] = n_override
    if t_override > 0:
        overrides["T"] = t_override
    if y_in_override > 0:
        overrides["y_in"] = y_in_override
    if y_out_override > 0:
        overrides["y_out"] = y_out_override
    if h_override > 0.0:
        overrides["H"] = h_override
    if d_override > 0.0:
        overrides["D"] = d_override
    # ---- 二期新增：拓扑几何覆盖（哨兵语义与 CLI 一致） ----
    if topology_override:
        overrides["topology"] = topology_override
    if flow_axis_override:
        overrides["flow_axis"] = flow_axis_override
    if space_radius_override >= 0.0:
        overrides["space_radius"] = space_radius_override
    return _run_training_with_config(
        Config(**overrides), preset, max_batches, checkpoint_path, backup, tag, arch
    )


def _run_training_with_config(
    config: Config,
    preset: str,
    max_batches: int,
    checkpoint_path: str,
    backup: bool,
    tag: str,
    arch: str = "neuron3d",
) -> Dict[str, float]:
    """以**已构造好的配置**执行阶段 B 训练（`run_full_training` 的内部实现）。

    参数
    ----
    config : Config
        已生效的完整配置（由调用方负责覆盖与校验）。
    preset : str
        预设名（仅用于日志与 checkpoint 元数据）。
    max_batches : int
        每个 epoch 最多处理的 batch 数（0 表示全量）。
    checkpoint_path : str
        checkpoint 保存路径（空串表示默认路径）。
    backup : bool
        覆盖已有 checkpoint 前是否先备份为 `<path>.bak`。
    tag : str
        限批验证产物名标签。

    返回
    ----
    Dict[str, float]
        最终统计字典（含 checkpoint 绝对路径与总耗时）。
    """
    # 设备解析：auto -> 有 CUDA 用 CUDA，否则 CPU
    device = get_device(config.device)

    log_info("=" * 78)
    log_info(f"阶段 B 正式训练开始（preset={preset}, arch={arch}）")
    log_info(f"配置：{config.describe()}")
    log_info(f"实际设备：{device}")
    log_info("=" * 78)

    t0 = time.perf_counter()
    model, train_loader, test_loader = build_model_and_data(
        config, device, max_batches, arch
    )
    # 优化器：weight_decay > 0 时用 AdamW，否则维持既有 Adam（默认路径逐位不变）
    if config.weight_decay > 0.0:
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        log_info(f"优化器：AdamW(lr={config.lr}, weight_decay={config.weight_decay})")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        log_info(f"优化器：Adam(lr={config.lr})")
    # 学习率调度：cosine -> CosineAnnealingLR(T_max=epochs)，逐 epoch 步进
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    if config.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.epochs)
        )
        log_info(f"学习率调度：CosineAnnealingLR(T_max={max(1, config.epochs)})")
    elif config.lr_schedule != "none":
        raise ValueError(f"不支持的 lr_schedule={config.lr_schedule!r}")
    if config.grad_clip > 0.0:
        log_info(f"梯度裁剪：clip_grad_norm_(max_norm={config.grad_clip})")

    last_loss = float("nan")
    test_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        last_loss, _, n_batches = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            max_batches=max_batches,
            grad_clip=config.grad_clip,
        )
        test_acc = evaluate(model, test_loader, device, max_batches=max_batches)
        current_lr = float(optimizer.param_groups[0]["lr"])
        log_info(
            f"[epoch {epoch}/{config.epochs}] loss={last_loss:.4f} "
            f"test_acc={test_acc * 100:.2f}% lr={current_lr:.6f} (batches={n_batches})"
        )
        if scheduler is not None:
            scheduler.step()

    # ---- 保存 checkpoint（限批模式自动隔离到 _verify/，保护正式产物） ----
    save_path = resolve_checkpoint_path(checkpoint_path, max_batches, config, tag)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 覆盖前先备份既有产物（checkpoints/ 未被 git 跟踪，覆盖后无法按位恢复）
    backup_path = backup_existing_checkpoint(save_path, enabled=backup)
    if backup_path:
        log_info(f"[产物保护] 覆盖前已备份既有 checkpoint：{backup_path}")
    stats = model.get_connection_stats()
    # 二期拓扑统计（含 forward 逐轮记录的动态 2a 覆盖；MLP 路径返回空字典）
    topo_stats = model.get_topology_stats() if arch == "neuron3d" else {}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "connection_stats": stats,
            "topology_stats": topo_stats,
            "test_acc": test_acc,
            "epochs": config.epochs,
            "batches_per_epoch": (max_batches if max_batches > 0 else None),
            "preset": preset,
            "arch": arch,
            "topology": config.topology,
            "flow_axis": config.flow_axis,
            "space_radius": config.space_radius,
            "lr_schedule": config.lr_schedule,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "readout_bias": config.readout_bias,
            "grad_clip": config.grad_clip,
        },
        save_path,
    )
    elapsed = time.perf_counter() - t0
    log_info(f"checkpoint 已保存：{os.path.abspath(save_path)}（总耗时 {elapsed:.1f}s）")
    if max_batches > 0:
        log_info(
            f"[产物保护] 本次为限批验证跑（--max-batches={max_batches}），"
            f"已写入 {VERIFY_CHECKPOINT_DIR}；正式产物默认路径 {CHECKPOINT_PATH} 未被写入"
        )

    # ---- 最终参数统计 ----
    log_info("-" * 78)
    log_info("最终参数统计：")
    log_info(f"  架构                 ：{arch}")
    log_info(f"  可学习参数总数       ：{model.count_parameters()}")
    log_info(f"  连接稀疏度(密度)     ：{stats['sparsity']:.6f}")
    log_info(f"  零元素占比(zero_ratio)：{stats['zero_ratio']:.6f}")
    log_info(f"  边数 E               ：{int(stats['num_edges'])}")
    log_info(f"  平均出度             ：{stats['avg_out_degree']:.3f}")
    log_info(f"  tau                  ：{stats['tau']:.6f}")
    if topo_stats:
        # 二期：拓扑指纹（几何方向性 + 2a 覆盖 + 连通性）必须随产物一起落盘
        log_info(f"  拓扑                 ：{config.topology} / 流向轴 {config.flow_axis}"
                 f" / 球半径 {config.effective_space_radius:.6f}")
        log_info(f"  逆向边占比           ：{topo_stats['reverse_edge_ratio']:.6f}"
                 f"（{int(topo_stats['reverse_edges'])}/{int(topo_stats['num_edges'])}）")
        log_info(f"  流向轴 gap mean/min/max：{topo_stats['axis_gap_mean']:.4f} / "
                 f"{topo_stats['axis_gap_min']:.4f} / {topo_stats['axis_gap_max']:.4f}")
        log_info(f"  神经元流向轴高度     ：mean={topo_stats['neuron_axis_mean']:.4f}, "
                 f"min={topo_stats['neuron_axis_min']:.4f}, "
                 f"max={topo_stats['neuron_axis_max']:.4f}, "
                 f"std={topo_stats['neuron_axis_std']:.4f}")
        log_info(f"  静态 2a 覆盖         ：{topo_stats['connected_output_ratio']:.6f}")
        log_info(f"  动态 2a 覆盖(末轮)   ：{topo_stats['out_nonzero_coverage_last']:.6f}"
                 f"（全程均值 {topo_stats['out_nonzero_coverage_mean']:.6f}）")
        log_info(f"  弱连通分量数         ：{int(topo_stats['weak_components'])}")
        log_info(f"  最大分量占比         ：{topo_stats['largest_component_ratio']:.6f}")
    log_info(f"  最终 test_acc        ：{test_acc * 100:.2f}%")
    log_info(f"  优化器               ：{'AdamW' if config.weight_decay > 0 else 'Adam'}")
    log_info(f"  lr_schedule          ：{config.lr_schedule}")
    log_info(f"  dropout / readout_bias：{config.dropout} / {config.readout_bias}")
    log_info(f"  grad_clip            ：{config.grad_clip}")
    log_info(f"  checkpoint           ：{os.path.abspath(save_path)}")
    log_info("-" * 78)

    return {
        "test_acc": float(test_acc),
        "loss": float(last_loss),
        "sparsity": float(stats["sparsity"]),
        "zero_ratio": float(stats["zero_ratio"]),
        "tau": float(stats["tau"]),
        "num_edges": float(stats["num_edges"]),
        "params": float(model.count_parameters()),
        "checkpoint": os.path.abspath(save_path),
        "elapsed_s": float(elapsed),
    }


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口：``--smoke-test`` 只跑阶段 A，否则跑阶段 B 正式训练。

    参数
    ----
    argv : Optional[List[str]]
        命令行参数（None 表示使用 sys.argv）。

    返回
    ----
    int
        进程退出码：
        * `0` 成功（阶段 A 全部判据通过，或阶段 B 正常结束）；
        * `1` 阶段 A 验收失败；
        * `2` 命令行参数非法（`validate_override_args` 抛出的 `ValueError`，
          例如 `--y-in -1`）。

    说明
    ----
    覆盖类参数的校验与构造统一走 `validate_override_args` + `apply_overrides`，
    两条路径（冒烟 / 正式训练）共用同一实现，避免出现两套并存逻辑。
    """
    args = parse_args(argv)
    try:
        # 唯一校验入口：负数等非法取值必须**显式报错并以非 0 退出码结束**
        validate_override_args(args)
        # 线程设置：在构建模型/数据之前生效，使线程数影响全部后续计算
        apply_threads(args.threads)
        if args.smoke_test:
            ok = run_smoke_test(
                device_override=args.device,
                criterion_conflict=args.criterion_conflict,
                config_override=build_smoke_config(args),
                arch=args.arch,
            )
            return 0 if ok else 1
        run_full_training(
            config=apply_overrides(PRESETS[args.preset], args),
            preset=args.preset,
            max_batches=args.max_batches,
            checkpoint_path=args.checkpoint,
            backup=args.backup,
            tag=args.tag,
            arch=args.arch,
        )
    except ValueError as exc:
        log_error(f"参数校验失败：{exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())