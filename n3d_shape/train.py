"""N3D 神经元空间**形状变体**训练入口（球体 / 立方体 / 圆柱体 + FCC 规则堆积 + 两阶段双副本展开）。

形状 = 生长度量，不是裁剪掩码
----------------------------
本模块把神经元的空间形状从**球体**泛化为**立方体**与**圆柱体**，做法是替换 FCC 放置中
"从中心向外取最近 N 个"的**排序度量**（`sphere: ‖p‖2` / `cube: ‖p‖∞` /
`cylinder: max(‖p_xy‖2, |p_axis|/λ)`）。**不改连接判据、不改突触半球切分、不引入随机放置、
不改训练循环与数据管线。** `shape="sphere"` 为默认分支，必须与二期 `n3d_sphere` 张量级逐位一致。

两种运行方式
------------
* 阶段 A（冒烟测试）：``python n3d_shape/train.py --smoke-test``
  使用 `SMALL_CONFIG`，仅跑 1 个 batch 的前向 + 反向，逐条核对**形状感知判据**
  （见 `SMOKE_CRITERIA_DOC` 与 `run_smoke_test` 中 `checks` 列表的逐条构造）。
* 阶段 B（正式训练）：``python n3d_shape/train.py``
  使用 `DEFAULT_CONFIG`，训练 `epochs` 轮，逐 epoch 打印 loss 与 test_acc。

命令行参数（几何相关）
----------------------
--shape {sphere,cube,cylinder}
                        神经元空间形状（缺省沿用预设 sphere）。形状只改放置的生长度量。
--cyl-aspect R          圆柱长径比 λ = c/r（c 沿流向轴半高、r 横截面半径；缺省 1.0）。
                        **仅 --shape cylinder 生效**；其它形状显式给出会报错（拒绝静默无效参数）。
--flow-axis {x,y,z}     全局流向轴：输入突触取负半球（-axis）、输出突触取正半球（+axis）
--space-radius R        形状特征尺度：0/缺省 = 该形状公式下界 R_min；显式值须落在 [R_min, R_max]
--input-scope S         输入层驱动判据 any_isolated / all_isolated（缺省沿用预设）
--readout-scope S       读出判据 any_isolated / all_isolated（缺省沿用预设）
--placement P           神经元放置方式，当前仅 fcc（FCC 规则堆积）

数据集通用层（第 6 轮新增：数据集可插拔）
----------------------------------------
--dataset NAME          数据集来源：mnist（缺省，回归锚点）/ synthetic / npz / csv / json。
                        注册表见 `data.DATASET_SPECS`；`--dataset mnist` **委派既有
                        `get_mnist_loaders`**，签名 / 行为 / 加载顺序 / 归一化逐字不变。
--data-root PATH        MNIST 数据根目录（缺省 data/mnist；仅 mnist 使用）。
--dataset-path PATH     npz / csv / json 的数据文件路径（相对路径按当前工作目录解析）。
--input-dim D          覆盖输入维数（缺省 = 取数据集规格；与规格声明值冲突时报错）。
--output-dim C         覆盖类别数（缺省 = 取数据集规格；与规格声明值冲突时报错）。
--num-samples M         仅 synthetic：总样本数（0 = 取缺省 4000）。
--norm-mean / --norm-std
                        归一化统计量（两者必须同时给出；缺省 = 训练集现场统计并写日志）。
                        归一化口径与 `--dataset` 的取值一起写入运行日志，便于事后复核。

产物保护
--------
* 本模块产物目录为 `checkpoints/n3d_shape/`，与一期 `checkpoints/` 与二期
  `checkpoints/n3d_sphere/` **物理隔离**，两期既有产物不会被本模块的任何命令写入；
* 验证类运行（`--smoke-test` / `--max-batches > 0`）一律写入
  `checkpoints/n3d_shape/_verify/`，**绝不覆盖正式产物**；
* 限批与正式全量产物的文件名含 **shape（+圆柱长径比）** / flow_axis / 两个 scope / seed
  等配置指纹，不同配置互不覆盖（历史纠正：同名互覆导致筛选记录丢失；
  形状维度若不入指纹，N/H/D/seed/scope 相同的三种形状会**互相覆盖**）。
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

# 兼容"以脚本方式运行"（python n3d_shape/train.py）与"作为包导入"两种情形
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _reconfigure_stdio() -> None:
    """把 stdout / stderr 重配为 UTF-8（errors='replace'），消除 GBK 控制台崩溃（离朱 DEF-1）。

    背景：Windows 默认 stdout 编码常为 **gbk**，而本模块的日志/判据文本含大量中文与数学记号。
    一旦某个字符不在 GBK 码表内，`print` / `log_info` 会抛 `UnicodeEncodeError`
    （实测历史：`=>` U+21D2、`d` U+2202、`^2` U+00B2、`[!]` U+26A0、`-` U+2212），
    使**验收命令在默认控制台下退码 2 / 1** —— 这是**输出层**缺陷，不是业务逻辑缺陷。

    处置为**两道防线**（对外部调用者最稳）：
    1. 本函数在**入口**处把 stdout / stderr 重配为 UTF-8 且 `errors='replace'`
       （Python 3.7+ 才有 `reconfigure`，缺失时静默跳过）；
    2. 源码中**全部非 GBK 字符已替换为 ASCII 等价记号**（`=>` / `dL/dalpha` / `^2` / `[!]` …），
       故即使 1 未生效（例如把本模块当库导入且未调用入口），也不会再有输出层崩溃。

    参数：无。返回：None。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # 非文本流 / 老版本解释器：静默跳过（ASCII 替换已提供第二道防线）
            pass


_reconfigure_stdio()

try:
    from .config import (
        DATASET_CHOICES,
        DATASET_SPECS,
        Config,
        DEFAULT_CONFIG,
        FLOW_AXIS_CHOICES,
        GEO_FIELD_CHOICES,
        HIGHACC_CONFIG,
        PLACEMENT_CHOICES,
        SCOPE_CHOICES,
        SHAPE_CHOICES,
        SMALL_CONFIG,
    )
    from .data import MNIST_MEAN, MNIST_STD, build_dataloaders
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
        DATASET_CHOICES,
        DATASET_SPECS,
        Config,
        DEFAULT_CONFIG,
        FLOW_AXIS_CHOICES,
        GEO_FIELD_CHOICES,
        HIGHACC_CONFIG,
        PLACEMENT_CHOICES,
        SCOPE_CHOICES,
        SHAPE_CHOICES,
        SMALL_CONFIG,
    )
    from data import MNIST_MEAN, MNIST_STD, build_dataloaders  # type: ignore
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

# 预设名 -> 配置对象（--preset 使用；缺省为 default）
PRESETS: Dict[str, Config] = {
    "small": SMALL_CONFIG,
    "default": DEFAULT_CONFIG,
    "highacc": HIGHACC_CONFIG,
}

# 归一化参数的"未提供"哨兵（第 6 轮新增）：`None` 而非浮点值 —— 均值的合法取值含
# 0.0（把数据平移到原点附近是常见做法），若用 -1.0 之类浮点哨兵就会与合法值冲突。
# 因此 CLI 侧 `--norm-mean` / `--norm-std` 的 default 是 `None`，只判 `is not None`。
NORM_SENTINEL_NOTE: str = "norm_mean / norm_std 的哨兵是 None（0.0 是合法均值）"

# 阶段 A 验收阈值
GRAD_NORM_MIN: float = 0.0    # 所有可学习参数梯度范数必须 > 0
SMOKE_TIME_MAX: float = 120.0  # CPU 单 batch 前向+反向耗时上限（秒）
NN_DIST_TOL: float = 1e-5     # 最近邻距 = 2H 的容差
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# 本模块产物目录与一期物理隔离（一期产物 n3d_model_*.pt 位于 checkpoints/ 根）
MODULE_CHECKPOINT_DIR_NAME: str = "n3d_shape"
CHECKPOINT_DIR: str = os.path.join(PROJECT_ROOT, "checkpoints", MODULE_CHECKPOINT_DIR_NAME)
CHECKPOINT_PATH: str = os.path.join(CHECKPOINT_DIR, "model.pt")
# 验证类运行（冒烟测试 / 限批短跑）的独立写入目录，避免覆盖正式产物
VERIFY_CHECKPOINT_DIR: str = os.path.join(CHECKPOINT_DIR, "_verify")

# 新架构的冒烟判据清单（仅作文档与日志自检；实际判定在 run_smoke_test 内逐条构造）
# 文案一律使用 ASCII 编号（不用 [1][2][3] 等圈号）：Windows GBK 控制台无法编码圈号，
#    会让 --help/日志/退出码路径抛 UnicodeEncodeError（历史缺陷）。
SMOKE_CRITERIA_DOC: Tuple[str, ...] = (
    "1 前向输出形状 == [B, output_dim]（真实断言，非恒真）",
    "2 反向无错误（全部可学习参数都有梯度；缺失梯度记为 -1.0）",
    "3 参与 loss 的参数梯度范数 > 0（W_out 断言非零梯度列数落在 [1, |S_out|]）",
    "4 loss 非 NaN/Inf",
    "5 S_in 非空（阶段 1 真正被输入层驱动）",
    "6 S_out 非空（readout 真正有信号）",
    "7 神经元级连接数 == 去重后的神经元对数（同一神经元对只算一条连接）",
    "8 图为无环 DAG 且每条边严格上行（z_A < z_B）",
    "9 最近邻距恰恰等于 2H（FCC 规则堆积契约）",
    "10 形状特征尺度落在 [R_min, R_max] 内",
    "11 神经元级连接数 E > 0 且平均出度 > 0",
    "12 不存在 [N*y_out, N*y_in] 形状的权重张量（未 materialize dense 矩阵）",
    "13 readout 严格口径：h 非零列都属于 S_out，且 h 逐位 == a_up 掩码积",
    "14 阶段 2 递推顺序 == 流向轴升序（topo_matches_axis_order == 1）",
    "15 [形状感知] 形状生长度量与层数落地：shape 编码/λ/外接系数一致、"
    "选取度量 == 按形状度量取最近 N 个的实测值、层数 K == 唯一流向轴坐标数",
    "16 CPU 单 batch 前向+反向耗时 < 120s",
)


def dataset_name_parts(
    config: Config,
    baseline: Optional[Config] = None,
    short_circuit: bool = True,
) -> Tuple[str, str]:
    """返回产物名的**数据集命名段**：`(dataset_part, dims_part)`（第 6 轮新增，防撞名）。

    口径（`smoke_fingerprint` / `config_fingerprint` / `full_checkpoint_name` / 冒烟默认判定
    四处**必须逐字一致**）
    ------------------------------------------------------------------------------
    * `dataset_part = ""` 当 `config.dataset == "mnist"`（**缺省来源不加段**，保证既有
      120 个 `full_*` / `verify_*` 产物名**逐字不变**）；否则为 `_ds{name}`；<br>
    * `dims_part = ""` 当 `(input_dim, output_dim)` 等于**该数据集规格的缺省值**；
      否则为 `_d{D}x{C}`。

    "MNIST 且维度取规格缺省"这一组合因此**恒为空段** —— 这正是"既有产物名逐字不变"的
    形式化条件，并由 `assert_dataset_name_distinguishable` 与冒烟默认判定共同守护。

    `baseline`（可选，用于**短路口径**）
    ---------------------------------
    当调用方传入 `baseline=SMALL_CONFIG`（冒烟默认判定路径）且
    `config.dataset == baseline.dataset` 且**两者解析后的维度相同**时，直接返回
    `("", "")`。理由：`Config.__init__` 的两个入口（`Config()` 缺省构造 vs
    `apply_overrides` 用 `to_dict()` 重建，此时 `input_dim` / `output_dim` 是**显式 int**）
    会得到**语义完全相同**的配置对象。既然配置语义相同，产物名也必须逐字相同 ——
    否则"裸跑冒烟"与"显式传 `--input-dim 784` 的冒烟"会写同一份默认产物的两个名字，
    与 `apply_overrides` 的"值等价即复用基线对象"纪律冲突。**默认开启该短路口径**；
    数据集与维度确有差异时照常生成段。

    参数
    ----
    config : Config
        本次生效的配置。
    baseline : Optional[Config]
        可选的基准确认配置（通常 `SMALL_CONFIG`）；`None` 表示不做短路。
    short_circuit : bool
        是否允许上述短路口径（缺省 True）。

    返回
    ----
    Tuple[str, str]
        `(dataset_part, dims_part)`，例如 `("_dssynthetic", "")` 或 `("", "_d64x4")`。
    """
    if short_circuit and baseline is not None:
        if str(config.dataset) == str(baseline.dataset) and (
            int(config.effective_input_dim) == int(baseline.effective_input_dim)
            and int(config.effective_output_dim) == int(baseline.effective_output_dim)
        ):
            return "", ""
    dataset_part = "" if str(config.dataset) == "mnist" else f"_ds{config.dataset}"
    spec = DATASET_SPECS.get(str(config.dataset))
    dims_part = ""
    if spec is not None and (
        int(config.effective_input_dim) != int(spec.input_dim)
        or int(config.effective_output_dim) != int(spec.num_classes)
    ):
        dims_part = f"_d{int(config.effective_input_dim)}x{int(config.effective_output_dim)}"
    return dataset_part, dims_part


def assert_dataset_name_distinguishable() -> None:
    """**可区分性断言**：同配置不同数据集的产物名必须互不相同（第 6 轮新增）。

    口径
    ----
    在同一份基线（`SMALL_CONFIG`）上只改 `dataset`，逐个数据集取
    `smoke_fingerprint` / `config_fingerprint` / `full_checkpoint_name` **三处指纹**，
    断言"任意两个数据集的指纹两两不同"。这条断言把"数据集维度漏进指纹"这一
    必然导致**同名互覆**的缺陷挡在入口处（与形状 / `fc_dim` / `geo_field` 的历史
    纠正记录同源）。

    参数：无。

    返回
    ----
    None

    异常
    ------
    ValueError
        任一数据集的任一指纹为空，或两个数据集在某处指纹相同（即命名段缺失）时抛出，
        报文列出冲突的数据集对与指纹值。
    """
    variants: Dict[str, Tuple[str, str, str]] = {}
    for name in DATASET_CHOICES:
        spec = DATASET_SPECS[name]
        # [!] `npz` / `csv` / `json` 的注册表维度是**占位 0**
        #     （"由数据文件现场决定"），因此构造这类配置必须显式给出维度。
        #     这里刻意给出与 mnist 相同的 (784, 10)：本断言只关心"数据集维度是否进指纹"，
        #     且这样能让"命名段本身"成为唯一的差异来源，判据最锐利。
        cfg = Config(
            **{
                **SMALL_CONFIG.to_dict(),
                "dataset": name,
                "input_dim": int(spec.input_dim) if int(spec.input_dim) > 0 else 784,
                "output_dim": (
                    int(spec.num_classes) if int(spec.num_classes) > 0 else 10
                ),
            }
        )
        variants[name] = (
            smoke_fingerprint(cfg),
            config_fingerprint(cfg, 1),
            full_checkpoint_name(cfg),
        )
    for name, fp in variants.items():
        if not all(fp):
            raise ValueError(
                f"[可区分性断言] dataset={name!r} 的产物名指纹为空：{fp}"
                f"（数据集命名段缺失，会导致同名互覆）"
            )
    names = sorted(variants)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if variants[a] == variants[b]:
                raise ValueError(
                    f"[可区分性断言] 同配置不同数据集的产物名相同：dataset={a!r} 与 "
                    f"dataset={b!r} 的三处指纹均为 {variants[a]}（数据集维度未进指纹）"
                )
    return None


def smoke_fingerprint(config: Config) -> str:
    """返回冒烟产物的配置指纹段（容量 + 几何 + **形状** + 判据 + 训练规模维度）。

    命名段：
    `N{N}_y{y_in}x{y_out}_H{H}_D{D}_pl{placement}_ax{axis}_is{scope}_rs{scope}`
    `_bs{batch_size}[_fc{n}]_nosyn[_R{space_radius}]_s{seed}`，**非 sphere 形状前置
    `{shape_tag}_`**。`_nosyn` 为第 4 轮新增的**格式段（恒定插入）**，紧随 `_fc{n}` 段之后。

    设计动机（皋陶审查 info 项）：冒烟产物原先只由 `flow_axis` / 两个 scope / `arch`
    决定，改变 `N` / `y` / `H` / `D` / `seed` 会写同一文件而互相覆盖；后续把容量与几何
    维度纳入指纹。

    [!] 历史缺陷（离朱第 10 轮 D1）：`is_default_smoke`（决定是否退化为默认冒烟产物名）已把
    `batch_size` 与 `space_radius` 纳入判定，但指纹**未**包含这两个维度 —— 于是
    `--batch-size 64`、`--space-radius 0.9`、以及两者同时覆盖的配置会落回**同一个文件名**
    而互相覆盖（实测后者会重写前者的取证产物）。指纹必须与默认判定**维度严格对齐**：
    默认判定包含的维度，指纹也必须包含。故此处补入 `batch_size`（始终）与
    `space_radius`（非 0 时，0 表示取 R_min 且已由判定覆盖）。

    [!] **形状维度（本模块新增的防撞名硬要求）**：指纹**必须**含形状（圆柱还须含长径比），
    否则 `N / H / D / seed / scope` 相同的三种形状会**互相覆盖**同一份取证产物 ——
    正是历史纠正记录中 `verify_<bpe>.pt` 同名互覆导致取证失效的同一类缺陷。
    形状段放在指纹**开头**，便于目视区分；`sphere` 分支仍带 `shapesphere_` 段，
    但 `is_default_smoke` 退化路径（默认组合 → `smoke_nosyn.pt`，见
    `smoke_checkpoint_path`）不受影响。

    参数
    ----
    config : Config
        冒烟运行实际生效的配置。

    返回
    ----
    str
        指纹字符串（不含 arch，arch 由 `smoke_checkpoint_path` 单独拼接）。
    """
    radius_part = (
        f"_R{config.space_radius:g}" if float(config.space_radius) > 0.0 else ""
    )
    # [!] fc_dim（第 3 轮新增）**必须入指纹**：否则同 N/seed/scope 的不同 fc_dim
    #     会写同一个文件名而互相覆盖（历史纠正 #1 的同类缺陷）。
    #     口径：`fc_dim == 0`（默认关闭）**不加段**（该段口径与改动前逐字相同；
    #     但产物名整体因下一行的 `_nosyn` 段而不再与旧格式产物同名）。
    fc_part = f"_fc{int(config.fc_dim)}" if int(config.fc_dim) != 0 else ""
    # [!] `nosyn`（第 4 轮新增）**格式段，恒定插入**：产物不再落盘 8 个突触类张量
    #     （`persistent=False`），故产物名必须与旧格式产物可区分（防"新旧格式同名互覆"）。
    #     口径：**无条件下加段**；与 `_fc{n}` 段的相对位置 = `_nosyn` **紧随 `_fc{n}` 之后**
    #     （`fc_dim == 0` 时紧随 scope 段之后）、位于 `_s{seed}` 之前。同一口径已同步
    #     `config_fingerprint` / `full_checkpoint_name` 与 README §19。
    nosyn_part = "_nosyn"
    # ---- 几何权重场（第 5 轮新增）：`geo_field` 族 ----
    # [!] 命名段 `_geo{mode}`（**仅 `geo_field != "none"` 时插入**）：否则同 N/seed/scope
    #     的"关闭"与"开启"两种产物会写同一个文件名而互相覆盖（历史纠正 #1 的同类缺陷）。
    #     口径（三处指纹**必须逐字一致**）：
    #       位置 = 紧随 `_fc{n}` 段之后（`fc_dim == 0` 时紧随 scope 段之后）、
    #              **`_nosyn` 段之前**、`_s{seed}` 之前；
    #       `geo_field == "none"` **不加段** -> 既有产物名逐字不变。
    #     段值含 RBF 阶数 `k`：`_geoadditive_k12`（同档不同 k 是两套不同几何场，必须可区分）。
    geo_part = (
        (
            f"_geo{config.geo_field}_k{int(config.geo_rbf_k)}"
            + ("_sd" if bool(config.geo_signed_delta) else "")
            # [!] `_a{alpha_init:g}`：离朱 DEF-7 —— `geo_alpha_init` 原先**未进指纹**，
            #     使 `--geo-alpha-init 0.4` 与默认 `1.0` 生成**同名产物**并**静默互覆**
            #     （实测 SHA `01e56cb7…` -> `7b89b4ef…`）。口径：**仅非默认值（!= 1.0）
            #     时插入**，故默认组合的产物名逐字不变（既有产物名与 README 口径不受影响）。
            + (f"_a{float(config.geo_alpha_init):g}"
               if float(config.geo_alpha_init) != 1.0 else "")
        )
        if str(config.geo_field) != "none" else ""
    )
    # ---- 数据集通用层（第 6 轮新增）：`_ds{name}` / `_d{D}x{C}` 两段 ----
    # [!] 命名段位置口径（四处逐字一致）：`..._rs{scope}` -> [_fc{n}] -> [_geo...]
    #     -> [_ds{name}] -> [_d{D}x{C}] -> _nosyn -> [_R{space_radius}] -> _s{seed}。
    #     `mnist` + 规格缺省维度时两段**均为空** => 既有产物名逐字不变（零回归）。
    ds_part, dim_part = dataset_name_parts(config)
    return (
        f"{config.shape_tag()}_"
        f"N{config.N}_y{config.y_in}x{config.y_out}"
        f"_H{config.H:g}_D{config.D:g}"
        f"_pl{config.placement}"
        f"_ax{config.flow_axis}"
        f"_is{scope_abbrev(config.input_scope)}"
        f"_rs{scope_abbrev(config.readout_scope)}"
        f"_bs{config.batch_size}"
        f"{fc_part}"
        f"{geo_part}"
        f"{ds_part}"
        f"{dim_part}"
        f"{nosyn_part}"
        f"{radius_part}"
        f"_s{config.seed}"
    )


def smoke_checkpoint_path(
    flow_axis: str,
    input_scope: str,
    readout_scope: str,
    arch: str = "neuron3d",
    fingerprint: str = "",
) -> str:
    """返回冒烟测试产物的路径（文件名含配置指纹，防止不同配置互覆）。

    命名规则：
    * 默认配置（`neuron3d` + `flow_axis=z` + 两个 `any_isolated` + 无指纹）用
      默认产物名 `smoke_nosyn.pt`（原历史名 `smoke.pt`）—— 第 4 轮加入 `_nosyn`
      格式段（8 个突触类张量 `persistent=False`、不落盘），使**新版瘦身产物与旧版
      同名产物分开留痕**，旧产物上的冻结 SHA256 断言不被触碰；`is_default_smoke`
      判定的**维度不变**（`_nosyn` 是格式常量，与 config 无关），其职责仍是
      **防止非默认配置静默覆盖**该默认产物；
    * 其它情形：`smoke[_ar{arch}]_{fingerprint}_ax{axis}_is{scope[:3]}_rs{scope[:3]}.pt`
      —— 传入 `fingerprint`（见 `smoke_fingerprint`）时容量与几何维度也进文件名。

    **arch 维度**（离朱第 8 轮实测 M3，已修复）：冒烟产物名原先只由 flow_axis 与两个
    scope 决定，`--arch mlp` 与主模型会写同一个产物名而互相覆盖（先跑 mlp 冒烟
    会让 `verify_sphere_dag.py` 的 R7 读到 mlp 产物、缺 `topology_stats` 而误报 FAIL）。
    现把 arch 纳入指纹：非 `neuron3d` 时文件名插入 `_ar{arch}`，两种架构各自留痕。

    参数
    ----
    flow_axis : str
        全局流向轴（x / y / z）。
    input_scope : str
        输入层驱动判据（any_isolated / all_isolated）。
    readout_scope : str
        读出判据（any_isolated / all_isolated）。
    arch : str
        架构（`neuron3d` / `mlp`）；缺省 `neuron3d`，保持历史命名不变。
    fingerprint : str
        容量/几何维度指纹（通常来自 `smoke_fingerprint(config)`）；空串表示不附加。

    返回
    ----
    str
        冒烟测试 checkpoint 的绝对路径。
    """
    if (
        flow_axis == "z"
        and input_scope == "any_isolated"
        and readout_scope == "any_isolated"
        and arch == "neuron3d"
        and not fingerprint
    ):
        # [!] 第 4 轮：默认产物名由历史名 `smoke.pt` 改为 `smoke_nosyn.pt`（加格式段），
        #     使新版瘦身产物与旧版同名产物**不互相覆盖**（旧产物与其上的冻结 SHA256
        #     断言保持可复核）。这是**名字变更**，`is_default_smoke` 的判定维度不变。
        return os.path.join(VERIFY_CHECKPOINT_DIR, "smoke_nosyn.pt")
    arch_part = "" if arch == "neuron3d" else f"_ar{arch}"
    if fingerprint:
        # 指纹本身已含 N/y/H/D/pl/ax/is/rs/s 全部维度，无需再拼接一遍
        return os.path.join(VERIFY_CHECKPOINT_DIR, f"smoke{arch_part}_{fingerprint}.pt")
    return os.path.join(
        VERIFY_CHECKPOINT_DIR,
        f"smoke{arch_part}_ax{flow_axis}_is{input_scope[:3]}_rs{readout_scope[:3]}.pt",
    )


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
        flow_axis / space_radius / input_scope / readout_scope / placement 等字段。
    """
    parser = argparse.ArgumentParser(
        description="N3D 神经元空间形状变体（球体/立方体/圆柱体 + FCC 规则堆积）训练入口"
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
            "架构：neuron3d（默认，球形分层 DAG 主模型）/ mlp（对照基线 input->hidden->out）。"
            "mlp 走**完全相同**的训练循环、优化器、调度器、梯度裁剪与评估代码，"
            "仅替换模型构造，用于判定瓶颈在架构还是在数据。"
        ),
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=sorted(PRESETS.keys()),
        help="配置预设：small / default / highacc（缺省 default）",
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
    parser.add_argument("--lr", type=float, default=0.0, help="覆盖学习率（默认取预设值）")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=-1.0,
        help="覆盖权重衰减（> 0 时优化器切换为 AdamW；默认取预设值）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=0, help="覆盖批大小（默认取预设值）"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="覆盖神经元数量 N（默认取预设值）",
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
        help=(
            "覆盖神经元（突触云）半径 H（> 0；默认取预设值）。"
            "FCC 晶格常数 a=2*sqrt(2)*H，最近邻距恒为 2H"
        ),
    )
    parser.add_argument(
        "--d",
        type=float,
        default=0.0,
        help=(
            "覆盖连接距离阈值 D（> 0；越大多连接越多、边数 E 越大；默认取预设值）。"
            "**硬约束 D <= H**（连接半径不得超过接收/发送范围半径 H），D > H 会在构造"
            "配置时直接报错；D 过小会触发连通性下限校验（要求 E >= N、层数 >= 2、"
            "|S_in| >= 1、|S_out| >= 1）。注意：FCC 放置下最近邻距为 2H，D 明显小于 2H "
            "时几乎无连接"
        ),
    )
    # ---- 几何 CLI ----
    parser.add_argument(
        "--flow-axis",
        type=str,
        default="",
        choices=["", *FLOW_AXIS_CHOICES],
        help=(
            "全局流向轴 x / y / z（缺省 = 沿用预设 z）：输入突触取负半球 (-axis)、"
            "输出突触取正半球 (+axis)；即流向定义为从 -axis 指向 +axis"
        ),
    )
    parser.add_argument(
        "--space-radius",
        type=float,
        default=-1.0,
        help=(
            "形状特征尺度（哨兵 -1 = 未提供，沿用预设 0）。0 表示取该形状的公式下界 R_min"
            "（sphere: H*(N/0.7405)^(1/3) / cube: H*(N*pi/(6*0.7405))^(1/3) / "
            "cylinder: H*(N/(pi*lambda*0.7405))^(1/3)*2^(2/3)）；显式指定时必须落在 "
            "[R_min, R_max] 内，越界在 Config 构造期报错。语义随形状而变：球 = 半径、"
            "立方体 = 外接立方体半边长、圆柱 = 横截面半径（半高 = lambda * R）"
        ),
    )
    parser.add_argument(
        "--input-scope",
        type=str,
        default="",
        choices=["", *SCOPE_CHOICES],
        help=(
            "输入层驱动判据（缺省 = 沿用预设 any_isolated）：any_isolated = 神经元有 >=1 个"
            "输入突触孤立即进入 S_in；all_isolated = 全部 y_in 个输入突触都孤立才进入 S_in"
        ),
    )
    parser.add_argument(
        "--readout-scope",
        type=str,
        default="",
        choices=["", *SCOPE_CHOICES],
        help=(
            "读出判据（缺省 = 沿用预设 any_isolated）：any_isolated = 有 >=1 个输出突触孤立即"
            "进入 S_out；all_isolated = 全部 y_out 个输出突触都孤立才进入 S_out"
        ),
    )
    parser.add_argument(
        "--placement",
        type=str,
        default="",
        choices=["", *PLACEMENT_CHOICES],
        help="神经元放置方式（缺省 = 沿用预设 fcc）。当前仅支持 FCC 规则堆积",
    )
    # ---- 数据集通用层 CLI（第 6 轮新增；缺省 mnist = 回归锚点）----
    # [!] 这 8 个参数**全部**必须纳入 `validate_override_args`（取值合法性）
    #     与 `apply_overrides`（真正生效）+ `build_smoke_config` 的 explicit 判定，
    #     否则会被**静默丢弃**（与 `--shape` / `--fc-dim` / `--geo-field` 的历史陷阱同源）。
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        choices=["", *DATASET_CHOICES],
        help=(
            "数据集来源（缺省 = 沿用预设 mnist = 零回归锚点）："
            "mnist（IDX，委派既有 get_mnist_loaders，行为逐字不变）/ "
            "synthetic（确定性生成、非线性可分、不联网）/ "
            "npz（X[M,D] / y[M]）/ csv（末列标签）/ "
            "json（{\"X\": [[...]], \"y\": [...]} 对象形态 + .jsonl 逐行样本对象）。"
            "输入维数与类别数的缺省值由注册表 data.DATASET_SPECS 单一解析点给出。"
            "**空串 = 不覆盖**（与 --shape / --input-scope 同一哨兵口径）"
        ),
    )
    parser.add_argument(
        "--data-root",
        dest="data_root",
        type=str,
        default="",
        help="MNIST 数据根目录（缺省 = 沿用预设 data/mnist；仅 --dataset mnist 使用）",
    )
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        type=str,
        default="",
        help=(
            "数据集文件路径（npz / csv / json 必需；相对路径按当前工作目录解析）。"
            "json 支持 .json 对象形态与 .jsonl / .ndjson 逐行样本对象两种形态"
        ),
    )
    parser.add_argument(
        "--input-dim",
        dest="input_dim",
        type=int,
        default=None,
        help=(
            "输入维数（缺省 = None，取数据集规格；**必须 > 0**）。"
            "显式给出且与数据集规格声明值冲突时立即报错（拒绝静默不一致）"
        ),
    )
    parser.add_argument(
        "--output-dim",
        dest="output_dim",
        type=int,
        default=None,
        help=(
            "类别数（缺省 = None，取数据集规格；**必须 > 0**）。"
            "显式给出且与数据集规格声明值冲突时立即报错（拒绝静默不一致）"
        ),
    )
    parser.add_argument(
        "--num-samples",
        dest="num_samples",
        type=int,
        default=0,
        help="仅 --dataset synthetic：总样本数（0 = 取缺省 4000；必须 >= 0）",
    )
    parser.add_argument(
        "--norm-mean",
        dest="norm_mean",
        type=float,
        default=None,
        help=(
            "归一化均值（缺省 = None，由训练集现场统计并写日志）。"
            "**必须与 --norm-std 同时给出**（只给一个直接报错）。"
            "缺省会覆盖为该值；注意哨兵不是浮点值而是 None（0.0 是合法均值）"
        ),
    )
    parser.add_argument(
        "--norm-std",
        dest="norm_std",
        type=float,
        default=None,
        help=(
            "归一化标准差（缺省 = None，由训练集现场统计；给出时必须 > 0）。"
            "**必须与 --norm-mean 同时给出**。注意哨兵不是浮点值而是 None"
        ),
    )
    # ---- 形状 CLI（本模块新增维度；缺省空串 = 沿用预设 sphere）----
    # [!] 帮助文案一律 ASCII 数学记号（`||p||2` / `||p||inf` / `|p_axis|`）：
    #    Windows GBK 控制台无法编码 `‖`（U+2016）、下标 `2`（U+2082）、`∞`（U+221E）
    #    等字符，会让 `--help` 抛 UnicodeEncodeError（该模块已有同类历史缺陷记录）。
    parser.add_argument(
        "--shape",
        type=str,
        default="",
        choices=["", *SHAPE_CHOICES],
        help=(
            "神经元空间形状（缺省 = 沿用预设 sphere）：sphere = 球体（回归锚点）/ "
            "cube = 立方体 / cylinder = 圆柱体。形状只替换 FCC 放置中"
            "\"从中心向外取最近 N 个\"的**生长度量**"
            "（sphere: ||p||2 / cube: ||p||inf / cylinder: max(||p_xy||2, |p_axis|/lambda)），"
            "不改连接判据、不改突触半球切分、不引入随机放置。"
            "**形状会改变层数 K（架构深度）与 E / |S_in| / |S_out| / 参数量。**"
        ),
    )
    parser.add_argument(
        "--cyl-aspect",
        dest="cyl_aspect",
        type=float,
        default=-1.0,
        help=(
            "圆柱长径比 lambda = c / r（c = 沿流向轴半高、r = 横截面半径；哨兵 -1 = 未提供，"
            "沿用预设 1.0）。**仅 --shape cylinder 生效**：显式给出该参数但形状不是 "
            "cylinder 时会报错并退出码 2（拒绝静默无效参数）。lambda < 1 为扁平圆柱、"
            "lambda > 1 为细高圆柱；极端 lambda 会使形状尺寸窗口 [R_min, R_max] 变窄甚至为空"
            "（构造期报错）。"
        ),
    )
    # ---- fc_dim（第 3 轮新增）：两端全连接包裹开关 ----
    # [!] `default=None` 表示"未提供"，与 `0`（= 关闭）**必须区分**：
    #     若用 0 作哨兵，则"显式要求关闭"与"未提供"无法区分，无法做 explicit 判定。
    parser.add_argument(
        "--fc-dim",
        dest="fc_dim",
        type=int,
        default=None,
        help=(
            "两端全连接包裹开关（缺省 = None，不覆盖；缺省预设值为 0 = 关闭）："
            "0 = 关闭（默认，走现状路径，参数集与数值逐位不变）；"
            "-1 = 宽度跟随 N（两端有效宽度 H = N）；> 0 = 显式宽度 H = 该值。"
            "启用后结构为 x -> Linear(784->H)+b+ReLU -> 投影 P(|S_in|,H) -> "
            "N3D 核心（不变） -> h = a_up[S_out]（索引收集） -> "
            "Linear(|S_out|->H)+b+ReLU -> Linear(H->10)+b -> logits。"
            "启用时不再创建 W_in / W_out / W_out_bias。仅 --arch neuron3d 生效。"
        ),
    )
    parser.add_argument(
        "--geo-field",
        dest="geo_field",
        type=str,
        default="",
        choices=["", *GEO_FIELD_CHOICES],
        help=(
            "几何权重场开关（缺省 = 沿用预设 none；形状 = 生长度量 + 几何权重场）。"
            "none = 关闭（默认）：**连几何特征都不构造**，buffer 不注册、参数不创建，"
            "代码路径与改动前逐位一致；"
            "additive = RBF 加性档（本批实现的唯一档）：w_e = w_free[e] + "
            "alpha*(sum_k c_k*phi_k(feat_e) + c_0)，c 零初始化（故初始前向与关闭路径"
            "逐位相同）、alpha 初值 = --geo-alpha-init 且可学习；"
            "class_tied / mlp = **枚举已接受但本批未实现** —— 构造期显式报错（不静默降级）。"
            "几何场只改**权重取值**，不改连接判据、不改突触半球切分、不引入随机放置、"
            "不改训练循环与数据管线。仅 --arch neuron3d 生效。"
        ),
    )
    parser.add_argument(
        "--geo-rbf-k",
        dest="geo_rbf_k",
        type=int,
        default=0,
        help=(
            "RBF 基函数个数 k（> 0；0 = 不覆盖、沿用预设 12）。基中心 = 逐维分位点、"
            "宽度 = 逐维相邻中心间距均值（下限 5e-2），全部由确定性算法算出、无额外随机数"
            "消耗；详见 README 的「几何权重场（geo_field）」节"
        ),
    )
    parser.add_argument(
        "--geo-alpha-init",
        dest="geo_alpha_init",
        type=float,
        default=-1.0,
        help=(
            "几何场增益 alpha 的初值（>= 0；哨兵 -1 = 不覆盖、沿用预设 1.0）。"
            "注意 c 零初始化，故该值不影响开关开启时的初始前向"
        ),
    )
    parser.add_argument(
        "--seed",
        dest="seed_override",
        type=int,
        default=0,
        help=(
            "覆盖随机种子（默认沿用预设的 seed=42）。神经元位置由 FCC 规则堆积确定、"
            "**与 seed 无关**；seed 只影响突触采样（进而影响边集与边数）、参数初始化与数据打乱。"
            "负数报错、0 表示不覆盖"
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
            "checkpoint 保存路径（默认 checkpoints/n3d_shape/model.pt）。"
            "冒烟测试与 --max-batches 限批模式不会写入该路径，"
            "而是写入 checkpoints/n3d_shape/_verify/ 下，避免覆盖正式产物。"
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
        架构选择：`"neuron3d"`（默认，球形分层 DAG 主模型）或 `"mlp"`（对照基线）。
        **这是全流程中唯一按 arch 分支的地方**——训练循环、优化器、调度器、
        梯度裁剪与评估代码在两条路径下完全共用同一份实现。

    返回
    ----
    Tuple[nn.Module, DataLoader, DataLoader]
        (model, train_loader, test_loader)。

    数据来源（第 6 轮修订）
    ----------------------
    原先此处的调用点硬编码 `get_mnist_loaders(...)`，现改为通用入口
    `data.build_dataloaders(...)`。`config.dataset == "mnist"` 时该入口**委派**
    `get_mnist_loaders`（参数与返回顺序逐字不变），故 MNIST 路径零回归；
    其余来源（synthetic / npz / csv / json）由同一入口按注册表分派。
    解析后的数据集规格写入运行日志（含归一化口径）。

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
    # ---- 数据：唯一通用入口（mnist 分支委派 get_mnist_loaders，行为逐字不变）----
    train_loader, test_loader, data_spec = build_dataloaders(
        dataset=str(config.dataset),
        batch_size=config.batch_size,
        data_root=config.data_root,
        dataset_path=str(config.dataset_path or ""),
        num_workers=config.num_workers,
        seed=config.seed,
        input_dim=config.__dict__["input_dim"],
        output_dim=config.__dict__["output_dim"],
        num_samples=int(config.num_samples),
        norm_mean=config.norm_mean,
        norm_std=config.norm_std,
    )
    log_info(
        f"数据集装配完成：{data_spec.describe()}；"
        f"生效维度 input_dim={int(config.effective_input_dim)}, "
        f"output_dim={int(config.effective_output_dim)}（{config.dims_source}）"
    )

    # ---- 设备一致性断言（装配后立即校验，尽早暴露设备错误） ----
    for name, p in model.named_parameters():
        if p.device != device:
            raise RuntimeError(f"设备不一致：参数 {name} 在 {p.device}，期望 {device}")
    for name, b in model.named_buffers():
        if b.device != device:
            raise RuntimeError(f"设备不一致：buffer {name} 在 {b.device}，期望 {device}")
    probe_x, probe_y = next(iter(train_loader))
    if probe_x.device != device or probe_y.device != device:
        raise RuntimeError(
            f"设备不一致：首个 batch 数据在 {probe_x.device}/{probe_y.device}，期望 {device}"
        )
    log_info(
        f"设备一致性探针已通过（已校验 参数/buffer/首个 batch 均在 {device}）；"
        f"单 epoch batch 数 = len(train_loader) = {len(train_loader)}"
    )

    stats = model.get_connection_stats()
    if arch == "mlp":
        log_info(
            f"模型装配完成（arch=mlp 对照基线）：input_dim={config.input_dim}, "
            f"hidden_dim={config.hidden_dim}, output_dim={config.output_dim}, "
            f"可学习参数={model.count_parameters()}, 设备={device}"
        )
    else:
        topo_stats = model.get_topology_stats()
        log_info(
            f"模型装配完成：N={config.N}, y_in={config.y_in}, y_out={config.y_out}, "
            f"placement={config.placement}, flow_axis={config.flow_axis}, "
            f"space_radius={config.effective_space_radius:.6f} "
            f"[R_min={config.min_space_radius:.6f}, R_max={config.max_space_radius:.6f}], "
            f"input_scope={config.input_scope}, readout_scope={config.readout_scope}, "
            f"E(神经元级连接)={int(stats['num_edges'])}, "
            f"avg_out_degree={stats['avg_out_degree']:.4f}, max_out_degree={int(stats['max_out_degree'])}, "
            f"avg_in_degree={stats['avg_in_degree']:.4f}, max_in_degree={int(stats['max_in_degree'])}, "
            f"可学习参数={model.count_parameters()}, 设备={device}"
        )
        log_info(
            f"几何/判据指纹：FCC 晶格常数={topo_stats['lattice_constant']:.6f}，"
            f"最近邻距={topo_stats['nearest_neighbour_dist']:.9f}（应=2H={2.0 * config.H:.9f}），"
            f"放置半径={topo_stats['placement_radius']:.6f}，"
            f"流向轴高度 mean={topo_stats['neuron_axis_mean']:.4f} / "
            f"min={topo_stats['neuron_axis_min']:.4f} / max={topo_stats['neuron_axis_max']:.4f}，"
            f"S_in={int(topo_stats['num_in_scope'])}（孤立输入突触 "
            f"{int(stats['isolated_input_syn'])}/{config.n_input_syn}），"
            f"S_out={int(topo_stats['num_out_scope'])}（孤立输出突触 "
            f"{int(stats['isolated_output_syn'])}/{config.n_output_syn}），"
            f"双副本神经元数={int(topo_stats['dual_copy_count'])}"
        )
    if max_batches > 0:
        log_info(f"本轮每 epoch 仅处理前 {max_batches} 个 batch（--max-batches 限制）")
    return model, train_loader, test_loader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> float:
    """在给定数据集上计算分类准确率。

    参数
    ----
    model : nn.Module
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
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_batches: int = 0,
    collect_grads: bool = False,
    grad_clip: float = 0.0,
) -> Tuple[float, Dict[str, float], int, torch.Tensor, torch.Tensor]:
    """训练一个 epoch。

    参数
    ----
    model : nn.Module
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
        梯度范数裁剪阈值；> 0 时在 backward() 与 step() 之间执行 `clip_grad_norm_`。

    返回
    ----
    Tuple[float, Dict[str, float], int, torch.Tensor, torch.Tensor]
        (平均 loss, 梯度范数字典, 实际处理 batch 数, 末批输入 x, 末批 logits)。
        未收集梯度（`collect_grads=False`）时字典为空、末批 x/logits 返回空张量
        （阶段 A 用它们做"输出形状 == [B, output_dim]"的真实断言）。

    异常
    ------
    RuntimeError
        loss 为 NaN/Inf 时抛出（附带 epoch/batch 上下文）。
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    grad_norms: Dict[str, float] = {}
    last_logits: torch.Tensor = torch.empty(0)
    last_x: torch.Tensor = torch.empty(0)
    for i, (x, y) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)                        # 前向：阶段 1 -> 阶段 2（单遍逐层递推）-> readout
        loss = F.cross_entropy(logits, y)         # 分类损失
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"loss 出现 NaN/Inf：epoch={epoch}, batch={i}, loss={loss.item()}"
            )
        loss.backward()                           # 反向：梯度沿两阶段 + 双副本回传
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        if collect_grads and (max_batches > 0 and i == max_batches - 1):
            # 阶段 A：在最后一个已反向的 batch 上收集全部可学习参数的梯度范数
            # 注意：若启用梯度裁剪，此处收集到的是**裁剪前**的原始梯度范数；
            # 阶段 A 默认不使用裁剪（HIGHACC 才用），故不影响验收语义。
            grad_norms = tensor_grad_norms(model)
            last_logits = logits.detach()
            last_x = x.detach()
        if (i + 1) % 100 == 0:
            log_info(f"epoch {epoch} | batch {i + 1} | running_loss={total_loss / n_batches:.4f}")
    if n_batches == 0:
        raise RuntimeError("训练集为空：没有任何 batch 被处理")
    return total_loss / n_batches, grad_norms, n_batches, last_x, last_logits


def validate_override_args(args: argparse.Namespace) -> None:
    """校验全部"覆盖类"命令行参数的取值范围（**唯一校验入口**）。

    语义（必须严格遵守，`0` = 不覆盖）
    ---------------------------------
    * 取值范围型参数（`--n / --y-in / --y-out / --h / --d / --seed`）：
      **负数一律 `raise ValueError`**（由 `main()` 捕获后以非 0 退出码结束）；
      `0` / `0.0` 表示"未提供、不覆盖"；
    * `--epochs` / `--lr` / `--batch-size` / `--threads` 同理；
    * `--weight-decay`（哨兵 `-1.0`，`< -1.0` 报错）；
    * `--space-radius`（哨兵 `-1.0` = 未提供；`< -1.0` 报错；实际窗口校验在 `Config`）。

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
        任一覆盖参数非法时抛出，消息含具体选项名与数值。
    """
    rules: List[Tuple[str, float]] = [
        ("--n", args.n),
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
            raise ValueError(f"{name} 不能为负（0 表示不覆盖/不干预），当前 {name}={value}")
    if args.weight_decay < -1.0:
        raise ValueError(
            f"--weight-decay 不能为负（-1 表示不覆盖、0 表示覆盖为 0），"
            f"当前 --weight-decay={args.weight_decay}"
        )
    if args.space_radius < -1.0:
        raise ValueError(
            f"--space-radius 必须 >= 0（-1 表示未提供、0 表示取公式下界 R_min），"
            f"当前 --space-radius={args.space_radius}"
        )
    # ---- fc_dim 校验（第 3 轮新增）----
    # [!] 上述"负数一律报错"的通用规则**必须为 `--fc-dim` 单独放宽到 >= -1**：
    #     `fc_dim = -1` 是**有意义的取值**（宽度跟随 N），不是错误输入。
    #     故 `--fc-dim` 不进上面的 rules 列表，只在下面对 `< -1` 报错。
    if args.fc_dim is not None and int(args.fc_dim) < -1:
        raise ValueError(
            f"--fc-dim 只允许 -1（跟随 N）或 >= 0（0 表示关闭），"
            f"当前 --fc-dim={args.fc_dim}"
        )
    # 拒绝静默无效参数（与本模块 `--cyl-aspect` 的既有纪律一致）：
    # `fc_dim != 0` 只作用于 neuron3d 主模型，MLP 基线不读取该字段。
    if args.fc_dim is not None and int(args.fc_dim) != 0 and args.arch == "mlp":
        raise ValueError(
            f"--fc-dim 仅在 --arch neuron3d 时生效：当前 --arch=mlp 但 "
            f"--fc-dim={args.fc_dim}（非 0）。请移除 --fc-dim 或改用 --arch neuron3d"
            f"（拒绝静默无效参数）。"
        )
    # ---- geo_field 族校验（第 5 轮新增）----
    # [!] `--arch mlp` 搭配 `geo_field != none` 在 **CLI 层直接拒绝**（沿用 `--fc-dim`
    #     的既有先例）：几何权重场只作用于 N3D 主模型的稀疏边集，MLP 基线不读取该字段，
    #     静默接受会让用户误以为"MLP 也开了几何场"。
    if args.geo_field and args.geo_field != "none" and args.arch == "mlp":
        raise ValueError(
            f"--geo-field 仅在 --arch neuron3d 时生效：当前 --arch=mlp 但 "
            f"--geo-field={args.geo_field}（非 none）。请移除 --geo-field 或改用 "
            f"--arch neuron3d（拒绝静默无效参数）。"
        )
    # `--geo-rbf-k` 哨兵 0 = 不覆盖；显式给出时必须 > 0（负数由下面的通用规则拦下）
    if args.geo_rbf_k < 0:
        raise ValueError(
            f"--geo-rbf-k 不能为负（0 表示不覆盖/沿用预设 12），"
            f"当前 --geo-rbf-k={args.geo_rbf_k}"
        )
    # `--geo-alpha-init` 哨兵 -1.0 = 不覆盖；显式给出时必须 >= 0
    if args.geo_alpha_init < -1.0:
        raise ValueError(
            f"--geo-alpha-init 必须 >= 0（-1 表示不覆盖、沿用预设 1.0），"
            f"当前 --geo-alpha-init={args.geo_alpha_init}"
        )
    # ---- 形状参数校验（本模块新增）----
    # `--cyl-aspect` 哨兵 -1.0 = 未提供；显式给出时必须 > 0，且形状必须是 cylinder。
    # [!] 这里做"非 cylinder + 显式 λ"的前置拒绝：Config 只能看到"非默认值"，看不到
    #    "用户是否显式传了 λ=1.0"，故该语义必须在 CLI 层判定（否则 `--shape cube --cyl-aspect 1.0`
    #    会被静默接受）。Config 层仍保留"非默认值即报错"的第二道防线。
    if args.cyl_aspect != -1.0 and args.cyl_aspect <= 0.0:
        raise ValueError(
            f"--cyl-aspect 必须 > 0（λ = c/r；-1 表示未提供、沿用预设 1.0），"
            f"当前 --cyl-aspect={args.cyl_aspect}"
        )
    if args.cyl_aspect != -1.0 and args.shape != "cylinder":
        raise ValueError(
            f"--cyl-aspect 仅在 --shape cylinder 时生效：当前 --shape={args.shape or 'sphere(缺省)'} "
            f"但显式给出了 --cyl-aspect={args.cyl_aspect}。请改用 --shape cylinder "
            f"或移除 --cyl-aspect（拒绝静默无效参数）。"
        )
    # ---- 数据集通用层校验（第 6 轮新增）----
    # [!] 维度哨兵是 `None`（**不是浮点/整数哨兵**）：`--input-dim 0` 与
    #     `--output-dim 0` 都是**非法显式值**，必须在 CLI 层就报错 ——
    #     否则 `0` 会被当成"未提供"而静默取规格，与计划验收第 6 条
    #     "input_dim=0 提前报错"直接冲突。
    if args.input_dim is not None and int(args.input_dim) <= 0:
        raise ValueError(
            f"--input-dim 必须 > 0（缺省 = 不覆盖、取数据集规格），"
            f"当前 --input-dim={args.input_dim}"
        )
    if args.output_dim is not None and int(args.output_dim) <= 0:
        raise ValueError(
            f"--output-dim 必须 > 0（缺省 = 不覆盖、取数据集规格），"
            f"当前 --output-dim={args.output_dim}"
        )
    if int(args.num_samples) < 0:
        raise ValueError(
            f"--num-samples 必须 >= 0（0 表示取该数据集的缺省样本数），"
            f"当前 --num-samples={args.num_samples}"
        )
    # 归一化：哨兵是 None；两者必须**同时**给出（只给一个直接报错，拒绝半套统计量）
    if (args.norm_mean is None) != (args.norm_std is None):
        raise ValueError(
            f"--norm-mean 与 --norm-std 必须同时给出（缺省 = 两者都不给、由训练集现场统计）："
            f"当前 --norm-mean={args.norm_mean}, --norm-std={args.norm_std}"
        )
    if args.norm_std is not None and not (float(args.norm_std) > 0.0):
        raise ValueError(
            f"--norm-std 必须 > 0，当前 --norm-std={args.norm_std}"
        )
    # 未注册的数据集名由 argparse 的 choices 直接拒绝；此处再校验需要外部文件的来源
    # 是否给了路径（空路径在 `build_dataloaders` 里也会报错，这里提前到 CLI 层更友好）。
    # [!] 只看 `args.dataset` **非空**（用户确实给了）的情形：留空 = 沿用预设的 mnist，
    #     此时 `--dataset-path` 对 mnist 无意义，不该被当成"缺路径"而误报。
    if args.dataset:
        _spec = DATASET_SPECS.get(str(args.dataset))
        if _spec is not None and _spec.kind in ("npz", "csv", "json") and not args.dataset_path:
            raise ValueError(
                f"--dataset {args.dataset} 需要 --dataset-path 指定数据文件路径"
                f"（拒绝静默使用空路径）。"
            )
    # 其余字符串型几何参数由 argparse 的 choices 直接拒绝非法取值（空串 = 未提供）


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
        非 None 时，在**确实发生覆盖**时打印该告警。

    返回
    ----
    Config
        覆盖后的配置对象；无任何覆盖时返回 `base` 本身。
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
    if args.weight_decay >= 0.0:
        overrides["weight_decay"] = args.weight_decay
        changed = True
    if args.batch_size > 0:
        overrides["batch_size"] = args.batch_size
        changed = True
    if args.n > 0:
        overrides["N"] = args.n
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
    # ---- 几何覆盖（哨兵：空串 / -1.0 = 未提供） ----
    if args.flow_axis:
        overrides["flow_axis"] = args.flow_axis
        changed = True
    if args.space_radius >= 0.0:
        overrides["space_radius"] = args.space_radius
        changed = True
    if args.input_scope:
        overrides["input_scope"] = args.input_scope
        changed = True
    if args.readout_scope:
        overrides["readout_scope"] = args.readout_scope
        changed = True
    if args.placement:
        overrides["placement"] = args.placement
        changed = True
    # ---- 形状覆盖（本模块新增；必须与 to_dict 的字段名严格对齐，否则被静默丢弃）----
    if args.shape:
        overrides["shape"] = args.shape
        changed = True
    if args.cyl_aspect != -1.0:
        overrides["cyl_aspect"] = args.cyl_aspect
        changed = True
    # ---- fc_dim 覆盖（第 3 轮新增；None = 未提供；`0` 是**合法覆盖值**）----
    if args.fc_dim is not None:
        overrides["fc_dim"] = int(args.fc_dim)
        changed = True
    # ---- 几何权重场覆盖（第 5 轮新增；空串 = 未提供，`none` 是**合法覆盖值**）----
    if args.geo_field:
        overrides["geo_field"] = args.geo_field
        changed = True
    if args.geo_rbf_k > 0:
        overrides["geo_rbf_k"] = int(args.geo_rbf_k)
        changed = True
    if args.geo_alpha_init >= 0.0:
        overrides["geo_alpha_init"] = float(args.geo_alpha_init)
        changed = True
    if args.seed_override > 0:
        log_info(f"seed 覆盖：{overrides['seed']} -> {args.seed_override}")
        overrides["seed"] = args.seed_override
        changed = True
    # ---- 数据集通用层覆盖（第 6 轮新增；哨兵："" / None / 0）----
    # [!] 这一段是"数据集可插拔"的**唯一生效路径**：漏一个字段就会被
    #     `Config(**base.to_dict())` 静默丢弃（历史陷阱同源）。
    if args.dataset:
        overrides["dataset"] = str(args.dataset)
        changed = True
        # [!] **换数据集时必须清掉上一份配置遗留的显式维度**（本模块第 6 轮的关键纪律）：
        #     `to_dict()` 里 `input_dim` / `output_dim` 可能是**显式值**
        #     （预设自身写着 784/10，或上一轮显式给了值），那些数字属于**上一个数据集**。
        #     若原样带进新数据集，`Config` 构造期会因"显式值与新规格冲突"直接报错
        #     （例如 `--preset default --dataset synthetic` 会因 `784 != 64` 失败），
        #     而用户的意图显然是"用新数据集自己的维度"。
        #     为什么清在这里、而不是在 `Config.__post_init__`：只有本入口**确知**
        #     "数据集确实被切换了"，因此可以安全区分
        #       (a) 遗留值（继承自上一步）-> 清成哨兵，交给规格解析；
        #       (b) 用户显式给出的冲突值 -> 仍走下方的显式写入与解析期报错。
        #     在构造期按"值等于 mnist 规格就清除"会把 (b) 也静默放过，属放宽纪律。
        overrides.pop("input_dim", None)
        overrides.pop("output_dim", None)
    if args.data_root:
        overrides["data_root"] = str(args.data_root)
        changed = True
    if args.dataset_path:
        overrides["dataset_path"] = str(args.dataset_path)
        changed = True
    if int(args.num_samples) > 0:
        overrides["num_samples"] = int(args.num_samples)
        changed = True
    if args.norm_mean is not None:
        overrides["norm_mean"] = float(args.norm_mean)
        overrides["norm_std"] = float(args.norm_std)
        changed = True
    # [!] `--input-dim` / `--output-dim` 的写法与其它字段**不同**（刻意）：
    #     只有当显式值**偏离**"该数据集规格的缺省值"时才写进 overrides。
    #     理由：`to_dict()` 写出的是哨兵（`None` = 未给出），若把"与规格一致的显式值"
    #     （如 mnist 的 784/10）也写进去，`Config(...) == base` 会因**原始字段值**不同
    #     而判不相等，于是 `apply_overrides` 的"值等价即复用基线对象"优化失效，
    #     同一语义的配置会产出不同字节的产物（离朱第 11 轮 D1 的同类缺陷）。
    #     语义等价的显式值，其**生效值**本来就等于规格缺省值，故等价性完全保留。
    if args.input_dim is not None and int(args.input_dim) != int(base.effective_input_dim):
        overrides["input_dim"] = int(args.input_dim)
        changed = True
    if args.output_dim is not None and int(args.output_dim) != int(base.effective_output_dim):
        overrides["output_dim"] = int(args.output_dim)
        changed = True
    if args.readout_bias is not None:
        overrides["readout_bias"] = bool(args.readout_bias)
        changed = True
    if not changed:
        return base
    result = Config(**overrides)
    if result == base:
        # 显式给出的覆盖值与基线**逐字段完全相同**（值等价）：复用基线对象本身。
        # 这不只是省一次构造 —— `torch.save` 的**字节**取决于 pickle 的记忆化（memo），
        # 而 memo 依赖**对象身份**而非取值：
        #   * 裸跑走 `SMALL_CONFIG` 单例时，`input_scope` 与 `readout_scope` 指向同一个
        #     `"any_isolated"` 字符串对象，第二次出现被写成 memo 引用（`h\xfd`）；
        #   * 显式传 `--input-scope any_isolated` 时，取值来自 argparse 构造的**等值新
        #     字符串**，两个位置分别内联写出（`X\x0c\0\0\0any_isolated`）。
        # 于是"语义完全相同"的两次运行产出**不同字节**的冒烟产物（离朱第 11 轮 D1：
        # data.pkl 4097 vs 4117 字节、1520 字节差异，而 loss/config/grad_norms/全部张量
        # 逐位相等）。复用基线对象即让"同一语义配置 => 同一字节"，
        # 使产物 SHA256 重新成为可靠的等价判据。
        log_info(
            "显式覆盖参数与基线取值逐字段一致（值等价）：复用基线配置对象，"
            "不构造新对象（保证同语义配置产出同字节产物）"
        )
        return base
    if warn_message:
        log_warn(warn_message)
    return result


def build_smoke_config(args: argparse.Namespace) -> Config:
    """构造冒烟测试（阶段 A）使用的配置。

    覆盖语义
    --------
    仅当用户**显式**给出覆盖参数时才覆盖，并通过日志提示"冒烟测试配置已被覆盖"。
    新增的几何参数（`--flow-axis` / `--space-radius` / `--input-scope` /
    `--readout-scope` / `--placement`）**与本模块新增的形状参数（`--shape` / `--cyl-aspect`）**
    必须纳入下方 `explicit` 判定，否则会被静默丢弃。

    **基线规则**：冒烟路径固定以 `SMALL_CONFIG` 为基线，只有当显式指定**非 default**
    预设时才换基线。

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
        log_warn(
            f"冒烟测试基线已被 --preset {args.preset} 替换："
            f"N={base.N}, batch_size={base.batch_size}（非 SMALL_CONFIG 规模）"
        )
    explicit = (
        args.preset != "default"
        or args.n > 0
        or args.y_in > 0
        or args.y_out > 0
        or args.h > 0.0
        or args.d > 0.0
        or args.seed_override > 0
        or args.lr > 0.0
        or args.weight_decay >= 0.0
        or args.batch_size > 0
        or args.readout_bias is not None
        or bool(args.device)
        # 几何参数：必须计入 explicit，否则 --input-scope all_isolated 会被静默丢弃
        or bool(args.flow_axis)
        or args.space_radius >= 0.0
        or bool(args.input_scope)
        or bool(args.readout_scope)
        or bool(args.placement)
        # 形状参数（本模块新增）：必须计入 explicit，否则 --shape cube 会被静默丢弃
        or bool(args.shape)
        or args.cyl_aspect != -1.0
        # fc_dim（第 3 轮新增）：必须计入 explicit，否则 --fc-dim 会被静默丢弃
        or args.fc_dim is not None
        # geo_field 族（第 5 轮新增）：必须计入 explicit，否则 --geo-field 会被静默丢弃
        or bool(args.geo_field)
        or args.geo_rbf_k > 0
        or args.geo_alpha_init >= 0.0
        # 数据集通用层（第 6 轮新增）：必须计入 explicit，否则 --dataset / --dataset-path
        # / --input-dim / --output-dim / --num-samples / --norm-mean / --norm-std
        # 会被静默丢弃（`--data-root` 亦同）。
        or bool(args.dataset)
        or bool(args.data_root)
        or bool(args.dataset_path)
        or args.input_dim is not None
        or args.output_dim is not None
        or int(args.num_samples) > 0
        or args.norm_mean is not None
        or args.norm_std is not None
    )
    if not explicit:
        return SMALL_CONFIG
    return apply_overrides(
        base,
        args,
        warn_message=(
            "冒烟测试配置已被显式覆盖（--preset/--n/--y-in/--y-out/--h/--d/--seed/--lr/"
            "--flow-axis/--space-radius/--input-scope/--readout-scope/--placement/"
            "--shape/--cyl-aspect/--fc-dim/--geo-field/--dataset/--data-root/"
            "--dataset-path/--input-dim/--output-dim/--num-samples/--norm-mean/--norm-std "
            "之一）："
            f"预设={args.preset}；阶段 A 的默认基线仅在默认组合下成立"
        ),
    )


def run_smoke_test(
    device_override: str = "",
    config_override: Optional[Config] = None,
    arch: str = "neuron3d",
) -> bool:
    """阶段 A 冒烟测试：验证前向/反向跑通并逐条核对**新架构验收标准**。

    新架构判据集合见 `SMOKE_CRITERIA_DOC`（旧架构的 tau / 连接稀疏度 / 边级参数数
    等判据已随架构失效，不再使用）。

    参数
    ----
    device_override : str
        设备覆盖（空串表示使用配置的 device；当 `config_override` 非空时忽略）。
    config_override : Optional[Config]
        完整的配置覆盖（通常来自 `build_smoke_config`）。为 None 时使用 `SMALL_CONFIG`。
    arch : str
        架构选择（`neuron3d` / `mlp`）。

    返回
    ----
    bool
        通过返回 True；有失败项打印 [FAIL] 并返回 False。

    产物保护
    --------
    冒烟测试只跑 1 个 batch，其 checkpoint 写入
    `checkpoints/n3d_shape/_verify/`，**不会覆盖 `checkpoints/n3d_shape/model.pt`
    等正式产物，也绝不触碰一期 `checkpoints/` 下的任何文件**。
    """
    if config_override is not None:
        config: Config = config_override
    else:
        config = SMALL_CONFIG
        if device_override:
            config = Config(**{**config.to_dict(), "device": device_override})
    device = get_device(config.device)

    log_info("=" * 78)
    log_info(
        "阶段 A 冒烟测试开始"
        f"（{'SMALL_CONFIG' if config is SMALL_CONFIG else '已覆盖/已换预设的配置'}）"
    )
    log_info(f"配置：{config.describe()}")
    log_info(f"实际设备：{device}")
    log_info("=" * 78)

    t0 = time.perf_counter()
    model, train_loader, _ = build_model_and_data(config, device, max_batches=1, arch=arch)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # ---- 阶段 A 约束：仅跑 1 个 batch 的前向 + 反向 ----
    avg_loss, grad_norms, n_batches, x_last, last_logits = train_one_epoch(
        model, train_loader, optimizer, device, epoch=1, max_batches=1, collect_grads=True
    )
    elapsed = time.perf_counter() - t0

    is_neuron3d = arch == "neuron3d"
    stats = model.get_connection_stats()
    topo_stats: Dict[str, float] = model.get_topology_stats() if is_neuron3d else {}
    selfcheck: Dict[str, float] = (
        model.connectivity_selfcheck() if is_neuron3d else {}
    )
    param_total = model.count_parameters()

    # ---- 打印各可学习参数的梯度范数 ----
    log_info("-" * 78)
    log_info(f"阶段 A 结果：loss={avg_loss:.6f}，处理 batch 数={n_batches}，耗时={elapsed:.2f}s")
    log_info("各可学习参数梯度范数（L2）：")
    if not grad_norms:
        log_error("未能收集到任何梯度，反向传播可能未执行")
    # [!] 离朱 DEF-5：结构性零梯度参数（`geo_alpha`，场系数零初始化 => dL/dalpha == 0）
    #     原先把标记打成 `[BAD]`，与随后判据 `[3]` 的 PASS 结论并列出现，会误导人工复核。
    #     现按 `model._geo_zero_grad_params` 显式标为 `[EXPECTED-ZERO]`（豁免标签），
    #     仅**非豁免**参数才可能落到 `[BAD]`。
    expected_zero = set(getattr(model, "_geo_zero_grad_params", ()) or ())
    for name, gnorm in grad_norms.items():
        if name in expected_zero:
            flag = "EXPECTED-ZERO"
        else:
            flag = "OK " if gnorm > GRAD_NORM_MIN else "BAD"
        log_info(f"  [{flag}] {name:<28s} grad_norm={gnorm:.6e}")
    log_info(f"可学习参数总数：{param_total}")
    log_info(
        f"连接统计（神经元级）：E={int(stats['num_edges'])}, "
        f"avg_out_degree={stats['avg_out_degree']:.4f}, max_out_degree={int(stats['max_out_degree'])}, "
        f"avg_in_degree={stats['avg_in_degree']:.4f}, max_in_degree={int(stats['max_in_degree'])}, "
        f"num_layers={int(stats['num_layers'])}"
    )
    if topo_stats:
        log_info(
            f"形状/几何统计：{config.shape_spec.describe()}（circum_coef="
            f"{config.circum_coef:.6f}, 外接半径 ρ={config.shape_circum_radius:.6f}）, "
            f"placement={config.placement}, flow_axis={config.flow_axis}, "
            f"R={topo_stats['space_radius']:.6f} [R_min={config.min_space_radius:.6f}, "
            f"R_max={config.max_space_radius:.6f}], 放置半径={topo_stats['placement_radius']:.6f}, "
            f"选取度量={topo_stats['selection_metric']:.6f}"
            f"(落在特征空间内={int(topo_stats['selection_metric_within_space'])}), "
            f"extents xy/axis={topo_stats['neuron_extent_xy']:.6f}/"
            f"{topo_stats['neuron_extent_axis']:.6f}, 层数 K={int(topo_stats['num_layers_true'])}, "
            f"最近邻距={topo_stats['nearest_neighbour_dist']:.9f}(=2H={2.0 * config.H:.9f}), "
            f"S_in={int(topo_stats['num_in_scope'])}, S_out={int(topo_stats['num_out_scope'])}, "
            f"双副本神经元数={int(topo_stats['dual_copy_count'])}, "
            f"代表连接间距 mean/min/max={topo_stats['edge_dist_mean']:.4f}/"
            f"{topo_stats['edge_dist_min']:.4f}/{topo_stats['edge_dist_max']:.4f}"
        )
        log_info(
            f"形状放置诊断（W2/I1，可复核证据）：placement_radius="
            f"{topo_stats['placement_radius']:.6f}, R_max={topo_stats['max_space_radius']:.6f}, "
            f"place/R_max={topo_stats['placement_radius_over_rmax']:.4f}"
            f"(within_R_max={int(topo_stats['placement_within_rmax'])}), "
            f"place/space_radius={topo_stats['placement_radius_over_rmin']:.4f}"
            f"(within_space={int(topo_stats['placement_within_space'])}), "
            f"place/rho={topo_stats['placement_within_circum_ratio']:.4f}"
            f"(within_rho={int(topo_stats['placement_within_circum'])})"
        )
        log_info(
            f"DAG 自检：无环={int(selfcheck['dag_acyclic'])}, "
            f"全部边严格上行={int(selfcheck['all_edges_uphill'])}, "
            f"拓扑序覆盖全部神经元={int(selfcheck['topo_covers_all'])}, "
            f"拓扑序==流向轴升序={int(selfcheck['topo_matches_axis_order'])}"
        )
    dense_weight_tensors = model.count_dense_weight_tensors()
    if is_neuron3d:
        log_info(
            f"稀疏性实测：神经元级连接 E={int(stats['num_edges'])}，"
            f"可能连接数=N*(N-1)={config.N * (config.N - 1)}，"
            f"连接权重参数数={int(model.edge_weight.numel())}（按连接构建，每条连接 1 个标量）"
        )
        log_info(
            f"  -> 若把神经元级连接 materialize 成 [N, N] 稠密矩阵，权重参数将从 "
            f"{param_total} 膨胀到 "
            f"{param_total - int(model.edge_weight.numel()) + config.N * config.N}"
        )
    else:
        log_info(
            f"arch=mlp 对照基线：input {config.input_dim} -> hidden {config.hidden_dim} "
            f"-> 输出 {config.output_dim}，可学习参数 {param_total}"
        )
    log_info("=" * 78)

    # ---- 逐条核对阶段 A 验收标准（新架构口径） ----
    checks: List[Tuple[str, bool, str]] = []
    # [1] 真实断言：前向输出形状必须是 [B, output_dim]
    logits_shape_ok = (
        tuple(last_logits.shape) == (int(x_last.shape[0]), int(config.output_dim))
    )
    checks.append(
        (
            "[1] 前向输出形状 == [B, output_dim]",
            logits_shape_ok,
            f"logits.shape={tuple(last_logits.shape)}，期望 "
            f"({int(x_last.shape[0])}, {int(config.output_dim)})",
        )
    )
    # [2] 真实断言：全部可学习参数都必须拿到梯度（tensor_grad_norms 把缺失梯度记为 -1.0）
    missing_grads = [n for n, v in grad_norms.items() if v < 0.0]
    checks.append(
        (
            "[2] 反向无错误（全部可学习参数都有梯度）",
            len(grad_norms) > 0 and not missing_grads,
            f"收集到 {len(grad_norms)} 个参数的梯度"
            + (f"；缺失梯度：{missing_grads}" if missing_grads else "，无缺失"),
        )
    )
    # [3] 参与 loss 的参数梯度范数必须 > 0（**不绑定参数名**，两种 arch 通用）。
    #     口径说明（严格 readout 的直接推论）：`h[n] = a_up[n]` 仅当 n ∈ S_out，否则
    #     h[n] = 0，故 neuron3d 的 `W_out` **非 S_out 列**不参与计算图、梯度恒为 0。
    #     因此判据分两部分：
    #       (a) 输出层参数（neuron3d 为 `W_out` / MLP 为 `fc2.*`）至少有一个的梯度范数 > 0
    #           —— 等价于"读出层真正参与了 loss"；
    #       (b) 其余全部可学习参数的梯度范数 > 0。
    #     [!] 历史缺陷（离朱第 9 轮 M1）：判据原先硬编码 `grad_norms.get("W_out", -1.0)`，
    #        而 `MLPBaseline` 的输出层参数名是 `fc2.weight` / `fc2.bias`，取不到就会拿到
    #        哨兵 -1.0 而误判 FAIL（`--arch mlp` 冒烟退出码 1）。故改为按 arch 解析参数名。
    #     [!] 历史缺陷（皋陶第 2 轮 F9）：曾额外断言 neuron3d 的"W_out 非零梯度列数
    #        == |S_out|"。该等式在"某 S_out 神经元的 pre-activation 在整个 batch 上全为负"
    #        时不成立 —— 此时其 ReLU 输出为 0，`h` 的对应列合法地为 0，故该列不参与 loss、
    #        梯度为 0（ReLU 的正常行为，spec 与 README 均已写明）。现改为**下界断言**：
    #        非零梯度列数必须 >= 1（且必须 <= |S_out|，因为非 S_out 列结构性为 0）。
    # [!] fc_dim 路径没有 `W_out`（输出层是 (5)(6) 两个全连接层），故按启用状态解析名字；
    #     `fc_dim == 0` 时该分支与改动前**逐字等价**，默认路径行为不变。
    _fc_on = bool(is_neuron3d and getattr(model, "fc_enabled", False))
    if _fc_on:
        out_layer_names = ("head_weight", "head_bias")
    elif is_neuron3d:
        out_layer_names = ("W_out",)
    else:
        out_layer_names = ("fc2.weight", "fc2.bias")
    out_layer_gnorms = [float(grad_norms.get(n, -1.0)) for n in out_layer_names]
    other_gnorms = {
        n: v for n, v in grad_norms.items() if n not in set(out_layer_names)
    }
    # [!] 几何权重场（第 5 轮新增）的**结构性零梯度**参数：`geo_alpha` 的梯度是
    #     `dL/dα = Σ_e Δ_e · 场(φ_e)`，而场系数 `c`（`geo_rbf_theta`）**零初始化**，
    #     初始时 `场(φ_e) == 0` => `dL/dα` **恒为 0**（后续步骤经 `c` 回传即非零）。
    #     这与"W_out 的非 S_out 列结构性零梯度"同源，属**设计预期**而非缺陷，
    #     故从"其余参数必须 > 0"的判据中**显式豁免**，并在判据文本中逐条列出。
    geo_zero_struct = [
        n for n in getattr(model, "_geo_zero_grad_params", ()) if n in other_gnorms
    ]
    for n in geo_zero_struct:
        other_gnorms.pop(n, None)
    zero_others = [n for n, v in other_gnorms.items() if not (v > GRAD_NORM_MIN)]
    out_layer_ok = any(v > GRAD_NORM_MIN for v in out_layer_gnorms)
    # neuron3d 额外核对"非零梯度列数落在 [1, |S_out|]"（严格 readout 的结构性下界/上界）
    out_col_participating = (
        None
        if (not is_neuron3d or _fc_on)
        else int((model.W_out.grad.abs().sum(dim=0) > 0).sum().item())
    )
    grad_positive_ok = out_layer_ok and len(zero_others) == 0
    if is_neuron3d and not _fc_on:
        s_out_size = int(topo_stats["num_out_scope"])
        grad_positive_ok = grad_positive_ok and (
            1 <= int(out_col_participating) <= s_out_size
        )
    checks.append(
        (
            "[3] 参与 loss 的参数梯度范数 > 0",
            grad_positive_ok,
            (
                f"输出层参数 {dict(zip(out_layer_names, out_layer_gnorms))}；"
                + (
                    f"W_out 非零梯度列数={out_col_participating}"
                    f"（须落在 [1, |S_out|={int(topo_stats['num_out_scope'])}]，"
                    f"等式不成立是 ReLU 死神经元的正常行为）；"
                    if (is_neuron3d and not _fc_on)
                    else ""
                )
                + f"其余参数梯度范数={'全部 > 0' if not zero_others else zero_others}"
                + (
                    "；结构性零梯度（设计预期，已豁免）："
                    + ", ".join(
                        f"{n}={float(grad_norms.get(n, float('nan'))):.3e}"
                        for n in geo_zero_struct
                    )
                    + "（几何场系数零初始化 => dL/dalpha == 0，梯度经 c 回传后为非零）"
                    if geo_zero_struct
                    else ""
                )
            ),
        )
    )
    finite_loss = bool(avg_loss == avg_loss) and abs(avg_loss) != float("inf")
    checks.append(("[4] loss 非 NaN/Inf", finite_loss, f"loss={avg_loss:.6f}"))
    if is_neuron3d:
        checks.append(
            (
                "[5] S_in 非空（阶段 1 真正被输入层驱动）",
                int(topo_stats["num_in_scope"]) > 0,
                f"|S_in|={int(topo_stats['num_in_scope'])}（judge={config.input_scope}）",
            )
        )
        checks.append(
            (
                "[6] S_out 非空（readout 真正有信号）",
                int(topo_stats["num_out_scope"]) > 0,
                f"|S_out|={int(topo_stats['num_out_scope'])}（judge={config.readout_scope}）",
            )
        )
        # 唯一性判据（"同一神经元对只算一条连接"的可执行形式）：
        # 把 [E, 2] 的 (起点, 终点) 逐行编码成整数对 (A, B)，去重后条数必须仍等于 E。
        # 注意不是 max_in_degree <= 1 —— 入度是"有多少上游神经元指向我"，
        # 与"同一上游神经元对我是否只出现一次"是两件事。
        pair_keys = torch.stack(
            [model.edge_src.to(torch.long), model.edge_dst.to(torch.long)], dim=1
        ).unique(dim=0)
        unique_pairs_ok = int(pair_keys.shape[0]) == int(model.num_edges)
        checks.append(
            (
                "[7] 神经元级连接数 == 去重后的神经元对数（同一神经元对只算一条）",
                unique_pairs_ok,
                f"E={int(model.num_edges)}，去重后神经元对数={int(pair_keys.shape[0])}，"
                f"max_in_degree={int(stats['max_in_degree'])}",
            )
        )
        checks.append(
            (
                "[8] 图为无环 DAG 且每条边严格上行（z_A < z_B）",
                int(selfcheck["dag_acyclic"]) == 1 and int(selfcheck["all_edges_uphill"]) == 1,
                f"dag_acyclic={int(selfcheck['dag_acyclic'])}, "
                f"all_edges_uphill={int(selfcheck['all_edges_uphill'])}",
            )
        )
        nn_ok = abs(topo_stats["nearest_neighbour_dist"] - 2.0 * config.H) <= NN_DIST_TOL
        checks.append(
            (
                "[9] 最近邻距 == 2H（FCC 规则堆积契约）",
                nn_ok,
                f"实测={topo_stats['nearest_neighbour_dist']:.9f}，2H={2.0 * config.H:.9f}",
            )
        )
        radius_ok = (
            config.min_space_radius - 1e-9
            <= config.effective_space_radius
            <= config.max_space_radius + 1e-9
        )
        checks.append(
            (
                "[10] 形状特征尺度落在 [R_min, R_max] 内",
                radius_ok,
                f"shape={config.shape_spec.describe()}, R={config.effective_space_radius:.6f}, "
                f"R_min={config.min_space_radius:.6f}, R_max={config.max_space_radius:.6f}",
            )
        )
        checks.append(
            (
                "[11] 神经元级连接数 E > 0 且平均出度 > 0",
                int(stats["num_edges"]) > 0 and stats["avg_out_degree"] > 0.0,
                f"E={int(stats['num_edges'])}，avg_out_degree={stats['avg_out_degree']:.4f}",
            )
        )
        checks.append(
            (
                "[12] 不存在 [N*y_out, N*y_in] 形状的权重张量",
                dense_weight_tensors == 0,
                f"命中 {dense_weight_tensors} 个（应为 0；syn_dist 属几何 buffer，不计入）",
            )
        )
        # [13] readout 严格口径的真实断言：h 的非零列集合必须恰为 S_out。
        # 用一次真实前向的中间量取证（不依赖任何自报字段）。
        with torch.no_grad():
            _a_in_probe = model.stage1_input_driven(x_last)
            _a_up_probe = model.stage2_recurrence(_a_in_probe)
            _h_probe = model.readout_activations(_a_up_probe)
        h_nonzero_cols = int((_h_probe.abs().sum(dim=0) > 0).sum().item())
        s_out_size = int(model.out_scope_mask.sum().item())
        # 口径（与 spec / README 完全一致）：
        #   (a) **只断言子集方向**：h 的非零列必须都属于 S_out；
        #   (b) "掩码机制正确"改为**直接结构核对**：h 必须逐位等于
        #       `(a_up * out_scope_mask)` 的转置形式 —— 即掩码确实作用在 a_up 上；
        #   (c) 下界：至少有一列非零（否则读出层完全无信号，S_out 形同虚设）。
        # [!] 历史缺陷（皋陶第 2 轮 F9）：曾断言 `h 非零列数 == |S_out|`。该等式在
        #    "某 S_out 神经元的 pre-activation 在整个探针 batch 上全为负"时不成立 ——
        #    此时其 ReLU 输出为 0，h 的对应列合法地为 0（ReLU 的正常行为，
        #    spec 与 README 均已写明）。故删除等式断言，改为子集 + 结构核对 + 下界。
        h_cols_subset = bool(
            ((_h_probe.abs().sum(dim=0) > 0) & ~model.out_scope_mask).sum().item() == 0
        )
        h_struct_ok = bool(
            torch.equal(
                _h_probe,
                (_a_up_probe * model.out_scope_mask.unsqueeze(1).to(_a_up_probe.dtype))
                .transpose(0, 1),
            )
        )
        checks.append(
            (
                "[13] readout 严格口径（h 非零列都属于 S_out，且 h 逐位 == a_up 掩码积）",
                h_cols_subset and h_struct_ok and h_nonzero_cols >= 1,
                f"h 非零列数={h_nonzero_cols}（须 >= 1 且 <= |S_out|={s_out_size}）；"
                f"非 S_out 列被屏蔽={h_cols_subset}；"
                f"h == (a_up * out_scope_mask) 逐位成立={h_struct_ok}",
            )
        )
        checks.append(
            (
                "[14] 阶段 2 递推顺序 == 流向轴升序（topo_matches_axis_order == 1）",
                int(selfcheck["topo_matches_axis_order"]) == 1,
                f"topo_matches_axis_order={int(selfcheck['topo_matches_axis_order'])}",
            )
        )
        # [15] 形状感知判据（本模块新增，取代二期"15 条"中的第 15 条耗时判据的位置）：
        #   把"形状"这件事做成**可复核的真实断言**，而不是只打印一行配置。三层核对：
        #   (a) 配置 → 模型：shape / λ / 外接系数必须逐字段一致（防"配置改了但模型没读到"）；
        #   (b) 度量自洽：模型自报的 `selection_metric` 必须等于"用 `_shape_metric` 直接对
        #       `neuron_pos` 复算得到的最大度量"——这条能抓住"度量写对了但排序用了别的键"
        #       这类最隐蔽的错位（**已知陷阱**：只改裁剪掩码等于什么都没改）；
        #   (c) 分层自洽：层数 K 必须等于唯一流向轴坐标数，且 K >= 2（架构深度有效）。
        metric_recheck = model._shape_metric(
            model.neuron_pos.to(torch.float64)
        )
        metric_selfconsistent = bool(
            abs(float(metric_recheck.max().item()) - float(model.selection_metric)) <= 1e-6
        )
        shape_cfg_ok = (
            str(model.shape) == str(config.shape)
            and float(model.cyl_aspect) == float(config.cyl_aspect)
            and float(model.circum_coef) == float(config.circum_coef)
        )
        lay_k = int(topo_stats["num_layers_true"])
        layers_shape_ok = lay_k == int(torch.unique(model.neuron_pos[:, model.flow_axis_index]).numel())
        checks.append(
            (
                "[15] 形状生长度量与层数落地（config<->model 一致 / 度量自洽 / K 自洽）",
                shape_cfg_ok and metric_selfconsistent and layers_shape_ok and lay_k >= 2,
                f"shape={model.shape}, λ={model.cyl_aspect:g}, circum_coef={model.circum_coef:.6f} "
                f"(配置一致={shape_cfg_ok})；selection_metric 复算={float(metric_recheck.max()):.6f} "
                f"vs 自报={model.selection_metric:.6f}（自洽={metric_selfconsistent}）；"
                f"K={lay_k}（唯一轴坐标数自洽={layers_shape_ok}）；"
                f"度量落在特征空间内={model.selection_metric_within_space}"
            )
        )
    else:
        # [5]（mlp 分支）：结构断言泛化 + **占位感知**（第 21 轮的修订与回修）
        # ---------------------------------------------------------------
        # 原先把 `input_dim == 784 and output_dim == 10` **硬编码**在判据里，
        # 换成非 MNIST 数据集后该判据必然 FAIL（而模型其实完全正确）。
        # 第 21 轮改为"与 `DATASET_SPECS[dataset]` 的声明值比对"，但**漏了占位语义**：
        # `npz` / `csv` / `json` 的注册表声明值是**占位 `0`**（含义 = "由数据文件现场解析"），
        # 这三个来源又**必须**显式给 `--input-dim` / `--output-dim`，故生效维度恒 `!= 0`
        # => 判据恒为 False => 这三个来源的**一切合法配置**的 mlp 冒烟必判 FAIL
        # （实测：`--smoke-test --arch mlp --dataset npz --dataset-path ... --input-dim 64
        # --output-dim 8` 时 [FAIL]，而同口径 synthetic 为 [PASS]）。
        # 本次回修把该判据改成与 `data.resolve_dims` **同一口径的占位感知比对**：
        #   * 声明值为**占位**（`<= 0`，即 npz / csv / json）-> 跳过规格比对，
        #     只断言"生效维度为正"（真值一致性已由数据层 `_check_dims` 在加载后用
        #     真实列数 / 最大标签校验，此处不重复也不越权）；
        #   * 声明值为**真值**（mnist 784/10、synthetic 64/8）-> 保持原判据
        #     （与规格声明值严格相等）。
        # MNIST 情形（784 / 10）下与最初的硬编码断言**完全等价**（回归锚点不变）。
        _spec = DATASET_SPECS[str(config.dataset)]
        _eff_in = int(config.effective_input_dim)
        _eff_out = int(config.effective_output_dim)
        _spec_in = int(_spec.input_dim)
        _spec_out = int(_spec.num_classes)
        # 占位判定与 `data.resolve_dims` 完全同源（`<= 0` 即占位）
        _in_placeholder = _spec_in <= 0
        _out_placeholder = _spec_out <= 0
        _in_ok = (_eff_in > 0) if _in_placeholder else (_eff_in == _spec_in)
        _out_ok = (_eff_out > 0) if _out_placeholder else (_eff_out == _spec_out)
        # [!] 皋陶 info 项（第 21 轮回修）：原先写作
        #     `int(config.input_dim) == int(config.effective_input_dim)` —— 由于
        #     `Config.__getattribute__` 把这两个名字**读取时解析为同一来源**（`_derived`），
        #     该断言**恒真**（无区分力，属"同源自洽"）。现改为读**原始字段值**
        #     （`config.__dict__`，即"显式给出的值 / 未给出哨兵"），与生效值做**真实的**
        #     一致性核对：原始值为 `None`（未给出）或与生效值相等都算通过；
        #     二者不等说明构造期解析出了问题（那才是真缺陷）。
        _raw_decls = (
            config.__dict__.get("input_dim"),
            config.__dict__.get("output_dim"),
        )
        _raw_consistent = all(
            raw is None or int(raw) == int(eff)
            for raw, eff in zip(_raw_decls, (_eff_in, _eff_out))
        )
        _dims_match = _in_ok and _out_ok and _raw_consistent
        checks.append(
            (
                "[5] arch=mlp 结构正确（input -> hidden -> output，dims 与数据集规格口径一致）",
                _dims_match,
                f"dataset={config.dataset}；input_dim={_eff_in}"
                f"（{('规格 占位，由数据决定' if _in_placeholder else '规格 ' + str(_spec_in))}），"
                f"hidden_dim={config.hidden_dim}，output_dim={_eff_out}"
                f"（{('规格 占位，由数据决定' if _out_placeholder else '规格 ' + str(_spec_out))}）；"
                f"原始声明值={_raw_decls}（与生效值一致={_raw_consistent}）"
            )
        )
    checks.append(
        ("[16] CPU 单 batch 前向+反向耗时 < 120s", elapsed < SMOKE_TIME_MAX, f"实际 {elapsed:.2f}s")
    )

    log_info("阶段 A 验收标准逐条核对（形状感知判据集合）：")
    all_ok = True
    for name, ok, detail in checks:
        log_info(f"  [{'PASS' if ok else 'FAIL'}] {name} —— {detail}")
        all_ok = all_ok and ok

    # ---- 冒烟测试产物写入独立目录，避免覆盖正式 checkpoint ----
    # 命名规则（唯一事实来源 = `smoke_checkpoint_path`，本处只决定是否退化）：
    #   * **完全默认组合**（neuron3d + SMALL_CONFIG 规模 + flow_axis=z + 两个
    #     any_isolated）→ `_verify/smoke_nosyn.pt`（第 4 轮起：原历史名 `smoke.pt`
    #     加 `_nosyn` 格式段，使新版瘦身产物与旧版同名产物分开留痕，
    #     旧产物上的冻结 SHA256 断言不被触碰）；
    #   * 其它任何组合（换 arch / 流向轴 / scope / **N、y、H、D、seed 被覆盖**）→
    #     `_verify/smoke[_ar{arch}]_{完整配置指纹}.pt`，保证不同配置不互相覆盖。
    # [!] 历史缺陷（离朱第 9 轮 M2）：曾无条件传入指纹，使默认路径**永远不再写**
    #    `smoke.pt`，而 README/spec 与 `verify_sphere_dag.py` 仍指向它 —— 于是读到
    #    上一版代码留下的陈旧产物（数值与文档不符且含已删除字段）。
    #    [第 4 轮提醒] 本次是**有意**改名（`smoke.pt` -> `smoke_nosyn.pt`），故 README /
    #    spec 中的默认产物名必须**同步改名**，不得再指向旧名（否则重演 M2 的陈旧产物缺陷）。
    # [!] 历史缺陷（皋陶第 2 轮 F12）：`is_default_smoke` 原先只比较 arch / flow_axis /
    #    两个 scope，未纳入容量与 seed —— 于是 `--smoke-test --n 32`、`--preset default`、
    #    `--seed 7` 等会**静默覆盖**默认冒烟产物（用一个不同配置的结果冒充默认产物）。
    #    故现按 `SMALL_CONFIG` 的全部几何/容量维度逐一比对。
    small = SMALL_CONFIG
    is_default_smoke = (
        arch == "neuron3d"
        and int(config.N) == int(small.N)
        and int(config.y_in) == int(small.y_in)
        and int(config.y_out) == int(small.y_out)
        and float(config.H) == float(small.H)
        and float(config.D) == float(small.D)
        and int(config.seed) == int(small.seed)
        and config.flow_axis == small.flow_axis
        and config.placement == small.placement
        and config.input_scope == small.input_scope
        and config.readout_scope == small.readout_scope
        and float(config.space_radius) == float(small.space_radius)
        # 形状维度（本模块新增）**必须纳入默认判定**：否则 `--smoke-test --shape cube`
        # 会静默覆盖默认产物（用立方体结果冒充"默认球体产物"），使既有引用读到错误取证。
        # 指纹与默认判定的维度必须严格对齐（与二期 batch_size / space_radius 的历史缺陷同源）。
        and config.shape == small.shape
        and float(config.cyl_aspect) == float(small.cyl_aspect)
        # fc_dim（第 3 轮新增）**必须纳入默认判定**：否则 `--smoke-test --fc-dim -1`
        # 会静默覆盖默认产物（用"两端全连接包裹"的结果冒充默认产物）。
        and int(config.fc_dim) == int(small.fc_dim)
        # geo_field（第 5 轮新增）**必须纳入默认判定**：否则
        # `--smoke-test --geo-field additive` 会静默覆盖默认产物（用"开了几何权重场"的
        # 结果冒充默认产物）。`geo_field == "none"` 时指纹不加 `_geo` 段，故此处比对
        # 与"段口径"天然自洽；几何场档位不同即判定为"非默认"。
        and str(config.geo_field) == str(small.geo_field)
        and bool(config.geo_signed_delta) == bool(small.geo_signed_delta)
        # batch_size 进指纹粒度之外，但它直接影响探针 batch 的规模，故一并比对，
        # 避免"几何相同但批大小不同"的配置覆盖默认产物
        and int(config.batch_size) == int(small.batch_size)
        # 数据集通用层（第 6 轮新增）**必须纳入默认判定**：否则 `--smoke-test
        # --dataset synthetic` 会静默覆盖默认产物（用别的数据集的冒烟结果冒充
        # "默认 MNIST 冒烟产物"）。判定口径 = 命名段逐字相同：
        #   `dataset_name_parts(config, baseline=small)` 比对 `("", "")`，
        #    既覆盖 `dataset`，也覆盖"显式维度偏离规格缺省值"的情形；
        #    `baseline=small` 开启短路口径（语义等价的显式 784/10 视为缺省）。
        # [!] 这里**复用指纹自身的函数**而不是手写维度清单：指纹与默认判定的维度
        #     因此天然对齐（下面的等价守卫还会再钉一次）。
        #     另**显式**比对 `dataset` 本身：短路口径会把"数据集不同但生效维度恰好相同"
        #     的配置也短路成 `("", "")`（属预期），故必须补这一条，否则
        #     `--dataset synthetic --input-dim 784 --output-dim 10` 会被误判为默认组合。
        and str(config.dataset) == str(small.dataset)
        and dataset_name_parts(config, baseline=small) == dataset_name_parts(small)
    )
    # [!] `_nosyn`（第 4 轮新增）是**格式常量**（与 config 无关），因此**不进入**上式判定：
    #     默认组合写 `_verify/smoke_nosyn.pt`（**判定维度不变，仅默认产物名加格式段**）。
    #     但"默认判定必须覆盖指纹的全部 config 维度"这条不变式仍须成立：一旦某个维度只进了
    #     `smoke_fingerprint` 而漏了 `is_default_smoke`，一个**非默认配置**就会被当成默认，
    #     从而**静默覆盖默认冒烟产物**（用不同配置的结果冒充默认产物）。
    #     故此处加一道等价守卫（指纹是 config 的纯函数，故"指纹与默认配置逐字相同" <=>
    #     "判定覆盖了指纹的全部 config 维度"）；守卫自动覆盖**今后**往指纹里加维度却忘记
    #     同步判定的情形，无需再维护一张手写维度清单。
    if is_default_smoke and smoke_fingerprint(config) != smoke_fingerprint(small):
        raise ValueError(
            "is_default_smoke 与 smoke_fingerprint 的维度不对齐："
            f"当前指纹={smoke_fingerprint(config)!r}，默认配置指纹="
            f"{smoke_fingerprint(small)!r}，但 `is_default_smoke` 判定为真 —— "
            "继续执行会**静默覆盖默认冒烟产物** `_verify/smoke_nosyn.pt`。"
            "请把该维度补进 `is_default_smoke` 判定（指纹与默认判定的维度必须严格对齐）。"
        )
    fingerprint = "" if is_default_smoke else smoke_fingerprint(config)
    smoke_path = smoke_checkpoint_path(
        config.flow_axis,
        config.input_scope,
        config.readout_scope,
        arch,
        fingerprint,
    )
    os.makedirs(VERIFY_CHECKPOINT_DIR, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "connection_stats": stats,
            "topology_stats": topo_stats,
            "dag_selfcheck": selfcheck,
            "stage": "smoke",
            "arch": arch,
            "loss": float(avg_loss),
            "grad_norms": grad_norms,
            # ---- 形状维度（本模块新增；冒烟产物自带形状取证信息）----
            "shape": config.shape,
            "cyl_aspect": config.cyl_aspect,
            "shape_tag": config.shape_tag(),
            "circum_coef": config.circum_coef,
            "shape_circum_radius": config.shape_circum_radius,
            "selection_metric": float(getattr(model, "selection_metric", float("nan"))),
            "selection_metric_within_space": bool(
                getattr(model, "selection_metric_within_space", False)
            ),
            # ---- W2/I1：放置半径 vs 窗口/外接半径的诊断量（产物自带可复核证据）----
            "placement_radius": float(getattr(model, "placement_radius", float("nan"))),
            "placement_radius_over_rmax": float(
                getattr(model, "placement_radius_over_rmax", float("nan"))
            ),
            "placement_radius_over_rmin": float(
                getattr(model, "placement_radius_over_rmin", float("nan"))
            ),
            "placement_within_rmax": bool(getattr(model, "placement_within_rmax", False)),
            "placement_within_space": bool(getattr(model, "placement_within_space", False)),
            "placement_within_circum": bool(getattr(model, "placement_within_circum", False)),
            "placement_within_circum_ratio": float(
                getattr(model, "placement_within_circum_ratio", float("nan"))
            ),
            # ---- 数据集通用层（第 6 轮新增；产物自带来源/维度/归一化取证）----
            # [!] `input_dim` / `output_dim` 写**生效值**（规格解析后），另写规格声明值，
            #     使"配置 -> 规格 -> 生效值"三者的关系在产物里可复核。
            "dataset": config.dataset,
            "dataset_path": str(config.dataset_path or ""),
            "dataset_kind": DATASET_SPECS[str(config.dataset)].kind,
            "num_samples": int(config.num_samples),
            "input_dim": int(config.effective_input_dim),
            "output_dim": int(config.effective_output_dim),
            "dataset_spec_input_dim": int(DATASET_SPECS[str(config.dataset)].input_dim),
            "dataset_spec_num_classes": int(DATASET_SPECS[str(config.dataset)].num_classes),
            "dims_source": config.dims_source,
            "norm_mean_arg": config.norm_mean,
            "norm_std_arg": config.norm_std,
            # ---- 几何权重场（第 5 轮新增；冒烟产物自带可复核取证信息）----
            "geo_field": config.geo_field,
            "geo_rbf_k": int(config.geo_rbf_k),
            "geo_alpha_init": float(config.geo_alpha_init),
            "geo_signed_delta": bool(config.geo_signed_delta),
            "geo_edge_feature_names": (
                list(model._geo_feature_names) if getattr(model, "geo_enabled", False) else []
            ),
        },
        smoke_path,
    )
    log_info(f"[产物保护] 冒烟测试 checkpoint 写入独立路径（不覆盖正式产物）：{smoke_path}")
    log_info(
        "[产物格式] nosyn：8 个突触类张量（syn_dist / input_syn_pos / output_syn_pos / "
        "representative_syn_out / representative_syn_input / input_isolated_mask / "
        "output_isolated_mask / neuron_conn_mask）为 persistent=False、不进入 state_dict；"
        "edge_dist 与全部索引拓扑量仍持久化（n3d_viz 契约）"
    )
    log_info(
        f"[产物保护] 正式 checkpoint 默认路径未被写入：{CHECKPOINT_PATH}"
        f"（该路径仅在正式全量训练时写入）"
    )
    log_info(f"阶段 A 冒烟测试结论：{'全部通过' if all_ok else '存在失败项'}")
    log_info("=" * 78)
    return all_ok


def scope_abbrev(scope: str) -> str:
    """把判据取值压缩为产物文件名用的短标记（any_isolated -> any，all_isolated -> all）。

    参数
    ----
    scope : str
        `"any_isolated"` 或 `"all_isolated"`。

    返回
    ----
    str
        短标记（`"any"` / `"all"`）；未知取值原样返回（防御性）。

    异常
    ------
    ValueError
        `scope` 不是两个合法取值之一时抛出。
    """
    if scope == "any_isolated":
        return "any"
    if scope == "all_isolated":
        return "all"
    raise ValueError(
        f"scope 仅允许 'any_isolated' 或 'all_isolated'，当前 scope={scope!r}"
    )


def config_fingerprint(config: Config, max_batches: int, tag: str = "") -> str:
    """生成限批验证产物的配置指纹文件名（保证不同配置互不覆盖）。

    命名格式：
    `verify_<bpe>_{shape_tag}_N{N}_y{y_in}x{y_out}_H{H}_D{D}_pl{placement}_ax{flow_axis}`
    `_is{scope}_rs{scope}[_fc{n}]_nosyn_s{seed}[_<tag>].pt`

    * **`{shape_tag}`（本模块新增，防撞名硬要求）**：形状段
      （`shapesphere` / `shapecube` / `shapecylinder_a{λ}`）。缺少该段时，
      `N / H / D / seed / scope` 完全相同的三种形状会写入**同一个文件名**而互相覆盖 ——
      正是历史纠正记录中 `verify_<bpe>.pt` 同名互覆导致取证失效的同一类缺陷。
      该段**始终**输出（含 sphere），保证"同配置不同形状产物名互不相同"这条断言
      对任意两两组合都成立；
    * `_s{seed}`：种子影响突触采样（边集与边数随 seed 变），指纹必须含 seed，
      否则只改 `--seed` 的限批跑会互相覆盖；
    * `_ax{flow_axis}` / `_is{input_scope}` / `_rs{readout_scope}`：几何与判据维度，
      不同取值是**两套不同拓扑**，必须各自留痕；
    * `_pl{placement}`：放置方式（当前恒为 fcc，显式记录便于扩展）。

    参数
    ----
    config : Config
        本次运行实际生效的配置。
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
    """
    if tag and not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError(f"--tag 仅允许字母/数字/下划线/连字符，当前 tag={tag!r}")
    # fc_dim != 0 时插入 `_fc{n}` 段；`fc_dim == 0` 不加段 -> 该段口径与改动前逐字相同。
    fc_part = f"_fc{int(config.fc_dim)}" if int(config.fc_dim) != 0 else ""
    # `_nosyn`（第 4 轮新增）恒定插入，紧随 `_fc{n}` 段之后、`_s{seed}` 之前
    # （产物不再落盘 8 个突触类张量，格式段必须进名字以与旧格式产物区分）。
    nosyn_part = "_nosyn"
    # `_geo{mode}_k{k}`（第 5 轮新增）：**仅 `geo_field != "none"` 时插入**，
    # 位置 = 紧随 `_fc{n}` 段之后、`_nosyn` 段之前（与 `smoke_fingerprint` /
    # `full_checkpoint_name` 三处**逐字同一口径**）。
    geo_part = (
        (
            f"_geo{config.geo_field}_k{int(config.geo_rbf_k)}"
            + ("_sd" if bool(config.geo_signed_delta) else "")
            # [!] `_a{alpha_init:g}`：离朱 DEF-7 —— `geo_alpha_init` 原先**未进指纹**，
            #     使 `--geo-alpha-init 0.4` 与默认 `1.0` 生成**同名产物**并**静默互覆**
            #     （实测 SHA `01e56cb7…` -> `7b89b4ef…`）。口径：**仅非默认值（!= 1.0）
            #     时插入**，故默认组合的产物名逐字不变（既有产物名与 README 口径不受影响）。
            + (f"_a{float(config.geo_alpha_init):g}"
               if float(config.geo_alpha_init) != 1.0 else "")
        )
        if str(config.geo_field) != "none" else ""
    )
    # ---- 数据集通用层（第 6 轮新增）：`_ds{name}` / `_d{D}x{C}`；口径与段位见
    #      `smoke_fingerprint` 的同一段注释（三处必须逐字一致）。 ----
    ds_part, dim_part = dataset_name_parts(config)
    name = (
        f"verify_{max_batches}"
        f"_{config.shape_tag()}"
        f"_N{config.N}"
        f"_y{config.y_in}x{config.y_out}"
        f"_H{config.H:g}"
        f"_D{config.D:g}"
        f"_pl{config.placement}"
        f"_ax{config.flow_axis}"
        f"_is{scope_abbrev(config.input_scope)}"
        f"_rs{scope_abbrev(config.readout_scope)}"
        f"{fc_part}"
        f"{geo_part}"
        f"{ds_part}"
        f"{dim_part}"
        f"{nosyn_part}"
        f"_s{config.seed}"
    )
    if tag:
        name = f"{name}_{tag}"
    return f"{name}.pt"


def full_checkpoint_name(config: Config, tag: str = "") -> str:
    """生成正式全量产物的文件名（配置指纹，去掉限批语义前缀）。

    命名格式：
    `full_{shape_tag}_N{N}_y{y_in}x{y_out}_H{H}_D{D}_pl{placement}_ax{flow_axis}`
    `_is{...}_rs{...}[_fc{n}]_nosyn_s{seed}[_tag].pt`

    * **`{shape_tag}`（本模块新增）**：形状段（含圆柱长径比），防"同配置不同形状互覆"；
    * 历史缺陷（离朱第 2 轮 D4）：早期实现直接复用 `config_fingerprint`，产出
      `full_verify_0_...` 这种带限批前缀的误导性名字；此处独立构造、不带该前缀。
    * **`{fc_part}`（第 3 轮新增）**：`fc_dim != 0` 时插入 `_fc{n}` 段，防止同
      `N`/`seed` 的不同 `fc_dim` 同名互覆；`fc_dim == 0` **不加段**（该段口径与改动前逐字相同）。
    * **`_nosyn`（第 4 轮新增，恒定插入）**：产物格式段 —— 8 个突触类张量自本轮起
      `persistent=False`、不再落盘，故产物名必须与旧格式产物可区分（防"新旧格式同名互覆"）。
      位置口径：**紧随 `_fc{n}` 段之后**（`fc_dim == 0` 时紧随 scope 段之后）、
      位于 `_s{seed}` 之前；与 `smoke_fingerprint` / `config_fingerprint` 三处一致。

    参数
    ----
    config : Config
        本次运行实际生效的配置。
    tag : str
        可选标签。

    返回
    ----
    str
        产物文件名（不含目录）。
    """
    fc_part = f"_fc{int(config.fc_dim)}" if int(config.fc_dim) != 0 else ""
    # `_nosyn`（第 4 轮新增）恒定插入，紧随 `_fc{n}` 段之后、`_s{seed}` 之前
    # （产物不再落盘 8 个突触类张量，格式段必须进名字以与旧格式产物区分）。
    nosyn_part = "_nosyn"
    # `_geo{mode}_k{k}`（第 5 轮新增）：**仅 `geo_field != "none"` 时插入**，
    # 位置 = 紧随 `_fc{n}` 段之后、`_nosyn` 段之前（三处指纹逐字同一口径）。
    geo_part = (
        (
            f"_geo{config.geo_field}_k{int(config.geo_rbf_k)}"
            + ("_sd" if bool(config.geo_signed_delta) else "")
            # [!] `_a{alpha_init:g}`：离朱 DEF-7 —— `geo_alpha_init` 原先**未进指纹**，
            #     使 `--geo-alpha-init 0.4` 与默认 `1.0` 生成**同名产物**并**静默互覆**
            #     （实测 SHA `01e56cb7…` -> `7b89b4ef…`）。口径：**仅非默认值（!= 1.0）
            #     时插入**，故默认组合的产物名逐字不变（既有产物名与 README 口径不受影响）。
            + (f"_a{float(config.geo_alpha_init):g}"
               if float(config.geo_alpha_init) != 1.0 else "")
        )
        if str(config.geo_field) != "none" else ""
    )
    # ---- 数据集通用层（第 6 轮新增）：`_ds{name}` / `_d{D}x{C}`；口径与段位见
    #      `smoke_fingerprint` 的同一段注释（三处必须逐字一致）。 ----
    ds_part, dim_part = dataset_name_parts(config)
    name = (
        f"full_{config.shape_tag()}"
        f"_N{config.N}"
        f"_y{config.y_in}x{config.y_out}"
        f"_H{config.H:g}"
        f"_D{config.D:g}"
        f"_pl{config.placement}"
        f"_ax{config.flow_axis}"
        f"_is{scope_abbrev(config.input_scope)}"
        f"_rs{scope_abbrev(config.readout_scope)}"
        f"{fc_part}"
        f"{geo_part}"
        f"{ds_part}"
        f"{dim_part}"
        f"{nosyn_part}"
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
       `checkpoints/n3d_shape/_verify/`：用户显式给了 `--checkpoint` 则沿用其文件名，
       否则使用带配置指纹的 `verify_<...>.pt`（`config` 为 None 时退化为 `verify_<bpe>.pt`），
       **绝不覆盖正式产物**；发生重定位时会打印「重定位前路径 -> 重定位后路径」；
    2. `checkpoint_override` 非空 -> 使用用户显式指定的路径（相对路径按工程根目录解析）；
    3. 否则 -> 正式全量默认路径：与 `DEFAULT_CONFIG` 完全同配置时用 `model.pt`，
       其它配置使用 `full_<指纹>.pt`，保证不同几何/判据/容量的实验点各自留痕。

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
    if config is not None and not is_default_config(config):
        path = os.path.join(CHECKPOINT_DIR, full_checkpoint_name(config, tag))
        log_info(f"[产物保护] 正式全量训练使用配置指纹产物名（不同配置互不覆盖）：{path}")
        return path
    return CHECKPOINT_PATH


def is_default_config(config: Config) -> bool:
    """判断配置是否与 `DEFAULT_CONFIG` 完全一致（仅影响默认产物文件名）。

    该判定只影响**默认产物文件名**（与 `DEFAULT_CONFIG` 一致时保持 `model.pt` 语义），
    不影响任何数值计算。

    参数
    ----
    config : Config
        本次生效的配置。

    返回
    ----
    bool
        `True` 表示与 `DEFAULT_CONFIG` 的全部字段一致。
    """
    return config.to_dict() == DEFAULT_CONFIG.to_dict()


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
        `0`（默认）= 不干预；正数则调用 `torch.set_num_threads(threads)`。

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
    batch_size_override: int = 0,
    readout_bias_override: Optional[bool] = None,
    n_override: int = 0,
    y_in_override: int = 0,
    y_out_override: int = 0,
    h_override: float = 0.0,
    d_override: float = 0.0,
    flow_axis_override: str = "",
    space_radius_override: float = -1.0,
    input_scope_override: str = "",
    readout_scope_override: str = "",
    placement_override: str = "",
    shape_override: str = "",
    cyl_aspect_override: float = -1.0,
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
        `checkpoints/n3d_shape/_verify/`，避免覆盖正式产物。
    checkpoint_path : str
        checkpoint 保存路径（空串表示使用默认 `checkpoints/n3d_shape/model.pt`）。
    backup : bool
        覆盖已有 checkpoint 前是否先备份为 `<path>.bak`。默认 True。
    preset : str
        预设名（small / default / highacc）。默认 "default"。
    lr_override / weight_decay_override / batch_size_override / readout_bias_override /
    n_override / y_in_override / y_out_override / h_override / d_override :
        兼容模式（`config is None`）下的字段覆盖；哨兵语义与 CLI 一致。
    flow_axis_override / space_radius_override / input_scope_override /
    readout_scope_override / placement_override :
        几何与判据字段覆盖（空串 / -1.0 表示不覆盖）。
    shape_override / cyl_aspect_override :
        **形状字段覆盖**（本模块新增；空串 / -1.0 表示不覆盖）。`shape_override`
        取 `sphere` / `cube` / `cylinder`；`cyl_aspect_override` 仅对 cylinder 生效。
    tag : str
        追加到限批验证产物文件名末尾的标签（仅限批模式生效）。
    arch : str
        架构选择（neuron3d / mlp）。

    返回
    ----
    Dict[str, float]
        最终统计：{"test_acc", "loss", "num_edges", "avg_out_degree", "params",
        "checkpoint", "elapsed_s"}。

    说明
    ----
    **覆盖与校验的唯一入口是 `apply_overrides` / `validate_override_args`**
    （`main()` 传入的 `config`）。本函数不再自行处理覆盖，避免出现两套并存实现。
    当 `config is None` 时，退回到"由各 `*_override` 参数拼装"的兼容模式。
    优化器：`weight_decay > 0` 用 AdamW，否则用 Adam。
    """
    if config is not None:
        return _run_training_with_config(
            config, preset, max_batches, checkpoint_path, backup, tag, arch
        )
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
    if batch_size_override > 0:
        overrides["batch_size"] = batch_size_override
    if readout_bias_override is not None:
        overrides["readout_bias"] = bool(readout_bias_override)
    if n_override > 0:
        overrides["N"] = n_override
    if y_in_override > 0:
        overrides["y_in"] = y_in_override
    if y_out_override > 0:
        overrides["y_out"] = y_out_override
    if h_override > 0.0:
        overrides["H"] = h_override
    if d_override > 0.0:
        overrides["D"] = d_override
    if flow_axis_override:
        overrides["flow_axis"] = flow_axis_override
    if space_radius_override >= 0.0:
        overrides["space_radius"] = space_radius_override
    if input_scope_override:
        overrides["input_scope"] = input_scope_override
    if readout_scope_override:
        overrides["readout_scope"] = readout_scope_override
    if placement_override:
        overrides["placement"] = placement_override
    if shape_override:
        overrides["shape"] = shape_override
    if cyl_aspect_override != -1.0:
        overrides["cyl_aspect"] = cyl_aspect_override
    return _run_training_with_config(
        Config(**overrides), preset, max_batches, checkpoint_path, backup, tag, arch
    )


def build_param_groups(
    model: torch.nn.Module, config: Config
) -> List[Dict[str, object]]:
    """构造优化器的 **param groups**（几何权重场参数排除 weight decay）。

    职责
    ----
    把"几何场参数 `geo_rbf_theta` / `geo_alpha` 排除 `weight_decay`"这一口径**集中到一个
    可被外部调用的函数**里（皋陶 F1 修复的配套改动）。理由：该逻辑原先内联在
    `_run_training_with_config` 中，导致验证脚本只能**重新实现一遍**（同源自洽、
    无法守护真实代码路径 —— 实测把 F1 缺陷注入 `train.py` 后，重实现的判据**抓不到**）。

    口径
    ----
    * `geo_field == "none"`（无几何参数）：**单一 param group**，且**必须显式携带**
      `"weight_decay": float(config.weight_decay)`；
    * `geo_field != "none"`：两个 group —— 第 0 组为其余参数（携带 `config.weight_decay`），
      第 1 组为几何参数（恒 `0.0`）。

    [!] 为什么"必须显式携带"（皋陶 F1，error）：本模块的调用点以
    `AdamW(param_groups, lr=..., weight_decay=0.0)` 构造，构造器参数会**覆盖** group 的
    默认值；若 group 字典里**没有** `weight_decay` 键，该组的 wd 就取构造器给的 `0.0` ——
    即把关闭路径的 weight decay **静默关成 0**（改动前是
    `AdamW(model.parameters(), lr, weight_decay=config.weight_decay)`），属关闭路径行为回归。

    参数
    ----
    model : torch.nn.Module
        已构造的模型（`named_parameters()` 的顺序即 group 内参数顺序）。
    config : Config
        本次生效的配置（读取 `weight_decay`）。

    返回
    ----
    List[Dict[str, object]]
        可直接传给 `torch.optim.Adam` / `AdamW` 的 param groups 列表。
    """
    geo_param_names = ("geo_rbf_theta", "geo_alpha")
    geo_params = [p for n, p in model.named_parameters() if n in geo_param_names]
    other_params = [p for n, p in model.named_parameters() if n not in geo_param_names]
    if geo_params:
        return [
            {"params": other_params, "weight_decay": float(config.weight_decay)},
            {"params": geo_params, "weight_decay": 0.0},
        ]
    # 关闭路径：单一 group，**必须显式携带 weight_decay**（见上方 F1 说明）
    return [{"params": other_params, "weight_decay": float(config.weight_decay)}]


def build_optimizer(
    model: torch.nn.Module, config: Config
) -> torch.optim.Optimizer:
    """按配置构造优化器（`weight_decay > 0` -> AdamW，否则 Adam）。

    参数
    ----
    model : torch.nn.Module
        已构造的模型。
    config : Config
        本次生效的配置。

    返回
    ----
    torch.optim.Optimizer
        优化器实例；其 `param_groups` 的**生效** `weight_decay` 由
        `build_param_groups` 保证（`AdamW` 构造器传 `weight_decay=0.0`，
        真实值一律由各 param group 携带）。
    """
    groups = build_param_groups(model, config)
    if float(config.weight_decay) > 0.0:
        return torch.optim.AdamW(groups, lr=config.lr, weight_decay=0.0)
    return torch.optim.Adam(groups, lr=config.lr)


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
    arch : str
        架构选择。

    返回
    ----
    Dict[str, float]
        最终统计字典（含 checkpoint 绝对路径与总耗时）。
    """
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
    # ---- 参数分组（第 5 轮新增）：几何场参数**排除 weight decay** ----
    # 归因纯度要求：几何场参数 `geo_rbf_theta`（零初始化）与 `geo_alpha` 若被 `weight_decay`
    # 拉向 0，等价于对几何场施加**隐式正则**，使"场是否有效"的归因被污染。故按
    # **参数分组**把这两者放进 `weight_decay = 0` 的独立 group；其余参数照旧。
    # `clip_grad_norm_` 仍覆盖**全参数**（它接收 `model.parameters()`，与分组无关）。
    # [!] 分组与优化器构造已抽成 `build_param_groups` / `build_optimizer`（皋陶 F1 配套改动）：
    #     这样**验证脚本与变异驱动器可以直接调用真实代码路径**，而不是"重新实现一遍"
    #     （后者是同源自洽、抓不到注入了 F1 缺陷的 train.py —— 实测验证过）。
    geo_param_names = ("geo_rbf_theta", "geo_alpha")
    geo_params = [p for n, p in model.named_parameters() if n in geo_param_names]
    param_groups = build_param_groups(model, config)
    if geo_params:
        log_info(
            "几何权重场参数分组：geo_rbf_theta / geo_alpha 排除 weight_decay"
            f"（共 {sum(p.numel() for p in geo_params)} 个参数）；"
            f"其余 {sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in geo_params)}"
            f" 个参数按 weight_decay={float(config.weight_decay)}"
        )
    optimizer: torch.optim.Optimizer = build_optimizer(model, config)
    if config.weight_decay > 0.0:
        # [!] 皋陶审查 F1 第 2 点：日志必须按**生效值**打印（原先把 `config.weight_decay`
        #     直接打进日志，在修复前会打印 `weight_decay=1e-4` 而实际为 0，属**误导性日志**）。
        effective_wds = sorted(
            {float(g.get("weight_decay", 0.0)) for g in optimizer.param_groups}
        )
        log_info(
            f"优化器：AdamW(lr={config.lr}, 生效 weight_decay="
            f"{'|'.join(f'{w:g}' for w in effective_wds)}"
            f"{'，几何场参数组 weight_decay=0' if geo_params else ''})"
        )
    else:
        log_info(f"优化器：Adam(lr={config.lr})")
    # ---- 契约断言（F1 常驻防线）：生效 wd 必须与 config 逐组一致 ----
    # 关闭路径（单组）必须恰好等于 `config.weight_decay`；开启路径第 0 组同值、几何组恒 0。
    _expr = [float(g.get("weight_decay", 0.0)) for g in optimizer.param_groups]
    if geo_params:
        assert len(_expr) == 2 and _expr[0] == float(config.weight_decay) and _expr[1] == 0.0, (
            "[契约失败] 几何场参数分组的生效 weight_decay 与配置不符："
            f"实测 {_expr}，期望 [{float(config.weight_decay)}, 0.0]"
        )
    else:
        assert len(_expr) == 1 and _expr[0] == float(config.weight_decay), (
            "[契约失败] 关闭路径的生效 weight_decay 与配置不符（F1 回归！）："
            f"实测 {_expr}，期望 [{float(config.weight_decay)}]"
        )
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
        last_loss, _, n_batches, _, _ = train_one_epoch(
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
    backup_path = backup_existing_checkpoint(save_path, enabled=backup)
    if backup_path:
        log_info(f"[产物保护] 覆盖前已备份既有 checkpoint：{backup_path}")
    stats = model.get_connection_stats()
    topo_stats = model.get_topology_stats() if arch == "neuron3d" else {}
    dag_selfcheck = model.connectivity_selfcheck() if arch == "neuron3d" else {}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "connection_stats": stats,
            "topology_stats": topo_stats,
            "dag_selfcheck": dag_selfcheck,
            "test_acc": test_acc,
            "epochs": config.epochs,
            "batches_per_epoch": (max_batches if max_batches > 0 else None),
            "preset": preset,
            "arch": arch,
            "placement": config.placement,
            "flow_axis": config.flow_axis,
            "space_radius": config.space_radius,
            # ---- 形状维度（本模块新增；产物必须自带形状取证信息）----
            "shape": config.shape,
            "cyl_aspect": config.cyl_aspect,
            "shape_tag": config.shape_tag(),
            # ---- fc_dim（第 3 轮新增；产物自带开关与有效宽度取证）----
            "fc_dim": int(config.fc_dim),
            "fc_width": int(config.fc_width),
            # ---- 几何权重场（第 5 轮新增；产物自带开关 / 基 / 特征列取证）----
            # [!] `edge_geo_feat` 等特征张量为 `persistent=False`（与突触类 buffer 同口径），
            #     故复核须回到 `config + seed` 重算；此处只落盘**标量取证字段**。
            "geo_field": config.geo_field,
            "geo_rbf_k": int(config.geo_rbf_k),
            "geo_hidden": int(config.geo_hidden),
            "geo_alpha_init": float(config.geo_alpha_init),
            "geo_signed_delta": bool(config.geo_signed_delta),
            "geo_edge_feature_names": (
                list(model._geo_feature_names) if getattr(model, "geo_enabled", False) else []
            ),
            "geo_alpha_final": (
                float(model.geo_alpha.detach().item())
                if getattr(model, "geo_enabled", False) else 0.0
            ),
            "geo_theta_norm_final": (
                float(model.geo_rbf_theta.detach().norm().item())
                if getattr(model, "geo_enabled", False) else 0.0
            ),
            "circum_coef": config.circum_coef,
            "shape_circum_radius": config.shape_circum_radius,
            "selection_metric": float(getattr(model, "selection_metric", float("nan"))),
            "selection_metric_within_space": bool(
                getattr(model, "selection_metric_within_space", False)
            ),
            # ---- W2/I1：放置半径 vs 窗口/外接半径的诊断量（产物自带可复核证据）----
            "placement_radius": float(getattr(model, "placement_radius", float("nan"))),
            "placement_radius_over_rmax": float(
                getattr(model, "placement_radius_over_rmax", float("nan"))
            ),
            "placement_radius_over_rmin": float(
                getattr(model, "placement_radius_over_rmin", float("nan"))
            ),
            "placement_within_rmax": bool(getattr(model, "placement_within_rmax", False)),
            "placement_within_space": bool(getattr(model, "placement_within_space", False)),
            "placement_within_circum": bool(getattr(model, "placement_within_circum", False)),
            "placement_within_circum_ratio": float(
                getattr(model, "placement_within_circum_ratio", float("nan"))
            ),
            "effective_space_radius": config.effective_space_radius,
            "min_space_radius": config.min_space_radius,
            "max_space_radius": config.max_space_radius,
            # ---- 数据集通用层（第 6 轮新增；正式产物同样自带来源/维度取证）----
            "dataset": config.dataset,
            "dataset_path": str(config.dataset_path or ""),
            "dataset_kind": DATASET_SPECS[str(config.dataset)].kind,
            "num_samples": int(config.num_samples),
            "input_dim": int(config.effective_input_dim),
            "output_dim": int(config.effective_output_dim),
            "dataset_spec_input_dim": int(DATASET_SPECS[str(config.dataset)].input_dim),
            "dataset_spec_num_classes": int(DATASET_SPECS[str(config.dataset)].num_classes),
            "dims_source": config.dims_source,
            "norm_mean_arg": config.norm_mean,
            "norm_std_arg": config.norm_std,
            "input_scope": config.input_scope,
            "readout_scope": config.readout_scope,
            "lr_schedule": config.lr_schedule,
            "weight_decay": config.weight_decay,
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
    log_info(f"  空间形状             ：{config.shape_spec.describe()}"
             f"（circum_coef={config.circum_coef:.6f}，λ={config.cyl_aspect:g}）")
    log_info(f"  可学习参数总数       ：{model.count_parameters()}")
    log_info(f"  神经元级连接数 E     ：{int(stats['num_edges'])}")
    log_info(f"  平均/最大出度        ：{stats['avg_out_degree']:.4f} / {int(stats['max_out_degree'])}")
    log_info(f"  平均/最大入度        ：{stats['avg_in_degree']:.4f} / {int(stats['max_in_degree'])}")
    log_info(f"  层数（流向轴）K      ：{int(stats['num_layers'])}")
    if topo_stats:
        log_info(
            f"  几何                 ：placement={config.placement} / 流向轴 {config.flow_axis} "
            f"/ R={config.effective_space_radius:.6f}（R_min={config.min_space_radius:.6f}, "
            f"R_max={config.max_space_radius:.6f}）"
        )
        log_info(
            f"  形状放置诊断         ：selection_metric={topo_stats['selection_metric']:.6f}, "
            f"placement_radius={topo_stats['placement_radius']:.6f}, "
            f"metric_in_space={int(topo_stats['selection_metric_within_space'])}, "
            f"extent_xy={topo_stats['neuron_extent_xy']:.6f}, "
            f"extent_axis={topo_stats['neuron_extent_axis']:.6f}"
        )
        log_info(f"  最近邻距 / 2H        ：{topo_stats['nearest_neighbour_dist']:.9f} / {2.0 * config.H:.9f}")
        log_info(
            f"  判据                 ：input_scope={config.input_scope}, "
            f"readout_scope={config.readout_scope}；S_in={int(stats['num_in_scope'])}, "
            f"S_out={int(stats['num_out_scope'])}"
        )
        log_info(f"  双副本神经元数       ：{int(topo_stats['dual_copy_count'])}")
        log_info(
            f"  DAG 自检             ：无环={int(dag_selfcheck.get('dag_acyclic', 0))}, "
            f"严格上行={int(dag_selfcheck.get('all_edges_uphill', 0))}"
        )
    log_info(f"  最终 test_acc        ：{test_acc * 100:.2f}%")
    log_info(f"  优化器               ：{'AdamW' if config.weight_decay > 0 else 'Adam'}")
    log_info(f"  lr_schedule          ：{config.lr_schedule}")
    log_info(f"  readout_bias         ：{config.readout_bias}")
    log_info(
        f"  geo_field            ：{config.geo_field}"
        f"（k={config.geo_rbf_k}, alpha_init={config.geo_alpha_init}, "
        f"signed_delta={int(bool(config.geo_signed_delta))}）"
    )
    log_info(f"  grad_clip            ：{config.grad_clip}")
    log_info(f"  checkpoint           ：{os.path.abspath(save_path)}")
    log_info("-" * 78)

    return {
        "test_acc": float(test_acc),
        "loss": float(last_loss),
        "num_edges": float(stats["num_edges"]),
        "avg_out_degree": float(stats["avg_out_degree"]),
        "max_out_degree": float(stats["max_out_degree"]),
        "num_in_scope": float(stats["num_in_scope"]),
        "num_out_scope": float(stats["num_out_scope"]),
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
        * `2` 命令行参数非法（`validate_override_args` 抛出的 `ValueError`）。
    """
    args = parse_args(argv)
    try:
        validate_override_args(args)
        # ---- 数据集命名可区分性断言（第 6 轮新增，常驻防线）----
        # 代价极小（5 个数据集 x 3 处指纹），但它把"数据集维度漏进指纹 -> 同名互覆"
        # 这类缺陷挡在**任何**产物写入之前；缺失该段的后果是静默覆盖既有取证产物。
        assert_dataset_name_distinguishable()
        apply_threads(args.threads)
        if args.smoke_test:
            ok = run_smoke_test(
                device_override=args.device,
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
            shape_override=args.shape,
            cyl_aspect_override=args.cyl_aspect,
        )
    except ValueError as exc:
        log_error(f"参数校验失败：{exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
