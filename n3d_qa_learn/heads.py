"""n3d_qa_learn 的输出头：``D`` 维 ``q`` 头 + 两种互斥的候选打分实现（全局开关）。

结构（固定，不可配置）
----------------------
::

    x  ->  [N3D 后端：BackendAdapter.features]  ->  f in R^[B, D]
                                                    |
                                              q = q_head(f) in R^[B, D]
                                                    |
              +-------------------------------------+-------------------------------------+
              |                                                                           |
     [index] logits = s * (q @ A^T)，A in R^[C+1, D] 为**可学习候选键表**   [pointer] logits = s * (q @ K^T)
              |      （答案表 + 末位「不相关」）                                          K in R^[B, L, D] 由输入提供
              +-------------------------------------+-------------------------------------+

两个实现的关键差别（必须守住）
------------------------------
* **index（索引生成式）**：``A`` 是**模型参数**，随产物落盘；答案表内容与口径指纹写入 meta，
  加载时校验，不一致立即报错。**只有这一模式有候选键参数**。
* **pointer（指针 Softmax）**：候选键表 ``K`` **由输入提供**（来自确定性文本特征），
  **不存任何固定候选参数表**；``A`` 参数在该模式下**根本不被创建** —— 这是可断言的结构事实。
* 两模式**共用同一个训练出的 ``q``**（``q_head`` 是唯一被两模式共享的打分前端）。

``D`` 的落点
------------
``q`` 的维度恒等于 ``D``（**不是**类别数宽度）。``describe()`` 与产物 meta 必须落 ``D``。

对齐机制（本轮新增，默认**关闭**）
--------------------------------
本模块新增两条**可选**的对齐通路，把「N3D 读出」与「编码器特征空间」显式耦合：

* **机制 B（``align_mode="proj_supcon"``）**：N3D 支路接一个线性**投影头** ``D -> D``
  （:attr:`N3DQAConfig.proj_dim` 控制开关），对齐损失 = 批内监督对比损失
  （:func:`supcon_loss`，同答案互为正样本）；总损失 = ``ce + λ · supcon``；
* **机制 C（``align_mode="distill"``）**：不建投影头，对齐损失 = 显式蒸馏
  ``0.5 · [(1 - cos(n3d, 本样本编码器特征)) + (1 - cos(n3d, 本样本所属类质心))]``，
  类质心**批内现场算、detach（不参与梯度）**；总损失 = ``ce + λ · distill``。

纪律：``align_mode="off"`` 时**不创建任何模块、不消耗全局 RNG**，``query()`` 的路径与
改动前**逐字符相同**；投影头初始化走**局部 generator**（固定种子
:data:`PROJ_INIT_SEED`），绝不复用 ``q_head`` 的近恒等先验（那是 ``raw`` 档的先验）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backends import BackendAdapter

#: 输出模式（全局开关，二选一；**不做级联**）。
OUTPUT_MODES: Tuple[str, ...] = ("index", "pointer")

#: ``q`` 头的输入来源模式（见 :attr:`N3DQAConfig.head_input_mode`）。
HEAD_INPUT_MODES: Tuple[str, ...] = ("n3d", "concat", "raw")

#: 「不相关」类在 answer 表 / 候选键表中的固定下标口径：恒为**末位**。
IRRELEVANT_AT_END: bool = True

#: 答案表口径：``"centroid"``（默认，由训练样本的**逐类质心**确定性算出并归一化后固化）
#: 或 ``"free"``（自由可学习参数）。
ANSWER_TABLE_MODES: Tuple[str, ...] = ("centroid", "free")

#: 对齐机制（全局开关；取值即口径，禁止改名）。
#: ``off`` = 关闭（默认，逐位复现历史行为）；
#: ``proj_supcon`` = 机制 B（投影头 + 监督对比损失）；
#: ``distill`` = 机制 C（显式蒸馏：本样本编码器特征 + 所属类质心）。
ALIGN_MODES: Tuple[str, ...] = ("off", "proj_supcon", "distill")

#: 投影头初始化口径（唯一注册点；**都不允许近恒等**）。
PROJ_INIT_MODES: Tuple[str, ...] = ("xavier_uniform", "orthogonal")

#: 投影头初始化使用的**局部 generator 种子**（冻结常量）。
#: 口径纪律：用局部 ``torch.Generator`` 播种，绝不消耗全局 RNG（否则会改变同一次运行里
#: 后续所有采样，破坏对照组之间的可比性）。
PROJ_INIT_SEED: int = 20261105

#: 对齐目标（**逐项可报**；两项同时进入 ``distill`` 损失，并分别登记）。
#: ``self_encoder_feature`` = 目标 (i)：本样本的编码器原始特征（逐样本蒸馏）；
#: ``class_centroid`` = 目标 (ii)：本样本所属类的质心（批内现场算、detach、不参与梯度）。
ALIGN_TARGETS: Tuple[str, ...] = ("self_encoder_feature", "class_centroid")

#: 监督对比损失的温度（冻结常量；**唯一实现落点在本模块**，``exp_repr`` 只做再导出）。
SUPCON_TEMPERATURE: float = 0.07


@dataclass(frozen=True)
class N3DQAConfig:
    """``n3d_qa_learn`` 模型的完整配置（构造期不变量）。

    参数
    ----
    dim : int
        连接参数 ``D``（同时是后端 ``input_dim`` / ``output_dim`` 与 ``q`` 的宽度）。
    output_mode : str
        ``"index"`` 或 ``"pointer"``（全局一个开关）。
    head_input_mode : str
        **喂给 ``q`` 头的连接口径**（本模块最重要的实测口径之一，三种取值）：
        ``"n3d"`` = 只用 N3D 读出向量 ``[B, D]``（最保守的 N3D 口径）；
        ``"concat"`` = 原始 ``D`` 维文本特征（L2 归一、经无仿射 ``LayerNorm``）与
        N3D 读出向量做**凸混合** ``alpha * raw + (1-alpha) * n3d``（``alpha`` 可学习、
        ``sigmoid`` 参数化，两路都不会被抹掉）；
        ``"raw"`` = 直通原始 ``D`` 维文本特征（N3D 读出**不进入** ``q``）。
        **现场实测（同一数据切分、同一 ``seed=42``，不得凭直觉选择）**：主测试集
        宏平均准确率 ``raw 0.2479`` / ``concat 0.0000~0.028`` / ``n3d 0.000~0.083``。
        即：本任务的可分性信息集中在确定性文本特征上，把 N3D 读出**混合进** ``q``
        反而显著有害（N3D 读出的低秩结构在 1e3 量级样本上会把判据带偏）。
        该字段只改变 ``q`` 的取值口径，**不改变**「问题 -> ``q`` -> 候选键打分」
        的整体结构与后端（N3D）的挂载方式。
    label_smoothing : float
        交叉熵标签平滑；``>= 0``。
    normalize_query : bool
        是否对 ``q`` 做 L2 归一化（默认 ``True``）。
        存在理由（实测口径）：不归一化时 ``q`` 的模长是自由参数，训练会走"把
        ``q`` 的模长放大到几个数量级"这条捷径，在 1e3 量级样本上迅速过拟合
        （实测：主测试集宏平均准确率从 ``0.2271`` 掉到 ``0.16`` 以下）。
        归一化后 ``logits`` 是有界余弦结构，优化更稳。
    logit_scale_init : float
        **可学习** logits 尺度 ``s``（逆温度）的初值（``> 0``），两种模式共用。
        存在理由：``q`` 与候选键均为 L2 归一化向量时 ``q @ K^T`` 落在 ``[-1, 1]``，
        若 ``s`` 恒为 1 则 logits 尺度极小、交叉熵饱和在 ``log C`` 附近、梯度信号过弱。
    mix_logit_init : float
        ``concat`` 模式下两路凸混合权重 ``alpha = sigmoid(mix_logit)`` 的初值
        （``> 0`` 偏向 N3D 读出，``< 0`` 偏向原始文本特征）。宽度口径见
        :attr:`N3DQAConfig.head_input_mode`。
    answer_table_mode : str
        ``index`` 模式下候选键表的来源（``"centroid"`` 默认 / ``"free"``）。
        ``"centroid"`` = 由训练样本的**逐类质心**确定性算出、L2 归一化后**固化为
        buffer**（不是可学习参数）。存在理由（现场实测，不得凭直觉删除）：本任务
        每类仅 6~22 个训练样本，自由可学习的 ``C x D`` 答案表会在训练集上饱和到
        100% 而主测试集宏平均准确率掉到 ``0.09`` 量级；质心口径给出强几何先验，
        同一数据切分下实测 ``0.28`` 以上。该口径**不改变**「答案表 -> 候选键 ->
        ``q @ A^T``」的打分结构，只是把答案表的取值来源由"自由学习"改为"训练样本质心"。
    align_mode : str
        **对齐机制**（默认 ``"off"`` = 关闭，逐位复现历史行为）。取值见
        :data:`ALIGN_MODES`。开启时**要求** ``head_input_mode ∈ {"concat", "n3d"}``：
        对齐的左边是 N3D 读出，``raw`` 档下 ``adapter.features`` 根本不被调用
        （现场实测），此时"开启对齐"只能是静默空转，故构造期直接报错。
    align_lambda : float
        对齐损失权重 ``λ``（总损失 = ``ce + λ · align``）。``align_mode == "off"``
        时必须为 ``0.0``；开启时必须 ``> 0``。
    proj_dim : int
        **投影头宽度开关**（仅机制 B 使用）：``0`` = 关闭（默认，**不创建任何模块**）；
        ``-1`` = 跟随 ``D``（即 ``D -> D``，本轮确认口径）；``> 0`` = 显式宽度，
        必须等于 ``D``（该投影头位于 N3D 支路与混合之间，混合要求两路同维），否则
        构造期报错。``> 0`` 时 ``align_mode`` 必须为 ``"proj_supcon"``。
    proj_init : str
        投影头初始化口径（取值见 :data:`PROJ_INIT_MODES`）。**不得沿用近恒等**
        （``q_head`` 的近恒等是 ``raw`` 档的直通先验，与投影头的职责无关）。
    """

    dim: int
    output_mode: str = "index"
    head_input_mode: str = "raw"
    label_smoothing: float = 0.0
    normalize_query: bool = True
    logit_scale_init: float = 10.0
    learn_logit_scale: bool = False
    mix_logit_init: float = -2.0
    answer_table_mode: str = "centroid"
    align_mode: str = "off"
    align_lambda: float = 0.0
    proj_dim: int = 0
    proj_init: str = "xavier_uniform"

    def __post_init__(self) -> None:
        if int(self.dim) < 1:
            raise ValueError(f"N3DQAConfig.dim 必须 >= 1，当前 {self.dim}")
        if self.output_mode not in OUTPUT_MODES:
            raise ValueError(
                f"N3DQAConfig.output_mode 仅允许 {list(OUTPUT_MODES)}，"
                f"当前 {self.output_mode!r}"
            )
        if float(self.label_smoothing) < 0.0:
            raise ValueError(
                f"N3DQAConfig.label_smoothing 必须 >= 0，当前 {self.label_smoothing}"
            )
        if self.head_input_mode not in HEAD_INPUT_MODES:
            raise ValueError(
                f"N3DQAConfig.head_input_mode 仅允许 {list(HEAD_INPUT_MODES)}，"
                f"当前 {self.head_input_mode!r}"
            )
        if self.answer_table_mode not in ANSWER_TABLE_MODES:
            raise ValueError(
                f"N3DQAConfig.answer_table_mode 仅允许 {list(ANSWER_TABLE_MODES)}，"
                f"当前 {self.answer_table_mode!r}"
            )
        if not (float(self.logit_scale_init) > 0.0):
            raise ValueError(
                "N3DQAConfig.logit_scale_init 必须 > 0，"
                f"当前 {self.logit_scale_init}"
            )
        # ---- 对齐机制（本轮新增；默认档必须逐位复现历史行为）----
        if self.align_mode not in ALIGN_MODES:
            raise ValueError(
                f"N3DQAConfig.align_mode 仅允许 {list(ALIGN_MODES)}，"
                f"当前 {self.align_mode!r}"
            )
        if self.proj_init not in PROJ_INIT_MODES:
            raise ValueError(
                f"N3DQAConfig.proj_init 仅允许 {list(PROJ_INIT_MODES)}，"
                f"当前 {self.proj_init!r}"
            )
        if int(self.proj_dim) < -1:
            raise ValueError(
                f"N3DQAConfig.proj_dim 只允许 -1（跟随 D）或 >= 0（0 = 关闭），"
                f"当前 {self.proj_dim}"
            )
        if int(self.proj_dim) > 0 and int(self.proj_dim) != int(self.dim):
            raise ValueError(
                "投影头位于 N3D 支路与凸混合之间，混合要求两路同维 D；"
                f"proj_dim={self.proj_dim} 与 dim={self.dim} 不一致。"
                "请用 proj_dim=-1（跟随 D）或 proj_dim=0（关闭）"
            )
        if self.align_mode == "off":
            if int(self.proj_dim) != 0:
                raise ValueError(
                    "对齐关闭（align_mode='off'）时不得启用投影头："
                    f"proj_dim={self.proj_dim}。关闭时不得创建任何模块、不得消耗 RNG"
                )
            if float(self.align_lambda) != 0.0:
                raise ValueError(
                    "对齐关闭（align_mode='off'）时 align_lambda 必须为 0.0，"
                    f"当前 {self.align_lambda}"
                )
        else:
            if self.head_input_mode not in ("concat", "n3d"):
                raise ValueError(
                    f"align_mode={self.align_mode!r} 要求 head_input_mode ∈ "
                    f"{{'concat', 'n3d'}}（对齐的左边是 N3D 读出）；当前 "
                    f"head_input_mode={self.head_input_mode!r}：raw 档下 "
                    "adapter.features 从不被调用，开启对齐只能是静默空转，故直接拒绝"
                )
            if not (float(self.align_lambda) > 0.0):
                raise ValueError(
                    f"align_mode={self.align_mode!r} 时 align_lambda 必须 > 0，"
                    f"当前 {self.align_lambda}"
                )
            if int(self.proj_dim) != 0 and self.align_mode != "proj_supcon":
                raise ValueError(
                    "投影头只属于机制 B（align_mode='proj_supcon'）；"
                    f"当前 align_mode={self.align_mode!r} 但 proj_dim={self.proj_dim}"
                )


class N3DQA(nn.Module):
    """N3D 问答学习模型：后端特征提取器 + ``D`` 维 ``q`` 头 + （index 模式的）候选键表。

    参数
    ----
    adapter : BackendAdapter
        连接契约代理层给出的后端适配器。
    n_answers : int
        答案类别数 ``C``（**不含**「不相关」类）。``index`` 模式下候选键表为 ``[C+1, D]``，
        末位下标 ``C`` 恒为「不相关」。``pointer`` 模式下该参数只用于记录类数，
        不产生任何参数。
    config : N3DQAConfig
        模型配置。

    关键不变量（构造期）
    ------------------
    * ``q_head`` 的输出维恒为 ``D``；
    * ``index`` 模式下 ``answer_table.shape == [C+1, D]``；
    * ``pointer`` 模式下**不存在** ``answer_table`` 属性（``hasattr`` 为 ``False``）。
    """

    def __init__(
        self,
        adapter: BackendAdapter,
        n_answers: int,
        config: N3DQAConfig,
    ) -> None:
        super().__init__()
        if int(n_answers) < 1:
            raise ValueError(f"n_answers 必须 >= 1（至少一个答案类），当前 {n_answers}")
        self.adapter = adapter
        self.backend_name: str = adapter.name
        self.n_answers: int = int(n_answers)
        self.config = config
        dim = int(config.dim)
        if dim != int(adapter.input_dim):
            raise ValueError(
                f"N3DQAConfig.dim={dim} 与后端连接参数 D={adapter.input_dim} 不一致；"
                "连接参数只有唯一来源（代理层），拒绝两处各写一个维度"
            )

        # ---- 共享的打分前端：D 维 q 头（两模式共用同一个训练出的 q） ----
        # 输入宽度恒为 D（concat 模式先在**特征层**做凸混合再进头，见 `query()`）
        self.q_head = nn.Linear(dim, dim, bias=True)
        if config.head_input_mode == "concat":
            # 对 N3D 读出做 LayerNorm（**elementwise_affine=False**，不引入可学习参数）
            self.backbone_norm = nn.LayerNorm(dim, elementwise_affine=False)
            # 两路的凸混合权重（标量，sigmoid 参数化 -> 恒在 (0,1)，两路谁都不会被抹掉）
            self.mix_logit = nn.Parameter(torch.tensor(float(config.mix_logit_init)))

        # ---- 投影头（**仅机制 B**；默认关闭时一个模块都不建、一点 RNG 都不消耗）----
        # 口径：线性 D -> D，位于 N3D 支路与凸混合之间（见 `n3d_branch` / `query`）。
        self.proj: Optional[nn.Linear] = None
        if int(config.proj_dim) != 0:
            # [!] nn.Linear.__init__ 的 reset_parameters 会消耗**全局** RNG，进而改变同一次
            #     运行里后续所有采样（数据打乱用独立的 torch.Generator，但骨干/其它模块的
            #     初始化顺序会漂移）。这里用 fork_rng 把全局 RNG 状态原样保存并恢复，
            #     保证「启用投影头」不改变其它任何模块的随机数消耗。
            with torch.random.fork_rng(devices=[]):
                self.proj = nn.Linear(dim, dim, bias=True)

        # ---- 候选键表：**仅 index 模式**存在 ----
        # 两种来源：free（可学习参数）/ centroid（训练样本质心，构造后由
        # `set_answer_table_from_centroids` 确定性写入；**不作为可学习参数**）。
        if config.output_mode == "index":
            if config.answer_table_mode == "free":
                self.answer_table = nn.Parameter(torch.empty(self.n_answers + 1, dim))
            else:
                self.register_buffer("answer_table", torch.zeros(self.n_answers + 1, dim))
        # pointer 模式：候选键由输入提供，**不创建任何固定候选参数表**

        # ---- 两模式共用的可学习 logits 尺度 ----
        # 尺度在 test 侧默认**固定**（见 N3DQAConfig.learn_logit_scale 的实测理由）
        self.logit_scale = nn.Parameter(
            torch.tensor(float(config.logit_scale_init)),
            requires_grad=bool(config.learn_logit_scale),
        )

        self.reset_parameters()

    # -- 初始化 -----------------------------------------------------------
    #: ``q`` 头权重的零初始化扰动幅度（冻结常量；见 ``reset_parameters`` 的实测理由）。
    HEAD_INIT_NOISE: float = 0.02

    def reset_parameters(self) -> None:
        """确定性初始化（不消耗全局 RNG：使用局部 generator，种子固定为 42+7）。

        初始化口径（现场实测选定，不得凭直觉改成均匀小随机）
        ---------------------------------------------------
        ``q_head`` 权重初始化为 **[近恒等]**：``W = I_block + 0.02 * U(-1,1)``，其中
        ``I_block`` 把「原始文本特征」这一路（``concat`` 模式下位于输入的后半段）
        直通到输出，偏置零初始化，``logit_scale`` 初值由配置给出。

        实测理由：均匀小随机初始化 + 全量可学习头部在 1e3 量级样本上会**在训练集上
        迅速饱和到 100%**、主测试集宏平均准确率掉到 ``0.08`` 量级；近恒等初始化
        等价于"从余弦最近邻出发再微调"，同一数据切分下实测可达 ``0.25`` 以上。
        """
        gen = torch.Generator().manual_seed(49)
        with torch.no_grad():
            dim = int(self.config.dim)
            w = self.q_head.weight
            w.zero_()
            noise = (torch.rand(w.shape, generator=gen) * 2.0 - 1.0) * self.HEAD_INIT_NOISE
            w.copy_(noise)
            # 近恒等直通：混合向量 -> q（初始时 q 基本等于混合向量本身）
            w += torch.eye(dim)
            self.q_head.bias.zero_()
            if self.config.output_mode == "index" and self.config.answer_table_mode == "free":
                self.answer_table.uniform_(
                    -1.0 / max(1.0, dim ** 0.5), 1.0 / max(1.0, dim ** 0.5), generator=gen
                )
            if self.proj is not None:
                self._init_projection(dim, gen)

    #: 投影头初始化的局部 generator 种子偏移（与 ``q_head`` 的 49 分开，互不串用）。
    PROJ_INIT_SEED_OFFSET: int = 17

    def _init_projection(self, dim: int, gen: torch.Generator) -> None:
        """投影头的**确定性**初始化（局部 generator；口径见 :data:`PROJ_INIT_MODES`）。

        口径声明（**不得沿用近恒等**）
        ----------------------------
        ``q_head`` 的近恒等（``W = I + 0.02·U(-1,1)``）是 ``raw`` 档"直通原始文本特征"
        这一先验的载体，与投影头的职责（把 N3D 读出重参数化到一个对齐空间）无关；
        若沿用近恒等，机制 B 在起点上等价于"没有投影头"，测不出任何东西。故此处：

        * ``xavier_uniform``（默认）：``W ~ U(-b, b)``，``b = sqrt(6 / (dim + dim))``，
          偏置零初始化 —— 与 PyTorch ``nn.Linear`` 的默认分布族同形，但**由局部
          generator 生成**，不消耗全局 RNG；
        * ``orthogonal``：由局部 generator 生成高斯矩阵再做 ``QR`` 正交化（范数保持，
          排除"投影头把 N3D 支路量级放缩"这一混杂因素）。

        两种口径都用 ``torch.Generator().manual_seed(PROJ_INIT_SEED)``（+ 偏移）播种，
        因此同一 ``dim`` 下**逐位可复现**，且与全局 RNG 状态无关。
        """
        g = torch.Generator().manual_seed(int(PROJ_INIT_SEED) + int(self.PROJ_INIT_SEED_OFFSET))
        weight = self.proj.weight  # type: ignore[union-attr]
        if self.config.proj_init == "orthogonal":
            raw = torch.randn((dim, dim), generator=g)
            q, r = torch.linalg.qr(raw)
            # QR 的符号规范化（确定性）：让 R 的对角元为正，避免同一旋转的符号歧义
            signs = torch.sign(torch.diagonal(r))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            weight.copy_(q * signs.unsqueeze(0))
        else:
            bound = float((6.0 / float(dim + dim)) ** 0.5)
            weight.copy_(
                (torch.rand((dim, dim), generator=g) * 2.0 - 1.0) * bound
            )
        self.proj.bias.zero_()  # type: ignore[union-attr]

    # -- 打分 -------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        """候选空间宽度：``index`` = ``C+1``；``pointer`` = ``C+1``（候选集口径同宽）。"""
        return self.n_answers + 1

    def answer_index(self) -> int:
        """「不相关」类下标（恒为末位 ``C``）。"""
        return int(self.n_answers)

    def n3d_branch(self, features: torch.Tensor,
                   readout: Optional[torch.Tensor] = None) -> torch.Tensor:
        """N3D 支路的**对外表示** ``[B, D]``：骨干读出 ->（concat 口径）LayerNorm ->（启用的）投影头。

        口径（**顺序固定，不得调换**）
        ----------------------------
        ``readout = adapter.features(features)``
        -> ``concat`` 口径下先过无仿射 ``LayerNorm``（压掉 N3D 读出与 raw 支路约 236 倍的
        量级差，见 :attr:`N3DQAConfig.head_input_mode` 的实测口径）
        -> 启用时再过线性投影头（``D -> D``）-> 返回。

        参数
        ----
        features : torch.Tensor
            ``[B, D]`` 原始文本特征。
        readout : Optional[torch.Tensor]
            已算好的骨干读出 ``[B, D]``；给出时不再重复前向。用于让
            :meth:`query` 与 :meth:`alignment_loss` **共用同一次骨干前向**（既省一半算力，
        也排除"两处各前向一次"带来的口径歧义）。

        返回
        ----
        torch.Tensor
            ``[B, D]``（**不做** L2 归一化；需要归一化时由调用方显式做）。
        """
        f = self.adapter.features(features) if readout is None else readout
        norm = getattr(self, "backbone_norm", None)
        if self.config.head_input_mode == "concat" and norm is not None:
            f = norm(f)
        if self.proj is not None:
            f = self.proj(f)
        return f

    def alignment_representation(self, features: torch.Tensor,
                                 readout: Optional[torch.Tensor] = None) -> torch.Tensor:
        """对齐所用的表示：**L2 归一化后的** :meth:`n3d_branch`（``[B, D]``）。

        投影头存在时即"投影后"的表示（机制 B）；不存在时即 N3D 读出本身。
        """
        return F.normalize(self.n3d_branch(features, readout), dim=1)

    def alignment_parameters(self) -> List[nn.Parameter]:
        """对齐机制新增的可学习参数（投影头）；未启用时返回空列表。"""
        if self.proj is None:
            return []
        return [p for p in self.proj.parameters()]

    def query(self, features: torch.Tensor,
              n3d_branch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``q = q_head(f)``：``[B, D] -> [B, D]``（**不是**类别数宽度的分类头）。

        当 ``config.normalize_query`` 为 ``True`` 时对 ``q`` 做 **L2 归一化**，
        使后续 ``q · key`` 成为有界余弦结构（见 ``N3DQAConfig.normalize_query``）。

        参数
        ----
        features : torch.Tensor
            ``[B, D]`` 原始文本特征。
        n3d_branch : Optional[torch.Tensor]
            已算好的 :meth:`n3d_branch` 输出（``[B, D]``）；给出时不再重复骨干前向。
            **默认路径逐字符不变**：``None`` 时对 ``concat`` / ``n3d`` 两档的行为与
            改动前完全一致；``raw`` 档**从不**触碰 N3D 支路（现场登记的结构事实：
            raw 档下 ``adapter.features`` 不被调用、骨干不在计算图上）。
        """
        raw = F.normalize(features, dim=1)
        if self.config.head_input_mode == "raw":
            # 直通口径：q = 原始 D 维文本特征（归一化后）-> 头（近恒等初始化）
            q = raw
        elif self.config.head_input_mode == "concat":
            n3d = self.n3d_branch(features) if n3d_branch is None else n3d_branch
            alpha = torch.sigmoid(self.mix_logit)
            q = alpha * raw + (1.0 - alpha) * n3d
        else:
            q = self.n3d_branch(features) if n3d_branch is None else n3d_branch
        q = self.q_head(q)
        if int(q.shape[1]) != int(self.config.dim):
            raise RuntimeError(
                f"q 的维度必须为 D={self.config.dim}，实际 {q.shape[1]}"
            )
        if self.config.normalize_query:
            q = F.normalize(q, dim=1)
        return q

    def logits_from_index(self, q: torch.Tensor) -> torch.Tensor:
        """索引生成式实现：``logits = q @ A^T``（``A`` 为可学习候选键表 ``[C+1, D]``）。"""
        if self.config.output_mode != "index":
            raise RuntimeError(
                "logits_from_index 仅在 output_mode='index' 时可用，"
                f"当前 output_mode={self.config.output_mode!r}"
            )
        if q.dim() != 2 or int(q.shape[1]) != int(self.config.dim):
            raise ValueError(f"q 必须为 2D [B, D={self.config.dim}]，当前 {tuple(q.shape)}")
        if self.config.normalize_query:
            keys = F.normalize(self.answer_table, dim=1)
        else:
            keys = self.answer_table
        return (q @ keys.transpose(0, 1)) * self.logit_scale

    def logits_from_pointer(self, q: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """指针 Softmax 实现：``logits = q @ K^T``（``K`` **由输入提供**）。

        参数
        ----
        q : torch.Tensor
            ``[B, D]``。
        keys : torch.Tensor
            ``[B, L, D]`` 的候选键表（来自确定性文本特征）。

        返回
        ----
        torch.Tensor
            ``[B, L]`` 的候选 logits。
        """
        if self.config.output_mode != "pointer":
            raise RuntimeError(
                "logits_from_pointer 仅在 output_mode='pointer' 时可用，"
                f"当前 output_mode={self.config.output_mode!r}"
            )
        if keys.dim() != 3:
            raise ValueError(
                f"keys 必须为 3D [B, L, D]，当前 shape={tuple(keys.shape)}"
            )
        if int(keys.shape[0]) != int(q.shape[0]):
            raise ValueError(
                f"keys 与 q 的 batch 维不一致：{keys.shape[0]} vs {q.shape[0]}"
            )
        if int(keys.shape[2]) != int(self.config.dim):
            raise ValueError(
                f"keys 最后一维必须为 D={self.config.dim}，当前 {keys.shape[2]}"
            )
        raw = torch.bmm(q.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)  # [B, L]
        return raw * self.logit_scale

    def logits(
        self,
        features: torch.Tensor,
        keys: Optional[torch.Tensor] = None,
        q: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """统一打分入口（按全局开关分派到两个实现之一）。

        参数
        ----
        features : torch.Tensor
            ``[B, D]`` 输入特征。
        keys : Optional[torch.Tensor]
            ``pointer`` 模式必需的 ``[B, L, D]`` 候选键表；``index`` 模式必须为 ``None``。
        q : Optional[torch.Tensor]
            已算好的 ``[B, D]`` 查询（避免重复前向）；为 ``None`` 时自行计算。

        返回
        ----
        torch.Tensor
            ``[B, C+1]``（index）或 ``[B, L]``（pointer）的 logits。
        """
        if q is None:
            q = self.query(features)
        if self.config.output_mode == "index":
            if keys is not None:
                raise ValueError(
                    "index 模式不接受 keys 参数（候选键表是模型参数，不是输入）"
                )
            return self.logits_from_index(q)
        if keys is None:
            raise ValueError("pointer 模式必须提供 keys=[B, L, D] 候选键表")
        return self.logits_from_pointer(q, keys)

    def forward(self, features: torch.Tensor, keys: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向：``[B, D] -> [B, 候选数]`` logits（见 ``logits``）。"""
        return self.logits(features, keys)

    # -- 损失 -------------------------------------------------------------
    def cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """交叉熵（支持逐样本权重与标签平滑；**不在此处做任何类别重排**）。

        参数
        ----
        logits : torch.Tensor
            ``[B, L]``。
        targets : torch.Tensor
            ``[B]`` 目标下标（``index`` 模式 = 答案表下标；``pointer`` 模式 = 候选位次）。
        sample_weight : Optional[torch.Tensor]
            ``[B]`` 逐样本权重（用于「不相关」类的降权）。为 ``None`` 时全 1。

        返回
        ----
        torch.Tensor
            标量损失。
        """
        if logits.dim() != 2:
            raise ValueError(f"logits 必须为 2D [B, L]，当前 {tuple(logits.shape)}")
        if targets.dim() != 1 or int(targets.shape[0]) != int(logits.shape[0]):
            raise ValueError(
                f"targets 必须为 1D [B={logits.shape[0]}]，当前 {tuple(targets.shape)}"
            )
        if int(targets.max().item()) >= int(logits.shape[1]):
            raise ValueError(
                f"targets 最大值 {int(targets.max().item())} 超出候选数 {logits.shape[1]}"
            )
        per = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=float(self.config.label_smoothing),
        )
        if sample_weight is None:
            return per.mean()
        w = sample_weight.to(per.dtype)
        if int(w.numel()) != int(per.numel()):
            raise ValueError(
                f"sample_weight 长度 {int(w.numel())} 与 batch {int(per.numel())} 不一致"
            )
        return (per * w).sum() / w.sum().clamp_min(1e-12)

    # -- 对齐损失（本轮新增；作为 ce 的**附加项**，默认关闭时不参与计算）----
    def alignment_loss(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        n_classes: int,
        *,
        readout: Optional[torch.Tensor] = None,
        branch: Optional[torch.Tensor] = None,
        temperature: float = SUPCON_TEMPERATURE,
    ) -> Dict[str, Any]:
        """对齐损失 ``align``（**返回分项**，供报告逐项登记）。

        口径（显式固定，不随运行变化）
        -----------------------------
        * 表示 ``z = alignment_representation(features)``（L2 归一化；机制 B 下为**投影后**）；
        * **机制 B（``proj_supcon``）**：``align = supcon_loss(z, targets, C)``，
          同答案互为正样本、温度 = :data:`SUPCON_TEMPERATURE`；
        * **机制 C（``distill``）**：``align = 0.5·(L_i + L_ii)``，其中
          ``L_i = mean(1 - cos(z, 本样本编码器原始特征))``（目标 (i)，逐样本蒸馏）、
          ``L_ii = mean(1 - cos(z, 批内所属类质心))``（目标 (ii)）。类质心
          **批内现场算、detach、不参与梯度**，且与 ``z`` 一样先做 L2 归一化；
        * 「不相关」样本（``target == C``）**不参与**两种对齐（它们没有正样本对，
          其"类质心"也不是答案类质心）；被排除的数量在 ``n_excluded`` 中可见。

        参数
        ----
        features : torch.Tensor
            ``[B, D]`` 原始文本特征（**也是**目标 (i)(ii) 的来源空间）。
        targets : torch.Tensor
            ``[B]`` 目标下标；等于 ``n_classes`` 的是「不相关」样本。
        n_classes : int
            答案类别数 ``C``。
        readout : Optional[torch.Tensor]
            已算好的骨干读出（避免重复前向）。
        branch : Optional[torch.Tensor]
            已算好的 :meth:`n3d_branch` 输出（**优先于** ``readout``）。
        temperature : float
            机制 B 的温度（``> 0``）。

        返回
        ----
        Dict[str, Any]
            ``{"mode", "applicable", "loss", "lambda", "weighted", "supcon",
            "distill_self", "distill_centroid", "n_used", "n_excluded",
            "skipped_reason"}``。``loss`` / ``weighted`` 是**可反传**的标量张量；
            其余为已 detach 的浮点/整数（供报告落盘）。
            ``applicable=False`` 表示本 batch 无可用对齐项（对应 ``loss=None``），
            调用侧按「跳过该 batch 的对齐项」处置并计数可见。
        """
        mode = str(self.config.align_mode)
        base: Dict[str, Any] = {
            "mode": mode,
            "applicable": False,
            "loss": None,
            "lambda": float(self.config.align_lambda),
            "weighted": None,
            "supcon": None,
            "distill_self": None,
            "distill_centroid": None,
            "n_used": 0,
            "n_excluded": 0,
            "skipped_reason": None,
        }
        if mode == "off":
            base["skipped_reason"] = "align_mode='off'（对齐关闭）"
            return base
        if targets.dim() != 1 or int(targets.shape[0]) != int(features.shape[0]):
            raise ValueError(
                f"targets 必须为 1D [B={features.shape[0]}]，当前 {tuple(targets.shape)}"
            )
        keep = targets != int(n_classes)
        n_used = int(keep.sum().item())
        base["n_excluded"] = int(targets.shape[0]) - n_used
        base["n_used"] = n_used
        if n_used < 1:
            base["skipped_reason"] = "批内无（非不相关）样本"
            return base

        branch_out = branch if branch is not None else (
            None if readout is None else self.n3d_branch(features, readout=readout)
        )
        z = (self.alignment_representation(features) if branch_out is None
             else F.normalize(branch_out, dim=1))
        if mode == "proj_supcon":
            loss = supcon_loss(z, targets, int(n_classes), temperature=float(temperature))
            if loss is None:
                base["skipped_reason"] = "批内无同类正样本对（supcon 的锚点集合为空）"
                return base
            value = float(loss.detach().item())
            base.update({
                "applicable": True, "loss": loss, "supcon": value,
                "weighted": loss * float(self.config.align_lambda),
            })
            return base
        if mode == "distill":
            dim = int(self.config.dim)
            if int(features.shape[1]) != dim:
                raise ValueError(
                    f"features 必须为 2D [B, D={dim}]，当前 {tuple(features.shape)}"
                )
            with torch.no_grad():
                raw = F.normalize(features, dim=1)
                # 批内逐类质心（现场算、detach）：index_add_ 向量化，禁止逐类 Python 循环
                cent = torch.zeros(
                    int(n_classes) + 1, dim, dtype=raw.dtype, device=raw.device
                )
                cnt = torch.zeros(
                    int(n_classes) + 1, dtype=raw.dtype, device=raw.device
                )
                cent.index_add_(0, targets, raw)
                cnt.index_add_(0, targets, torch.ones_like(targets, dtype=raw.dtype))
                cent = cent / cnt.clamp_min(1.0).unsqueeze(1)
                cent = F.normalize(cent, dim=1)
            if int(z.shape[1]) != dim:
                raise ValueError(
                    f"对齐表示必须为 D={dim} 维，当前 {z.shape[1]}"
                )
            cos_self = (z * raw).sum(dim=1)
            cos_cent = (z * cent[targets]).sum(dim=1)
            l_self = (1.0 - cos_self)[keep].mean()
            l_cent = (1.0 - cos_cent)[keep].mean()
            loss = 0.5 * (l_self + l_cent)
            base.update({
                "applicable": True, "loss": loss,
                "distill_self": float(l_self.detach().item()),
                "distill_centroid": float(l_cent.detach().item()),
                "weighted": loss * float(self.config.align_lambda),
                "n_used": n_used,
            })
            return base
        raise ValueError(f"未知对齐机制 {mode!r}；合法集合 = {list(ALIGN_MODES)}")

    # -- 答案表（centroid 口径） ------------------------------------------
    @torch.no_grad()
    def set_answer_table_from_centroids(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        n_classes: int,
    ) -> None:
        """按训练样本的**逐类质心**写入答案表（centroid 口径）。

        参数
        ----
        features : torch.Tensor
            ``[N, D]`` 训练样本的**原始文本特征**（与 ``q`` 头所用特征同源）。
        targets : torch.Tensor
            ``[N]`` 样本目标下标；等于 ``n_classes`` 的样本归入「不相关」位。
        n_classes : int
            答案类别数 ``C``（候选键表行数 = ``C + 1``，末位「不相关」）。

        关键不变量
        ----------
        * 每个候选键的 L2 范数为 1（零质心除外：该位保持零向量）；
        * 该方法是**确定性**的（无随机数），同一 ``(features, targets)`` 恒得同一表；
        * 仅在 ``output_mode == "index"`` 时可用。
        """
        if self.config.output_mode != "index":
            raise RuntimeError("set_answer_table_from_centroids 仅在 index 模式可用")
        dim = int(self.config.dim)
        if features.dim() != 2 or int(features.shape[1]) != dim:
            raise ValueError(
                f"features 必须为 2D [N, D={dim}]，当前 {tuple(features.shape)}"
            )
        table = torch.zeros(int(n_classes) + 1, dim, dtype=features.dtype,
                            device=features.device)
        for c in range(int(n_classes) + 1):
            mask = targets == c
            count = int(mask.sum().item())
            if count == 0:
                continue
            table[c] = features[mask].mean(dim=0)
        norms = table.norm(dim=1, keepdim=True).clamp_min(1e-12)
        table = table / norms
        self.answer_table.copy_(table)

    # -- 自描述 -----------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        """自描述（写进产物 meta；**``D`` 必须出现在这里**）。"""
        info: Dict[str, Any] = {
            "backend": self.backend_name,
            "dim": int(self.config.dim),
            "output_mode": str(self.config.output_mode),
            "label_smoothing": float(self.config.label_smoothing),
            "n_answers": int(self.n_answers),
            "candidate_width": int(self.output_dim),
            "irrelevant_index": int(self.answer_index()),
            "has_answer_table": bool(self.config.output_mode == "index"),
            "answer_table_mode": str(self.config.answer_table_mode),
            "head_parameters": int(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            ),
            "backbone_parameters": int(
                sum(p.numel() for p in self.adapter.model.parameters() if p.requires_grad)
            ),
        }
        info["head_input_mode"] = str(self.config.head_input_mode)
        info["normalize_query"] = bool(self.config.normalize_query)
        info["logit_scale"] = float(self.logit_scale.detach().item())
        # ---- 对齐机制（本轮新增；默认关闭时全为关闭口径，便于事后回答"这份数字带没带对齐"）
        info["align_mode"] = str(self.config.align_mode)
        info["align_lambda"] = float(self.config.align_lambda)
        info["proj_dim"] = int(self.config.proj_dim)
        info["proj_init"] = str(self.config.proj_init)
        info["has_projection_head"] = bool(self.proj is not None)
        info["alignment_parameters"] = int(
            sum(p.numel() for p in self.alignment_parameters())
        )
        return info

    def trainable_parameter_names(self) -> List[str]:
        """逐参数名列示（供"梯度非零"断言现场枚举，禁止凭记忆手写）。

        口径说明
        --------
        ``q`` 头与候选键表是 ``N3DQA`` **自己的**参数（``named_parameters`` 可见）；
        后端 N3D 权重通过 ``self.adapter.model`` 持有（**不是** ``N3DQA`` 的子模块，
        故不出现在 ``named_parameters`` 里），这里显式加 ``"backbone."`` 前缀列出。
        """
        names = [f"head.{n}" for n, p in self.named_parameters() if p.requires_grad]
        names.extend(
            f"backbone.{n}"
            for n, p in self.adapter.model.named_parameters()
            if p.requires_grad
        )
        return names

    def zero_grad_parameters(self, probe_features: torch.Tensor,
                             probe_keys: Optional[torch.Tensor],
                             probe_targets: torch.Tensor) -> List[str]:
        """诊断接口：返回**首步梯度恒为 0** 的参数名（结构性零梯度豁免名单）。

        当前实现下应为空列表（无结构性零梯度参数）；该方法存在是为了让
        "梯度非零"断言可以把豁免项**显式列出**而不是放宽为"大部分非零"。
        """
        self.zero_grad(set_to_none=True)
        lg = self.logits(probe_features, probe_keys)
        loss = self.cross_entropy(lg, probe_targets)
        loss.backward()
        zeros: List[str] = []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None or float(p.grad.abs().sum().item()) == 0.0:
                zeros.append(f"head.{n}")
        for n, p in self.adapter.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None or float(p.grad.abs().sum().item()) == 0.0:
                zeros.append(f"backbone.{n}")
        self.zero_grad(set_to_none=True)
        return zeros


def supcon_loss(
    q: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    temperature: float = SUPCON_TEMPERATURE,
) -> Optional[torch.Tensor]:
    """监督对比损失（同答案问题为正样本；**批内构造、不采样**）。

    **唯一实现落点在本模块**（``exp_repr.supcon_loss`` 只做再导出，保证机制 B 与
    ``loss_mode="supcon"`` 用的是同一段代码，不存在两份实现漂移的风险）。

    口径（显式固定，不随运行变化）
    -----------------------------
    * 锚点集合 ``I`` = 批内**答案表内**样本中「至少有一个同类正样本」的那些样本；
      「不相关」样本（``target == n_classes``）**不参与**（它们没有正样本）；
    * 正样本集合 ``P(i)`` = 批内与 ``i`` 同答案类、且不等于 ``i`` 的样本；
    * 负样本 = 批内其余样本（含其它类别），归一化后按 ``q @ q^T / temperature`` 打分；
    * 损失 = ``-1/|I| * sum_i [ 1/|P(i)| * sum_{p in P(i)} log softmax_i(p) ]``。

    参数
    ----
    q : torch.Tensor
        ``[B, D]`` 的 **L2 归一化**表示（:meth:`N3DQA.query` 的输出，或机制 B 下的
        投影后 N3D 表示）。
    targets : torch.Tensor
        ``[B]`` 目标下标；等于 ``n_classes`` 的是「不相关」样本。
    n_classes : int
        答案类别数 ``C``。
    temperature : float
        温度（``> 0``）。

    返回
    ----
    Optional[torch.Tensor]
        标量损失；批内可用锚点不足（``|I| == 0``）时返回 ``None``
        （调用侧按「跳过该 batch」处置，并在诊断里计数可见）。
    """
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature 必须 > 0，当前 {temperature}")
    keep = targets != int(n_classes)
    z = q[keep]
    y = targets[keep]
    if int(z.shape[0]) < 2:
        return None
    eye = torch.eye(int(z.shape[0]), dtype=torch.bool, device=z.device)
    sim = (z @ z.transpose(0, 1)) / float(temperature)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & (~eye)
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if int(valid.sum().item()) == 0:
        return None
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # [!] 必须用 ``where`` 而不是 ``* pos``：``-inf * False`` 在 PyTorch 里是 ``nan``
    #     （对角线被填成 ``-inf``，乘 0 仍得 nan），会污染整个 batch 的损失。
    log_prob = torch.where(pos, log_prob, torch.zeros_like(log_prob))
    per_anchor = log_prob.sum(dim=1) / pos_count.clamp_min(1)
    return -per_anchor[valid].mean()


__all__ = [
    "OUTPUT_MODES",
    "HEAD_INPUT_MODES",
    "ANSWER_TABLE_MODES",
    "IRRELEVANT_AT_END",
    "ALIGN_MODES",
    "PROJ_INIT_MODES",
    "PROJ_INIT_SEED",
    "ALIGN_TARGETS",
    "SUPCON_TEMPERATURE",
    "N3DQAConfig",
    "N3DQA",
    "supcon_loss",
]