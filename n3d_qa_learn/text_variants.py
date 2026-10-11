"""n3d_qa_learn 第三步 Phase 1：文本侧数据改造（4 支变体 + 难负例 + 硬门禁 + 零训练评测）。

背景（现场已核实，见 README 第十七节）
--------------------------------------
步骤 2 的文本侧口径一直是**自匹配恒等任务**：查询行本身在检索池内、金标 = 自身行，
故「原文支」的余弦检索 `cos = 1.0` ⇒ `R@1 = 1.0` 是**结构性**结果，不构成任务。

本模块把文本侧改造成「**多表述 -> 原文**」的真实检索任务：

* 对每一行原文 `T` 生成 4 支**问题侧**表述（**答案恒为原文**，与 1a / 步骤 2 的
  `outputs = 行原文` 一致）：`原文` / `关键词`（KeyBERT）/ `摘要`（抽取式）/
  `自适应`（按 `char_len` 分档）；
* 引入**难负例**：同文档他行（同 `file`）/ 相邻行（`line_no +/- 1`）/
  编码器近邻他行（确定性抽取、固定 seed）；
* **特征差异硬门禁**：对**可约简行**（`row_is_irreducible == False`）逐位断言各支特征与「原文支」
  **不同**（违反即 `AssertionError`，`hard_gate.passed` 直接取该断言结果）；**不可约简行**（任何抽取算子都只能返回它自身）
  的重合比例**如实登记**；短行档的「自适应支」重合也单独登记（**修正说明**：本条此前写成「逐位硬门禁」，
  但 `IDENTITY_GATED_VARIANTS` 为空时该断言循环不执行、实际不生效，隐阈提及已在 R67 回修轮修正为上述可约简行断言）；
* **零训练评测**：纯余弦检索（**不上任何模型、不训练**），分档报告 `R@1` / `R@5`。

口径边界（**如实登记，不得包装**）
----------------------------------
抽取式的「关键词」/「摘要」是 **信息减少**（从原文里删内容），**不是**真正的
paraphrase（换词换语序）。因此本轮的结论只能说明「**把问题侧改造成原文的子集**之后
检索任务不再是恒等任务、且仍然可解」，**不能**据此声称「解决了换说法泛化」。

零新依赖之外的两项**模块级可选依赖**（登记在 `requirements-qa.txt`）
----------------------------------------------------------------------
`keybert`（关键词抽取）与 `jieba`（中文分词）。二者**只在生成「关键词支」时需要**；
凡缺失一律**可读报错 + 退码非 0**，绝不静默跳过或回退成别的变体。
**可选依赖后置（R67 修正）**：本模块**导入期不加载 keybert**（基类用 `object` 兑底），
因此缺 keybert 时「摘要支 / 难负例 / 特征门禁 / 零训练检索」仍可正常使用；
只有在**实际构造关键词后端**时才抛可读 `TextVariantDependencyError`。

上游边界（零改动）
------------------
只读 import `n3d_qa.zh_features`（中文确定性特征）与 `n3d_qa_learn` 内既有模块；
不修改 `entry_table.py` / `robust_eval.py`（1a 考卷）与 `variant_b.py`（第二步），
不触碰任何上游模块的源码 / 产物 / `current_spec.md`。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .step2 import (
    BUCKETS_PER_ORDER,  # noqa: F401  (口径透传，便于报告引用)
    N_GRAM_ORDERS,  # noqa: F401  (口径透传，便于报告引用)
    TextRow,
    ZhBagVectorizer,
    canonical_dumps,
    load_row_index,
    load_text_rows,
    resolve_product_dir,
    sha256_bytes,
)

# ---------------------------------------------------------------------------
# 全局纪律
# ---------------------------------------------------------------------------

#: 本模块**唯一**随机种子（单 seed、冻结、确定性；不消耗全局 RNG）。
TEXT_VARIANTS_SEED: int = 20261010

#: 变体产物格式版本（结构变化必须递增）。
TEXT_VARIANTS_SCHEMA_VERSION: str = "text-variants-v1"

#: 运行期日志字面量的编码纪律（沿用 `step2_run._ASCII_SAFE_LOG_NOTE` 的同一纪律）。
ASCII_SAFE_LOG_NOTE: str = (
    "运行时日志字面量必须能被 Windows 控制台默认编码 cp936（GBK）表示；"
    "本模块新增的日志一律只用 ASCII + 汉字，禁止 U+2212 / U+21D2 一类字符。"
    "docstring / 注释 / 产物 JSON（UTF-8）不受此限。"
)

#: 缺依赖的统一说明（供 CLI / 报告引用）。
TEXT_VARIANT_DEPENDENCY_NOTE: str = (
    "关键词支依赖 keybert（含 scikit-learn 传递依赖）与 jieba；"
    "二者缺失一律**可读报错 + 退码非 0**，不静默跳过。安装："
    "python -m pip install keybert jieba（登记在 n3d_qa_learn/requirements-qa.txt；"
    "根 requirements.txt 不动）。"
)

# ---------------------------------------------------------------------------
# 冻结口径常量（全部可枚举；判定规则与判定实现由同一常量派生）
# ---------------------------------------------------------------------------

#: 变体名（顺序敏感：第 0 支恒为锚点「原文」）。
VARIANT_ANCHOR: str = "原文"
VARIANT_KEYWORD: str = "关键词"
VARIANT_SUMMARY: str = "摘要"
VARIANT_ADAPTIVE: str = "自适应"
VARIANT_ORDER: Tuple[str, ...] = (
    VARIANT_ANCHOR,
    VARIANT_KEYWORD,
    VARIANT_SUMMARY,
    VARIANT_ADAPTIVE,
)

#: 非锚点支（需要与锚点逐位比较的三支）。
NON_ANCHOR_VARIANTS: Tuple[str, ...] = tuple(
    name for name in VARIANT_ORDER if name != VARIANT_ANCHOR
)
# --- 关键词支（KeyBERT）-----------------------------------------------------

#: 关键词 top-k 的允许区间（构造期校验；**不得**超出）。
KEYWORD_MIN_TOP_K: int = 1
KEYWORD_MAX_TOP_K: int = 5

#: 关键词 top-k 的**判定档**（默认档 = 3；其余档只在 --keyword-top-k 下显式启用）。
KEYWORD_TOP_K: int = 3

#: 候选 n-gram 范围（jieba 分词后重新拼装为 1~3 元候选词）。
KEYWORD_NGRAM_RANGE: Tuple[int, int] = (1, 3)

#: 每行保留的候选词上限（**确定性**截断：候选词已按 文档频次降序 -> 词面升序
#: 排序，故取前 N 个是确定的）。
#:
#: **取值论证（与 `KEYWORD_TOP_K` 的关系，必须显式）**
#: --------------------------------------------------
#: 该上限作用于「**交给评分的候选词集**」，而最终输出只取 `KEYWORD_TOP_K = 3` 个。
#: 若 `MAX_CANDIDATES <= TOP_K`，则 top-k 选择对输出**完全无作用**（候选全部入选），
#: 关键词支退化为「取文档频次最高的前 k 个候选」—— 这是 2026-10-11 回修轮发现的
#: **调参残留**（当时取 5）。现取 **32**：远大于 `TOP_K=3`（使 top-k 真正生效），
#: 同时保持每行评分候选在可控量级（现场实测：POS 过滤后单行候选中位 6 / p90 72）。
#: 实测影响（同一数据集）：该值 5 -> 32 使候选词全局唯一数从 3248 升到 ~10464，
#: 嵌入预热成本约×3.2；而关键词支的输出文本随之变化（已重算产物）。
KEYWORD_MAX_CANDIDATES: int = 32

#: 候选词的**语料级文档频次下限**（只保留在全语料中出现
#: 于 >= 该数量行的候选词）。**现场实测为何需要这一步**：2665 行的
#: POS 过滤候选词共 **33664 个唯一词**（而（行, 候选）对只有 67846），即**大部分词
#: 只在单行出现**；而每个唯一词都要走一次 bge-m3 前向（CPU 上 ~0.2 s）
#: ⇒ 全量冷缓存预估 **~3.5 小时**。只在单行出现的词对关键词评分几乎无価值（它
#: 的唯一作用就是在它所在的那一行里排名），却占了绝大部分成本。
KEYWORD_MIN_DOC_FREQ: int = 2

#: 单行候选词被词表过滤到空时，回退保留的候选数（**保证任何行
#: 都不会静默地整体失去该支**）。
KEYWORD_MIN_CANDIDATES: int = 3

#: 关键词候选只保留的 jieba 词性（**冻结、可枚举**）：
#: 名词类（n/nr/ns/nt/nz/ng）/ 动词（v/vd/vn）/ 形容词（a/ad/an）/
#: 英文（eng）/ 数词与量词（m/q）/ 缩写与习语（j/l/i）/ 处所（s）。
#: 被排除的主要是：标点 w、助词 u、连词 c、介词 p、副词 d、代词 r、
#: 叹词 e/o、拟声词等（它们对「多表述 -> 原文」的词面匹配贡献极小，却会让
#: 候选集与嵌入 memo 爆炸）。
KEYWORD_POS_KEEP: Tuple[str, ...] = (
    "n", "nr", "ns", "nt", "nz", "ng",
    "v", "vd", "vn",
    "a", "ad", "an",
    "eng", "m", "q", "j", "l", "i", "s",
)

#: 每次交给编码器的**文档行数**。口径上必须固定：文档嵌入与词嵌入都是纯前向，
#: 批量大小只影响速度，不影响取值。
KEYWORD_BATCH_SIZE: int = 32

#: 每次交给编码器的**候选词数**。候选词短（多数 1~6 字），故可用远大于
#: :data:`KEYWORD_BATCH_SIZE` 的批。实测证据：同工作量下批量 8 -> 256 把接口调用数量降低两个
#: 数量级，而取值逐位不变（纯前向 + L2 归一化）。
KEYWORD_WORD_BATCH_SIZE: int = 256

#: 关键词支内部评分（文档嵌入 + 候选词嵌入）所用编码器的 ``max_length``。
#:
#: 取值 **512**，与注释一致的现场依据：冻结文本行的 ``char_len`` 最大仅 **293**（全量 2665 行实测），
#: token 数远低于 512，故**不会发生任何截断**；512 只是一个上界。
#: 而此前评测（区引入期代码）向 `HFTextEncoder` 传的是文本行角色的冻结 8192，
#: 使 20 行 / 400 个唯一词的词嵌入耗时 20.3 s；降到 512 后同量工作显著下降。
#:
#: **截断取证（R68 小收口轮落地）**：`build_keyword_backend` 返回的取证字典里
#: ``word_max_length`` 取自**编码器实际生效**的 ``max_length``（不再回抄本常量）；
#: `probe` 随后用 ``HFTextEncoder.encode_batch_with_stats`` 对**最长的若干行文本**
#: 现场实测 ``n_tokens`` / ``n_truncated``，把 ``truncation_observed`` 一并写进
#: `keyword_backend` 取证（落在 `variants_meta.json` 与 `probe_summary.json`）。
KEYWORD_ENCODER_MAX_LENGTH: int = 512

#: 是否启用 MMR 多样性（**关闭**：MMR 的选取依赖候选相对顺序，跨版本可能漂移）。
KEYWORD_USE_MMR: bool = False

#: 关键词支的最少字数（不足则**处置为回退原文**并如实登记，不产空串 / NaN）。
KEYWORD_MIN_CHARS: int = 1

#: KeyBERT 后端（与 encoders 注册表同名）。
KEYBERT_BACKEND_REPO_ENCODER: str = "repo-encoder"
KEYBERT_BACKEND_SENTENCE_TRANSFORMERS: str = "sentence-transformers"

#: 是否把「文档自身嵌入」也算作候选包含判定（KeyBERT 默认 True；固定 True 保证确定性口径）。
#: ``doc_embeddings`` 口径（**现场核实**：keybert 0.9.0 的 ``doc_embeddings`` 形参是 ``np.ndarray``（文档嵌入数组），
#: **不是**老版本的布尔开关；传 ``True`` 会在 ``doc_embeddings[index].reshape(1, -1)`` 处抛
#: ``TypeError: 'bool' object is not subscriptable``）。
#: 本模块取 ``None`` = 由 keybert 在本次调用内**现算**文档嵌入（确定性：纯前向）。
KEYWORD_DOC_EMBEDDINGS: Any = None

# --- 摘要支（抽取式）-------------------------------------------------------

#: 摘要最多保留几句。
SUMMARY_MAX_SENTENCES: int = 2

#: 摘要最多保留几字（截断到最后一个非空白字符）。
SUMMARY_MAX_CHARS: int = 60

#: 摘要最少字数（不足则**处置为回退原文**并如实登记）。
#: 实测口径：该值曾取 4，但现场实测有整行仅 3 字的行（如「各省、」），
#: 4 会把它们一律打回原文；取 1 使「非空子串」成为唯一门限（更小 = 更宽）。
SUMMARY_MIN_CHARS: int = 1

#: 句级切分标点（中英文句末标点 + 换行）。
SUMMARY_SENTENCE_DELIMITERS: str = "。！？!?；;…" + chr(10) + chr(13)

#: 摘要是否剥离尾部空白。
SUMMARY_STRIP_TRAILING_WHITESPACE: bool = True

#: 子句级切分标点（**现场实测**：2092/2665 行只有单句，其中 ``b.md`` 2403 行
#: 中位数 41 字，句级抽取对它们无效，必须再下沉一级到子句边界）。
SUMMARY_CLAUSE_DELIMITERS: str = "，,、：:；;。！？!?…" + chr(10) + chr(13)

#: 子句级抽取的字数上限（先取首子句，再截断到该上限）。现场实测 24 字可覆盖
#: 中位首标点位置（18）与 p75（24）。
SUMMARY_CLAUSE_MAX_CHARS: int = 24

#: 句级抽取失效时的**兜底**字数上限（直接取首 N 字；仍是原文连续子串）。
SUMMARY_HEAD_MAX_CHARS: int = 12

#: 兜底抽取启用的 ``char_len`` 阈值（现场实测 ``char_len`` 分布：min 1 / p25 22 /
#: 中位 43 / p75 64 / max 293；**短于该阈值的行一律不启用兜底截断**，
#: 以避免把整行就是几字短片的行截成无信息碎片）。
SUMMARY_HEAD_FALLBACK_BELOW_CHARS: int = 12

# --- 自适应支（按 char_len 分档）------------------------------------------

#: 分档阈值（升序，char_len 连续区间）。
ADAPTIVE_TIERS: Tuple[Tuple[str, int, int], ...] = (
    ("short", 0, 24),
    ("mid", 25, 64),
    ("long", 65, 10 ** 9),
)

#: 自适应支在短行档**允许**与原文支逐位相同（须如实登记重合比例）。
ADAPTIVE_ALLOW_IDENTITY_TIER: str = "short"

#: 自适应支 mid 档的子句抽取上限（= 摘要支同口径）。
ADAPTIVE_MID_MAX_CHARS: int = 24

#: 自适应支 long 档的子句抽取上限（比 mid 档更宽）。
ADAPTIVE_LONG_MAX_CHARS: int = 42

#: 自适应支各档的构造口径（显式、可枚举）。
ADAPTIVE_RULE: str = (
    "short(0-24) => 关键词支（KeyBERT top-k）；"
    "mid(25-64) => 子句级抽取（首子句，上限 24 字 + 兜底 12 字）；"
    "long(>=65) => 子句级抽取（上限 42 字 + 兜底 12 字）；"
    "任一支回退为原文时自适应支等同于该回退结果（逐位可判）"
)

# --- 难负例 ------------------------------------------------------------------

#: 难负例类别（顺序敏感，报告分列）。
NEGATIVE_KINDS: Tuple[str, ...] = ("same_file", "adjacent_line", "encoder_neighbor")

#: 每类难负例的最大条数。**现场实测为何取 16**：`same_file` 在 `b.md`
#: 上有 2402 条他行，不限制时全量 2665 行产生 **580 万对**（话话 139 MB 的
#: `negatives.json`），且使「难负例小池」退化为近似全量池、对照失去意义；
#: 取 16 与 `RANDOM_NEGATIVE_COUNT` / `NEGATIVE_POOL_SIZE` 同量级，使两种小池设定**池大小完全一致**（只差负例内容）。
#: 选取口径 = **按 |line_no 差| 最近**（并列按 row_id 升序），比「取前 16 行」更接近「难负例」语义。
HARD_NEGATIVE_CAP: int = 16

#: 随机负例条数（与难负例条数**分开报**）。取与 `HARD_NEGATIVE_CAP`
#: 同值，使「难负例小池」与「随机负例小池」**池大小一致**，
#: 从而对照只反映负例**内容**的差异（而非池大小）。
RANDOM_NEGATIVE_COUNT: int = 16

#: 「难负例小池」设定的每行负例上限（池 = 自身 + 最多该数量条负例）。
NEGATIVE_POOL_SIZE: int = 16

#: 编码器近邻的 top-n（确定性抽取；并列按 row_id 升序打破平局）。
ENCODER_NEIGHBOR_TOPK: int = 3

#: 编码器近邻使用的编码器（**只用仓库自有 models/bge-m3**）。
ENCODER_NEIGHBOR_ENCODER: str = "bge-m3"

#: 编码器近邻使用的角色。
ENCODER_NEIGHBOR_ROLE: str = "text_line"

#: 编码器近邻使用的本地模型目录。
ENCODER_NEIGHBOR_SOURCE: str = os.path.join("models", "bge-m3")

#: 编码器近邻用编码器的截断长度（冻结文本行 ``char_len``
#: 最大 293 => token 数远低于 2048，**不会截断**）。
NEIGHBOR_ENCODER_MAX_LENGTH: int = 2048

#: 编码器近邻的每批行数。
ENCODER_NEIGHBOR_BATCH: int = 64

#: **允许与原文支逐位相同**的支：它们的重合比例**必须**如实登记，
#: 但**不**作为硬门禁。**现场实测驱动**：
#: * 关键词支的冻结回退口径（抽不出 / 与原文逐字相同 -> 原文）
#:   在全量 2665 行中**真实触发 9 行**，故它**不可能**被要求零重合；
#: * 自适应支在短行档同理；
#: * 摘要支**不在此列**：三级抽取阶梯保证它对任何非空行都能给出
#:   **真子串**，故它仍然是硬门禁。
IDENTITY_REGISTERED_VARIANTS: Tuple[str, ...] = (VARIANT_KEYWORD, VARIANT_SUMMARY, VARIANT_ADAPTIVE)

#: **硬门禁**的支（``n_identical`` 必须为 0）。
#:
#: **为何为空**：数据集里存在一类**不可约简行**（定义见
#: :data:`IRREDUCIBLE_RULE`；由 `row_is_irreducible` **独立于抽取实现**枚举得出）：
#: 对它们**任何抽取算子**都只能返回它自身，冻结回退口径因此也只能返回原文；
#: 故**不能**对任何一支要求「全量零重合」。
#:
#: **具体行数（可复算）**：它由 `row_is_irreducible` 在**当前评估行集**
#: 上现场算出，写进产物 `feature_gate.json` 的
#: ``hard_gate.irreducible_rows``（与 ``hard_gate.reducible_rows``），
#: **不在源码里臆写任何数字**（此前此处曾出现一个无现场出处的行数，
#: 已在 R67 回修轮删除）。
#:
#: 强门禁因此落在 ``feature_gate`` 内的**可约简行上严格为零**，
#: 并且摘要支的重合行数**必须精确等于**现场独立算出的不可约简行数。
IDENTITY_GATED_VARIANTS: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# 判定规则（**先声明、规则与判定同源**：本节常量是 verdict 的唯一来源）
# ---------------------------------------------------------------------------

#: 检索池规模（``n3d_qa`` 冻结文本行全量；现场实测 2665）。
RETRIEVAL_POOL_SIZE: int = 2665

#: 随机基线 = 1 / 检索池规模。
RANDOM_BASELINE: float = 1.0 / float(RETRIEVAL_POOL_SIZE)

#: 「锚点支 R@1 必须等于」的值（结构性锚点）。
ANCHOR_EXPECTED_RECALL_AT_1: float = 1.0

#: 锚点相等的容差（显式）。
ANCHOR_TOLERANCE: float = 1e-12

#: 「显著 < 1.0」的显式数值定义：1 - R@1 >= SIGNIFICANT_GAP。
SIGNIFICANT_GAP: float = 0.05

#: 「约等于 1.0」的容差。
CLOSE_TO_ONE_TOLERANCE: float = 0.01

#: 「约等于 0」的容差。
CLOSE_TO_ZERO_TOLERANCE: float = 1e-9

#: 「高于随机基线」的显式余量。
ABOVE_BASELINE_MARGIN: float = 1e-6

#: 结论键（机器可读）。
VERDICTS: Tuple[str, ...] = (
    "anchor_not_structural",
    "variant_too_close",
    "variant_too_far",
    "task_valid_not_identity",
    "undecided",
    "mixed",
)

#: 判定粒度的短别名（统一参数名，供报告与 CLI 引用）。
ANCHOR_TOL: float = ANCHOR_TOLERANCE
IS_IDENTITY_TOL: float = CLOSE_TO_ONE_TOLERANCE
IS_ZERO_TOL: float = CLOSE_TO_ZERO_TOLERANCE
GAP_TOL: float = SIGNIFICANT_GAP
BASELINE_TOL: float = ABOVE_BASELINE_MARGIN
BASE_TOL: float = RANDOM_BASELINE

#: 粒度守卫规则（沿用 1a 的 GRANULARITY_GUARD_RULE，**含等号**）。
GRANULARITY_GUARD_RULE: str = (
    "2665 行 => 最小可分辨粒度 = 1/2665 = 0.00037523452157598497；"
    "两支 R@1 落差 <= 1/2665 者视为同一粒度、**不得**作为判定依据，须标注并单列"
)

#: 粒度（唯一取值点）。
GRANULARITY: float = 1.0 / float(RETRIEVAL_POOL_SIZE)


def _build_verdict_rule() -> str:
    """由常量**派生**判定规则文本（与 :data:VERDICT_RULE 必须逐字相同）。"""
    return (
        "先声明、规则与判定同源（由同一常量派生）："
        "① 锚点支「原文」R@1 必须 == 1.0（容差 %g），否则 verdict_key = anchor_not_structural；"
        "② 非锚点支（关键词/摘要/自适应）：若 1 - R@1 <= %g（约等于 1.0）=> variant_too_close（改造无效）；"
        "若 R@1 <= %g（约等于 0）=> variant_too_far（任务不可解）；"
        "若 1 - R@1 >= %g（显著 < 1.0）且 R@1 - %g >= %g（显著 > 随机基线 %g）"
        "=> task_valid_not_identity（改造有效）；其余落在判定间隙、一律记 undecided 并如实登记。"
        "粒度守卫（含等号）：落差 <= %g 者不得作为判定依据，须标注并单列。"
    ) % (
        ANCHOR_TOLERANCE,
        CLOSE_TO_ONE_TOLERANCE,
        CLOSE_TO_ZERO_TOLERANCE,
        SIGNIFICANT_GAP,
        RANDOM_BASELINE,
        ABOVE_BASELINE_MARGIN,
        RANDOM_BASELINE,
        GRANULARITY,
    )


#: 判定规则文本（唯一口径来源；进产物与 Markdown）。
VERDICT_RULE: str = _build_verdict_rule()

assert VERDICT_RULE == _build_verdict_rule(), "VERDICT_RULE 与常量派生结果不一致"

#: 「如实登记」的统一文本：抽取式不是 paraphrase。
EXTRACTIVE_NOT_PARAPHRASE_NOTE: str = (
    "关键词支与摘要支都是**抽取式**（从原文里删内容），属**信息减少**，"
    "**不是**真 paraphrase（换词换语序）。因此本轮的结论只能说明"
    "「问题侧被改造成原文的子集之后不再是恒等任务、且仍可解」，"
    "**不得**据此声称「解决了换说法泛化」。"
)

#: 本轮**不接任何损失**的显式声明（是否接损失为独立决策，不得擅自引入）。
QA_GROUPING_NO_LOSS_NOTE: str = (
    "本轮只做**数据组织与取证**：把同答案的多问题显式组织成组、给出规模与指纹。"
    "**不接任何损失**、不训练、不改任何模型；是否把组结构接入损失是**独立决策**，"
    "不在本轮范围内。"
)
# ---------------------------------------------------------------------------
# 依赖可用性（缺依赖一律可读报错，不静默跳过）
# ---------------------------------------------------------------------------


class TextVariantDependencyError(RuntimeError):
    """关键词支所需的模块级依赖（keybert / jieba）不可用。"""


def _require_keybert() -> Any:
    """import keybert；缺失时给出**可读报错**（附安装口径）。"""
    try:
        from keybert import KeyBERT
    except Exception as exc:
        raise TextVariantDependencyError(
            "关键词支需要 keybert（及其传递依赖 scikit-learn），当前不可用："
            + type(exc).__name__ + ": " + str(exc) + "。安装口径见 "
            "n3d_qa_learn/requirements-qa.txt：python -m pip install keybert jieba"
            "（本模块不自动安装依赖）。"
        ) from exc
    return KeyBERT


def _require_jieba() -> Any:
    """import jieba；缺失时给出**可读报错**（附安装口径）。"""
    try:
        import jieba
    except Exception as exc:
        raise TextVariantDependencyError(
            "关键词支的中文分词需要 jieba，当前不可用："
            + type(exc).__name__ + ": " + str(exc) + "。安装口径见 "
            "n3d_qa_learn/requirements-qa.txt：python -m pip install keybert jieba"
            "（本模块不自动安装依赖）。"
        ) from exc
    return jieba


# ---------------------------------------------------------------------------
# 变体定义
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantDef:
    """一个变体支的**完整可枚举定义**（口径常量的唯一视图）。"""

    name: str
    is_anchor: bool
    description: str
    params: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        """可 JSON 化的定义视图。"""
        return {
            "name": str(self.name),
            "is_anchor": bool(self.is_anchor),
            "description": str(self.description),
            "params": dict(self.params),
        }


def variant_definitions(
    keyword_top_k: int = KEYWORD_TOP_K,
    summary_max_sentences: int = SUMMARY_MAX_SENTENCES,
    summary_max_chars: int = SUMMARY_MAX_CHARS,
) -> Tuple[VariantDef, ...]:
    """返回 4 支变体的冻结定义（顺序 = :data:VARIANT_ORDER）。"""
    top_k = int(keyword_top_k)
    if not (KEYWORD_MIN_TOP_K <= top_k <= KEYWORD_MAX_TOP_K):
        raise ValueError(
            "关键词 top-k 必须落在 [" + str(KEYWORD_MIN_TOP_K) + ", "
            + str(KEYWORD_MAX_TOP_K) + "]，当前 " + str(top_k)
        )
    if int(summary_max_sentences) < 1:
        raise ValueError("摘要句数上限必须 >= 1，当前 " + str(summary_max_sentences))
    if int(summary_max_chars) < 1:
        raise ValueError("摘要字数上限必须 >= 1，当前 " + str(summary_max_chars))
    return (
        VariantDef(
            name=VARIANT_ANCHOR,
            is_anchor=True,
            description="锚点支：行原文逐字不动（答案也是原文，故本支是结构性恒等锚点）",
            params={"transform": "identity", "normalization": "none"},
        ),
        VariantDef(
            name=VARIANT_KEYWORD,
            is_anchor=False,
            description="关键词支：KeyBERT（仓库自有 bge-m3 编码器）+ jieba 候选词，取 top-k 词拼接",
            params={
                "backend": KEYBERT_BACKEND_REPO_ENCODER,
                "encoder": ENCODER_NEIGHBOR_ENCODER,
                "source": ENCODER_NEIGHBOR_SOURCE,
                "tokenizer": "jieba",
                "ngram_range": list(KEYWORD_NGRAM_RANGE),
                "top_k": top_k,
                "use_mmr": KEYWORD_USE_MMR,
                "doc_embeddings": KEYWORD_DOC_EMBEDDINGS,
                "min_chars": KEYWORD_MIN_CHARS,
                "empty_policy": "回退原文并登记（不产 NaN / 零范数）",
            },
        ),
        VariantDef(
            name=VARIANT_SUMMARY,
            is_anchor=False,
            description=(
                "摘要支：抽取式（三级阶梯 —— 句级 / 子句级 / 兜底截断，"
                "结果恒为原文的连续子串）"
            ),
            params={
                "extractive": True,
                "levels": ["sentence", "clause", "head"],
                "sentence_delimiters": SUMMARY_SENTENCE_DELIMITERS,
                "clause_delimiters": SUMMARY_CLAUSE_DELIMITERS,
                "max_sentences": int(summary_max_sentences),
                "max_chars": int(summary_max_chars),
                "clause_max_chars": SUMMARY_CLAUSE_MAX_CHARS,
                "head_max_chars": SUMMARY_HEAD_MAX_CHARS,
                "head_fallback_below_chars": SUMMARY_HEAD_FALLBACK_BELOW_CHARS,
                "min_chars": SUMMARY_MIN_CHARS,
                "empty_policy": "回退原文并登记（不产 NaN / 零范数）",
            },
        ),
        VariantDef(
            name=VARIANT_ADAPTIVE,
            is_anchor=False,
            description="自适应支：按 char_len 分档取关键词支 / 首子句抽取（短行档允许 == 原文支）",
            params={
                "tiers": [[str(t), int(lo), int(hi)] for t, lo, hi in ADAPTIVE_TIERS],
                "rule": ADAPTIVE_RULE,
                "short_tier_allows_identity": ADAPTIVE_ALLOW_IDENTITY_TIER,
                "mid_max_chars": ADAPTIVE_MID_MAX_CHARS,
                "long_max_chars": ADAPTIVE_LONG_MAX_CHARS,
                "head_max_chars": SUMMARY_HEAD_MAX_CHARS,
                "keyword_top_k": top_k,
            },
        ),
    )


# ---------------------------------------------------------------------------
# 文本处理小工具（确定性；不消耗全局 RNG）
# ---------------------------------------------------------------------------


def variant_text_hash(text: str) -> str:
    """变体文本的 SHA256（UTF-8 裸字节）。"""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def adaptive_tier_of(char_len: int) -> str:
    """char_len -> 自适应分档名（区间左闭右闭；越界归 long）。"""
    n = int(char_len)
    for name, lo, hi in ADAPTIVE_TIERS:
        if int(lo) <= n <= int(hi):
            return str(name)
    return str(ADAPTIVE_TIERS[-1][0])


def split_sentences(text: str, delimiters: str = SUMMARY_SENTENCE_DELIMITERS) -> List[str]:
    """按句末标点切句（**保留标点**，故每句仍是原文的连续子串）。

    原文里若没有任何句末标点（典型：标题行、编号行），整段作为**一句**返回。
    """
    raw = str(text)
    if not raw:
        return []
    out: List[str] = []
    buf: List[str] = []
    for ch in raw:
        buf.append(ch)
        if ch in delimiters:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return [s for s in out if s.strip() != ""]


def first_clause(text: str, delimiters: str = SUMMARY_CLAUSE_DELIMITERS) -> str:
    """首子句（**含**边界标点；无边界标点时返回整段）。结果恒为原文前缀子串。"""
    raw = str(text)
    for index, char in enumerate(raw):
        if char in delimiters:
            return raw[: index + 1]
    return raw


def clause_extract(
    text: str,
    max_chars: int = SUMMARY_CLAUSE_MAX_CHARS,
    min_chars: int = SUMMARY_MIN_CHARS,
) -> str:
    """子句级抽取：取首子句，再截断到 max_chars 字（尾部空白剥离）。

    不变量
    ------
    * 结果是**原文的连续子串**（抽取式；不改写任何字符）；
    * 短于 min_chars 时返回空串（由调用方按冻结口径回退，本函数不抛）。
    """
    raw = str(text)
    if not raw:
        return ""
    kept = first_clause(raw)
    if len(kept) > int(max_chars):
        kept = kept[: int(max_chars)]
    if SUMMARY_STRIP_TRAILING_WHITESPACE:
        kept = kept.rstrip()
    if len(kept) < int(min_chars):
        return ""
    return kept


def head_extract(
    text: str,
    max_chars: int = SUMMARY_HEAD_MAX_CHARS,
    below_chars: int = SUMMARY_HEAD_FALLBACK_BELOW_CHARS,
    min_chars: int = SUMMARY_MIN_CHARS,
) -> str:
    """兜底抽取：char_len > below_chars 时取首 max_chars 字，否则返回空串。

    为什么需要这一步：句级与子句级抽取对「整行无任何标点」的**头部无标点长行**
    都会原样返回整行（等于没抽取）。此时只需一个**明确、可判**的兜底。
    阈值比较必须用严格大于：把整行就是几字短片的行截成碎片会让该支失去信息。
    """
    raw = str(text)
    if not raw:
        return ""
    if len(raw) <= int(below_chars):
        return ""
    kept = raw[: int(max_chars)]
    if SUMMARY_STRIP_TRAILING_WHITESPACE:
        kept = kept.rstrip()
    if len(kept) < int(min_chars):
        return ""
    return kept


def extractive_summary(
    text: str,
    max_sentences: int = SUMMARY_MAX_SENTENCES,
    max_chars: int = SUMMARY_MAX_CHARS,
    min_chars: int = SUMMARY_MIN_CHARS,
    clause_max_chars: int = SUMMARY_CLAUSE_MAX_CHARS,
    head_max_chars: int = SUMMARY_HEAD_MAX_CHARS,
) -> str:
    """抽取式摘要（**三级抽取**，仍是原文的连续子串）。

    判定阶梯（**显式、可判、纯确定性**）
    ----------------------------------
    一、**句级**：取前 max_sentences 句（按句末标点切分，保留标点）；
    二、**子句级**：一级与原文逐字相同时，取首子句再截断到 clause_max_chars 字；
    三、**兜底**：二级仍与原文相同时，在 char_len > SUMMARY_HEAD_FALLBACK_BELOW_CHARS
        的前提下取首 head_max_chars 字。

    不变量
    ------
    * 结果是**原文的连续子串**（抽取式，不改写任何字符）；
    * 只做尾部空白剥离（SUMMARY_STRIP_TRAILING_WHITESPACE）；
    * 返回空串表示三级都抽不出东西 —— 由调用方按冻结口径回退，本函数不抛。
    """
    raw = str(text)
    if not raw:
        return ""
    sentences = split_sentences(raw)
    if sentences:
        kept = "".join(sentences[: max(1, int(max_sentences))])
        if len(kept) > int(max_chars):
            kept = kept[: int(max_chars)]
        if SUMMARY_STRIP_TRAILING_WHITESPACE:
            kept = kept.rstrip()
        if len(kept) >= int(min_chars) and kept != raw:
            return kept
    clause = clause_extract(raw, max_chars=clause_max_chars, min_chars=min_chars)
    if clause and clause != raw:
        return clause
    head = head_extract(raw, max_chars=head_max_chars, min_chars=min_chars)
    if head and head != raw:
        return head
    # 三级都没给出**真子串**（典型：整行就是 1~3 个字，
    # 任何刷清都等于它自身）-> 返回空，由调用方按冻结口径处置。
    return ""


def jieba_tokenizer() -> Callable[[str], List[str]]:
    """返回基于 jieba 的**确定性**分词器（冻结口径：不过滤词性）。

    为什么必须自定义
    ----------------
    KeyBERT 默认的 CountVectorizer 按**空格**切词，对中文等于**整句一个 token**，
    关键词抽取直接失效。这里显式注入 jieba 分词，并关掉 CountVectorizer 自带的
    ``token_pattern``（中文词大多长度 < 2，会被默认 pattern 整批丢掉）。
    """
    jieba = _require_jieba()
    try:  # 冻结 jieba 初始化（其词典前缀构建是确定性的；此调用仅为显式声明）
        jieba.initialize()
    except Exception:
        pass

    def _tokenize(doc: str) -> List[str]:
        return [w for w in jieba.lcut(str(doc)) if w.strip() != ""]

    return _tokenize


def jieba_pos_tokenizer() -> Callable[[str], List[str]]:
    """返回基于 jieba 词性标注的**内容词**分词器（主路径口径）。

    为什么必须过滤词性（现场实测驱动）
    ------------------------------------
    不过滤时候选词包含大量标点 / 助词 / 单字碎片，1~3 元展开后全局唯一词达
    **52753** 个（嵌入 memo 约 **216 MB**，且为它们全部算一次嵌入需数十分钟）。
    只保留内容词后，现场实测（单行候选数）中位 **6** / p90 **72**，全局唯一降到 **33664**。

    词性集合由 :data:`KEYWORD_POS_KEEP` **冻结并可枚举**；
    标注本身是 jieba 带的确定性词典查找，不消耗全局 RNG。
    """
    jieba = _require_jieba()
    try:
        from jieba import posseg
    except Exception as exc:
        raise TextVariantDependencyError(
            "关键词支需要 jieba.posseg：" + type(exc).__name__ + ": " + str(exc)
        ) from exc
    try:
        jieba.initialize()
    except Exception:
        pass
    keep = frozenset(KEYWORD_POS_KEEP)

    def _tokenize(doc: str) -> List[str]:
        return [
            str(word)
            for word, flag in posseg.cut(str(doc))
            if str(flag) in keep and str(word).strip() != ""
        ]

    return _tokenize
# ---------------------------------------------------------------------------
# 仓库自有编码器 -> KeyBERT 后端薄适配（避免引入 sentence-transformers）
# ---------------------------------------------------------------------------


def _keybert_base_embedder() -> Any:
    """KeyBERT 的 ``BaseEmbedder``（缺失时回退 ``object``，**不阻断本模块导入**）。

    **为什么不直接在导入期报错（R67 修正）**：本模块除了关键词支，
    还提供摘要支 / 难负例 / 特征门禁 / 零训练检索等**不依赖 keybert**的能力。
    若在 `class RepoEncoderEmbedder(BaseEmbedder)` 处就抛 ImportError，连这些能力也会被阻断。
    现在改为：导入期缺 keybert 时基类取 ``object``，真正需要 keybert 的时候
    （:func:`build_keyword_backend`）再抛可读 `TextVariantDependencyError`。
    """
    try:
        from keybert.backend import BaseEmbedder
    except Exception:
        return object
    return BaseEmbedder


class RepoEncoderEmbedder(_keybert_base_embedder()):
    """把 :class:`n3d_qa_learn.encoders.HFTextEncoder` 适配成 KeyBERT 可用后端。

    **必须**继承 ``keybert.backend.BaseEmbedder``
    ------------------------------------------
    keybert 0.9.0 的 ``select_backend()`` 对「不是 BaseEmbedder 实例」的对象一律
    **回落到** ``SentenceTransformerBackend("paraphrase-multilingual-MiniLM-L12-v2")``，
    即**静默尝试下载一个模型**。继承 BaseEmbedder 是「**只复用仓库自有 bge-m3、
    绝不下载**」这条契约的**唯一**实现方式。

    同时必须提供 ``embed``（keybert 0.9.0 调用 ``self.model.embed(...)``），
    而不只是 ``encode``。

    不变量
    ------
    * 返回 ``float32[n, dim]``；编码器本身已 L2 归一化（余弦 = 内积）；
    * 不消耗全局 RNG（编码器是纯前向）。
    """

    def __init__(self, encoder: Any) -> None:
        super().__init__(embedding_model=encoder)
        self.encoder = encoder

    def embed_words(self, words: Any) -> np.ndarray:
        """候选**词**的嵌入（与文档嵌入共用**同一个**编码器实例）。"""
        return self.encode(words)

    def embed(self, documents: Any, verbose: bool = False) -> np.ndarray:
        """``List[str] -> float32[n, D]``（keybert 的**唯一**调用入口）。"""
        return self.encode(documents)

    def encode(self, sentences: Any, **kwargs: Any) -> np.ndarray:
        """list[str] -> float32[n, D]（kwargs 一律忽略：截断长度已由角色冻结）。"""
        if isinstance(sentences, str):
            items: Sequence[str] = [sentences]
        else:
            items = [str(s) for s in sentences]
        mat = np.asarray(self.encoder.encode_batch(list(items)), dtype=np.float32)
        return mat.reshape((len(items), int(self.encoder.dim)))

    def get_sentence_embedding_dimension(self) -> int:
        """KeyBERT 可选调用：嵌入维度。"""
        return int(self.encoder.dim)


class SentenceTransformerEmbedder(_keybert_base_embedder()):
    """``sentence-transformers`` 后端的**薄适配**（仅在显式允许回退时使用）。

    为何需要它：回退路径原本直接返回 ``SentenceTransformer`` 对象，
    它缺 ``embed_words``，使下一步的词嵌入报
    ``AttributeError: 'SentenceTransformer' object has no attribute 'embed_words'``
    —— 不是可读报文。本适配器把 ``encode`` / ``embed`` / ``embed_words``
    统一到同一入口（且仍以 :func:`_keybert_base_embedder` 为基类，
    使 keybert 的 ``select_backend`` 不会再二次包装）。
    """

    def __init__(self, model: Any) -> None:
        super().__init__(embedding_model=model)
        self.model = model

    def encode(self, sentences: Any, **kwargs: Any) -> np.ndarray:
        """``list[str] -> float32[n, D]``。"""
        items = [str(sentences)] if isinstance(sentences, str) else [str(s) for s in sentences]
        mat = self.model.encode(items, show_progress_bar=False)
        return np.asarray(mat, dtype=np.float32)

    def embed(self, documents: Any, verbose: bool = False) -> np.ndarray:
        """keybert 0.9.0 调用的入口。"""
        return self.encode(documents)

    def embed_words(self, words: Any) -> np.ndarray:
        """候选**词**的嵌入（与文档嵌入共用同一模型）。"""
        return self.encode(words)

    def get_sentence_embedding_dimension(self) -> int:
        """KeyBERT 可选调用：嵌入维度。"""
        return int(self.model.get_sentence_embedding_dimension())


def build_keyword_backend(
    encoder_name: str = ENCODER_NEIGHBOR_ENCODER,
    source: str = ENCODER_NEIGHBOR_SOURCE,
    role: str = ENCODER_NEIGHBOR_ROLE,
    *,
    allow_sentence_transformers_fallback: bool = True,
    cache_dir: str = "",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """构造关键词支的编码后端（**优先复用仓库自有 bge-m3**）。

    返回
    ----
    Dict[str, Any]
        `{"kind", "model_object", "encoder", "note"}`：
        `kind = "repo-encoder"` 时 `model_object` 是本模块的薄适配器；
        不可行且允许回退时 `kind = "sentence-transformers"`（**如实登记**）。

    异常
    ----
    RuntimeError
        两条路径都不可用时（可读报文；调用方据此退码非 0）。
    """
    primary_error = ""
    try:
        from .encoders import ENCODER_BGE_M3, EncoderConfig, build_vectorizer

        name = str(encoder_name) or str(ENCODER_BGE_M3)
        kwargs: Dict[str, Any] = {"name": name, "role": str(role), "source": str(source)}
        if cache_dir:
            kwargs["cache_dir"] = str(cache_dir)
            kwargs["use_cache"] = bool(use_cache)
        kwargs["max_length"] = int(
            int(kwargs.get("max_length", 0) or 0) or KEYWORD_ENCODER_MAX_LENGTH
        )
        encoder = build_vectorizer(EncoderConfig(**kwargs))
        backend = RepoEncoderEmbedder(encoder)
        return {
            "kind": KEYBERT_BACKEND_REPO_ENCODER,
            "model_object": backend,
            "encoder": encoder,
            # 取证取**编码器实际生效**的 max_length（而非回抄常量），
            # 使「注释称 512」与「运行时真的是 512」可被产物独立核对。
            "word_max_length": int(getattr(encoder, "max_length", KEYWORD_ENCODER_MAX_LENGTH)),
            "note": (
                "复用仓库自有编码器（n3d_qa_learn.encoders.HFTextEncoder 薄适配），"
                "未引入 sentence-transformers"
            ),
        }
    except Exception as exc:
        primary_error = type(exc).__name__ + ": " + str(exc)

    # **缺 keybert 绝不回退**（R67 修正）：若 keybert 本身不可用，回退到
    # sentence-transformers 也无法跑通（KeyBERT 构造不了），且会去**实际构造**
    # SentenceTransformer（离线下报错冗长、联网时可能触发下载）。
    # 此处先判定依赖可用性，给出**可读的依赖缺失报文**。
    _require_keybert()

    if not allow_sentence_transformers_fallback:
        raise RuntimeError(
            "关键词支的编码后端不可用：仓库自有编码器路径失败（" + primary_error
            + "），且已禁止 sentence-transformers 回退。请检查 models/bge-m3 是否齐全。"
        )

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "关键词支的编码后端不可用：① 仓库自有编码器路径失败（" + primary_error
            + "）；② sentence-transformers 回退也不可用（" + type(exc).__name__ + ": "
            + str(exc) + "）。请检查 models/bge-m3，或安装 sentence-transformers。"
        ) from exc

    model = SentenceTransformer(str(source) if source else "BAAI/bge-m3")
    return {
        "kind": KEYBERT_BACKEND_SENTENCE_TRANSFORMERS,
        "model_object": SentenceTransformerEmbedder(model),
        "encoder": None,
        "note": (
            "回退到 sentence-transformers 后端（**须在报告中如实登记**）："
            "仓库自有编码器路径失败原因 = " + primary_error
            + "；已用 SentenceTransformerEmbedder 适配（embed / embed_words 可用）"
        ),
    }


def _build_jieba_vectorizer(tokenizer: Callable[[str], List[str]]) -> Any:
    """构造注入 jieba 分词的 CountVectorizer（KeyBERT 的候选词来源）。

    `token_pattern=None` **必须有**：CountVectorizer 默认的 `(?u)\\b\\w\\w+\\b`
    会丢掉全部单字词，而中文单字词非常常见。
    """
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,
        lowercase=False,
        ngram_range=tuple(int(x) for x in KEYWORD_NGRAM_RANGE),
    )


# ---------------------------------------------------------------------------
# 变体生成
# ---------------------------------------------------------------------------


class VariantGenerator:
    """4 支变体的生成器（**确定性**；不消耗全局 RNG）。

    参数
    ----
    keyword_top_k : int
        关键词支 top-k（同时决定自适应支 short 档的 top-k）。
    summary_max_sentences / summary_max_chars : int
        摘要支上限。
    keyword_backend : Optional[Dict[str, Any]]
        :func:uild_keyword_backend 的返回值；`None` = 本实例**不生成关键词支**
        （此时对关键词支与自适应支的请求会抛可读异常，绝不静默产出别的变体）。
    seed : int
        保留字段（本类全部路径都是确定性的；仅用于登记）。
    """

    def __init__(
        self,
        keyword_top_k: int = KEYWORD_TOP_K,
        summary_max_sentences: int = SUMMARY_MAX_SENTENCES,
        summary_max_chars: int = SUMMARY_MAX_CHARS,
        *,
        keyword_backend: Optional[Dict[str, Any]] = None,
        seed: int = TEXT_VARIANTS_SEED,
        vocab_path: str = "",
    ) -> None:
        self.defs = variant_definitions(
            keyword_top_k=keyword_top_k,
            summary_max_sentences=summary_max_sentences,
            summary_max_chars=summary_max_chars,
        )
        self.keyword_top_k = int(keyword_top_k)
        self.summary_max_sentences = int(summary_max_sentences)
        self.summary_max_chars = int(summary_max_chars)
        self.seed = int(seed)
        self.keyword_backend = keyword_backend
        #: 语料级候选词表路径（**空 = 用缓存默认位置**）。
        #: probe 会把它指向本次产物目录下的 ``keyword_vocab.json``，
        #: 使词表成为**可审计产物**（warning 12），并避免 ``--limit`` 演练污染全量词表。
        self.vocab_path = str(vocab_path)
        self._kw_model: Any = None
        self._tokenizer: Optional[Callable[[str], List[str]]] = None
        self._word_emb: Optional[Dict[str, np.ndarray]] = None
        self._vocab: Optional[frozenset] = None
        # 逐支计数（用于产物取证，不是判定依据）
        self.counters: Dict[str, Dict[str, int]] = {
            name: {"generated": 0, "fallback_to_original": 0, "empty_source": 0}
            for name in VARIANT_ORDER
        }

    # -- 关键词支 ---------------------------------------------------------
    def _keyword_model(self) -> Any:
        """惰性构造 KeyBERT（**唯一**构造点）。"""
        if self._kw_model is not None:
            return self._kw_model
        if self.keyword_backend is None:
            raise TextVariantDependencyError(
                "本实例未装配关键词后端（keyword_backend=None）；"
                "关键词支与自适应支不可用。请用 build_keyword_backend(...) 构造后再传。"
            )
        KeyBERT = _require_keybert()
        if self._tokenizer is None:
            self._tokenizer = jieba_tokenizer()
        model_obj = self.keyword_backend["model_object"]
        self._kw_model = KeyBERT(model=model_obj)
        return self._kw_model

    def keyword_candidates(self, text: str) -> List[str]:
        """确定性候选词表（jieba 分词 + 1~3 元，按 文档频次 -> 词面 排序后取前 N）。

        **确定性**：``CountVectorizer.get_feature_names_out()`` 已按 文档频次降序 ->
        词面升序 排序（sklearn 的固定口径），本函数只做「保留前
        :data:`KEYWORD_MAX_CANDIDATES` 个」的确定性截断。
        """
        count = _build_jieba_vectorizer(self._jieba_tokenizer())
        try:
            count.fit([str(text)])
        except ValueError:
            # 整行都是停用词 / 标点 -> 无候选（上层按冻结口径回退原文）
            return []
        words = [str(w) for w in count.get_feature_names_out()]
        vocab = self._vocabulary()
        if vocab:
            kept = [w for w in words if w in vocab]
            if kept:
                words = kept
            else:
                # 整行都是 df == 1 的孤词：保留原始前 N 个（确定性）
                words = words[: int(KEYWORD_MIN_CANDIDATES)]
        return words[: int(KEYWORD_MAX_CANDIDATES)]

    def _vocabulary(self) -> frozenset:
        """语料级候选词表（**惰性构造一次**；从已缓存的词典文件读取）。

        目的：把候选词限制到在全语料中出现于 >= :data:`KEYWORD_MIN_DOC_FREQ` 行的那些。
        词表由 :func:`build_keyword_vocabulary` 产出并落盘（可复算）；未落盘时返回空字典
        （= 不过滤，行为退回未加该步的口径）。
        """
        if self._vocab is None:
            path = str(self.vocab_path) or keyword_vocabulary_path()
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                self._vocab = frozenset(str(w) for w in payload.get("words", []))
            else:
                self._vocab = frozenset()
        return self._vocab

    def _jieba_tokenizer(self) -> Callable[[str], List[str]]:
        """**唯一** jieba 词性过滤分词器实例（惰性构造一次，避免逐行重建）。"""
        if self._tokenizer is None:
            self._tokenizer = jieba_pos_tokenizer()
        return self._tokenizer

    def _word_embedding(self, words: Sequence[str]) -> np.ndarray:
        """候选词的嵌入（**进程级 memo**，键 = 词面本身；纯前向、确定性）。

        缓存未命中的词以 :data:`KEYWORD_WORD_BATCH_SIZE` 为批交给编码器
        （避免进入编码器自己的小批逐条前向）。
        """
        if self._word_emb is None:
            self._word_emb = {}
        missing = [w for w in dict.fromkeys(words) if w not in self._word_emb]
        if missing:
            backend = self.keyword_backend["model_object"]
            step = max(1, int(KEYWORD_WORD_BATCH_SIZE))
            for b0 in range(0, len(missing), step):
                chunk = missing[b0 : b0 + step]
                mat = np.asarray(backend.embed_words(chunk), dtype=np.float32)
                for i, word in enumerate(chunk):
                    self._word_emb[word] = mat[i]
        return np.asarray([self._word_emb[w] for w in words], dtype=np.float32)

    def keyword_variants_batch(self, texts: Sequence[str]) -> List[str]:
        """**批量**生成关键词支文本。

        评分口径（与 keybert 0.9.0 的 cosine 分支逐字同源）
        ------------------------------------------------
        ``distances = cosine_similarity(doc_embedding, candidate_embeddings)``；
        取分数降序的前 ``top_k``（**并列不重排**，保持候选顺序 = 文档频次序）。
        文档与候选词均经仓库自有编码器（CLS pooling + L2 归一化），故余弦 = 内积。
        """
        if self.keyword_backend is None:
            # **缺依赖/缺后端必须在入口处可读报错**：
            # 批量路径的第一趟之前就拦下，避免在 ``_word_embedding``
            # 里报 ``TypeError: 'NoneType' object is not subscriptable``；
            # 更不得在候选为空时**静默把原文当成关键词支结果**。
            raise TextVariantDependencyError(
                "本实例未装配关键词后端（keyword_backend=None）；"
                "关键词支与自适应支不可用。"
                "请用 build_keyword_backend(...) 构造后再传。"
            )
        items = [str(t) for t in texts]
        out: List[str] = []
        # 第一趟：全量候选词（确定性）+ 它们的嵌入（大批，一次算全）
        candidates_all = [self.keyword_candidates(t) for t in items]
        flat: List[str] = []
        index_of: Dict[str, int] = {}
        for cand in candidates_all:
            for word in cand:
                if word not in index_of:
                    index_of[word] = len(flat)
                    flat.append(word)
        if flat:
            self._word_embedding(flat)
        # 第二趟：文档嵌入（按 KEYWORD_BATCH_SIZE 分批）+ 评分
        candidates = candidates_all
        step = max(1, int(KEYWORD_BATCH_SIZE))
        for b0 in range(0, len(items), step):
            chunk = items[b0 : b0 + step]
            cand_chunk = candidates[b0 : b0 + step]
            model = self._keyword_model()
            doc_emb = np.asarray(model.model.embed(chunk), dtype=np.float32)
            for index, raw in enumerate(chunk):
                cand = cand_chunk[index]
                if not cand:
                    self._record_fallback(raw)
                    out.append(raw)
                    continue
                mat = self._word_embedding(cand)
                scores = mat @ doc_emb[index]
                order = np.argsort(-scores, kind="stable")[: int(self.keyword_top_k)]
                terms: List[str] = []
                for pos in order.tolist():
                    word = str(cand[pos]).strip()
                    if word and word not in terms:
                        terms.append(word)
                joined = " ".join(terms)
                if len(joined) < int(KEYWORD_MIN_CHARS) or joined == raw:
                    self._record_fallback(raw)
                    out.append(raw)
                else:
                    self.counters[VARIANT_KEYWORD]["generated"] += 1
                    out.append(joined)
        return out

    def _record_fallback(self, raw: str) -> None:
        """登记一次「回退原文」（空文本另计 ``empty_source``）。"""
        if raw.strip() == "":
            self.counters[VARIANT_KEYWORD]["empty_source"] += 1
        self.counters[VARIANT_KEYWORD]["fallback_to_original"] += 1

    def keyword_terms(self, text: str) -> Tuple[List[str], List[float]]:
        """单行的关键词词表 + 分数（**只读**，不改变任何计数器）。

        实现口径与 :meth:`keyword_variants_batch` 的批量路径**逐字同源**：
        同一份候选词、同一公式（归一化嵌入的内积 = 余弦）、同一 top-k 选取。
        本方法只用于取证 / 调试，主路径一律走批量。
        """
        raw = str(text)
        cand = self.keyword_candidates(raw)
        if not cand:
            return [], []
        model = self._keyword_model()
        doc_emb = np.asarray(model.model.embed([raw]), dtype=np.float32)[0]
        word_emb = self._word_embedding(cand)
        scores = word_emb @ doc_emb
        order = np.argsort(-scores, kind="stable")[: int(self.keyword_top_k)]
        terms: List[str] = []
        scored: List[float] = []
        for pos in order.tolist():
            word = str(cand[pos]).strip()
            if word == "" or word in terms:
                continue
            terms.append(word)
            scored.append(float(scores[pos]))
        return terms, scored

    def keyword_variant(self, text: str) -> str:
        """关键词支文本（**回退口径**：抽不出 / 太短 / 与原文逐字相同 -> 原文）。"""
        return self.keyword_variants_batch([str(text)])[0]

    def summary_variant(self, text: str, *, max_sentences: int = 0) -> str:
        """摘要支文本（**回退口径**：结果太短 / 与原文逐字相同 -> 原文）。"""
        raw = str(text)
        if raw.strip() == "":
            self.counters[VARIANT_SUMMARY]["empty_source"] += 1
            self.counters[VARIANT_SUMMARY]["fallback_to_original"] += 1
            return raw
        limit = int(max_sentences) if int(max_sentences) > 0 else self.summary_max_sentences
        out = extractive_summary(raw, max_sentences=limit, max_chars=self.summary_max_chars)
        if len(out) < int(SUMMARY_MIN_CHARS) or out == raw:
            self.counters[VARIANT_SUMMARY]["fallback_to_original"] += 1
            return raw
        self.counters[VARIANT_SUMMARY]["generated"] += 1
        return out

    # -- 自适应支 ---------------------------------------------------------
    def adaptive_variant(
        self, text: str, char_len: int = -1, keyword_text: Optional[str] = None
    ) -> str:
        """自适应支文本（按 char_len 分档）。

        分档口径（与 :data:`ADAPTIVE_RULE` 逐字同源）
        ------------------------------------------
        * ``short``（0-24）  -> 关键词支（KeyBERT top-k）；
        * ``mid``（25-64）   -> 首子句 + 兜底（上限 :data:`ADAPTIVE_MID_MAX_CHARS`）；
        * ``long``（>=65）   -> 首子句 + 兜底（上限 :data:`ADAPTIVE_LONG_MAX_CHARS`）。

        回退口径：任一步抽不出内容 -> 原文（并计数登记），不产空串 / NaN。
        ``keyword_text`` 非空时直接复用（避免重复调用 KeyBERT）。
        """
        raw = str(text)
        n = int(char_len) if int(char_len) >= 0 else len(raw)
        tier = adaptive_tier_of(n)
        out = ""
        if tier == "short":
            if keyword_text is not None:
                out = str(keyword_text)
            else:
                out = self.keyword_variant(raw)
            if not out:
                out = raw
        else:
            cap = ADAPTIVE_MID_MAX_CHARS if tier == "mid" else ADAPTIVE_LONG_MAX_CHARS
            out = clause_extract(raw, max_chars=cap, min_chars=SUMMARY_MIN_CHARS)
            if not out or out == raw:
                out = head_extract(
                    raw,
                    max_chars=SUMMARY_HEAD_MAX_CHARS,
                    below_chars=SUMMARY_HEAD_FALLBACK_BELOW_CHARS,
                    min_chars=SUMMARY_MIN_CHARS,
                )
            if not out:
                out = raw
        if out == raw:
            self.counters[VARIANT_ADAPTIVE]["fallback_to_original"] += 1
        else:
            self.counters[VARIANT_ADAPTIVE]["generated"] += 1
        return out

    # -- 统一入口 ---------------------------------------------------------
    def variant(
        self, name: str, text: str, char_len: int = -1,
        keyword_text: Optional[str] = None,
    ) -> str:
        """按名字生成单支变体文本（``keyword_text`` 非空时复用已算好的关键词支）。"""
        key = str(name)
        if key == VARIANT_ANCHOR:
            return str(text)
        if key == VARIANT_KEYWORD:
            if keyword_text is not None:
                return str(keyword_text)
            return self.keyword_variant(str(text))
        if key == VARIANT_SUMMARY:
            return self.summary_variant(str(text))
        if key == VARIANT_ADAPTIVE:
            return self.adaptive_variant(
                str(text), char_len=char_len, keyword_text=keyword_text
            )
        raise KeyError("未知变体名 " + repr(name) + "；允许值 = " + str(list(VARIANT_ORDER)))

    def all_variants(
        self, text: str, char_len: int = -1, keyword_text: Optional[str] = None
    ) -> Dict[str, str]:
        """一次生成全部 4 支（同一 ``char_len`` 与同一份关键词结果口径）。"""
        return {
            name: self.variant(
                name, text, char_len=char_len, keyword_text=keyword_text
            )
            for name in VARIANT_ORDER
        }

    def variants_batch(self, texts: Sequence[str]) -> List[Dict[str, str]]:
        """**批量**生成 4 支（关键词支按批聚合，避免逐行调用 KeyBERT 的开销）。"""
        items = [str(t) for t in texts]
        keywords = self.keyword_variants_batch(items)
        return [
            self.all_variants(raw, char_len=len(raw), keyword_text=kw)
            for raw, kw in zip(items, keywords)
        ]

    def definition_manifest(self) -> Dict[str, Any]:
        """定义清单（进产物；含全部冻结口径常量）。"""
        return {
            "schema_version": TEXT_VARIANTS_SCHEMA_VERSION,
            "seed": int(self.seed),
            "variant_order": list(VARIANT_ORDER),
            "non_anchor_variants": list(NON_ANCHOR_VARIANTS),
            "definitions": [d.as_dict() for d in self.defs],
            "keyword_backend": (
                None
                if self.keyword_backend is None
                else {
                    "kind": str(self.keyword_backend.get("kind", "")),
                    "note": str(self.keyword_backend.get("note", "")),
                }
            ),
            "extractive_not_paraphrase_note": EXTRACTIVE_NOT_PARAPHRASE_NOTE,
            "ascii_safe_log_note": ASCII_SAFE_LOG_NOTE,
            "granularity_guard_rule": GRANULARITY_GUARD_RULE,
            "verdict_rule": VERDICT_RULE,
            "random_baseline": float(RANDOM_BASELINE),
            "granularity": float(GRANULARITY),
        }

#: 语料级候选词表的**缓存回退**路径。
#:
#: **已作废 / 仅供直接调用 API 时回退**（R68 小收口轮标注）：
#: * `probe` 一律把词表落进**产物目录** `keyword_vocab.json`（从**全量行表**构建），
#:   并通过 `VariantGenerator(vocab_path=...)` 显式指向该产物 —— 这是唯一被消费的词表；
#: * 本常量只有在**直接调用库函数**、未指定 `vocab_path` 时才被读取；
#: * 盘上该路径的**旧文件已删除**（见 README §17.18.14），避免与产物词表混淆。
KEYWORD_VOCAB_FILE: str = os.path.join(
    "checkpoints", "qa_learn", "_cache", "text_variants", "keyword_vocab.json"
)


def keyword_vocabulary_path() -> str:
    """候选词表的**缓存回退**路径（首选是产物目录里的 ``keyword_vocab.json``）。"""
    return str(KEYWORD_VOCAB_FILE)


def build_keyword_vocabulary(
    texts: Sequence[str],
    *,
    min_doc_freq: int = KEYWORD_MIN_DOC_FREQ,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    write: bool = True,
) -> Dict[str, Any]:
    """构造**语料级候选词表**（文档频次 >= ``min_doc_freq``）并可选落盘。

    参数
    ----
    texts : Sequence[str]
        全语料行文本（本轮 = 全量 2665 行）。
    min_doc_freq : int
        文档频次下限。
    tokenizer : Optional[Callable[[str], List[str]]]
        分词器；空 = :func:`jieba_pos_tokenizer`（内容词过滤）。
    write : bool
        是否把词表写到 :func:`keyword_vocabulary_path`。

    返回
    ----
    Dict[str, Any]
        ``{n_rows, n_pairs, n_distinct, min_doc_freq, n_words, fingerprint, path}``。
    """
    token = tokenizer if tokenizer is not None else jieba_pos_tokenizer()
    counter: Dict[str, int] = {}
    n_pairs = 0
    for text in texts:
        try:
            count = _build_jieba_vectorizer(token)
            count.fit([str(text)])
            words = [str(w) for w in count.get_feature_names_out()]
        except ValueError:
            words = []
        n_pairs += len(words)
        for word in set(words):
            counter[word] = counter.get(word, 0) + 1
    kept = sorted(w for w, freq in counter.items() if int(freq) >= int(min_doc_freq))
    payload: Dict[str, Any] = {
        "schema_version": TEXT_VARIANTS_SCHEMA_VERSION,
        "min_doc_freq": int(min_doc_freq),
        "n_rows": int(len(texts)),
        "n_pairs": int(n_pairs),
        "n_distinct": int(len(counter)),
        "n_words": int(len(kept)),
        "words": kept,
        "rule": (
            "词表 = {候选词 : 全语料文档频次 >= min_doc_freq}；"
            "候选词 = jieba（词性过滤）+ 1~3 元 n-gram"
        ),
        "fingerprint": sha256_bytes(canonical_dumps(kept)),
    }
    if write:
        path = keyword_vocabulary_path()
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1))
        payload["path"] = path
    return payload


# ---------------------------------------------------------------------------
# 变体产物构造（4 x N 条文本 + SHA256 指纹）
# ---------------------------------------------------------------------------


def build_variant_records(
    rows: Sequence[TextRow],
    generator: VariantGenerator,
    *,
    limit: int = 0,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """为每行生成 4 支变体文本，返回**逐行记录**（顺序 = rows 顺序）。

    参数
    ----
    rows : Sequence[TextRow]
        文本行（只读）。
    generator : VariantGenerator
        变体生成器。
    limit : int
        只处理前 limit 行（0 = 全量）；用于**小批演练**。
    on_progress : Optional[Callable[[int, int], None]]
        进度回调 (已处理, 总数)。

    返回
    ----
    List[Dict[str, Any]]
        逐行 = {row_id, file, line_no, char_len, adaptive_tier, variants:{支名: 文本},
        variant_sha256:{支名: sha}, fallback_flags:{支名: bool}}。
    """
    picked = list(rows[: int(limit)]) if int(limit) > 0 else list(rows)
    out: List[Dict[str, Any]] = []
    total = len(picked)
    raws = [str(row.text) for row in picked]
    batched = generator.variants_batch(raws)
    for i, row in enumerate(picked):
        raw = raws[i]
        char_len = len(raw)
        texts = batched[i]
        out.append(
            {
                "row_id": str(row.row_id),
                "file": str(row.file),
                "line_no": int(row.line_no),
                "char_len": int(char_len),
                "adaptive_tier": adaptive_tier_of(char_len),
                "variants": {k: str(v) for k, v in texts.items()},
                "variant_sha256": {k: variant_text_hash(v) for k, v in texts.items()},
                "fallback_flags": {k: bool(texts[k] == raw) for k in VARIANT_ORDER},
            }
        )
        if on_progress is not None and ((i + 1) % 200 == 0 or (i + 1) == total):
            on_progress(i + 1, total)
    return out


def serialize_variant_records(records: Sequence[Dict[str, Any]]) -> bytes:
    """把逐行记录序列化成**规范化 JSONL 字节**（sort_keys + 紧凑分隔符）。

    规范化是「**连续两次运行逐字节一致**」这条现场证据的前提：不依赖字典插入顺序、
    不写任何挂钟字段。
    """
    lines = [
        json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for rec in records
    ]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def variant_fingerprint(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """变体产物的**可复算指纹**（逐支 + 总体）。

    返回
    ----
    Dict[str, Any]
        {n_rows, blob_sha256, per_variant: {支名: {sha256, non_empty, n_chars_total,
        fallback_to_original, distinct_from_original}}}。
    """
    per: Dict[str, Dict[str, Any]] = {}
    for name in VARIANT_ORDER:
        chunks = [str(rec["variants"][name]).encode("utf-8") for rec in records]
        blob = b"".join(len(c).to_bytes(4, "big") + c for c in chunks)
        per[name] = {
            "sha256": sha256_bytes(blob),
            "non_empty": int(sum(1 for c in chunks if c)),
            "n_chars_total": int(sum(len(str(rec["variants"][name])) for rec in records)),
            "fallback_to_original": int(
                sum(1 for rec in records if bool(rec["fallback_flags"][name]))
            ),
            "distinct_from_original": int(
                sum(
                    1
                    for rec in records
                    if str(rec["variants"][name]) != str(rec["variants"][VARIANT_ANCHOR])
                )
            ),
        }
    blob_all = serialize_variant_records(records)
    return {
        "n_rows": int(len(records)),
        "blob_bytes": int(len(blob_all)),
        "blob_sha256": sha256_bytes(blob_all),
        "per_variant": per,
    }


def variant_product_files(
    records: Sequence[Dict[str, Any]],
    generator: VariantGenerator,
    *,
    source_rows_sha256: str = "",
    product_dir: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """构造**全部落盘产物**的 {相对路径: {text, sha256, bytes}} 视图。

    产物清单（全部写 checkpoints/qa_learn/_verify/text_variants/）
    --------------------------------------------------------------
    * variants.jsonl —— 4 x N 条变体文本（规范化 JSONL）；
    * variants_meta.json —— 口径常量 + 逐支指纹。
    """
    blob = serialize_variant_records(records)
    meta: Dict[str, Any] = {
        "schema_version": TEXT_VARIANTS_SCHEMA_VERSION,
        "seed": int(generator.seed),
        "variant_order": list(VARIANT_ORDER),
        "definitions": [d.as_dict() for d in generator.defs],
        "keyword_backend": (
            None
            if generator.keyword_backend is None
            else {
                "kind": str(generator.keyword_backend.get("kind", "")),
                "note": str(generator.keyword_backend.get("note", "")),
            }
        ),
        "source_product_dir": str(product_dir),
        "source_rows_sha256": str(source_rows_sha256),
        "n_rows": int(len(records)),
        "fingerprint": variant_fingerprint(records),
        "generator_counters": {k: dict(v) for k, v in sorted(generator.counters.items())},
        "extractive_not_paraphrase_note": EXTRACTIVE_NOT_PARAPHRASE_NOTE,
        "granularity_guard_rule": GRANULARITY_GUARD_RULE,
        "verdict_rule": VERDICT_RULE,
    }
    if extra:
        meta.update(extra)
    meta_text = json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=1)
    return {
        "variants.jsonl": {
            "text": blob.decode("utf-8"),
            "sha256": sha256_bytes(blob),
            "bytes": int(len(blob)),
        },
        "variants_meta.json": {
            "text": meta_text,
            "sha256": sha256_bytes(meta_text.encode("utf-8")),
            "bytes": int(len(meta_text.encode("utf-8"))),
        },
    }


# ---------------------------------------------------------------------------
# 难负例构造
# ---------------------------------------------------------------------------


def hard_negative_sets(
    rows: Sequence[TextRow],
    *,
    cap: int = HARD_NEGATIVE_CAP,
) -> Dict[str, Dict[str, List[str]]]:
    """构造**结构性**难负例（同文档他行 / 相邻行）。

    参数
    ----
    rows : Sequence[TextRow]
        全量文本行（顺序 = 产物顺序）。
    cap : int
        每类最大条数（0 = 不限）。

    返回
    ----
    Dict[str, Dict[str, List[str]]]
        {row_id: {"same_file": [...], "adjacent_line": [...]}}（row_id 升序）。
    """
    items = list(rows)
    by_file: Dict[str, List[Tuple[int, str]]] = {}
    by_line: Dict[Tuple[str, int], str] = {}
    for row in items:
        by_file.setdefault(str(row.file), []).append((int(row.line_no), str(row.row_id)))
        by_line[(str(row.file), int(row.line_no))] = str(row.row_id)
    for key in by_file:
        by_file[key].sort(key=lambda t: (t[0], t[1]))

    limit = int(cap)
    out: Dict[str, Dict[str, List[str]]] = {}
    for row in items:
        rid = str(row.row_id)
        fid = str(row.file)
        lno = int(row.line_no)
        # 按 |line_no 差| 最近选取（并列按 row_id 升序），使 `same_file` 真正近似「难负例」语义。
        cand_same = [(abs(int(ln) - lno), str(other)) for ln, other in by_file.get(fid, []) if str(other) != rid]
        cand_same.sort(key=lambda t: (t[0], t[1]))
        same = [other for _, other in cand_same]
        adj: List[str] = []
        for delta in (1, -1):
            other = by_line.get((fid, lno + delta))
            if other is not None and other != rid and other not in adj:
                adj.append(other)
        if limit > 0:
            same = same[:limit]
            adj = adj[:limit]
        out[rid] = {"same_file": same, "adjacent_line": adj}
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def random_negative_sets(
    row_ids: Sequence[str],
    *,
    count: int = RANDOM_NEGATIVE_COUNT,
    seed: int = TEXT_VARIANTS_SEED,
) -> Dict[str, List[str]]:
    """随机负例（**确定性**：局部 random.Random(seed)，不消耗全局 RNG）。"""
    ids = [str(x) for x in row_ids]
    if len(ids) < 2:
        raise ValueError("random_negative_sets 需要至少 2 行")
    kk = max(1, int(count))
    rng = random.Random(int(seed))
    out: Dict[str, List[str]] = {}
    for rid in ids:
        pool = [x for x in ids if x != rid]
        picked = rng.sample(pool, k=min(kk, len(pool)))
        out[str(rid)] = [str(x) for x in picked]
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def encoder_neighbor_sets(
    rows: Sequence[TextRow],
    encoder: Any,
    *,
    topk: int = ENCODER_NEIGHBOR_TOPK,
    batch_size: int = 64,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """编码器近邻他行（**确定性**；并列按 row_id 升序打破平局）。

    参数
    ----
    rows : Sequence[TextRow]
        全量文本行。
    encoder : Any
        满足 encoders 统一接口的编码器（encode_matrix(texts) -> float32[L, D]）。
    topk : int
        每行取多少个近邻。
    batch_size : int
        编码批大小。
    on_progress : Optional[Callable[[int, int], None]]
        进度回调。

    返回
    ----
    Tuple[Dict[str, List[str]], Dict[str, Any]]
        ({row_id: [邻居 row_id, ...]}, 取证字典)。
    """
    items = list(rows)
    n = len(items)
    if n == 0:
        raise ValueError("encoder_neighbor_sets 需要至少 1 行")
    kk = max(1, int(topk))
    dim = int(getattr(encoder, "dim"))
    step = max(1, int(batch_size))
    mat = np.zeros((n, dim), dtype=np.float32)
    for b0 in range(0, n, step):
        chunk = items[b0 : b0 + step]
        mat[b0 : b0 + len(chunk)] = np.asarray(
            encoder.encode_matrix([str(r.text) for r in chunk]), dtype=np.float32
        ).reshape((len(chunk), dim))
        if on_progress is not None:
            on_progress(min(n, b0 + len(chunk)), n)

    norms = np.linalg.norm(mat, axis=1)
    zero_rows = int(np.sum(norms <= 1e-12))
    sims = mat @ mat.T
    np.fill_diagonal(sims, -np.inf)
    ids = np.asarray([str(r.row_id) for r in items], dtype=object)
    out: Dict[str, List[str]] = {}
    all_cos: List[float] = []
    for i in range(n):
        # **全序选取（R67 修正）**：此前只在「按 cos 降序的前 kk*4 个」窗口内二次排序，
        # 当窗口截断处存在同分项时，可能漏掉 row_id 更小的同分项，与注释承诺的
        # 「并列按 row_id 升序」不完全一致。现改为**对全部 n-1 个候选**按
        # (-cos, row_id) 全序排序后取前 kk，语义与注释严格一致。
        order = sorted(
            (int(j) for j in range(n) if int(j) != i),
            key=lambda j: (-float(sims[i, j]), str(ids[j])),
        )
        picked = [str(ids[j]) for j in order[:kk]]
        out[str(ids[i])] = picked
        all_cos.extend(float(sims[i, j]) for j in order[:kk])
    evidence = {
        "encoder_dim": int(dim),
        "n_rows": int(n),
        "topk": int(kk),
        "zero_norm_rows": int(zero_rows),
        "cos_min": float(min(all_cos)) if all_cos else 0.0,
        "cos_max": float(max(all_cos)) if all_cos else 0.0,
        "ties_broken_by_row_id": True,
        "rule": (
            "确定性抽取：cos = 归一化特征内积（float32）；排除自身；"
            "对全部 n-1 个候选按**全序键 (-cos, row_id 升序)** 排序后取前 topk"
        ),
    }
    return dict(sorted(out.items(), key=lambda kv: kv[0])), evidence
def build_negative_manifest(
    rows: Sequence[TextRow],
    *,
    include_encoder_neighbors: bool = True,
    encoder: Any = None,
    random_count: int = RANDOM_NEGATIVE_COUNT,
    hard_cap: int = HARD_NEGATIVE_CAP,
    encoder_topk: int = ENCODER_NEIGHBOR_TOPK,
    seed: int = TEXT_VARIANTS_SEED,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """构造**可复算的负例清单**（随机 / 同文档他行 / 相邻行 / 编码器近邻）。

    返回
    ----
    Dict[str, Any]
        {per_kind, counts, fingerprint, evidence, rule}。
    """
    items = list(rows)
    hard = hard_negative_sets(items, cap=int(hard_cap))
    per_kind: Dict[str, Dict[str, List[str]]] = {
        "same_file": {k: list(v["same_file"]) for k, v in hard.items()},
        "adjacent_line": {k: list(v["adjacent_line"]) for k, v in hard.items()},
    }
    evidence: Dict[str, Any] = {}
    if include_encoder_neighbors:
        if encoder is None:
            raise ValueError("include_encoder_neighbors=True 时必须提供 encoder")
        enc, enc_evidence = encoder_neighbor_sets(
            items, encoder, topk=int(encoder_topk), on_progress=on_progress
        )
        per_kind["encoder_neighbor"] = enc
        evidence["encoder_neighbor"] = enc_evidence
    per_kind["random"] = random_negative_sets(
        [str(r.row_id) for r in items], count=int(random_count), seed=int(seed)
    )

    counts = {
        kind: {
            "n_queries": int(len(per_kind[kind])),
            "n_queries_with_negative": int(
                sum(1 for v in per_kind[kind].values() if v)
            ),
            "n_pairs": int(sum(len(v) for v in per_kind[kind].values())),
            "min_per_query": int(
                min((len(v) for v in per_kind[kind].values()), default=0)
            ),
            "max_per_query": int(
                max((len(v) for v in per_kind[kind].values()), default=0)
            ),
        }
        for kind in per_kind
    }
    structural = {k: v for k, v in per_kind.items() if k != "random"}
    fingerprint = {
        "random": sha256_bytes(canonical_dumps(per_kind["random"])),
        "structural_sha256": sha256_bytes(canonical_dumps(structural)),
        "encoder_neighbor_sha256": (
            sha256_bytes(canonical_dumps(per_kind["encoder_neighbor"]))
            if "encoder_neighbor" in per_kind
            else ""
        ),
    }
    return {
        "per_kind": per_kind,
        "counts": counts,
        "fingerprint": fingerprint,
        "evidence": evidence,
        "encoder_neighbor_skipped": bool(not include_encoder_neighbors),
        "encoder_neighbor_skip_reason": (
            "" if include_encoder_neighbors else
            "调用方显式跳过（include_encoder_neighbors=False / CLI --no-encoder-neighbors）："
            "本产物**不含** encoder_neighbor 类，故难负例只落地 two 类（same_file / adjacent_line）。"
        ),
        "rule": (
            # 每一类只描述一次，顺序与 NEGATIVE_KINDS 一致（same_file -> adjacent_line
            # -> encoder_neighbor -> random），避免同一类被描述两次且次序错位。
            "same_file = 同 file 的他行中，按 |line_no 差| 最近取前 "
            + (str(int(hard_cap)) + " 条（并列按 row_id 升序；**抽出即全部入池**）"
               if int(hard_cap) > 0 else "全部（上限 0 = 不限；全部他行入池）")
            + "；"
            "adjacent_line = 同 file 且 line_no +/- 1 的行；"
            "encoder_neighbor = 编码器归一化特征余弦 top-n（确定性；全序排序键 = (-cos, row_id)）；"
            "random = 局部 random.Random(seed) 从全部他行中抽 "
            + str(int(random_count)) + " 条（**抽出即全部入池**）"
        ),
        "seed": int(seed),
    }


#: 硬门禁的**唯一口径文本**（R67 修正后）。
HARD_GATE_RULE: str = (
    "对**每一个可约简行**（row_is_irreducible == False），"
    "逐支（关键词 / 摘要 / 自适应）断言其 float32 特征与原文支**逐位不同**；"
    "违反即 AssertionError（fail-closed）；不可约简行（任何抽取算子只能返回它自身）"
    "的重合**如实登记**。passed = （无违反）且（摘要支重合数 == 不可约简行数）。"
)


IRREDUCIBLE_RULE: str = (
    "不可约简行 = 在三类算子可能产出的候选前缀集合中，"
    "**不存在**任何一个长度 >= SUMMARY_MIN_CHARS 且 != 原文的前缀（"
    "候选集合 = 首子句/句末切点前缀、首 SUMMARY_HEAD_MAX_CHARS 字、"
    "首 SUMMARY_MAX_CHARS 字）。该定义**不调用抽取实现**，故与实现相互独立。"
)


def row_is_irreducible(raw: str) -> bool:
    """该行是否不可约简（定义见 :data:`IRREDUCIBLE_RULE`）。

    **独立于抽取实现**：本函数**不调用** ``extractive_summary`` / ``clause_extract`` /
    ``head_extract``，而是直接枚举三类算子**可能产出的候选前缀**，
    并判定它们里是否存在一个**长度 >= SUMMARY_MIN_CHARS 且 != 原文** 的前缀。
    都不存在则不可约简。
    """
    text = str(raw)
    delims = SUMMARY_SENTENCE_DELIMITERS + SUMMARY_CLAUSE_DELIMITERS
    cut = None
    for index, char in enumerate(text):
        if char in delims:
            cut = index + 1
            break
    candidates: List[str] = []
    if cut is not None and cut < len(text):
        candidates.append(text[:cut])
    if len(text) > int(SUMMARY_HEAD_FALLBACK_BELOW_CHARS):
        candidates.append(text[: int(SUMMARY_HEAD_MAX_CHARS)])
    if len(text) > int(SUMMARY_MAX_CHARS):
        candidates.append(text[: int(SUMMARY_MAX_CHARS)])
    # 不得加入「去掉尾部标点」一类候选：冻结阶梯里**没有任何算子**
    # 会剥离尾部分隔符（仅做 rstrip 空白），故它们属于**不可约简**。
    for cand in candidates:
        trimmed = cand.rstrip()
        if len(trimmed) >= int(SUMMARY_MIN_CHARS) and trimmed != text:
            return False
    return True


# ---------------------------------------------------------------------------
# 特征差异硬门禁
# ---------------------------------------------------------------------------


def feature_matrix_for_texts(vectorizer: ZhBagVectorizer, texts: Sequence[str]) -> np.ndarray:
    """用步骤 2 同口径的确定性词袋编码（float32[L, D]，逐行 L2 归一化）。"""
    return np.asarray(vectorizer.encode_matrix([str(t) for t in texts]), dtype=np.float32)


def row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """两行余弦（零范数任一侧 -> 0.0；不产生 NaN）。"""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    """min / 分位 / max（空序列 -> 全 0.0，不产生 NaN）。"""
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25.0)),
        "median": float(np.percentile(arr, 50.0)),
        "p75": float(np.percentile(arr, 75.0)),
        "max": float(np.max(arr)),
    }


def feature_gate(
    records: Sequence[Dict[str, Any]],
    vectorizer: ZhBagVectorizer,
    *,
    tier_field: str = "adaptive_tier",
    coverage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """**特征差异硬门禁**：逐位断言、按行型分档、cos 分布。

    两类变体的处置**必须分开**（**现场实测驱动**，不是设计偏好）
    ----------------------------------------------------------------
    * :data:`IDENTITY_GATED_VARIANTS`（摘要支）：**硬门禁** ——
      ``n_identical`` 必须为 0（三级抽取阶梯保证它对任何非空行
      都能给出**真子串**）；违反即 ``assert`` 失败（fail-closed）。
    * :data:`IDENTITY_REGISTERED_VARIANTS`（关键词支 / 自适应支）：它们的
      **回退口径**（抽不出 / 与原文逐字相同 -> 原文）在现场**真实触发**
      （全量 2665 行中 **9 行**），故它们**不可能**被要求零重合；
      它们的重合比例**如实登记、不作硬门禁**。

    参数
    ----
    records : Sequence[Dict[str, Any]]
        变体记录（顺序敏感。
    vectorizer : ZhBagVectorizer
        步骤 2 同口径的确定性词袋向量化器。
    tier_field : str
        分档字段名。

    返回
    ----
    Dict[str, Any]
        ``per_variant`` / ``per_tier`` / ``hard_gate`` / ``identity_registration`` 等。
    """
    items = list(records)
    if not items:
        raise ValueError("feature_gate 需要至少 1 行记录")
    anchor_texts = [str(r["variants"][VARIANT_ANCHOR]) for r in items]
    anchor_mat = feature_matrix_for_texts(vectorizer, anchor_texts)

    per_variant: Dict[str, Any] = {}
    per_tier: Dict[str, Dict[str, Any]] = {}
    identical_by_variant: Dict[str, List[bool]] = {}
    for name in NON_ANCHOR_VARIANTS:
        texts = [str(r["variants"][name]) for r in items]
        mat = feature_matrix_for_texts(vectorizer, texts)
        identical = [bool(np.array_equal(mat[i], anchor_mat[i])) for i in range(len(items))]
        identical_by_variant[name] = identical
        cosines = [row_cosine(mat[i], anchor_mat[i]) for i in range(len(items))]
        n_identical = int(sum(1 for x in identical if x))
        per_variant[name] = {
            "n_rows": int(len(items)),
            "n_identical": int(n_identical),
            "identity_ratio": float(n_identical / max(1, len(items))),
            "n_distinct": int(len(items) - n_identical),
            "cos_distribution": _quantiles(cosines),
            "tier_field": str(tier_field),
            "hard_gate": bool(name in IDENTITY_GATED_VARIANTS),
            "identity_registered": bool(name in IDENTITY_REGISTERED_VARIANTS),
        }
        tiers = sorted({str(r.get(tier_field, "")) for r in items})
        block: Dict[str, Any] = {}
        for tier in tiers:
            idx = [i for i, r in enumerate(items) if str(r.get(tier_field, "")) == tier]
            block[tier] = {
                "n_rows": int(len(idx)),
                "n_identical": int(sum(1 for i in idx if identical[i])),
                "identity_ratio": float(
                    sum(1 for i in idx if identical[i]) / max(1, len(idx))
                ),
                "char_len_min": int(min((int(items[i]["char_len"]) for i in idx), default=0)),
                "char_len_max": int(max((int(items[i]["char_len"]) for i in idx), default=0)),
                "cos_distribution": _quantiles([cosines[i] for i in idx]),
            }
        per_tier[name] = block

    irreducible = [row_is_irreducible(str(r["variants"][VARIANT_ANCHOR])) for r in items]
    n_irreducible = int(sum(1 for x in irreducible if x))
    identity_counts = {n: int(per_variant[n]["n_identical"]) for n in NON_ANCHOR_VARIANTS}
    if IDENTITY_GATED_VARIANTS:
        for n in IDENTITY_GATED_VARIANTS:
            assert identity_counts[n] == 0, (
                "特征差异硬门禁失败：" + str(n) + " 支有 " + str(identity_counts[n])
                + " 行与原文支特征逐位相同，但该支不允许与原文支重合"
            )

    # --- **真实逐位门禁（R67 修正）** -------------------------------------------
    # 此前：`IDENTITY_GATED_VARIANTS` 为空时上面的 assert 循环**不执行**，
    # 而 `passed` 又只由「摘要支重合数 == 不可约简行数」决定，该等式又已被
    # 下方 assert 强制 ⇒ **passed 恒真风险**，计划要求的「逐位断言各支特征与原文支不同」
    # 实际上**没有任何一支被真正强制**。
    # 现在：对**每一个可约简行**（`row_is_irreducible == False`），**逐支**断言其特征与
    # 原文支**逐位不同**（违反即 AssertionError）；不可约简行（任何抽取算子都只能
    # 返回它自身）则只如实登记重合。
    reducible_idx = [i for i in range(len(items)) if not irreducible[i]]
    violations: List[Dict[str, Any]] = []
    for name in NON_ANCHOR_VARIANTS:
        for i in reducible_idx:
            if identical_by_variant[name][i]:
                violations.append(
                    {
                        "variant": str(name),
                        "row_id": str(items[i]["row_id"]),
                        "char_len": int(items[i]["char_len"]),
                        "text_head": str(items[i]["variants"][VARIANT_ANCHOR])[:40],
                    }
                )
    assert not violations, (
        "特征差异硬门禁失败：在**可约简行**上发现 "
        + str(len(violations)) + " 处“变体特征 == 原文特征”（前 5 处 = "
        + str(violations[:5]) + "）"
    )
    assert int(per_variant[VARIANT_SUMMARY]["n_identical"]) == n_irreducible, (
        "摘要支不变量失败：重合行 = "
        + str(int(per_variant[VARIANT_SUMMARY]["n_identical"]))
        + " 不等于不可约简行数 " + str(n_irreducible)
        + "（可约简行上必须为零重合）"
    )
    # 门禁字段**只**承载门禁成员（``IDENTITY_GATED_VARIANTS``）的数值；
    # 登记类支的数值由 ``identity_registration`` 承载（两者不得混用）。
    gated = {n: int(identity_counts[n]) for n in IDENTITY_GATED_VARIANTS}

    registered = {
        n: {
            "n_identical": int(per_variant[n]["n_identical"]),
            "identity_ratio": float(per_variant[n]["identity_ratio"]),
        }
        for n in IDENTITY_REGISTERED_VARIANTS
    }
    n_any = int(
        sum(
            1
            for i in range(len(items))
            if any(identical_by_variant[n][i] for n in IDENTITY_REGISTERED_VARIANTS)
        )
    )

    short_tier = str(ADAPTIVE_ALLOW_IDENTITY_TIER)
    short_idx = [i for i, r in enumerate(items) if str(r.get(tier_field, "")) == short_tier]
    adaptive_identical = identical_by_variant[VARIANT_ADAPTIVE]
    n_short_identical = int(sum(1 for i in short_idx if adaptive_identical[i]))
    registration = {
        "tier": short_tier,
        "n_rows_in_tier": int(len(short_idx)),
        "n_identical_in_tier": int(n_short_identical),
        "identity_ratio_in_tier": float(n_short_identical / max(1, len(short_idx))),
        "n_identical_overall": int(sum(1 for x in adaptive_identical if x)),
        "identity_ratio_overall": float(
            sum(1 for x in adaptive_identical if x) / max(1, len(items))
        ),
        "allowed_by_design": True,
        "occurred": bool(n_short_identical > 0),
        "note": (
            "短行档的「自适应支」在设计上允许与原文支重合；"
            "本项单独如实登记（不作硬门禁）。"
        ),
    }
    return {
        "tier_field": str(tier_field),
        "n_rows": int(len(items)),
        "per_variant": per_variant,
        "per_tier": per_tier,
        "short_tier_identity_registration": registration,
        "hard_gate": {
            "rule": HARD_GATE_RULE,
            "gated_variants": list(IDENTITY_GATED_VARIANTS),
            "gated_n_identical": gated,
            "reducible_rows": int(len(reducible_idx)),
            "reducible_rows_with_identical_features": int(len(violations)),
            "assertion_passed": bool(not violations),
            "irreducible_rows": int(n_irreducible),
            "irreducible_rule": IRREDUCIBLE_RULE,
            "summary_identity_equals_irreducible": bool(
                int(per_variant[VARIANT_SUMMARY]["n_identical"]) == n_irreducible
            ),
            "passed": bool(
                not violations
                and all(gated[n] == 0 for n in IDENTITY_GATED_VARIANTS)
                and int(per_variant[VARIANT_SUMMARY]["n_identical"]) == n_irreducible
            ),
            "violations_head": violations[:20],
        },
        "coverage": dict(coverage) if coverage else {
            "n_rows": int(len(items)),
            "source_total_rows": None,
            "limit": None,
            "note": "未传入 coverage：本块只能声明本次评估的行数 n_rows",
        },
        "identity_registration": {
            "registered_variants": list(IDENTITY_REGISTERED_VARIANTS),
            "per_variant": registered,
            "n_rows_with_any_identical": int(n_any),
            "identity_ratio_overall": float(n_any / max(1, len(items))),
            "note": (
                "「关键词」与「自适应」支的回退口径在现场真实触发"
                "（全量 9 行），故其重合比例如实登记、不作硬门禁；"
                "「摘要」支三级抽取阶梯保证它对任何非空行都能给出真子串，"
                "它仍然是硬门禁。"
            ),
        },
        "rule": (
            "逐位差异 = float32 特征向量任一分量不等；"
            "硬门禁只对 " + str(list(IDENTITY_GATED_VARIANTS)) + " 生效（必须为 0）；"
            "IDENTITY_REGISTERED_VARIANTS = " + str(list(IDENTITY_REGISTERED_VARIANTS))
            + " 的重合比例如实登记"
        ),
    }


# ---------------------------------------------------------------------------
# 零训练评测（纯余弦检索）
# ---------------------------------------------------------------------------


GOLD_SET_RULE: str = (
    "金标集 = 自身行 ∪ 与查询行**原文逐字节相同**的其他行（仅接受逐字节相同，不接受"
    "模糊 / 归一化后相同）；hit@1 = top-1 落在金标集内；"
    "rank = 金标集成员在 top-k 列表中的最前 1-based 位置（不在则 -1）。"
    "存在理由：数据集存在原文逐字节相同的重复行，若只认自身行，"
    "锚点支 R@1 会因重复行占据 top-1 而**非结构性地** < 1.0。"
)


def duplicate_original_texts(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """原文逐字节相同的行分组（**只返回组内 >= 2 行**的那些组）。"""
    groups: Dict[str, List[str]] = {}
    for rec in records:
        key = str(rec["variants"][VARIANT_ANCHOR])
        groups.setdefault(key, []).append(str(rec["row_id"]))
    return {
        key: sorted(ids)
        for key, ids in sorted(groups.items(), key=lambda kv: kv[1][0])
        if len(ids) >= 2
    }


def gold_id_sets(records: Sequence[Dict[str, Any]]) -> Dict[str, frozenset]:
    """逐查询行的**金标集**（自身 + 原文逐字节相同的其他行）。"""
    groups: Dict[str, List[str]] = {}
    for rec in records:
        groups.setdefault(str(rec["variants"][VARIANT_ANCHOR]), []).append(str(rec["row_id"]))
    return {
        rid: frozenset(ids)
        for ids in groups.values()
        for rid in ids
    }


def _library_view(
    records: Sequence[Dict[str, Any]],
    variant: str,
    vectorizer: ZhBagVectorizer,
) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """库（检索池）视图：该支的变体特征 + 行 id -> 下标 映射 + 行 id 有序表。"""
    ids = [str(r["row_id"]) for r in records]
    texts = [str(r["variants"][variant]) for r in records]
    mat = feature_matrix_for_texts(vectorizer, texts)
    return mat, {rid: i for i, rid in enumerate(ids)}, ids


def evaluate_retrieval(
    records: Sequence[Dict[str, Any]],
    vectorizer: ZhBagVectorizer,
    *,
    variant: str,
    query_ids: Sequence[str],
    retrieval_ids: Sequence[str],
    topk: int = 5,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """**纯余弦检索**（零训练、无模型）：查询支在该支的检索池上检索。

    参数
    ----
    records : Sequence[Dict[str, Any]]
        变体记录（**全量行**，同时充当检索池）。
    vectorizer : ZhBagVectorizer
        步骤 2 同口径的确定性词袋向量化器。
    variant : str
        查询支名（4 支之一）。
    query_ids : Sequence[str]
        参与评测的查询行 id（顺序敏感）。
    retrieval_ids : Sequence[str]
        检索池行 id（**库支恒为「原文支」**；顺序敏感）。
    topk : int
        Recall@k 的 k。
    batch_size : int
        打分批大小。

    返回
    ----
    Dict[str, Any]
        {n, n_pool, recall_at_1, recall_at_5, rank_hist, misses_head, ...}。
    """
    items = {str(r["row_id"]): r for r in records}
    gold_sets = gold_id_sets(records)
    duplicate_hits = 0
    q_ids = [str(x) for x in query_ids]
    for rid in q_ids:
        if rid not in items:
            raise KeyError("查询行 " + repr(rid) + " 不在变体记录中")
    lib_mat, pos_of, _lib_ids = _library_view(records, VARIANT_ANCHOR, vectorizer)
    pool_ids = [str(x) for x in retrieval_ids]
    keep = [pos_of[rid] for rid in pool_ids]
    mat = lib_mat[np.asarray(keep, dtype=np.int64)]
    local_of = {rid: i for i, rid in enumerate(pool_ids)}

    kk = max(1, int(topk))
    q_texts = [str(items[rid]["variants"][variant]) for rid in q_ids]
    q_mat = feature_matrix_for_texts(vectorizer, q_texts)

    hits1 = 0
    hits5 = 0
    ranks: List[int] = []
    misses: List[Dict[str, Any]] = []
    step = max(1, int(batch_size))
    for b0 in range(0, len(q_ids), step):
        chunk = list(range(b0, min(len(q_ids), b0 + step)))
        sims = q_mat[chunk] @ mat.T
        order = np.argsort(-sims, axis=1, kind="stable")[:, :kk]
        for local_row, qi in enumerate(chunk):
            rid = q_ids[qi]
            gold = local_of.get(rid)
            if gold is None:
                raise ValueError("查询行 " + repr(rid) + " 不在检索池中（自检索要求同池）")
            rank_list = [int(x) for x in order[local_row].tolist()]
            # 注意：不得用 ``gold`` 作临时名（它是金标集映射）
            gold_ids = gold_sets[rid]
            rank = -1
            for position, value in enumerate(rank_list):
                if pool_ids[value] in gold_ids:
                    rank = position + 1
                    break
            ranks.append(int(rank))
            hits1 += int(rank == 1)
            hits5 += int(1 <= rank <= kk)
            if rank == 1 and pool_ids[rank_list[0]] != rid:
                duplicate_hits += 1
            if rank != 1 and len(misses) < 20:
                misses.append(
                    {
                        "row_id": rid,
                        "file": str(items[rid]["file"]),
                        "line_no": int(items[rid]["line_no"]),
                        "rank": int(rank),
                        "gold_ids": sorted(gold_ids),
                        "top1_row_id": pool_ids[rank_list[0]],
                    }
                )
    n = len(q_ids)
    return {
        "variant": str(variant),
        "n": int(n),
        "n_pool": int(len(pool_ids)),
        "topk": int(kk),
        "recall_at_1": float(hits1 / max(1, n)),
        "recall_at_5": float(hits5 / max(1, n)),
        "rank_1": int(sum(1 for r in ranks if r == 1)),
        "rank_miss_topk": int(sum(1 for r in ranks if r < 0)),
        "rank_hist": {
            str(i): int(sum(1 for r in ranks if r == i)) for i in range(1, kk + 1)
        },
        "misses_head": misses,
        "pool_rule": "检索池 = 该支特征键表全量；库支恒为「原文支」",
        "gold_set_rule": GOLD_SET_RULE,
        "duplicate_gold_hits_at_1": int(duplicate_hits),
    }


SMALL_POOL_RULE: str = (
    "\u300c\u96be\u8d1f\u4f8b\u5c0f\u6c60\u300d\u8bbe\u5b9a\uff1a\u5bf9**\u6bcf\u4e2a\u67e5\u8be2\u884c**\uff0c\u68c0\u7d22\u6c60 = { \u81ea\u8eab\u884c } \u222a "
    "{\u8be5\u884c\u7684\u96be\u8d1f\u4f8b\uff08same_file -> adjacent_line -> encoder_neighbor \u987a\u5e8f\uff09\u524d "
    "NEGATIVE_POOL_SIZE \u6761}\uff1b\u300c\u968f\u673a\u8d1f\u4f8b\u5c0f\u6c60\u300d\u540c\u6c60\u5927\u5c0f\uff0c\u53ea\u5c06"
    "\u8d1f\u4f8b\u6362\u6210 random \u7c7b\u3002\u4e24\u8005**\u6c60\u5927\u5c0f\u5b8c\u5168\u4e00\u81f4\u3001\u53ea\u5dee\u8d1f\u4f8b\u5185\u5bb9**\uff0c"
    "\u56e0\u6b64\u5176 R@1 \u4e4b\u5dee\u662f**\u771f\u6b63\u4f9d\u8d56 negatives.json \u5185\u5bb9**\u7684\u5bf9\u7167\u91cf\uff08\u800c\u975e\u6052\u4e3a 0 \u7684\u5047\u5bf9\u7167\uff09\u3002"
)

SMALL_POOL_PRECONDITION_RULE: str = (
    "\u96be\u8d1f\u4f8b\u5bf9\u7167 B - C \u53ea\u5728**\u9010\u67e5\u8be2\u884c\u7684\u6c60\u5927\u5c0f\u4e25\u683c\u76f8\u7b49**"
    "\u65f6\u624d**\u53ea\u662f\u300c\u8d1f\u4f8b\u5185\u5bb9\u300d\u4e4b\u5dee**\uff1b\u5426\u5219\u5b83\u540c\u65f6\u6df7\u5165\u300c\u6c60\u5927\u5c0f\u300d\u4e4b\u5dee\uff0c"
    "\u8bfb\u6cd5\u5fc5\u987b\u6539\u4e3a\u300c\u6df7\u5408\u5dee\u300d\u3002\u73b0\u573a\u5145\u5206\u6761\u4ef6\uff1a\u6bcf\u4e2a\u67e5\u8be2\u884c\u5747\u6709 "
    ">= NEGATIVE_POOL_SIZE \u6761**\u540c\u7c7b**\u8d1f\u4f8b\uff08\u672c\u8f6e\uff1a\u6bcf\u884c 16 \u6761 same_file \u4e0e 16 \u6761 random\uff09\uff0c"
    "\u4e14\u4e24\u8bbe\u5b9a\u7684**\u67e5\u8be2\u884c\u96c6\u76f8\u540c**\u3002\u8be5\u6761\u4ef6\u5bf9**\u66f4\u5c0f\u8bed\u6599\u6216\u5355\u6587\u4ef6\u8bed\u6599\u53ef\u80fd\u4e0d\u6210\u7acb**\uff0c"
    "\u6545 `zero_training_eval` \u5728\u8bc4\u6d4b\u524d**fail-closed \u65ad\u8a00**\uff0c\u5e76\u5728\u62a5\u544a\u91cc\u767b\u8bb0\u73b0\u573a\u6c60\u5927\u5c0f\u4e09\u5143\u7ec4\u3002"
)


FULL_POOL_RULE: str = (
    "\u300c\u5168\u91cf\u6c60\u300d\u8bbe\u5b9a\uff1a\u68c0\u7d22\u6c60 = \u672c\u6b21\u8bc4\u4f30\u7684\u5168\u90e8\u884c\uff08\u5e93\u652f\u6052\u4e3a\u300c\u539f\u6587\u652f\u300d\uff09\uff0c"
    "\u67e5\u8be2\u884c = \u5168\u90e8\u884c\u3002**\u672c\u8bbe\u5b9a\u4e0d\u6d88\u8d39 negatives.json**\uff08\u8d1f\u4f8b\u4e0d\u8fdb\u6c60\uff09\uff0c"
    "\u6545\u5b83\u662f\u300c\u4e0d\u53d7\u8d1f\u4f8b\u5f71\u54cd\u300d\u7684\u4e3b\u6307\u6807\u57fa\u7ebf\uff0c**\u4e0d\u5f97**\u88ab\u5f53\u4f5c\u96be\u8d1f\u4f8b\u5bf9\u7167\u3002"
)


@dataclass(frozen=True)
class NegativeSetting:
    """\u4e00\u4e2a**\u8d1f\u4f8b\u8bbe\u5b9a**\u7684\u51bb\u7ed3\u5b9a\u4e49\u3002"""

    key: str
    description: str
    query_rule: str
    pool_rule: str
    consumes_negatives: bool


#: \u4e09\u79cd\u8bbe\u5b9a\uff08**\u5206\u5f00\u62a5**\uff09\u3002
NEGATIVE_SETTINGS: Tuple[NegativeSetting, ...] = (
    NegativeSetting(
        key="full_pool_all_rows",
        description="\u8bbe\u5b9a A\uff08\u5168\u91cf\u6c60\uff09\uff1a\u68c0\u7d22\u6c60 = \u672c\u6b21\u8bc4\u4f30\u7684\u5168\u90e8\u884c\uff0c\u67e5\u8be2\u884c = \u5168\u90e8\u884c\uff1b\u4e0d\u6d88\u8d39\u8d1f\u4f8b\u6e05\u5355",
        query_rule="query_ids = \u672c\u6b21\u8bc4\u4f30\u7684\u5168\u90e8\u884c id",
        pool_rule=FULL_POOL_RULE,
        consumes_negatives=False,
    ),
    NegativeSetting(
        key="small_pool_hard",
        description="\u8bbe\u5b9a B\uff08\u96be\u8d1f\u4f8b\u5c0f\u6c60\uff09\uff1a\u6bcf\u67e5\u8be2\u884c\u7684\u6c60 = { \u81ea\u8eab } \u222a \u5176\u96be\u8d1f\u4f8b\uff08\u6700\u591a NEGATIVE_POOL_SIZE \u6761\uff09",
        query_rule="query_ids = \u6709 >=1 \u6761\u96be\u8d1f\u4f8b\u7684\u884c",
        pool_rule=SMALL_POOL_RULE,
        consumes_negatives=True,
    ),
    NegativeSetting(
        key="small_pool_random",
        description="\u8bbe\u5b9a C\uff08\u968f\u673a\u8d1f\u4f8b\u5c0f\u6c60\uff09\uff1a\u4e0e\u8bbe\u5b9a B **\u6c60\u5927\u5c0f\u5b8c\u5168\u4e00\u81f4**\uff0c\u53ea\u5c06\u8d1f\u4f8b\u6362\u4e3a random \u7c7b",
        query_rule="query_ids = \u6709 >=1 \u6761\u968f\u673a\u8d1f\u4f8b\u7684\u884c",
        pool_rule=SMALL_POOL_RULE,
        consumes_negatives=True,
    ),
)


def _negative_pool_for(
    rid: str,
    kinds: Sequence[str],
    per_kind: Dict[str, Dict[str, List[str]]],
    *,
    limit: int = NEGATIVE_POOL_SIZE,
) -> List[str]:
    """\u6309\u56fa\u5b9a\u7c7b\u522b\u987a\u5e8f\u53d6\u8be5\u884c\u7684\u8d1f\u4f8b\uff08\u786e\u5b9a\u6027\u622a\u65ad\u5230 ``limit``\uff0c\u53bb\u91cd\u3001\u4e0d\u542b\u81ea\u8eab\uff09\u3002"""
    out: List[str] = []
    for kind in kinds:
        block = per_kind.get(kind, {}) or {}
        for other in block.get(str(rid), []) or []:
            other = str(other)
            if other != str(rid) and other not in out:
                out.append(other)
            if len(out) >= int(limit):
                return out
    return out


def evaluate_retrieval_pools(
    records: Sequence[Dict[str, Any]],
    vectorizer: ZhBagVectorizer,
    *,
    variant: str,
    pools_by_query: Dict[str, Sequence[str]],
    topk: int = 5,
) -> Dict[str, Any]:
    """**\u9010\u67e5\u8be2\u6c60**\u7684\u7eaf\u4f59\u5f26\u68c0\u7d22\uff08\u96f6\u8bad\u7ec3\uff09\u3002

    \u4e0e :func:`evaluate_retrieval` \u7684\u533a\u522b\uff1a\u68c0\u7d22\u6c60**\u9010\u67e5\u8be2\u884c\u4e0d\u540c**\uff08\u7528\u4e8e\u300c\u96be\u8d1f\u4f8b\u5c0f\u6c60\u300d\u8bbe\u5b9a\uff09\u3002
    \u6bcf\u4e2a\u6c60\u5fc5\u987b\u5305\u542b\u67e5\u8be2\u884c\u81ea\u8eab\uff08\u5426\u5219\u91d1\u6807\u4e0d\u53ef\u8fbe\uff0c\u76f4\u63a5\u62a5\u9519\uff09\u3002
    """
    items = {str(r["row_id"]): r for r in records}
    gold_sets = gold_id_sets(records)
    lib_mat, pos_of, _ids = _library_view(records, VARIANT_ANCHOR, vectorizer)
    kk = max(1, int(topk))

    q_ids = [str(x) for x in pools_by_query.keys()]
    q_texts = [str(items[rid]["variants"][variant]) for rid in q_ids]
    q_mat = feature_matrix_for_texts(vectorizer, q_texts)

    hits1 = 0
    hits5 = 0
    ranks: List[int] = []
    pool_sizes: List[int] = []
    duplicate_hits = 0
    misses: List[Dict[str, Any]] = []
    for i, rid in enumerate(q_ids):
        pool = [str(x) for x in pools_by_query[rid]]
        if rid not in pool:
            raise ValueError("\u67e5\u8be2\u884c " + repr(rid) + " \u4e0d\u5728\u81ea\u5df1\u7684\u68c0\u7d22\u6c60\u4e2d")
        mat = lib_mat[np.asarray([pos_of[x] for x in pool], dtype=np.int64)]
        sims = mat @ q_mat[i]
        order = np.argsort(-sims, kind="stable")[:kk]
        gold_ids = gold_sets[rid]
        rank = -1
        for position, value in enumerate(order.tolist()):
            if pool[int(value)] in gold_ids:
                rank = position + 1
                break
        ranks.append(int(rank))
        pool_sizes.append(int(len(pool)))
        hits1 += int(rank == 1)
        hits5 += int(1 <= rank <= kk)
        if rank == 1 and pool[int(order[0])] != rid:
            duplicate_hits += 1
        if rank != 1 and len(misses) < 20:
            misses.append(
                {
                    "row_id": rid,
                    "rank": int(rank),
                    "pool_size": int(len(pool)),
                    "gold_ids": sorted(gold_ids),
                    "top1_row_id": pool[int(order[0])],
                }
            )
    n = len(q_ids)
    arr = np.asarray(pool_sizes if pool_sizes else [0], dtype=np.int64)
    return {
        "variant": str(variant),
        "n": int(n),
        "n_pool": None,
        "pool_size_min": int(np.min(arr)),
        "pool_size_median": float(np.median(arr)),
        "pool_size_max": int(np.max(arr)),
        "topk": int(kk),
        "recall_at_1": float(hits1 / max(1, n)),
        "recall_at_5": float(hits5 / max(1, n)),
        "rank_1": int(sum(1 for r in ranks if r == 1)),
        "rank_miss_topk": int(sum(1 for r in ranks if r < 0)),
        "rank_hist": {
            str(i): int(sum(1 for r in ranks if r == i)) for i in range(1, kk + 1)
        },
        "misses_head": misses,
        "pool_rule": SMALL_POOL_RULE,
        "gold_set_rule": GOLD_SET_RULE,
        "duplicate_gold_hits_at_1": int(duplicate_hits),
    }


def negative_setting_payload(
    manifest: Dict[str, Any],
    *,
    common_query_ids: Sequence[str],
) -> Dict[str, Any]:
    """\u628a\u8d1f\u4f8b\u6e05\u5355\u6574\u7406\u6210**\u4e09\u79cd\u8bbe\u5b9a + \u516c\u5171\u5b50\u96c6**\u7684\u8bc4\u6d4b\u89c6\u56fe\u3002

    \u5f02\u5e38
    ----
    RuntimeError
        \u8d1f\u4f8b\u6e05\u5355\u662f probe \u5199\u4e0b\u7684**\u8df3\u8fc7\u6807\u8bb0**\uff08\u884c\u6570 < 2 \u65e0\u6cd5\u62bd\u968f\u673a\u8d1f\u4f8b\uff09\u65f6\u629b\u51fa\u53ef\u8bfb\u62a5\u6587
        \uff08**\u4e0d**\u629b ``KeyError``\uff1a\u5b83\u4f1a\u88ab CLI \u8bef\u5f52\u7c7b\u6210\u300c\u975e\u6cd5\u53c2\u6570\u300d\u9000\u7801 2\uff09\u3002
    """
    if bool(manifest.get("skipped")):
        raise RuntimeError(
            "\u8d1f\u4f8b\u6e05\u5355\u4e3a\u8df3\u8fc7\u6807\u8bb0\uff08\u884c\u6570 < 2\uff09\uff1a"
            + str(manifest.get("reason", ""))
            + "\uff1b\u8bf7\u7528 >= 2 \u884c\u91cd\u8dd1 probe\u3002"
        )
    per_kind = manifest["per_kind"]
    all_ids = sorted(per_kind["random"].keys())
    hard_kinds = ("same_file", "adjacent_line", "encoder_neighbor")
    hard_pools = {
        rid: _negative_pool_for(rid, hard_kinds, per_kind) for rid in all_ids
    }
    random_pools = {
        rid: _negative_pool_for(rid, ("random",), per_kind) for rid in all_ids
    }
    hard_ids = sorted(rid for rid in all_ids if hard_pools[rid])
    random_ids = sorted(rid for rid in all_ids if random_pools[rid])
    common = [str(x) for x in common_query_ids if str(x) in set(all_ids)]
    return {
        "all_row_ids": list(all_ids),
        "hard_eligible_row_ids": list(hard_ids),
        "random_eligible_row_ids": list(random_ids),
        "common_subset_row_ids": list(common),
        "hard_pools": hard_pools,
        "random_pools": random_pools,
        "counts": {
            "n_all_rows": int(len(all_ids)),
            "n_hard_eligible_rows": int(len(hard_ids)),
            "n_random_eligible_rows": int(len(random_ids)),
            "n_common_subset": int(len(common)),
        },
        "per_kind_counts": manifest["counts"],
        "fingerprint": manifest["fingerprint"],
    }


def zero_training_eval(
    records: Sequence[Dict[str, Any]],
    vectorizer: ZhBagVectorizer,
    manifest: Dict[str, Any],
    *,
    query_split_ids: Sequence[str] = (),
    topk: int = 5,
    coverage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """**\u96f6\u8bad\u7ec3\u8bc4\u6d4b**\uff1a4 \u652f x 3 \u79cd\u8bbe\u5b9a\uff08+ \u516c\u5171\u5b50\u96c6\uff09\u3002

    \u4e09\u79cd\u8bbe\u5b9a\uff08**\u5fc5\u987b\u5206\u5f00\u62a5**\uff09
    --------------------------------
    * A ``full_pool_all_rows``\uff1a\u6c60 = \u5168\u90e8\u884c\uff08**\u4e0d\u6d88\u8d39\u8d1f\u4f8b**\uff09\u2014\u2014 \u4e3b\u6307\u6807\u57fa\u7ebf\uff1b
    * B ``small_pool_hard``\uff1a\u6bcf\u884c\u6c60 = {\u81ea\u8eab} \u222a \u96be\u8d1f\u4f8b\uff08\u2264 :data:`NEGATIVE_POOL_SIZE`\uff09\uff1b
    * C ``small_pool_random``\uff1a\u6bcf\u884c\u6c60 = {\u81ea\u8eab} \u222a \u968f\u673a\u8d1f\u4f8b\uff08\u540c\u6c60\u5927\u5c0f\uff09\u3002

    **\u96be\u8d1f\u4f8b\u5bf9\u7167** = B \u2212 C\uff08\u6c60\u5927\u5c0f\u5b8c\u5168\u4e00\u81f4\u3001\u53ea\u5dee\u8d1f\u4f8b\u5185\u5bb9\uff09\u3002
    A \u4e0e B/C \u7684\u5dee\u522b**\u540c\u65f6\u5305\u542b\u6c60\u5927\u5c0f\u4e0e\u8d1f\u4f8b\u5185\u5bb9**\uff0c\u6545\u4e0d\u4f5c\u4e3a\u96be\u8d1f\u4f8b\u5bf9\u7167\u3002
    """
    view = negative_setting_payload(manifest, common_query_ids=query_split_ids)
    all_ids = list(view["all_row_ids"])
    pool_all = list(all_ids)
    common = list(view["common_subset_row_ids"])

    def _block_full(variant: str, ids: Sequence[str]) -> Dict[str, Any]:
        return evaluate_retrieval(
            records, vectorizer,
            variant=variant, query_ids=ids, retrieval_ids=pool_all, topk=topk,
        )

    def _block_small(variant: str, ids: Sequence[str], pools: Dict[str, List[str]]) -> Dict[str, Any]:
        sub = {
            str(rid): [str(rid)] + [str(x) for x in pools[str(rid)]]
            for rid in ids
        }
        return evaluate_retrieval_pools(
            records, vectorizer, variant=variant, pools_by_query=sub, topk=topk,
        )

    hard_ids = list(view["hard_eligible_row_ids"])
    random_ids = list(view["random_eligible_row_ids"])

    # --- \u5c0f\u6c60\u8bbe\u5b9a\u7684**\u663e\u5f0f\u524d\u7f6e\u6761\u4ef6**\uff08R68 \u5c0f\u6536\u53e3\u8f6e\uff09------------------------------
    # B - C \u53ea\u6709\u5728\u4e24\u4fa7\u9010\u67e5\u8be2\u884c\u7684\u6c60\u5927\u5c0f\u4e25\u683c\u76f8\u7b49\u65f6\u624d\u53ea\u662f\u300c\u8d1f\u4f8b\u5185\u5bb9\u300d\u4e4b\u5dee\uff1b
    # \u5426\u5219\u5b83\u540c\u65f6\u6df7\u5165\u300c\u6c60\u5927\u5c0f\u300d\u4e4b\u5dee\uff0c\u8bfb\u6cd5\u5fc5\u987b\u6539\u3002\u672c\u8f6e\u73b0\u573a\u6ee1\u8db3\uff08\u6bcf\u884c 16 \u6761 same_file + 16 \u6761 random\uff09\uff0c
    # \u4f46\u5bf9\u66f4\u5c0f\u8bed\u6599 / \u5355\u6587\u4ef6\u8bed\u6599\u53ef\u80fd\u4e0d\u6210\u7acb\uff0c\u6545\u5728\u8fd9\u91cc fail-closed \u65ad\u8a00\u3002
    _same_query_set = bool(hard_ids == random_ids)
    _pool_size_equal = all(
        len(view["hard_pools"][rid]) == len(view["random_pools"][rid]) for rid in hard_ids
    )
    assert _same_query_set and _pool_size_equal, (
        "\u96be\u8d1f\u4f8b\u5c0f\u6c60\u8bbe\u5b9a\u7684\u524d\u7f6e\u6761\u4ef6\u4e0d\u6210\u7acb\uff1a" + SMALL_POOL_PRECONDITION_RULE
        + "\uff08\u73b0\u573a\uff1a\u67e5\u8be2\u884c\u96c6\u76f8\u540c = " + str(_same_query_set)
        + "\uff1b\u9010\u884c\u6c60\u5927\u5c0f\u76f8\u7b49 = " + str(_pool_size_equal)
        + "\uff1bn_hard = " + str(len(hard_ids)) + "\uff1bn_random = " + str(len(random_ids)) + "\uff09"
    )

    settings: Dict[str, Any] = {
        "full_pool_all_rows": {
            "description": NEGATIVE_SETTINGS[0].description,
            "query_ids_rule": NEGATIVE_SETTINGS[0].query_rule,
            "pool_rule": FULL_POOL_RULE,
            "consumes_negatives": False,
            "n_queries": int(len(all_ids)),
            "variants": {name: _block_full(name, all_ids) for name in VARIANT_ORDER},
        },
        "small_pool_hard": {
            "description": NEGATIVE_SETTINGS[1].description,
            "query_ids_rule": NEGATIVE_SETTINGS[1].query_rule,
            "pool_rule": SMALL_POOL_RULE,
            "consumes_negatives": True,
            "n_queries": int(len(hard_ids)),
            "negative_counts": view["per_kind_counts"],
            "variants": {
                name: _block_small(name, hard_ids, view["hard_pools"])
                for name in VARIANT_ORDER
            },
        },
        "small_pool_random": {
            "description": NEGATIVE_SETTINGS[2].description,
            "query_ids_rule": NEGATIVE_SETTINGS[2].query_rule,
            "pool_rule": SMALL_POOL_RULE,
            "consumes_negatives": True,
            "n_queries": int(len(random_ids)),
            "variants": {
                name: _block_small(name, random_ids, view["random_pools"])
                for name in VARIANT_ORDER
            },
        },
    }
    common_block: Dict[str, Any] = {}
    if common:
        common_block = {
            "query_ids_rule": "\u516c\u5171\u5b50\u96c6 = n3d_qa \u51bb\u7ed3\u5212\u5206\u7684\u67e5\u8be2\u884c\uff08\u4e09\u79cd\u8bbe\u5b9a\u90fd\u5728\u540c\u4e00\u6279\u67e5\u8be2\u884c\u4e0a\u8bc4\u6d4b\uff09",
            "n": int(len(common)),
            "variants_full_pool": {n: _block_full(n, common) for n in VARIANT_ORDER},
            "variants_small_pool_hard": {
                n: _block_small(n, common, view["hard_pools"]) for n in VARIANT_ORDER
            },
            "variants_small_pool_random": {
                n: _block_small(n, common, view["random_pools"]) for n in VARIANT_ORDER
            },
        }

    def _r1(setting: str, name: str) -> float:
        return float(settings[setting]["variants"][name]["recall_at_1"])

    # --- \u96be\u8d1f\u4f8b\u5bf9\u7167\uff08B - C\uff0c\u540c\u6c60\u5927\u5c0f\uff09\uff0c\u7c92\u5ea6\u5b88\u536b\u7528\u300c\u672c\u8bbe\u5b9a\u67e5\u8be2\u884c\u6570\u300d---
    delta: Dict[str, Any] = {}
    n_hard = max(1, int(settings["small_pool_hard"]["n_queries"]))
    gran_hard = 1.0 / n_hard
    for name in VARIANT_ORDER:
        hard = _r1("small_pool_hard", name)
        rnd = _r1("small_pool_random", name)
        gap = abs(float(hard - rnd))
        delta[name] = {
            "recall_at_1_small_pool_hard": float(hard),
            "recall_at_1_small_pool_random": float(rnd),
            "delta_hard_minus_random": float(hard - rnd),
            "gap_abs": float(gap),
            "granularity_used": float(gran_hard),
            "below_granularity": bool(gap <= gran_hard),
            "recall_at_1_full_pool_reference": float(_r1("full_pool_all_rows", name)),
            "note": (
                "\u96be\u8d1f\u4f8b\u5bf9\u7167 = \u540c\u6c60\u5927\u5c0f\u7684 B - C\uff08\u53ea\u5dee\u8d1f\u4f8b\u5185\u5bb9\uff09\uff1b"
                "\u843d\u5dee <= \u7c92\u5ea6 " + repr(gran_hard) + " \u8005\u4e0d\u5f97\u4f5c\u4e3a\u5224\u5b9a\u4f9d\u636e"
                if gap <= gran_hard else
                "\u96be\u8d1f\u4f8b\u5bf9\u7167 = \u540c\u6c60\u5927\u5c0f\u7684 B - C\uff1b\u843d\u5dee > \u7c92\u5ea6 "
                + repr(gran_hard) + "\uff0c\u53ef\u4f5c\u4e3a\u5bf9\u7167\u4f9d\u636e"
            ),
        }

    # --- \u5b9e\u6d4b\u53e3\u5f84\u4e0e\u58f0\u660e\u53e3\u5f84\u5206\u79bb\uff08warning 4\uff09---
    def _pool_stats(block: Dict[str, Any]) -> Dict[str, Any]:
        ent = block["variants"][VARIANT_ANCHOR]
        med = float(ent.get("pool_size_median") or 0.0)
        return {
            "n_queries": int(block["n_queries"]),
            "n_pool": ent.get("n_pool"),
            "pool_size_min": ent.get("pool_size_min"),
            "pool_size_median": med,
            "pool_size_max": ent.get("pool_size_max"),
        }

    measured = {
        "n_rows": int(len(records)),
        "source_total_rows": (coverage or {}).get("source_total_rows"),
        "limit": (coverage or {}).get("limit"),
        "granularity_measured": float(1.0 / max(1, len(records))),
        "granularity_measured_rule": "1 / n_rows\uff08\u672c\u6b21\u8bc4\u4f30\u7684\u884c\u6570\uff09",
        "random_baseline_measured_full_pool": float(1.0 / max(1, len(pool_all))),
        "random_baseline_measured_rule": (
            "\u5168\u91cf\u6c60\u7684\u968f\u673a\u57fa\u7ebf = 1/\u6c60\u5927\u5c0f\uff1b"
            "\u5c0f\u6c60\u8bbe\u5b9a\u7684\u6c60\u5927\u5c0f\u9010\u884c\u4e0d\u540c\uff0c\u5176\u57fa\u7ebf\u4e0d\u53ef\u7528\u5355\u4e00\u5e38\u91cf\u8868\u793a"
        ),
        "declared_constants": {
            "RETRIEVAL_POOL_SIZE": int(RETRIEVAL_POOL_SIZE),
            "RANDOM_BASELINE": float(RANDOM_BASELINE),
            "GRANULARITY": float(GRANULARITY),
            "note": (
                "\u4e0a\u5217\u4e09\u4e2a\u5e38\u91cf\u662f**\u5168\u8bed\u6599\u53e3\u5f84\u5b9a\u4e49**\uff08\u7528\u4e8e\u5224\u5b9a\u89c4\u5219\u6587\u672c\uff09\uff0c"
                "**\u4e0d\u5f97**\u76f4\u63a5\u5f53\u4f5c\u672c\u6b21\u8fd0\u884c\u7684\u5b9e\u6d4b\u57fa\u7ebf\uff1b"
                "\u672c\u6b21\u8fd0\u884c\u7684\u5b9e\u6d4b\u57fa\u7ebf\u89c1 random_baseline_measured_full_pool \u4e0e granularity_measured"
            ),
        },
        "per_setting": {
            "full_pool_all_rows": _pool_stats(settings["full_pool_all_rows"]),
            "small_pool_hard": _pool_stats(settings["small_pool_hard"]),
            "small_pool_random": _pool_stats(settings["small_pool_random"]),
        },
    }

    return {
        "settings": settings,
        "common_subset": common_block,
        "hard_vs_random_delta": delta,
        "measured_scope": measured,
        "small_pool_rule": SMALL_POOL_RULE,
        "full_pool_rule": FULL_POOL_RULE,
        "granularity_guard_rule": GRANULARITY_GUARD_RULE,
        "negatives_participation": {
            "full_pool_all_rows": False,
            "small_pool_hard": True,
            "small_pool_random": True,
            "note": (
                "**\u53ea\u6709 B/C \u4e24\u4e2a\u5c0f\u6c60\u8bbe\u5b9a\u771f\u6b63\u6d88\u8d39 negatives.json**\uff1b"
                "A\uff08\u5168\u91cf\u6c60\uff09\u4e0d\u6d88\u8d39\u8d1f\u4f8b\uff0c\u6545\u4e0d\u53ef\u5f53\u4f5c\u96be\u8d1f\u4f8b\u5bf9\u7167\u57fa\u7840\u3002"
            ),
        },
    }


# ---------------------------------------------------------------------------
# 判定（机器可读、单点来源）
# ---------------------------------------------------------------------------


def render_verdict(
    anchor_recall_at_1: float,
    recall_at_1_by_variant: Dict[str, float],
) -> Dict[str, Any]:
    """按 :data:VERDICT_RULE 产出**机器可读、单点来源**的判定结论。

    参数
    ----
    anchor_recall_at_1 : float
        锚点支（原文）的 R@1。
    recall_at_1_by_variant : Dict[str, float]
        非锚点支的 R@1。

    返回
    ----
    Dict[str, Any]
        {anchor_ok, anchor_detail, per_variant, verdict_key, verdict_keys, rule, constants}。
    """
    anchor_ok = bool(
        abs(float(anchor_recall_at_1) - ANCHOR_EXPECTED_RECALL_AT_1) <= ANCHOR_TOLERANCE
    )
    per_variant: Dict[str, Any] = {}
    keys: List[str] = []
    for name in NON_ANCHOR_VARIANTS:
        if name not in recall_at_1_by_variant:
            continue
        r1 = float(recall_at_1_by_variant[name])
        gap = float(1.0 - r1)
        if not anchor_ok:
            key = "anchor_not_structural"
        elif gap <= IS_IDENTITY_TOL:
            key = "variant_too_close"
        elif r1 <= IS_ZERO_TOL:
            key = "variant_too_far"
        elif gap >= GAP_TOL and (r1 - BASE_TOL) >= BASELINE_TOL:
            key = "task_valid_not_identity"
        else:
            key = "undecided"
        per_variant[name] = {
            "recall_at_1": r1,
            "one_minus_recall_at_1": gap,
            "above_random_baseline": float(r1 - BASE_TOL),
            "significant_gap": bool(gap >= GAP_TOL),
            "above_baseline": bool((r1 - BASE_TOL) >= BASELINE_TOL),
            "close_to_one": bool(gap <= IS_IDENTITY_TOL),
            "close_to_zero": bool(r1 <= IS_ZERO_TOL),
            "verdict_key": key,
            "is_identity_task": bool(gap <= IS_IDENTITY_TOL),
        }
        keys.append(key)
    if not anchor_ok:
        verdict_key = "anchor_not_structural"
    elif keys and all(k == "task_valid_not_identity" for k in keys):
        verdict_key = "task_valid_not_identity"
    elif keys and all(k == "variant_too_close" for k in keys):
        verdict_key = "variant_too_close"
    elif keys and all(k == "variant_too_far" for k in keys):
        verdict_key = "variant_too_far"
    elif keys and all(k == "undecided" for k in keys):
        verdict_key = "undecided"
    else:
        verdict_key = "mixed"
    return {
        "anchor_ok": bool(anchor_ok),
        "anchor_detail": {
            "recall_at_1": float(anchor_recall_at_1),
            "expected": float(ANCHOR_EXPECTED_RECALL_AT_1),
            "tolerance": float(ANCHOR_TOLERANCE),
            "abs_diff": float(
                abs(float(anchor_recall_at_1) - ANCHOR_EXPECTED_RECALL_AT_1)
            ),
        },
        "per_variant": per_variant,
        "verdict_key": str(verdict_key),
        "verdict_keys": list(keys),
        "rule": VERDICT_RULE,
        "constants": {
            "ANCHOR_EXPECTED_RECALL_AT_1": float(ANCHOR_EXPECTED_RECALL_AT_1),
            "ANCHOR_TOLERANCE": float(ANCHOR_TOLERANCE),
            "SIGNIFICANT_GAP": float(SIGNIFICANT_GAP),
            "CLOSE_TO_ONE_TOLERANCE": float(CLOSE_TO_ONE_TOLERANCE),
            "CLOSE_TO_ZERO_TOLERANCE": float(CLOSE_TO_ZERO_TOLERANCE),
            "RANDOM_BASELINE": float(RANDOM_BASELINE),
            "ABOVE_BASELINE_MARGIN": float(ABOVE_BASELINE_MARGIN),
            "GRANULARITY": float(GRANULARITY),
            "RETRIEVAL_POOL_SIZE": int(RETRIEVAL_POOL_SIZE),
        },
        "granularity_guard_rule": GRANULARITY_GUARD_RULE,
        "extractive_not_paraphrase_note": EXTRACTIVE_NOT_PARAPHRASE_NOTE,
    }


# ---------------------------------------------------------------------------
# QA 侧数据组织取证：「同答案的多问题」显式成组（本轮只取证，不接损失）
# ---------------------------------------------------------------------------


def qa_answer_groups(
    pairs: Sequence[Dict[str, Any]],
    *,
    answer_key_field: str = "answer_key",
    question_id_field: str = "qid",
    limit: int = 0,
) -> Dict[str, Any]:
    """把「同答案的多问题」显式组织成组并产出取证。

    参数
    ----
    pairs : Sequence[Dict[str, Any]]
        QA 问答对记录（每项至少含答案键与问题 id）。
    answer_key_field : str
        答案键字段名（默认 answer_key）。
    question_id_field : str
        问题 id 字段名（默认 qid）。
    limit : int
        只取前 limit 条（0 = 全部）。

    返回
    ----
    Dict[str, Any]
        {n_pairs, n_groups, group_size_hist, group_size_quantiles, fingerprint}。
    """
    items = list(pairs[: int(limit)]) if int(limit) > 0 else list(pairs)
    groups: Dict[str, List[str]] = {}
    missing = 0
    for rec in items:
        key = rec.get(answer_key_field)
        qid = rec.get(question_id_field)
        if key is None or qid is None:
            missing += 1
            continue
        groups.setdefault(str(key), [])
        qid_s = str(qid)
        if qid_s not in groups[str(key)]:
            groups[str(key)].append(qid_s)
    ordered = {k: sorted(v) for k, v in sorted(groups.items(), key=lambda kv: kv[0])}
    sizes = [len(v) for v in ordered.values()]
    hist: Dict[str, int] = {}
    for s in sizes:
        hist[str(int(s))] = hist.get(str(int(s)), 0) + 1
    arr = np.asarray(sizes if sizes else [0], dtype=np.int64)
    return {
        "n_pairs": int(len(items)),
        "n_pairs_missing_key": int(missing),
        "n_groups": int(len(ordered)),
        "n_questions_total": int(sum(sizes)),
        "group_size_hist": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        "group_size_quantiles": {
            "min": int(np.min(arr)),
            "p25": float(np.percentile(arr, 25.0)),
            "median": float(np.percentile(arr, 50.0)),
            "p75": float(np.percentile(arr, 75.0)),
            "max": int(np.max(arr)),
        },
        "fingerprint": sha256_bytes(canonical_dumps(ordered)),
        "groups": ordered,
        "no_loss_note": QA_GROUPING_NO_LOSS_NOTE,
    }


def candidate_group_map(
    pairs: Sequence[Dict[str, Any]],
    *,
    answer_key_field: str = "answer_key",
    question_id_field: str = "qid",
    limit: int = 0,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """逐**候选答案**的组视图（供 QA 侧数据组织取证直接消费）。

    返回
    ----
    Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]
        `({answer_key: {n_questions, qids}}, evidence)`；`evidence` 即
        :func:qa_answer_groups 的返回值（去掉逐组明细以保持体积可控）。
    """
    ev = qa_answer_groups(
        pairs,
        answer_key_field=answer_key_field,
        question_id_field=question_id_field,
        limit=limit,
    )
    per_answer = {
        key: {"n_questions": int(len(qids)), "qids": list(qids)}
        for key, qids in ev["groups"].items()
    }
    summary = {k: v for k, v in ev.items() if k != "groups"}
    return per_answer, summary



# ---------------------------------------------------------------------------
# 数据侧取证（只读；用于报告，不进任何判定）
# ---------------------------------------------------------------------------


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    """文件的 SHA256（流式；不整文件入内存）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _int_quantiles(values: Sequence[int]) -> Dict[str, float]:
    """整数序列的 min / p25 / 中位 / p75 / max（空序列 -> 全 0，不产生 NaN）。"""
    arr = np.asarray([int(v) for v in values], dtype=np.int64)
    if arr.size == 0:
        return {"min": 0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0}
    return {
        "min": int(np.min(arr)),
        "p25": float(np.percentile(arr, 25.0)),
        "median": float(np.percentile(arr, 50.0)),
        "p75": float(np.percentile(arr, 75.0)),
        "max": int(np.max(arr)),
    }


def rows_evidence(rows: Sequence[TextRow]) -> Dict[str, Any]:
    """文本行侧取证：规模、按文件的分布、``char_len`` 分布、分档计数。

    ``char_len`` 口径 = ``len(row.text)``（与冻结产物字段同名同口径）。
    """
    items = list(rows)
    by_file: Dict[str, int] = {}
    for row in items:
        by_file[str(row.file)] = by_file.get(str(row.file), 0) + 1
    lens = [len(str(r.text)) for r in items]
    tiers: Dict[str, int] = {}
    for n in lens:
        tier = adaptive_tier_of(n)
        tiers[tier] = tiers.get(tier, 0) + 1
    return {
        "n_rows": int(len(items)),
        "n_files": int(len(by_file)),
        "rows_per_file": dict(sorted(by_file.items())),
        "char_len_distribution": _int_quantiles(lens),
        "adaptive_tier_counts": dict(sorted(tiers.items())),
        "n_zero_char_len": int(sum(1 for n in lens if n == 0)),
        "n_below_keyword_min_chars": int(sum(1 for n in lens if n < KEYWORD_MIN_CHARS)),
        "n_below_summary_min_chars": int(sum(1 for n in lens if n < SUMMARY_MIN_CHARS)),
    }


def split_evidence(
    rows: Sequence[TextRow],
    row_index: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """库/查询划分取证（只读 ``doclines_row_index.jsonl``）。

    口径（**现场核实**）：``rows_for_this_query = 2`` 的行走查询侧（各带 1 正例 = 自身
    + 1 负例），``rows_for_this_query = 0`` 的行走库侧。
    """
    items = list(rows)
    index_map: Dict[str, Dict[str, Any]] = {}
    for rec in row_index:
        index_map[str(rec.get("row_id", ""))] = dict(rec)
    query_ids: List[str] = []
    library_ids: List[str] = []
    missing = 0
    dist: Dict[str, int] = {}
    for row in items:
        rid = str(row.row_id)
        rec = index_map.get(rid)
        if rec is None:
            missing += 1
            continue
        value = int(rec.get("rows_for_this_query", 0))
        dist[str(value)] = dist.get(str(value), 0) + 1
        if value > 0:
            query_ids.append(rid)
        else:
            library_ids.append(rid)
    return {
        "n_rows": int(len(items)),
        "n_query_rows": int(len(query_ids)),
        "n_library_rows": int(len(library_ids)),
        "n_missing_in_row_index": int(missing),
        "rows_for_this_query_distribution": dict(sorted(dist.items(), key=lambda kv: int(kv[0]))),
        "query_row_ids": list(query_ids),
        "library_row_ids": list(library_ids),
        "rule": (
            "rows_for_this_query > 0 => 查询行（各带 1 正例 = 自身 + 1 负例）；"
            "== 0 => 库行"
        ),
    }


# ---------------------------------------------------------------------------
# 报告渲染（Markdown；与 JSON 同源）
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 6) -> str:
    """数值格式化（float 保 ``digits`` 位；其余用 ``str``）。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return ("%." + str(int(digits)) + "f") % value
    return str(value)


def render_markdown(report: Dict[str, Any]) -> str:
    """把报告 JSON 渲染成 Markdown（**不引入任何新数字**，只做格式化）。"""
    lines: List[str] = []
    add = lines.append
    add("# n3d_qa_learn 第三步 Phase 1：文本侧数据改造 + 零训练验证")
    add("")
    add("> 本轮**零训练**（不上任何模型、不训练）；单 seed、确定性；纯余弦检索。")
    add("> 所有数字来自同一份报告 JSON（本文件不引入任何新数字）。")
    add("")

    add("## 1. 口径")
    add("")
    add("- 变体支：" + " / ".join(str(x) for x in report.get("variant_order", [])))
    add("- 答案恒为**行原文**（与 1a / 步骤 2 的 `outputs = 行原文` 一致）")
    add("- 检索池规模：" + _fmt(report.get("retrieval_pool_size")))
    add("- 随机基线：" + _fmt(report.get("random_baseline"), 12))
    add("- 粒度（全语料常量）：" + _fmt(report.get("granularity"), 12))
    _np = int(report.get("retrieval_pool_size") or 0)
    if _np > 0:
        add("- 粒度（**本次运行实测** = 1/检索池规模）："
            + _fmt(1.0 / _np, 12) + "（检索池 = " + str(_np) + "）")
    add("- 判定规则：" + str(report.get("verdict_rule", "")))
    add("- 粒度守卫：" + str(report.get("granularity_guard_rule", "")))
    add("")

    add("## 2. 变体构造取证")
    add("")
    vp = report.get("variant_product", {})
    add("| 项 | 值 |")
    add("| --- | --- |")
    add("| 行数 | " + _fmt(vp.get("n_rows")) + " |")
    add("| `variants.jsonl` 字节 | " + _fmt(vp.get("blob_bytes")) + " |")
    add("| `variants.jsonl` SHA256 | `" + str(vp.get("blob_sha256", "")) + "` |")
    add("")
    add("### 2.1 逐支指纹与回退计数")
    add("")
    add("| 支 | SHA256 | 非空 | 总字数 | 回退为原文 | 与原文逐字不同 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for name in report.get("variant_order", []):
        ent = (vp.get("per_variant", {}) or {}).get(str(name), {})
        add(
            "| " + str(name) + " | `" + str(ent.get("sha256", "")) + "` | "
            + _fmt(ent.get("non_empty")) + " | " + _fmt(ent.get("n_chars_total")) + " | "
            + _fmt(ent.get("fallback_to_original")) + " | "
            + _fmt(ent.get("distinct_from_original")) + " |"
        )
    add("")

    add("## 3. 特征差异硬门禁")
    add("")
    fg = report.get("feature_gate", {})
    add("| 支 | 与原文支逐位相同行数 | 重合比例 | 硬门禁 | cos min | cos 中位 | cos max |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for name in report.get("non_anchor_variants", []):
        ent = (fg.get("per_variant", {}) or {}).get(str(name), {})
        cos = ent.get("cos_distribution", {}) or {}
        add(
            "| " + str(name) + " | " + _fmt(ent.get("n_identical")) + " | "
            + _fmt(ent.get("identity_ratio")) + " | "
            + ("硬门禁" if ent.get("hard_gate") else "允许重合") + " | "
            + _fmt(cos.get("min")) + " | " + _fmt(cos.get("median")) + " | "
            + _fmt(cos.get("max")) + " |"
        )
    add("")
    reg = fg.get("short_tier_identity_registration", {}) or {}
    add(
        "**短行重合如实登记**：短行档（`" + str(reg.get("tier", "")) + "`）"
        + _fmt(reg.get("n_rows_in_tier")) + " 行中，自适应支与原文支特征逐位相同 "
        + _fmt(reg.get("n_identical_in_tier")) + " 行（档内比例 "
        + _fmt(reg.get("identity_ratio_in_tier")) + "；全量比例 "
        + _fmt(reg.get("identity_ratio_overall")) + "）。"
    )
    add("")

    add("## 4. 难负例清单")
    add("")
    ng = report.get("negatives", {})
    add("| 类别 | 有负例的查询行 | 负例对总数 | 每查询最少 | 每查询最多 |")
    add("| --- | --- | --- | --- | --- |")
    for kind, ent in sorted((ng.get("counts", {}) or {}).items()):
        add(
            "| " + str(kind) + " | " + _fmt(ent.get("n_queries_with_negative")) + " | "
            + _fmt(ent.get("n_pairs")) + " | " + _fmt(ent.get("min_per_query")) + " | "
            + _fmt(ent.get("max_per_query")) + " |"
        )
    add("")

    add("## 5. 零训练检索评测（分档，不混平均）")
    add("")
    _setting_order = ("full_pool_all_rows", "small_pool_hard", "small_pool_random")
    for _si, setting_key in enumerate(_setting_order):
        block = (report.get("retrieval_eval", {}).get("settings", {}) or {}).get(setting_key, {})
        add("### 5." + str(_si + 1) + " " + str(setting_key))
        add("")
        add(str(block.get("description", "")))
        add("")
        add("| 支 | n | 池 min/中位/max | R@1 | R@5 | rank=1 | 未进 top-5 |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for name in report.get("variant_order", []):
            ent = (block.get("variants", {}) or {}).get(str(name), {})
            pool_txt = (
                _fmt(ent.get("n_pool")) if ent.get("n_pool") is not None
                else (_fmt(ent.get("pool_size_min")) + "/" + _fmt(ent.get("pool_size_median"))
                      + "/" + _fmt(ent.get("pool_size_max")))
            )
            add(
                "| " + str(name) + " | " + _fmt(ent.get("n")) + " | "
                + pool_txt + " | " + _fmt(ent.get("recall_at_1")) + " | "
                + _fmt(ent.get("recall_at_5")) + " | " + _fmt(ent.get("rank_1")) + " | "
                + _fmt(ent.get("rank_miss_topk")) + " |"
            )
        add("")
        add("- 池口径：" + str(block.get("pool_rule", "")))
        add("- 是否消费 negatives.json：**" + _fmt(bool(block.get("consumes_negatives"))) + "**")
        add("")
    # **fix 1（R68）**：`zero_training_eval` 在 common_subset 下产出的是**三个分池键**
    # （variants_full_pool / variants_small_pool_hard / variants_small_pool_random），
    # **没有** `variants` 键；此前按 `common.get("variants")` 取行使该表**整表渲染为 None**。
    # 现按三个池键分别输出（与 §5.1–5.3 的口径一致）。
    common = report.get("retrieval_eval", {}).get("common_subset", {}) or {}
    if common:
        add("### 5.4 公共子集（" + _fmt(common.get("n")) + " 行；**三种池设定分别评测**）")
        add("")
        add("- 查询行口径：" + str(common.get("query_ids_rule", "")))
        add("")
        for _ctitle, _ckey in (
            ("全量池 A", "variants_full_pool"),
            ("难负例小池 B", "variants_small_pool_hard"),
            ("随机负例小池 C", "variants_small_pool_random"),
        ):
            _cblk = common.get(_ckey, {}) or {}
            add("**" + _ctitle + "**（`common_subset." + _ckey + "`）")
            add("")
            if not _cblk:
                add("本池设定缺数据（该键不存在）——**不渲染空表**")
                add("")
                continue
            add("| 支 | R@1 | R@5 | 池 min/中位/max |")
            add("| --- | --- | --- | --- |")
            for name in report.get("variant_order", []):
                ent = _cblk.get(str(name), {}) or {}
                if not ent:
                    add("| " + str(name) + " | 缺该支数据 | 缺该支数据 | 缺该支数据 |")
                    continue
                pool_txt = (
                    _fmt(ent.get("n_pool")) if ent.get("n_pool") is not None
                    else (_fmt(ent.get("pool_size_min")) + "/" + _fmt(ent.get("pool_size_median"))
                          + "/" + _fmt(ent.get("pool_size_max")))
                )
                add(
                    "| " + str(name) + " | " + _fmt(ent.get("recall_at_1")) + " | "
                    + _fmt(ent.get("recall_at_5")) + " | " + pool_txt + " |"
                )
            add("")

    add("### 5.5 难负例 vs 随机负例：同支 R@1 差")
    add("")
    add("| 支 | R@1（随机小池 C） | R@1（难负例小池 B） | 差 B-C | 落差 | 判据粒度 | 粒度判定 | R@1（全量池 A，参考） |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in report.get("variant_order", []):
        ent = (report.get("retrieval_eval", {}).get("hard_vs_random_delta", {}) or {}).get(str(name), {})
        add(
            "| " + str(name) + " | " + _fmt(ent.get("recall_at_1_small_pool_random")) + " | "
            + _fmt(ent.get("recall_at_1_small_pool_hard")) + " | "
            + _fmt(ent.get("delta_hard_minus_random")) + " | "
            + _fmt(ent.get("gap_abs")) + " | " + _fmt(ent.get("granularity_used")) + " | "
            + ("<= 粒度（不作判据）" if ent.get("below_granularity") else "> 粒度") + " | "
            + _fmt(ent.get("recall_at_1_full_pool_reference")) + " |"
        )
    add("")
    _ms = report.get("retrieval_eval", {}).get("measured_scope", {}) or {}
    if _ms:
        add("**实测口径 vs 声明口径（分离登记）**")
        add("")
        add("| 项 | 值 |")
        add("| --- | --- |")
        add("| 本次评估行数 n_rows | " + _fmt(_ms.get("n_rows")) + " |")
        add("| 源总行数 source_total_rows | " + _fmt(_ms.get("source_total_rows")) + " |")
        add("| limit | " + _fmt(_ms.get("limit")) + " |")
        add("| 实测粒度 = 1/n_rows | " + _fmt(_ms.get("granularity_measured"), 12) + " |")
        add("| 实测随机基线（全量池 = 1/池大小） | "
            + _fmt(_ms.get("random_baseline_measured_full_pool"), 12) + " |")
        add("| 声明常量 RETRIEVAL_POOL_SIZE / RANDOM_BASELINE / GRANULARITY | "
            + _fmt((_ms.get("declared_constants") or {}).get("RETRIEVAL_POOL_SIZE")) + " / "
            + _fmt((_ms.get("declared_constants") or {}).get("RANDOM_BASELINE"), 12) + " / "
            + _fmt((_ms.get("declared_constants") or {}).get("GRANULARITY"), 12) + " |")
        add("")
        add(str((_ms.get("declared_constants") or {}).get("note", "")))
        add("")

    # **fix 8（R68）**：小池设定的前置条件与现场一致性登记
    # （数据全部取自 `retrieval_eval.json` 已有字段，**不新增任何产物字段**）。
    _pst = (_ms.get("per_setting", {}) or {}) if _ms else {}
    if _pst:
        _pb = _pst.get("small_pool_hard", {}) or {}
        _pc = _pst.get("small_pool_random", {}) or {}
        _trip_b = (_pb.get("pool_size_min"), _pb.get("pool_size_median"), _pb.get("pool_size_max"))
        _trip_c = (_pc.get("pool_size_min"), _pc.get("pool_size_median"), _pc.get("pool_size_max"))
        add("### 5.6 小池设定的**前置条件**（B-C 何时才只是「内容差」）")
        add("")
        add("- 代码内 `assert`（fail-closed）：" + str(SMALL_POOL_PRECONDITION_RULE))
        add("")
        add("| 设定 | n_queries | 池 min | 池中位 | 池 max |")
        add("| --- | --- | --- | --- | --- |")
        for _k, _t in (("small_pool_hard", "B 难负例小池"),
                       ("small_pool_random", "C 随机负例小池")):
            _e = _pst.get(_k, {}) or {}
            add("| " + _t + " | " + _fmt(_e.get("n_queries")) + " | " + _fmt(_e.get("pool_size_min"))
                + " | " + _fmt(_e.get("pool_size_median")) + " | " + _fmt(_e.get("pool_size_max")) + " |")
        add("")
        add("- 现场一致性：两侧池大小三元组相同 = **" + _fmt(_trip_b == _trip_c)
            + "**；查询行集行数相同 = **" + _fmt(_pb.get("n_queries") == _pc.get("n_queries")) + "**")
        add("")

    add("## 6. 判定结论（机器可读、单点来源）")
    add("")
    vd = report.get("verdict", {})
    add("- `verdict_key` = **`" + str(vd.get("verdict_key", "")) + "`**")
    add("- 锚点支 R@1 = " + _fmt((vd.get("anchor_detail", {}) or {}).get("recall_at_1"))
        + "（期望 1.0，容差 " + _fmt((vd.get("anchor_detail", {}) or {}).get("tolerance"), 12)
        + "，实测差 " + _fmt((vd.get("anchor_detail", {}) or {}).get("abs_diff"), 12) + "）")
    add("")
    add("| 支 | R@1 | 1-R@1 | 显著<1.0 | 高于随机基线 | 结论键 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for name, ent in sorted((vd.get("per_variant", {}) or {}).items()):
        add(
            "| " + str(name) + " | " + _fmt(ent.get("recall_at_1")) + " | "
            + _fmt(ent.get("one_minus_recall_at_1")) + " | "
            + _fmt(ent.get("significant_gap")) + " | "
            + _fmt(ent.get("above_baseline")) + " | `" + str(ent.get("verdict_key", "")) + "` |"
        )
    add("")
    add("逐支结论键（有序）：`" + str(vd.get("verdict_keys", [])) + "`")
    add("")

    add("## 7. QA 侧数据组织取证（同答案多问题成组；**本轮不接损失**）")
    add("")
    qg = (report.get("qa_grouping", {}) or {}).get("summary", {})
    add("| 项 | 值 |")
    add("| --- | --- |")
    add("| 问答对 | " + _fmt(qg.get("n_pairs")) + " |")
    add("| 答案组数 | " + _fmt(qg.get("n_groups")) + " |")
    add("| 组内问题数中位 | " + _fmt((qg.get("group_size_quantiles", {}) or {}).get("median")) + " |")
    add("| 组内问题数最大 | " + _fmt((qg.get("group_size_quantiles", {}) or {}).get("max")) + " |")
    add("| 组指纹 SHA256 | `" + str(qg.get("fingerprint", "")) + "` |")
    add("")
    add(str(qg.get("no_loss_note", "")))
    add("")

    add("## 8. 如实登记的局限")
    add("")
    for note in report.get("honest_notes", []) or []:
        add("- " + str(note))
    add("")
    det = report.get("determinism", {}) or {}
    if det:
        add("## 9. 双跑逐字节一致（确定性证据）")
        add("")
        add("| 文件 | SHA256 | 字节 | 逐字节一致 |")
        add("| --- | --- | --- | --- |")
        for name, ent in sorted((det.get("files", {}) or {}).items()):
            add(
                "| `" + str(name) + "` | `" + str(ent.get("a_sha256", "")) + "` | "
                + _fmt(ent.get("a_bytes")) + " | "
                + _fmt(ent.get("bitwise_equal")) + " |"
            )
        add("")
        add("- 全部逐字节一致 = **" + _fmt(det.get("all_bitwise_equal")) + "**")
        add("- 对照目录 = `" + str(det.get("repro_dir", "")) + "`")
        add("")
    return "\n".join(lines) + "\n"
# ---------------------------------------------------------------------------
# 产物目录
# ---------------------------------------------------------------------------

#: 本轮**唯一**产物目录（验证类运行目录的子目录）。
TEXT_VARIANTS_DIR: str = os.path.join(
    "checkpoints", "qa_learn", "_verify", "text_variants"
)

#: 变体产物文件名（**冻结清单**）。
TEXT_VARIANTS_FILES: Tuple[str, ...] = (
    "variants.jsonl",
    "variants_meta.json",
    "negatives.json",
    "feature_gate.json",
    "retrieval_eval.json",
    "qa_grouping.json",
    "text_variants_report.json",
    "text_variants_report.md",
)

#: 双跑一致性证据文件名（**单独一族**：由对照运行写入，不参与主产物指纹）。
DETERMINISM_EVIDENCE_FILE: str = "determinism_check.json"


__all__ = [
    "ABOVE_BASELINE_MARGIN",
    "ADAPTIVE_ALLOW_IDENTITY_TIER",
    "ADAPTIVE_RULE",
    "ADAPTIVE_TIERS",
    "ANCHOR_EXPECTED_RECALL_AT_1",
    "ANCHOR_TOL",
    "ANCHOR_TOLERANCE",
    "ASCII_SAFE_LOG_NOTE",
    "BASE_TOL",
    "BASELINE_TOL",
    "BUCKETS_PER_ORDER",
    "CLOSE_TO_ONE_TOLERANCE",
    "CLOSE_TO_ZERO_TOLERANCE",
    "DETERMINISM_EVIDENCE_FILE",
    "ENCODER_NEIGHBOR_ENCODER",
    "ENCODER_NEIGHBOR_ROLE",
    "ENCODER_NEIGHBOR_SOURCE",
    "ENCODER_NEIGHBOR_TOPK",
    "EXTRACTIVE_NOT_PARAPHRASE_NOTE",
    "GAP_TOL",
    "GRANULARITY",
    "GRANULARITY_GUARD_RULE",
    "HARD_NEGATIVE_CAP",
    "IS_IDENTITY_TOL",
    "IDENTITY_GATED_VARIANTS",
    "IDENTITY_REGISTERED_VARIANTS",
    "HARD_GATE_RULE",
    "IRREDUCIBLE_RULE",
    "IRREDUCIBLE_RULE",
    "IS_ZERO_TOL",
    "KEYBERT_BACKEND_REPO_ENCODER",
    "KEYBERT_BACKEND_SENTENCE_TRANSFORMERS",
    "KEYWORD_DOC_EMBEDDINGS",
    "KEYWORD_MAX_TOP_K",
    "KEYWORD_MIN_CANDIDATES",
    "keyword_vocabulary_path",
    "build_keyword_vocabulary",
    "KEYWORD_VOCAB_FILE",
    "KEYWORD_MIN_CHARS",
    "KEYWORD_MIN_DOC_FREQ",
    "KEYWORD_MIN_TOP_K",
    "KEYWORD_BATCH_SIZE",
    "KEYWORD_MAX_CANDIDATES",
    "KEYWORD_NGRAM_RANGE",
    "KEYWORD_POS_KEEP",
    "KEYWORD_WORD_BATCH_SIZE",
    "KEYWORD_ENCODER_MAX_LENGTH",
    "KEYWORD_TOP_K",
    "KEYWORD_USE_MMR",
    "NEGATIVE_KINDS",
    "NEGATIVE_SETTINGS",
    "NON_ANCHOR_VARIANTS",
    "N_GRAM_ORDERS",
    "QA_GROUPING_NO_LOSS_NOTE",
    "RANDOM_BASELINE",
    "RANDOM_NEGATIVE_COUNT",
    "RETRIEVAL_POOL_SIZE",
    "RepoEncoderEmbedder",
    "SIGNIFICANT_GAP",
    "SUMMARY_MAX_CHARS",
    "SUMMARY_MAX_SENTENCES",
    "SUMMARY_MIN_CHARS",
    "SUMMARY_SENTENCE_DELIMITERS",
    "SUMMARY_STRIP_TRAILING_WHITESPACE",
    "TEXT_VARIANTS_DIR",
    "TEXT_VARIANTS_FILES",
    "TEXT_VARIANTS_SCHEMA_VERSION",
    "TEXT_VARIANTS_SEED",
    "TEXT_VARIANT_DEPENDENCY_NOTE",
    "SentenceTransformerEmbedder",
    "TextVariantDependencyError",
    "VARIANT_ADAPTIVE",
    "VARIANT_ANCHOR",
    "VARIANT_KEYWORD",
    "VARIANT_ORDER",
    "VARIANT_SUMMARY",
    "VERDICTS",
    "VERDICT_RULE",
    "VariantDef",
    "VariantGenerator",
    "adaptive_tier_of",
    "build_keyword_backend",
    "build_negative_manifest",
    "build_variant_records",
    "candidate_group_map",
    "encoder_neighbor_sets",
    "evaluate_retrieval",
    "extractive_summary",
    "file_sha256",
    "feature_gate",
    "gold_id_sets",
    "duplicate_original_texts",
    "GOLD_SET_RULE",
    "feature_matrix_for_texts",
    "hard_negative_sets",
    "jieba_pos_tokenizer",
    "jieba_tokenizer",
    "load_row_index",
    "load_text_rows",
    "SMALL_POOL_PRECONDITION_RULE",
    "SMALL_POOL_RULE",
    "FULL_POOL_RULE",
    "NEGATIVE_POOL_SIZE",
    "evaluate_retrieval_pools",
    "negative_setting_payload",
    "qa_answer_groups",
    "random_negative_sets",
    "render_markdown",
    "render_verdict",
    "resolve_product_dir",
    "row_cosine",
    "row_is_irreducible",
    "rows_evidence",
    "serialize_variant_records",
    "split_evidence",
    "split_sentences",
    "variant_definitions",
    "variant_fingerprint",
    "variant_product_files",
    "variant_text_hash",
    "zero_training_eval",
]
