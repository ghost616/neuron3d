"""轮 1：QA 侧**数据规模扩容**（C=10 -> C=298）+ **按比例留出** + **解题类任务拆分**。

本模块**只改数据口径与任务分层**，不改模型结构、不改打分链路、不做任何训练。
产物一律写 ``checkpoints/qa_learn/_verify/qa_scale/``（**不落盘裸特征矩阵、不产 zip、
无挂钟字段**）。

数据来源（**只读**，且必须是「现场枚举」而非「凭记忆写」）
----------------------------------------------------------
* ``checkpoints/qa_learn/dataset/answer_table.jsonl``（298 行；字段现场枚举 =
  ``['count', 'display', 'index', 'key', 'raw_variants', 'sources']``）
  —— **统一答案表**：``index`` 即类别下标，``key`` 是归一化答案键，``display`` 是展示文本，
  ``count`` 是该类在全部任务上的**正样本条数**。
* ``checkpoints/qa_learn/dataset/n3d_qa_all_pairs.jsonl``（50865 行；字段现场枚举 =
  ``['answer_table_size', 'answer_text', 'candidate_answer_index', 'candidate_answer_key',
  'label', 'pair_id', 'question_id', 'question_text', 'row_id', 'task']``）
  —— **问答对**：``label == 1`` 表示该候选类是该问题的正确答案之一。
* ``checkpoints/qa_learn/dataset/n3d_qa_all.npz``（``X`` ``[50865, 306]`` float32 +
  ``y`` 二值 + ``meta.task_of_row``）—— 与 pairs 文件**逐行同序**（现场核对：
  npz 行数 == pairs 行数 == 50865，``task_of_row`` 计数 == pairs 的 ``task`` 计数）。

**为什么以「冻结产物」而不是「QA JSON 缓存」为本轮口径**：本轮要在同一口径下比较
C=10 与 C=298，而 manifest 的 ``answer_table.classes = 298`` 只对**冻结产物**成立；
若改用缓存（``checkpoints/triviaqa/_cache``）会得到**另一套**答案空间（基数与类集合都
不同）⇒ 与 manifest 声明不可对齐。故本轮以冻结产物为准，并把缓存口径的既有基线
（C=10 / 149 训练 / 20 测试）**作为另一总体**单独列出对照，不混为一谈。

按比例留出的边界归属（**显式声明 + 现场用例**）
----------------------------------------------
留出 = **哈希比例抽稀**，单点公式（唯一实现 :func:`question_side`）::

    h = int(sha256("qgroup\\0{seed}\\0{task}\\0{question_id}"), 8B big-endian) % 100
    side = "test" if h < holdout_pct else "train"      # 左闭右开：h ∈ [0, holdout_pct)
    holdout_pct == 0   -> 恒为 train（空测试集）
    holdout_pct == 100 -> 恒为 test（空训练集）

**边界归属 = 「小于」（``<``），不是「小于等于」（``<=``）** —— 轮盘取
``[0, holdout_pct)`` 这 ``holdout_pct`` 个值 ⇒ 命中概率恰为 ``holdout_pct/100``。
现场用例（:func:`proportional_holdout_evidence` 的 ``boundary_probe``）：对 ``h == 20``
（``holdout_pct == 20``）的题目，本实现给出 ``'train'``；若把左闭右开改成左闭右闭
（``<=``），同一题会变成 ``'test'`` —— 两者**可区分**，故边界归属是**被现场证据钉住的**，
不是口头声明。

拆出解题类任务的依据（**分组=manifest 的 task 字段**）
------------------------------------------------------
``judge`` / ``choice`` 的答案类只有 **2 / 4** 类（现场：judge ``kept_keys = 2``、
choice ``kept_keys = 4``），且其答案**不由问题相似性决定**（「3 是奇数吗」与「3 是偶数吗」
问法极相似而答案相反）⇒ 实质是**解题**而非匹配。故按 manifest 的 ``task`` 字段分层：

* **匹配任务集**（``MATCH_TASKS``）= ``blank`` / ``solve`` / ``triviaqa``；
* **解题任务集**（``SOLVE_TASKS``）= ``judge`` / ``choice``。

该拆分只影响**报告分侧**，不影响同一个切分的定义（先切分、再按任务分组统计）。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

#: 模块版本（结构变化必须递增；进产物）。
QA_SCALE_VERSION: str = "n3dqa-qa-scale-v1"

#: ``n3d_qa`` 冻结产物目录（**只读**）。
DEFAULT_PRODUCT_DIR: str = os.path.join("checkpoints", "qa_learn", "dataset")

#: 统一答案表文件名（**只读**）。
ANSWER_TABLE_FILE: str = "answer_table.jsonl"

#: 全任务问答对文件名（**只读**；与 ``n3d_qa_all.npz`` 逐行同序）。
ALL_PAIRS_FILE: str = "n3d_qa_all_pairs.jsonl"

#: 全任务 npz 文件名（**只读**；本模块只读其 ``meta`` 用于交叉核对，**不落盘特征矩阵**）。
ALL_NPZ_FILE: str = "n3d_qa_all.npz"

#: manifest 文件名（**只读**；任务清单与答案表声明的来源）。
MANIFEST_FILE: str = "manifest.json"

#: 全量口径的类别数（= manifest ``answer_table.classes``；**现场核对后写入，不凭记忆**）。
FULL_CLASSES: int = 298

#: 基线（现状）口径：``max_classes=10`` / ``min_questions=8`` / ``test_every=3`` /
#: ``test_per_class=2``（与 ``entry_table.QA_*`` 及 ``exp_repr.BASE_*`` 逐项一致）。
BASELINE_MAX_CLASSES: int = 10
BASELINE_MIN_QUESTIONS: int = 8
BASELINE_TEST_EVERY: int = 3
BASELINE_TEST_PER_CLASS: int = 2

#: 默认留出比例（百分比；**整数**，避免浮点边界歧义）。
DEFAULT_HOLDOUT_PCT: int = 20

#: 留出比例的允许区间（``0`` = 全部训练、``100`` = 全部测试；两端都合法且必须可复算）。
HOLDOUT_PCT_RANGE: Tuple[int, int] = (0, 100)

#: 任务清单（**现场来自 manifest ``tasks``**；此处作为冻结常量登记以便离线复算）。
ALL_TASKS: Tuple[str, ...] = ("judge", "choice", "blank", "solve", "triviaqa")

#: **解题任务集**（答案类少且答案不由问题相似性决定）。
SOLVE_TASKS: Tuple[str, ...] = ("judge", "choice")

#: **匹配任务集**（其余任务）。
MATCH_TASKS: Tuple[str, ...] = ("blank", "solve", "triviaqa")

#: 任务分层规则文本（进产物）。
TASK_STRATIFICATION_RULE: str = (
    "按 manifest 的 task 字段分层：解题任务集 = (judge, choice) —— 现场 kept_keys 分别只有 "
    "2 / 4 类，答案由计算/选项判定而非问题相似性决定（如「3 是奇数吗」与「3 是偶数吗」"
    "问法极相似而答案相反）；匹配任务集 = (blank, solve, triviaqa)。该分层只影响报告分侧，"
    "不改变切分本身（先按 question_id 整组切分，再按任务分组统计）。"
)

#: 按比例留出口径文本。
HOLDOUT_RULE: str = (
    "留出 = 哈希比例抽稀，作用单位是**问题组**（同一 question_id 的全部行同侧）："
    "h = int(sha256('qgroup\\0{seed}\\0{task}\\0{question_id}'), 8B) % 100；"
    "h < holdout_pct 进测试侧，否则进训练侧。边界归属 = **小于**（左闭右开 "
    "[0, holdout_pct)），故命中概率恰为 holdout_pct/100；holdout_pct=0 时测试集为空、"
    "=100 时训练集为空（两端都必须可复算）。**不**使用 test_per_class 的逐类封顶语义，"
    "也**不**强制每类至少 1 条测试样本（后者会破坏比例性）；每类实际样本数在产物中"
    "逐类登记，偏差如实报告。"
)

#: 判据口径文本。
CRITERION_RULE: str = (
    "① 「macro >= 10/C（随机基线 x10）」：阈值 = 10/C，**仅当 10/C < 1.0 时适用**；"
    "C <= 10 时阈值 >= 1.0，而 macro <= 1.0 ⇒ 该项**恒不可满足**，一律标 "
    "applicable=False 并如实报出，**不得**当成通过项。②「macro >= 多数类基线 + 10pp」与 "
    "③「macro >= 多数类基线 + 5pp」是真正生效的两条。全部阈值为**闭区间下界**（判据用 "
    "`>=`），并在产物中给出阈值与基线的可行域边界核查（基线是否落在 0 / 1 饱和点）。"
)

#: 产物目录与守卫根。
VERIFY_ROOT: str = os.path.join("checkpoints", "qa_learn", "_verify")
DEFAULT_OUT_DIR: str = os.path.join(VERIFY_ROOT, "qa_scale")

#: 类别样本数分档（**顺序敏感**：第一档在最前；`hi=None` 表示无上界）。
CLASS_SIZE_BINS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10-49", 10, 49),
    ("50+", 50, None),
)


# ---------------------------------------------------------------------------
# 现场枚举工具
# ---------------------------------------------------------------------------


def _sha256_bytes(blob: bytes) -> str:
    """字节 SHA256（十六进制小写）。"""
    return hashlib.sha256(blob).hexdigest()


def _canonical_json(obj: Any) -> str:
    """规范化 JSON（``sort_keys`` + 紧凑分隔符；跨运行一致）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    """UTF-8 裸字节 SHA256。"""
    return _sha256_bytes(str(text).encode("utf-8"))


def sha256_file(path: str) -> str:
    """文件 SHA256（分块读，避免大文件一次性入内存）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_manifest(product_dir: str = DEFAULT_PRODUCT_DIR) -> Dict[str, Any]:
    """读取 ``n3d_qa`` 冻结产物的 ``manifest.json``（**只读**）。"""
    path = os.path.join(str(product_dir), MANIFEST_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少 manifest：{path!r}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_answer_table(product_dir: str = DEFAULT_PRODUCT_DIR) -> List[Dict[str, Any]]:
    """读取统一答案表（``answer_table.jsonl``，**只读**）。

    返回
    ----
    List[Dict[str, Any]]
        按 ``index`` 升序的类别条目；每条的字段现场枚举见 :func:`answer_table_fields`。
    """
    path = os.path.join(str(product_dir), ANSWER_TABLE_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少统一答案表：{path!r}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"统一答案表为空：{path!r}")
    rows.sort(key=lambda r: int(r["index"]))
    idx = [int(r["index"]) for r in rows]
    if idx != list(range(len(rows))):
        raise ValueError(
            f"统一答案表的 index 不是 0..{len(rows) - 1} 的连续序列（现场前 8 个 = {idx[:8]}）；"
            "拒绝在不可靠的类别下标上建答案空间"
        )
    return rows


def answer_table_fields(product_dir: str = DEFAULT_PRODUCT_DIR) -> Dict[str, Any]:
    """现场枚举统一答案表的**字段名**与条目数（禁止凭记忆手写）。"""
    rows = load_answer_table(product_dir)
    return {
        "file": ANSWER_TABLE_FILE,
        "n_rows": int(len(rows)),
        "fields": sorted(str(k) for k in rows[0].keys()),
        "first_row": {str(k): rows[0][k] for k in sorted(rows[0].keys())},
    }


@dataclass(frozen=True)
class QAPair:
    """冻结产物里的一条问答对（**只带本轮所需字段**）。

    参数
    ----
    row_index : int
        在 ``n3d_qa_all_pairs.jsonl`` 中的 0-based 行号（下单序，确定性）。
    row_id / pair_id / question_id : str
        行 id / 对 id / 问题 id（切分单位是 ``question_id``）。
    task : str
        任务名（manifest ``tasks`` 之一）。
    candidate_index : int
        候选答案在统一答案表中的下标。
    label : int
        ``1`` = 该候选是本题正确答案之一；``0`` = 错误候选。
    """

    row_index: int
    row_id: str
    pair_id: str
    question_id: str
    task: str
    candidate_index: int
    label: int


def pair_record_fields(product_dir: str = DEFAULT_PRODUCT_DIR) -> Dict[str, Any]:
    """现场枚举问答对记录的**字段名**（禁止凭记忆手写）。"""
    path = os.path.join(str(product_dir), ALL_PAIRS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少问答对文件：{path!r}")
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.readline().strip()
    if not first:
        raise RuntimeError(f"问答对文件为空：{path!r}")
    obj = json.loads(first)
    return {
        "file": ALL_PAIRS_FILE,
        "fields": sorted(str(k) for k in obj.keys()),
        "first_row_fields_only": {
            k: ("<str>" if isinstance(obj[k], str) else obj[k])
            for k in sorted(obj.keys())
        },
    }


def load_all_pairs(product_dir: str = DEFAULT_PRODUCT_DIR) -> List[QAPair]:
    """读取全任务问答对（**只读**；行号即下标的确定性顺序）。

    口径
    ----
    * 只保留可解析的行用于计数（正负样本都要，因为「行数」是规模指标）；
      建立每个问题的**正确类集合**时只取 ``label == 1`` 的行；
    * **``task`` 字段的取值（现场实测，必须显式登记）**：``n3d_qa_all_pairs.jsonl``
      的 ``task`` 列**恒为 ``'all'``**（它是「全任务合并产品」的标记，不是逐行任务）。
      **逐行真实任务**在 ``n3d_qa_all.npz`` 的 ``meta.task_of_row``（与本文件**逐行同序**）
      ⇒ 由 :func:`load_row_tasks` 提供，并在 :func:`build_scale_corpus` 里按**行下标**回填。
      这是「字段名必须现场枚举」这条纪律的直接后果：字段**存在**不等于它**承载**我们要的语义。
    """
    path = os.path.join(str(product_dir), ALL_PAIRS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少问答对文件：{path!r}")
    out: List[QAPair] = []
    with open(path, "r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                QAPair(
                    row_index=int(i),
                    row_id=str(obj["row_id"]),
                    pair_id=str(obj["pair_id"]),
                    question_id=str(obj["question_id"]),
                    task=str(obj["task"]),
                    candidate_index=int(obj["candidate_answer_index"]),
                    label=int(obj["label"]),
                )
            )
    if not out:
        raise RuntimeError(f"问答对文件为空：{path!r}")
    return out


def load_row_tasks(product_dir: str = DEFAULT_PRODUCT_DIR) -> Dict[str, Any]:
    """从 ``n3d_qa_all.npz`` 的 ``meta`` 读**逐行任务**（``task_of_row``）。

    只读 ``meta``（**不**把 ``X`` 特征矩阵读进产物）：本函数只取 ``meta`` 字段，
    返回的任务列表用于按行下标回填 ``QAPair.task``。

    返回
    ----
    Dict[str, Any]
        ``tasks``（去重后的任务清单，保序）、``task_of_row``（长度 = 行数）、
        ``n_rows``、``per_task_rows``、``meta_keys``、``source``。

    异常
    ------
    FileNotFoundError
        npz 缺失。
    ValueError
        ``task_of_row`` 长度与 pairs 行数不一致（拒绝在错位的数据上分层）。
    """
    import numpy as np

    path = os.path.join(str(product_dir), ALL_NPZ_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少全任务 npz：{path!r}")
    with np.load(path, allow_pickle=True) as data:
        if "meta" not in data:
            raise ValueError(f"npz {path!r} 缺少 meta 字段；实际字段 = {sorted(data.keys())}")
        meta = json.loads(str(data["meta"]))
        n_rows = int(data["y"].shape[0]) if "y" in data else 0
    if "task_of_row" not in meta:
        raise ValueError(
            f"npz meta 缺少 task_of_row；现场 meta 键 = {sorted(meta.keys())}；"
            "逐行任务无法确定，拒绝继续"
        )
    row_tasks = [str(x) for x in meta["task_of_row"]]
    if n_rows and len(row_tasks) != n_rows:
        raise ValueError(
            f"task_of_row 长度 {len(row_tasks)} 与 npz 行数 {n_rows} 不一致"
        )
    order: List[str] = []
    per_task: Dict[str, int] = {}
    for t in row_tasks:
        if t not in per_task:
            order.append(str(t))
        per_task[str(t)] = per_task.get(str(t), 0) + 1
    return {
        "source": ALL_NPZ_FILE + "::meta.task_of_row",
        "tasks": [str(t) for t in order],
        "task_of_row": row_tasks,
        "n_rows": int(len(row_tasks)),
        "per_task_rows": {str(k): int(v) for k, v in sorted(per_task.items())},
        "meta_keys": sorted(str(k) for k in meta.keys()),
        "npz_sha256": sha256_file(path),
        "feature_matrix_note": (
            "本函数**只读 npz 的 meta**；X（特征矩阵）不进入任何产物"
            "（验收要求：不落盘裸特征矩阵）"
        ),
    }


# ---------------------------------------------------------------------------
# 语料与答案空间（C=298 全量 / C=10 重建）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleCorpus:
    """扩容口径下的 QA 语料（答案空间 + 问题级正确类集合 + 逐行下标）。

    参数
    ----
    answer_keys : List[str]
        答案表类别键（**下标即本轮口径下的类别下标**；在 C=10 重建里只含被保留的 10 类）。
    answer_display : List[str]
        类别展示文本（与 ``answer_keys`` 同序）。
    class_row_counts : List[int]
        本轮口径下每类的**正样本条数**（取自冻结表 ``count`` 字段；用于确定性排序与基线）。
    class_global_index : List[int]
        本地类别下标 -> **冻结表里的全局类别下标**（截断后二者不同，必须显式可查）。
        **轮 8 口径**：``merge_map`` 开启时该项是 **canonical 下标**（见
        ``source["merge"]`` 与 ``source["index_space"]``）—— 归并后「本地 -> 全局」是
        1:多关系，完整簇成员只能从 ``merge_map.jsonl`` / ``MergePlan`` 查，不得从本字段反推。
    class_pos : Dict[int, int]
        全局类别下标 -> 本地下标（**截断后被排除的类不在其中**；查不到即该行不在本轮范围内）。
    pairs : List[QAPair]
        全部问答对（按文件行序；``task`` 已由 npz ``meta.task_of_row`` 回填）。
    question_gold : Dict[str, List[int]]
        问题 id -> 该问题的**正确类下标集合**（**全局下标**；升序去重）。
    question_task : Dict[str, str]
        问题 id -> 任务名。
    question_rows : Dict[str, List[int]]
        问题 id -> 该问题的行下标（升序）。
    class_question_counts : List[int]
        每类的**问题数**（按 ``question_gold`` 现场统计；与 ``class_row_counts`` 口径不同，
        后者是**正样本条数** —— 二者必须分别登记）。
    source : Dict[str, Any]
        来源取证（产物路径 + 文件 SHA256 + 现场枚举字段）。
    """

    answer_keys: List[str]
    answer_display: List[str]
    class_row_counts: List[int]
    class_global_index: List[int]
    class_pos: Dict[int, int]
    pairs: List[QAPair]
    question_gold: Dict[str, List[int]]
    question_task: Dict[str, str]
    question_rows: Dict[str, List[int]]
    class_question_counts: List[int]
    source: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        """答案类别数 ``C``。"""
        return int(len(self.answer_keys))

    @property
    def n_pairs(self) -> int:
        """问答对行数。"""
        return int(len(self.pairs))

    @property
    def n_questions(self) -> int:
        """问题数（切分单位）。"""
        return int(len(self.question_gold))

    def key_to_index(self) -> Dict[str, int]:
        """类别键 -> 类别下标。"""
        return {str(k): i for i, k in enumerate(self.answer_keys)}

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化摘要（**不落盘特征矩阵**）。"""
        return {
            "version": QA_SCALE_VERSION,
            "n_classes": int(self.n_classes),
            "n_pairs": int(self.n_pairs),
            "n_questions": int(self.n_questions),
            "answer_keys_sha256": sha256_text(_canonical_json(self.answer_keys)),
            "source": dict(self.source),
        }


def _median_int(values: Sequence[int]) -> float:
    """整数序列的中位数（偶数长度取中间两值均值；零新依赖）。"""
    arr = sorted(int(x) for x in values)
    size = int(len(arr))
    if size == 0:
        return 0.0
    if size % 2 == 1:
        return float(arr[size // 2])
    return (float(arr[size // 2 - 1]) + float(arr[size // 2])) / 2.0


def _canonicalize_answer_table(
    table: Sequence[Mapping[str, Any]],
    merge_map: Mapping[int, int],
    merge_labels: Optional[Mapping[int, Sequence[str]]],
) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[str, Any]]:
    """按**归并映射**把冻结表重建为 canonical 表（轮 8；默认路径**不**经过本函数）。

    参数
    ----
    table : Sequence[Mapping[str, Any]]
        冻结表条目（现场字段 = ``['count', 'display', 'index', 'key',
        'raw_variants', 'sources']``）。
    merge_map : Mapping[int, int]
        **已校验**的映射（原全局下标 -> canonical 下标，见
        :func:`n3d_qa_learn.answer_merge.validate_merge_map`）。
    merge_labels : Optional[Mapping[int, Sequence[str]]]
        canonical 下标 -> ``(canonical_key, canonical_display)``；``None`` 时按「成员内
        ``(-count, key)`` 最小者」取代表（确定性兜底，仍可复算）。

    返回
    ----
    Tuple[List[Dict[str, Any]], Dict[int, int], Dict[str, Any]]
        ``(canonical_table, candidate_map, merge_source)``。canonical 表与冻结表**同形**
        （``index`` / ``key`` / ``display`` / ``count``）⇒ 下游代码零改动；``candidate_map``
        用于改写 ``pairs[].candidate_index``；``merge_source`` 进 ``source["merge"]``。

    关键不变量（违反即抛）
    ---------------------
    1. 映射取值的集合必须稠密（由 ``validate_merge_map`` 保证）；
    2. 每个 canonical 必须至少有一个成员（空簇即抛）；
    3. canonical 的 ``count`` = 成员 ``count`` 之和（**归并不得增删样本**）。
    """
    n_canonical = int(len(set(int(v) for v in merge_map.values())))
    members: List[List[int]] = [[] for _ in range(n_canonical)]
    for original in sorted(int(k) for k in merge_map.keys()):
        members[int(merge_map[int(original)])].append(int(original))
    rows: List[Dict[str, Any]] = []
    for canonical in range(n_canonical):
        group = members[int(canonical)]
        if not group:
            raise ValueError(
                "归并映射产生的 canonical " + str(canonical) + " 没有任何成员；拒绝继续"
            )
        if merge_labels is not None:
            label = tuple(merge_labels[int(canonical)])
            key = str(label[0])
            display = str(label[1])
        else:
            representative = sorted(
                group,
                key=lambda g: (-int(table[int(g)]["count"]), str(table[int(g)]["key"])),
            )[0]
            key = str(table[int(representative)]["key"])
            display = str(table[int(representative)]["display"])
        rows.append(
            {
                "index": int(canonical),
                "key": key,
                "display": display,
                "count": int(sum(int(table[int(g)]["count"]) for g in group)),
                "merged_member_indices": [int(g) for g in group],
            }
        )
    merge_source: Dict[str, Any] = {
        "enabled": True,
        "n_original_classes": int(len(table)),
        "n_canonical_classes": int(n_canonical),
        "n_reductions": int(len(table) - n_canonical),
        "merge_map_sha256": sha256_text(
            _canonical_json({str(k): int(v) for k, v in sorted(merge_map.items())})
        ),
        "index_space": "canonical",
        "labels_rule": (
            "canonical_key / canonical_display 由 merge_labels 显式给出（来自 "
            "answer_merge 的归一化结果）；未给出时按「簇内 (-count, key) 最小者」取代表"
        ),
        "rule": (
            "归并**只消标签噪声、不增加样本总量**：canonical 的 count = 成员 count 之和，"
            "问题数与行数一个不增；pairs[].candidate_index 与 question_gold 一律改写为 "
            "canonical 下标（与 class_pos 的键空间一致）。上游 n3d_qa 产物**只读**。"
        ),
    }
    return rows, {int(k): int(v) for k, v in merge_map.items()}, merge_source


def build_scale_corpus(
    product_dir: str = DEFAULT_PRODUCT_DIR,
    *,
    max_classes: int = 0,
    min_class_questions: int = 0,
    merge_map: Optional[Mapping[int, int]] = None,
    merge_labels: Optional[Mapping[int, Sequence[str]]] = None,
) -> ScaleCorpus:
    """由冻结产物构建扩容语料。

    参数
    ----
    product_dir : str
        冻结产物目录（只读）。
    max_classes : int
        类别数上限（``0`` = 全量 298）。截断规则 = **按 ``(-count, key)`` 降序取前 N**
        （与 ``data.build_answer_space`` 的确定性档位同一排序键），目的是让
        「C=10 重建」与「C=298 全量」成为**同一总体上的同一规则**，唯一变量是类别数上限。
    min_class_questions : int
        **轮 9 · A 口径的类集门槛**（``0`` = 不过滤，**与轮 1 逐位不变**）。只保留
        「**问题数** >= 该门槛」的类别；口径是**每类的问题数**（该类的不同
        ``question_id`` 个数，同一问题只计一次），**不是**正样本行数、也不是冻结表
        ``count``；门槛判定用**筛选前**（全量类集、全部问题）的计数。与 ``max_classes``
        同时给定时顺序固定为「**先按门槛剔除，再按 ``(-count, key)`` 截断**」，并写入
        ``source["class_scope"]``。
    merge_map : Optional[Mapping[int, int]]
        **轮 8 · 答案表同义归并映射**（原全局下标 -> canonical 下标）。**默认 ``None``
        = 现状口径**（与轮 1 逐位不变）。给定时：按 canonical 聚合 ``count``、重建
        ``answer_keys`` / ``answer_display`` / ``class_pos``，并把 ``pairs[].candidate_index``
        与 ``question_gold`` 改写成 canonical 下标（见 :func:`_canonicalize_answer_table`）。
        映射由 :func:`n3d_qa_learn.answer_merge.build_merge_plan` 生成、经
        :func:`n3d_qa_learn.answer_merge.validate_merge_map` 校验（fail-closed）。
        **轮 9 口径声明**：A 口径**不使用**归并（high-risk 两簇语义未解决），两者不要一起开。
        **轮 10 · I4 互斥守卫**：``merge_map`` 非 ``None`` 且 ``min_class_questions > 0``
        时**直接抛 ``ValueError``**（不再只是「别一起开」的口头约定）。
    merge_labels : Optional[Mapping[int, Sequence[str]]]
        canonical 下标 -> ``(canonical_key, canonical_display)``；``merge_map`` 为
        ``None`` 时**必须**也为 ``None``（否则抛 ``ValueError``）。

    返回
    ----
    ScaleCorpus
        语料（答案空间、问题级正确类集合、逐行下标）。

    关键不变量（构造期，违反即抛）
    -----------------------------
    1. 冻结表 ``index`` 必须恰为 ``0..C-1``（由 :func:`load_answer_table` 保证）；
    2. 每一行的 ``candidate_answer_index`` 必须落在 ``[0, C_full)``（越界即抛）；
    3. 每个问题的正确类集合**非空**（否则该问题在所有任务上都没有正确答案）；
    4. 每个问题只属于**一个**任务（同 ``question_id`` 出现两个 ``task`` 即抛）；
    5. （轮 8）归并映射的定义域/值域与 ``merge_labels`` 必须彼此一致；归并不变量
       「canonical count 之和 == 原表 count 之和」由 :func:`_canonicalize_answer_table`
       与调用方共同保证；
    6. （轮 9）类集门槛按**问题数**口径判定；筛选后保留类集非空，且**金标全部落在保留类内**
       的问题才进语料（含越界金标的问题**整体**排除，不偷偷把「答案不在表内」变成「在表内」）。
    """
    # ---- 轮 10 · I4：**互斥守卫** —— 「既归并又按门槛筛选」的口径误用必须当场拒绝 ----
    # 理由：归并把类改写到 canonical 下标空间，而 min_class_questions 的门槛计数是
    # **筛选前**的全量类集口径；两者同时启用时，既无法判断门槛落在原表还是 canonical 表上，
    # 也会把轮 8（归并）与轮 9（A 口径，明确「归并关闭」）的结论混成一套数。
    if merge_map is not None and int(min_class_questions) > 0:
        raise ValueError(
            "merge_map（轮 8 答案表归并）与 min_class_questions（轮 9 · A 口径类集门槛）"
            "**不允许同时启用**：归并后的 canonical 下标空间与「筛选前全量类集」的门槛计数"
            "分属两套口径（A 口径明确「归并关闭」），同时启用会让两层结论互相污染。"
            "请分两轮分别跑：先归并（merge_map）或先按门槛筛选（min_class_questions），"
            "当前 merge_map 非 None 且 min_class_questions = "
            + str(int(min_class_questions))
            + "。"
        )
    table = load_answer_table(str(product_dir))
    full_c = int(len(table))
    # ---- 轮 8：可选归并（**默认关闭**；关闭时下面的代码路径与轮 1 逐位相同） ----------
    scope_table: Sequence[Mapping[str, Any]] = table
    scope_c = int(full_c)
    candidate_map: Optional[Dict[int, int]] = None
    merge_source: Dict[str, Any] = {}
    if merge_map is None and merge_labels is not None:
        raise ValueError(
            "merge_labels 只在 merge_map 给定时才允许（否则它没有任何作用对象）；"
            "拒绝静默忽略"
        )
    if merge_map is not None:
        import n3d_qa_learn.answer_merge as AM

        normalized_map = AM.validate_merge_map(
            merge_map, n_original=int(full_c), merge_labels=merge_labels
        )
        scope_table, candidate_map, merge_source = _canonicalize_answer_table(
            table, normalized_map, merge_labels
        )
        scope_c = int(len(scope_table))
    pairs = load_all_pairs(str(product_dir))
    # **逐行任务回填**（见 load_all_pairs 的口径说明）：pairs 文件的 task 列恒为 'all'，
    # 真实任务在 npz 的 meta.task_of_row，二者逐行同序。
    row_tasks = load_row_tasks(str(product_dir))
    if int(row_tasks["n_rows"]) != int(len(pairs)):
        raise ValueError(
            f"npz 行数 {row_tasks['n_rows']} 与 pairs 行数 {len(pairs)} 不一致；"
            "拒绝在没有逐行任务对齐的数据上做任务分层"
        )
    pairs = [
        QAPair(
            row_index=int(p.row_index),
            row_id=str(p.row_id),
            pair_id=str(p.pair_id),
            question_id=str(p.question_id),
            task=str(row_tasks["task_of_row"][int(p.row_index)]),
            candidate_index=int(p.candidate_index),
            label=int(p.label),
        )
        for p in pairs
    ]
    for pair in pairs:
        if not (0 <= int(pair.candidate_index) < full_c):
            raise ValueError(
                f"行 {pair.row_index} 的 candidate_answer_index = {pair.candidate_index} "
                f"越出统一答案表范围 [0, {full_c})；拒绝在越界下标上建答案空间"
            )
    # ---- 轮 8：把行与金标一律改写到 canonical 下标（**默认路径 candidate_map is None**）----
    if candidate_map is not None:
        # ① 折叠前的口径核查：原表内 (question_id, candidate) 是否唯一 ——
        #    只有唯一，后面的「金标对折叠数」才能**只**归因于归并。
        seen_pairs: Dict[Tuple[str, int], int] = {}
        n_duplicate_rows_before = 0
        dup_before_by_label: Dict[int, int] = {0: 0, 1: 0}
        for pair in pairs:
            dup_key = (str(pair.question_id), int(pair.candidate_index))
            if dup_key in seen_pairs:
                n_duplicate_rows_before += 1
                label_key = int(pair.label)
                dup_before_by_label[label_key] = dup_before_by_label.get(label_key, 0) + 1
            else:
                seen_pairs[dup_key] = int(pair.row_index)
        gold_raw: set = set()
        gold_merged: set = set()
        seen_after: Dict[Tuple[str, int], int] = {}
        n_duplicate_rows_after = 0
        n_positive_rows = 0
        for pair in pairs:
            canonical = int(candidate_map[int(pair.candidate_index)])
            after_key = (str(pair.question_id), int(canonical))
            if after_key in seen_after:
                n_duplicate_rows_after += 1
            else:
                seen_after[after_key] = int(pair.row_index)
            if int(pair.label) == 1:
                n_positive_rows += 1
                gold_raw.add((str(pair.question_id), int(pair.candidate_index)))
                gold_merged.add((str(pair.question_id), int(canonical)))
        merge_source["n_duplicate_rows_before_merge"] = int(n_duplicate_rows_before)
        merge_source["n_duplicate_rows_before_merge_by_label"] = {
            str(k): int(v) for k, v in sorted(dup_before_by_label.items())
        }
        merge_source["n_duplicate_rows_after_merge"] = int(n_duplicate_rows_after)
        merge_source["n_positive_rows"] = int(n_positive_rows)
        merge_source["n_gold_pairs_before_merge"] = int(len(gold_raw))
        merge_source["n_gold_pairs_after_merge"] = int(len(gold_merged))
        merge_source["n_gold_pairs_collapsed"] = int(len(gold_raw) - len(gold_merged))
        merge_source["gold_rows_unique_premise"] = bool(
            int(n_positive_rows) == int(len(gold_raw))
        )
        merge_source["duplicate_row_note"] = (
            "归并会把「同一问题的两个变体类金标」变成两行指向同一 canonical 的等价行；"
            "本轮**不去重**（故行数口径与归并前一致，仍为 50865 行），只逐项计数。"
            "重复行与原行同真值、同类，micro / macro **不受影响**（分子分母同倍变化）；"
            "但 unit='row' 的逐类计数会含重复，比较时须以 unit='question' 为主。"
            "**现场发现（如实登记）**：冻结的问答对文件里 (question_id, candidate) 本来"
            "就**不唯一**（n_duplicate_rows_before_merge 远大于 0，按 label 的分解见 "
            "n_duplicate_rows_before_merge_by_label）—— 该现象**不是**归并造成的；"
            "金标侧（label=1）是否唯一由 gold_rows_unique_premise 现场判定，"
            "只有它为 True 时 n_gold_pairs_collapsed 才能**只**归因于归并。"
        )
        # ② 改写为 canonical 下标（此后 pairs / question_gold / class_pos 同一空间）
        pairs = [
            QAPair(
                row_index=int(p.row_index),
                row_id=str(p.row_id),
                pair_id=str(p.pair_id),
                question_id=str(p.question_id),
                task=str(p.task),
                candidate_index=int(candidate_map[int(p.candidate_index)]),
                label=int(p.label),
            )
            for p in pairs
        ]
    question_gold: Dict[str, List[int]] = {}
    question_task: Dict[str, str] = {}
    question_rows: Dict[str, List[int]] = {}
    for pair in pairs:
        qid = str(pair.question_id)
        question_rows.setdefault(qid, []).append(int(pair.row_index))
        prev_task = question_task.get(qid)
        if prev_task is None:
            question_task[qid] = str(pair.task)
        elif str(prev_task) != str(pair.task):
            raise ValueError(
                f"问题 {qid!r} 同时出现在任务 {prev_task!r} 与 {pair.task!r} 下；"
                "切分与分侧统计都要求「一个问题只属于一个任务」，拒绝继续"
            )
        if int(pair.label) == 1:
            gold = question_gold.setdefault(qid, [])
            if int(pair.candidate_index) not in gold:
                gold.append(int(pair.candidate_index))
    empty_gold = sorted(q for q in question_task if not question_gold.get(q))
    if empty_gold:
        raise ValueError(
            f"有 {len(empty_gold)} 个问题没有任何正确答案（前几个 = {empty_gold[:5]}）；"
            "拒绝把无金标的问题放进评测"
        )
    for qid, gold in question_gold.items():
        gold.sort()
    # ---- 轮 9：类集筛选（按最小「问题数」门槛 + 可选截断；两者都关 ⇒ 全量，逐位同轮 1） ----
    # 口径（**必须显式**）：每类的「问题数」= 该类的**问题**个数（同一问题只计一次，
    # 用 (question_id, class) 去重），**不是**正样本行数。二者在场下恰好相等
    # （冻结对的 label=1 侧 (question_id, candidate) 唯一），但口径按「问题数」钉死。
    class_question_sets: List[set] = [set() for _ in range(int(scope_c))]
    for pair in pairs:
        if int(pair.label) == 1:
            class_question_sets[int(pair.candidate_index)].add(str(pair.question_id))
    class_question_counts_all = [int(len(s)) for s in class_question_sets]
    keep = list(range(int(scope_c)))
    n_classes_before_scope = int(len(keep))
    if int(min_class_questions) > 0:
        keep = [
            int(i)
            for i in keep
            if int(class_question_counts_all[int(i)]) >= int(min_class_questions)
        ]
    if int(max_classes) > 0:
        order = sorted(
            keep,
            key=lambda i: (-int(scope_table[i]["count"]), str(scope_table[i]["key"])),
        )
        keep = sorted(int(i) for i in order[: int(max_classes)])
    keep = sorted(int(i) for i in keep)
    if not keep:
        raise ValueError(
            "类集筛选后**没有任何类**被保留（min_class_questions="
            + str(int(min_class_questions))
            + " / max_classes="
            + str(int(max_classes))
            + "）；该口径不可评测，拒绝继续"
        )
    answer_keys = [str(scope_table[i]["key"]) for i in keep]
    answer_display = [str(scope_table[i]["display"]) for i in keep]
    class_row_counts = [int(scope_table[i]["count"]) for i in keep]
    class_pos = {int(g): pos for pos, g in enumerate(keep)}
    class_scope_source: Dict[str, Any] = {}
    if int(min_class_questions) > 0:
        kept_set = set(int(i) for i in keep)
        class_scope_source = {
            "enabled": True,
            "min_class_questions": int(min_class_questions),
            "count_unit": "questions",
            "n_classes_before": int(n_classes_before_scope),
            "n_classes_after": int(len(keep)),
            "kept_class_question_counts": {
                "min": int(min(class_question_counts_all[i] for i in keep)),
                "median": float(_median_int([class_question_counts_all[i] for i in keep])),
                "max": int(max(class_question_counts_all[i] for i in keep)),
                "total": int(sum(class_question_counts_all[i] for i in keep)),
            },
            "count_rule": (
                "门槛口径 = **每类的「问题数」**：该类的不同 question_id 个数（同一问题只计一次），"
                "**不是**正样本行数、也不是冻结表 count；门槛判定用**筛选前**（全量类集、"
                "全部问题）的计数，故 `kept_class_question_counts` 是 pre-scope 值；"
                "语料内实际每类问题数见 class_question_counts（post-scope，因含越界金标的问题"
                "被整体排除而略小）。"
            ),
            "n_question_class_incidences_kept": int(
                sum(class_question_counts_all[i] for i in keep)
            ),
            "n_questions_covered_any_kept": int(
                sum(
                    1
                    for gold in question_gold.values()
                    if any(int(g) in kept_set for g in gold)
                )
            ),
        }
    # ---- 类别截断时的**问题级作用域过滤**（C=10 重建的核心语义） --------------------
    # 截断到 C 类后，金标**全部落在保留类之内**的问题才属于本轮口径；金标含被截断类的
    # 问题必须**整体**排除（不能只删掉越界的那部分金标 —— 那会把「答案不在表内」偷偷
    # 变成「答案在表内」，是口径污染）。全量口径（max_classes=0）下本步不做任何过滤。
    n_questions_before_scope_filter = int(len(question_task))
    n_questions_out_of_scope = 0
    if int(max_classes) > 0 or int(min_class_questions) > 0:
        keep_qids: List[str] = []
        for qid in sorted(question_task.keys()):
            gold = question_gold.get(qid, [])
            if all(int(g) in class_pos for g in gold):
                keep_qids.append(str(qid))
        keep_set = set(keep_qids)
        n_questions_out_of_scope = int(n_questions_before_scope_filter - len(keep_qids))
        if not keep_qids:
            raise ValueError(
                "类集筛选（max_classes="
                + str(int(max_classes))
                + " / min_class_questions="
                + str(int(min_class_questions))
                + "）后**没有任何问题**的金标落在保留类内；该口径不可评测，拒绝继续"
            )
        old_rows: set = set()
        for qid in keep_qids:
            old_rows.update(int(x) for x in question_rows[str(qid)])
        kept_old = sorted(old_rows)
        remap = {int(o): i for i, o in enumerate(kept_old)}
        pairs = [
            QAPair(
                row_index=int(i),
                row_id=str(pairs[int(o)].row_id),
                pair_id=str(pairs[int(o)].pair_id),
                question_id=str(pairs[int(o)].question_id),
                task=str(pairs[int(o)].task),
                candidate_index=int(pairs[int(o)].candidate_index),
                label=int(pairs[int(o)].label),
            )
            for i, o in enumerate(kept_old)
        ]
        question_gold = {q: question_gold[q] for q in keep_qids}
        question_task = {q: question_task[q] for q in keep_qids}
        question_rows = {
            q: sorted(int(remap[int(x)]) for x in question_rows[str(q)])
            for q in keep_qids
        }
    class_question_counts = [0] * int(len(answer_keys))
    for qid, gold in question_gold.items():
        for cls in gold:
            pos = class_pos.get(int(cls))
            if pos is not None:
                class_question_counts[int(pos)] += 1
    source = {
        "product_dir": str(product_dir),
        "answer_table_file": ANSWER_TABLE_FILE,
        "answer_table_sha256": sha256_file(
            os.path.join(str(product_dir), ANSWER_TABLE_FILE)
        ),
        "all_pairs_file": ALL_PAIRS_FILE,
        "all_pairs_sha256": sha256_file(os.path.join(str(product_dir), ALL_PAIRS_FILE)),
        "manifest_sha256": sha256_file(os.path.join(str(product_dir), MANIFEST_FILE)),
        "row_tasks_source": str(row_tasks["source"]),
        "row_tasks_npz_sha256": str(row_tasks["npz_sha256"]),
        "row_tasks_per_task_rows": dict(row_tasks["per_task_rows"]),
        "row_tasks_field_note": (
            "n3d_qa_all_pairs.jsonl 的 task 列恒为 'all'（现场实测）；逐行真实任务取自 "
            "n3d_qa_all.npz 的 meta.task_of_row（与本文件逐行同序）"
        ),
        "full_classes_in_table": int(full_c),
        "max_classes": int(max_classes),
        "n_questions_before_scope_filter": int(n_questions_before_scope_filter),
        "n_questions_out_of_scope": int(n_questions_out_of_scope),
        "n_questions_after_scope_filter": int(len(question_gold)),
        "scope_filter_rule": (
            "**max_classes > 0 或 min_class_questions > 0 时都做问题级作用域过滤**：只有"
            "「金标全部落在保留类之内」的问题才属于本轮口径；金标含被筛选掉类的问题"
            "**整体**排除（不能只删越界的那部分金标 —— 那会把「答案不在表内」偷偷变成"
            "「答案在表内」）。保留类由 min_class_questions（每类问题数门槛，先按门槛剔除）"
            "和/或 max_classes（按 (-count, key) 截断）共同决定，因此**门槛筛选同样可能**"
            "让问题越界：n_questions_out_of_scope 每次运行现场统计，随门槛/截断参数变化"
            "（A 口径下非 0）；**只有两者都不启用**（默认档）才不进入本步过滤，"
            "n_questions_out_of_scope = 0。行下标在过滤后**重新编号**为 0..M-1 连续，"
            "保证「语料行号」只有一套。"
        ),
        "selection_rule": (
            "保留类 = 先按 min_class_questions 剔除「每类问题数 < 门槛」的类（口径 = 该类"
            "的不同 question_id 个数，一问只计一次），再按 (-count, key) 降序取前 "
            "max_classes（max_classes=0 表示不截断）；与 data.build_answer_space 的确定性"
            "排序键一致。两者同时给定时顺序固定为「先门槛、后截断」，组合语义见 "
            "scope_filter_rule。"
        ),
    }
    # **轮 8**：只在归并开启时追加 ``merge`` 键 —— 默认档的 ``source`` 逐字节不变。
    # **轮 9**：只在类集筛选开启时追加 ``class_scope`` 键（同理，默认档不加任何键）。
    if class_scope_source:
        class_scope_source["n_questions_excluded_by_scope_filter"] = int(
            n_questions_out_of_scope
        )
        class_scope_source["n_questions_after_scope_filter"] = int(len(question_gold))
        source["class_scope"] = dict(class_scope_source)
    if candidate_map is not None:
        source["merge"] = dict(merge_source)
        source["index_space"] = "canonical"
        source["index_space_rule"] = (
            "归并开启后，``pairs[].candidate_index`` / ``question_gold`` / ``class_pos`` 的键"
            "**全部**是 canonical 下标（0..C'-1）；``class_global_index`` 因此是 canonical "
            "下标而非冻结表全局下标，簇成员只能从 merge_map.jsonl / MergePlan 查。"
        )
    return ScaleCorpus(
        answer_keys=answer_keys,
        answer_display=answer_display,
        class_row_counts=class_row_counts,
        class_global_index=[int(g) for g in keep],
        class_pos=class_pos,
        pairs=pairs,
        question_gold=question_gold,
        question_task=question_task,
        question_rows=question_rows,
        class_question_counts=class_question_counts,
        source=source,
    )


# ---------------------------------------------------------------------------
# 按比例留出（哈希比例抽稀；边界归属 = 小于）
# ---------------------------------------------------------------------------


def question_hash(question_id: str, task: str, seed: int) -> int:
    """问题的留出轮盘值 ``h ∈ [0, 100)``（**唯一实现**；跨运行确定性）。

    公式（与 :data:`HOLDOUT_RULE` 逐字一致）::

        h = int(sha256("qgroup\\0{seed}\\0{task}\\0{question_id}"), 8B big-endian) % 100

    用 ``hashlib.sha256`` 而不是内置 ``hash()``：后者对 str 加了每进程随机盐，
    跨进程不稳定，会让「同参数双跑逐字节一致」不可能成立。
    """
    blob = f"qgroup\x00{int(seed)}\x00{str(task)}\x00{str(question_id)}".encode("utf-8")
    digest = hashlib.sha256(blob).digest()
    return int.from_bytes(digest[:8], "big") % 100


def question_side(question_id: str, task: str, seed: int, holdout_pct: int) -> str:
    """按比例留出判定**单点**：返回 ``'train'`` 或 ``'test'``。

    **边界归属（显式声明）**：``h < holdout_pct`` -> ``'test'``。
    即轮盘命中集合是**左闭右开**区间 ``[0, holdout_pct)``，命中概率恰为
    ``holdout_pct / 100``；``h == holdout_pct`` 归 **train**（不是 test）。
    ``holdout_pct`` 必须落在 :data:`HOLDOUT_PCT_RANGE` 内（``0`` 与 ``100`` 都合法）。
    """
    pct = int(holdout_pct)
    lo, hi = HOLDOUT_PCT_RANGE
    if not (int(lo) <= pct <= int(hi)):
        raise ValueError(
            f"holdout_pct 必须落在 [{lo}, {hi}]，当前 {pct}"
            "（0 = 全部训练、100 = 全部测试；两端都必须可复算）"
        )
    h = question_hash(str(question_id), str(task), int(seed))
    return "test" if h < pct else "train"


def question_side_with_hash(
    question_id: str, task: str, seed: int, holdout_pct: int
) -> Tuple[str, int]:
    """同 :func:`question_side`，但**一并返回轮盘值**（供边界取证）。"""
    h = question_hash(str(question_id), str(task), int(seed))
    pct = int(holdout_pct)
    lo, hi = HOLDOUT_PCT_RANGE
    if not (int(lo) <= pct <= int(hi)):
        raise ValueError(f"holdout_pct 必须落在 [{lo}, {hi}]，当前 {pct}")
    return ("test" if h < pct else "train"), int(h)


def proportional_holdout_evidence(seed: int, holdout_pct: int) -> Dict[str, Any]:
    """**边界归属的现场用例证据**：证明边界用的是 ``<`` 而不是 ``<=``。

    做法（不依赖任何真实数据）：枚举 ``h`` 的**边界邻域**（``holdout_pct-1``、
    ``holdout_pct``、``holdout_pct+1``）并给出本实现的判定，同时给出「若改为 ``<=``」
    的判定，从而把两者**可区分**这一事实钉成产物字段。此外给出 ``pct ∈ {0, 100}``
    的退化行为（空测试集 / 空训练集）。

    返回
    ----
    Dict[str, Any]
        ``boundary_probe``（三个邻域点的双侧判定）、``degenerate_cases``、
        ``implied_test_probability``（``holdout_pct/100`` 与区间表述）。
    """
    pct = int(holdout_pct)
    lo, hi = HOLDOUT_PCT_RANGE
    if not (int(lo) <= pct <= int(hi)):
        raise ValueError(f"holdout_pct 必须落在 [{lo}, {hi}]，当前 {pct}")
    probe: List[Dict[str, Any]] = []
    for h in (pct - 1, pct, pct + 1):
        if h < 0 or h > 99:
            continue
        # 本实现的判定（唯一实现 question_side 的左闭右开语义）
        ours = "test" if h < pct else "train"
        # 对照：若把边界改成左闭右闭（<=）
        alt = "test" if h <= pct else "train"
        probe.append(
            {
                "h": int(h),
                "implemented_rule_lt": str(ours),
                "alternative_rule_le": str(alt),
                "rules_distinguishable_at_this_point": bool(ours != alt),
            }
        )
    degenerate = {
        "pct_0": {
            "h_0": "train" if 0 < 0 else "train",
            "h_99": "test" if 99 < 0 else "train",
            "expected": "全部训练（测试集为空）",
        },
        "pct_100": {
            "h_0": "test" if 0 < 100 else "train",
            "h_99": "test" if 99 < 100 else "train",
            "expected": "全部测试（训练集为空）",
        },
    }
    return {
        "seed": int(seed),
        "holdout_pct": int(pct),
        "rule": "side = 'test' if h < holdout_pct else 'train'（左闭右开 [0, holdout_pct)）",
        "boundary_probe": probe,
        "degenerate_cases": degenerate,
        "implied_test_probability": {
            "as_fraction": f"{pct}/100",
            "as_float": float(pct) / 100.0,
            "interval": "[0, " + str(pct) + ")",
        },
        "why_not_le": (
            "若改成 h <= holdout_pct，命中集合变为 [0, holdout_pct] 共 holdout_pct+1 个值，"
            "命中概率变成 (holdout_pct+1)/100 —— 比例不再等于声明值。"
            "boundary_probe 给出了两者在 h == holdout_pct 处判定不同的现场证据。"
        ),
    }


# ---------------------------------------------------------------------------
# 切分与规模统计
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleSplit:
    """扩容口径下的训练 / 测试切分（**按 question_id 整组**）。

    参数
    ----
    train_questions : List[str]
        训练侧问题 id（升序）。
    test_questions : List[str]
        测试侧问题 id（升序）。
    train_rows : List[int]
        训练侧行下标（升序）。
    test_rows : List[int]
        测试侧行下标（升序）。
    seed / holdout_pct : int
        切分参数。
    """

    train_questions: List[str]
    test_questions: List[str]
    train_rows: List[int]
    test_rows: List[int]
    seed: int
    holdout_pct: int

    def summary(self) -> Dict[str, int]:
        """规模摘要。"""
        return {
            "n_train_questions": int(len(self.train_questions)),
            "n_test_questions": int(len(self.test_questions)),
            "n_train_rows": int(len(self.train_rows)),
            "n_test_rows": int(len(self.test_rows)),
        }


def make_scale_split(
    corpus: ScaleCorpus,
    *,
    holdout_pct: int = DEFAULT_HOLDOUT_PCT,
    seed: int = 42,
) -> ScaleSplit:
    """按比例留出做**问题组级**切分（同一 ``question_id`` 的全部行同侧）。

    关键不变量（构造期与返回前校验，违反即抛）
    ------------------------------------------
    1. ``train_rows`` 与 ``test_rows`` **不相交**、并集 = 全部行下标；
    2. 每个问题的**全部行**落在同一侧（按问题组切分的直接后果）；
    3. 两侧问题集不相交、并集 = 全部问题；
    4. 切分**完全确定**：只用 ``(seed, task, question_id)`` 的哈希，无全局 RNG。
    """
    pct = int(holdout_pct)
    lo, hi = HOLDOUT_PCT_RANGE
    if not (int(lo) <= pct <= int(hi)):
        raise ValueError(f"holdout_pct 必须落在 [{lo}, {hi}]，当前 {pct}")
    train_q: List[str] = []
    test_q: List[str] = []
    train_rows: List[int] = []
    test_rows: List[int] = []
    for qid in sorted(corpus.question_gold.keys()):
        side = question_side(str(qid), str(corpus.question_task[str(qid)]), int(seed), pct)
        rows = [int(x) for x in corpus.question_rows[str(qid)]]
        if side == "test":
            test_q.append(str(qid))
            test_rows.extend(rows)
        else:
            train_q.append(str(qid))
            train_rows.extend(rows)
    train_rows.sort()
    test_rows.sort()
    if set(train_rows) & set(test_rows):
        raise ValueError("切分不变量被破坏：训练行与测试行有交集")
    if sorted(train_rows + test_rows) != list(range(int(corpus.n_pairs))):
        raise ValueError(
            "切分不变量被破坏：训练行 ∪ 测试行 != 全部行下标；拒绝在残缺切分上出数"
        )
    if set(train_q) & set(test_q):
        raise ValueError("切分不变量被破坏：两侧问题集有交集")
    if sorted(train_q + test_q) != sorted(corpus.question_gold.keys()):
        raise ValueError("切分不变量被破坏：问题并集不等于全部问题")
    return ScaleSplit(
        train_questions=train_q,
        test_questions=test_q,
        train_rows=train_rows,
        test_rows=test_rows,
        seed=int(seed),
        holdout_pct=int(pct),
    )


def class_size_distribution(counts: Sequence[int]) -> Dict[str, Any]:
    """样本数分布的现场统计（min / median / max / 分位 / 分档直方图）。

    参数
    ----
    counts : Sequence[int]
        每类样本数（**顺序不参与**统计，只用于直方图分档）。

    返回
    ----
    Dict[str, Any]
        ``n`` / ``min`` / ``p25`` / ``median`` / ``p75`` / ``max`` / ``mean`` /
        ``zero_count`` / ``histogram``（分档见 :data:`CLASS_SIZE_BINS`）/ ``total``。
    """
    import numpy as np

    arr = np.asarray([int(x) for x in counts], dtype=np.int64)
    if arr.size == 0:
        return {
            "n": 0,
            "min": 0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "max": 0,
            "mean": 0.0,
            "zero_count": 0,
            "histogram": {str(name): 0 for name, _lo, _hi in CLASS_SIZE_BINS},
            "total": 0,
        }
    hist = {str(name): 0 for name, _lo, _hi in CLASS_SIZE_BINS}
    for value in arr.tolist():
        v = int(value)
        for name, lo, hi in CLASS_SIZE_BINS:
            if v >= int(lo) and (hi is None or v <= int(hi)):
                hist[str(name)] += 1
                break
        else:  # pragma: no cover - 分档表覆盖全部非负整数
            raise ValueError(f"无法分档的取值 {v}")
    return {
        "n": int(arr.size),
        "min": int(arr.min()),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "zero_count": int((arr == 0).sum()),
        "histogram": hist,
        "total": int(arr.sum()),
    }


def side_class_counts(
    corpus: ScaleCorpus,
    split: ScaleSplit,
    *,
    side: str,
    unit: str = "question",
) -> List[int]:
    """某一侧的**逐类样本数**（``unit='question'`` 按问题数、``'row'`` 按行数）。

    用「按类计数」而不是「按类平均」：类别覆盖与小样本类的真实可用量必须**逐类**可见。
    """
    key = str(side)
    if key not in ("train", "test"):
        raise ValueError(f"side 只能是 'train' / 'test'，当前 {side!r}")
    counts = [0] * int(corpus.n_classes)
    questions = split.train_questions if key == "train" else split.test_questions
    if str(unit) == "question":
        for qid in questions:
            for cls in corpus.question_gold[str(qid)]:
                pos = corpus.class_pos.get(int(cls))
                if pos is not None:
                    counts[int(pos)] += 1
        return counts
    rows = split.train_rows if key == "train" else split.test_rows
    for r in rows:
        pair = corpus.pairs[int(r)]
        if int(pair.label) == 1:
            pos = corpus.class_pos.get(int(pair.candidate_index))
            if pos is not None:
                counts[int(pos)] += 1
    return counts


def class_coverage(counts: Sequence[int]) -> Dict[str, Any]:
    """类别覆盖：有样本的类数与**显式列出**的零样本类（不得只写「恰好等于」）。"""
    zero = [int(i) for i, c in enumerate(counts) if int(c) == 0]
    return {
        "n_classes": int(len(counts)),
        "n_classes_with_samples": int(len(counts) - len(zero)),
        "n_classes_without_samples": int(len(zero)),
        "classes_without_samples_head": zero[:20],
        "rule": (
            "覆盖 = 该类在本侧有 >= 1 个样本（按声明单位计）。"
            "零样本类必须**逐个列出**（本节 classes_without_samples_head，最多 20 个）"
            "——不得用「恰好等于」一语带过；允许的额外项 = 列表长度超过 20 时截断。"
        ),
    }


def task_side_stats(
    corpus: ScaleCorpus,
    split: ScaleSplit,
    *,
    side: str,
) -> Dict[str, Dict[str, Any]]:
    """按**任务**分侧的规模统计（行数 / 问题数 / 类别覆盖 / 正样本条数）。

    返回
    ----
    Dict[str, Dict[str, Any]]
        键 = 任务名（:data:`ALL_TASKS` 的子集，按 manifest 现场清单），值 = 该任务在该侧
        的统计。
    """
    key = str(side)
    if key not in ("train", "test"):
        raise ValueError(f"side 只能是 'train' / 'test'，当前 {side!r}")
    rows = split.train_rows if key == "train" else split.test_rows
    questions = split.train_questions if key == "train" else split.test_questions
    out: Dict[str, Dict[str, Any]] = {}
    for task in ALL_TASKS:
        task_rows = [r for r in rows if str(corpus.pairs[int(r)].task) == str(task)]
        task_qs = [q for q in questions if str(corpus.question_task[str(q)]) == str(task)]
        classes: set = set()
        for qid in task_qs:
            for cls in corpus.question_gold[str(qid)]:
                classes.add(int(cls))
        out[str(task)] = {
            "n_rows": int(len(task_rows)),
            "n_questions": int(len(task_qs)),
            "n_answer_classes": int(len(classes)),
            "n_positive_rows": int(
                sum(1 for r in task_rows if int(corpus.pairs[int(r)].label) == 1)
            ),
            "answer_classes_head": sorted(classes)[:10],
        }
    return out


def grouped_stats(
    corpus: ScaleCorpus,
    split: ScaleSplit,
    *,
    side: str,
) -> Dict[str, Any]:
    """按**任务分组**（匹配 / 解题）汇总某一侧的统计。"""
    per_task = task_side_stats(corpus, split, side=str(side))
    out: Dict[str, Any] = {}
    for name, tasks in (("match", MATCH_TASKS), ("solve", SOLVE_TASKS)):
        rows = int(sum(int(per_task[str(t)]["n_rows"]) for t in tasks))
        qs = int(sum(int(per_task[str(t)]["n_questions"]) for t in tasks))
        pos = int(sum(int(per_task[str(t)]["n_positive_rows"]) for t in tasks))
        classes: set = set()
        for t in tasks:
            classes.update(per_task[str(t)]["answer_classes_head"])
        out[str(name)] = {
            "tasks": [str(t) for t in tasks],
            "n_rows": int(rows),
            "n_questions": int(qs),
            "n_positive_rows": int(pos),
            "n_answer_classes_lower_bound": int(len(classes)),
            "per_task": {str(t): dict(per_task[str(t)]) for t in tasks},
            "rule": TASK_STRATIFICATION_RULE if name == "solve" else "其余任务即为匹配任务集",
        }
    return out


# ---------------------------------------------------------------------------
# 判据退化修正（10/C 的适用性）+ 可行域边界核查
# ---------------------------------------------------------------------------


def macro_criterion_applicability(n_classes: int) -> Dict[str, Any]:
    """判据「macro >= 10/C」的**适用性**判定（阈值本身与适用性都必须现场算出）。

    口径
    ----
    * 阈值 ``threshold = 10 / C``；
    * ``applicable = threshold < 1.0``（**严格小于**）；而 ``macro <= 1.0``，
      故 ``threshold >= 1.0`` 时该项**恒不可满足**（不是"很难"，是"不可能"）；
    * ``threshold == 1.0`` 恰好落在边界上 —— 此时仍**不可满足**（macro 最多 1.0，
      要满足需要 ``macro >= 1.0`` 即全对；实现上该条件"可满足但要求满分"，
      按要求一律标 ``applicable=False`` 并**如实报出**，不得当成通过项）。
      为把这条边界归属钉死，产物同时给出 ``boundary_case`` 的现场判定。

    返回
    ----
    Dict[str, Any]
        ``n_classes`` / ``threshold`` / ``applicable`` / ``reason`` /
        ``boundary_case``（C=10 时的现场判定，用于证明边界归属）。
    """
    c = int(n_classes)
    if c <= 0:
        raise ValueError(f"n_classes 必须 >= 1，当前 {c}")
    threshold = 10.0 / float(c)
    applicable = bool(threshold < 1.0)
    boundary = {
        "n_classes_for_threshold_1": 10,
        "threshold_at_10": 10.0 / 10.0,
        "applicable_at_10": bool((10.0 / 10.0) < 1.0),
        "rule": (
            "边界归属 = **严格小于**：threshold < 1.0 才 applicable。"
            "C=10 时 threshold 恰为 1.0 ⇒ 不可满足（macro 上界就是 1.0，"
            "要满足需全对）⇒ applicable=False。"
        ),
    }
    return {
        "n_classes": int(c),
        "threshold": float(threshold),
        "applicable": bool(applicable),
        "boundary_rule": "applicable ⇔ 10/C < 1.0（严格小于）",
        "reason": (
            "阈值 10/C 落在可行域 (0, 1] 内且严格小于 1.0，判据可满足"
            if applicable
            else "阈值 10/C >= 1.0，而 macro <= 1.0 ⇒ 该项恒不可满足；"
            "按要求标 applicable=False 并如实报出，**不得**当成通过项"
        ),
        "boundary_case": boundary,
    }


def saturated_baseline_check(baseline: float, *, name: str = "majority_baseline") -> Dict[str, Any]:
    """**可行域边界核查**：「基线 + 5pp/10pp」这类判据的前提是基线不在饱和点。

    口径
    ----
    ``在可行域内部 ⇔ 0.0 < baseline < 1.0``（严格开区间）。基线为 ``1.0`` 时
    「+5pp/+10pp」不可能达到（上界 1.0）；基线为 ``0.0`` 时「>= 基线 + Xpp」虽然可满足，
    但该基线本身没有判别力，必须**标注**而不当作有效参照。
    """
    b = float(baseline)
    interior = bool(0.0 < b < 1.0)
    return {
        "name": str(name),
        "baseline": float(b),
        "in_feasible_interior": bool(interior),
        "saturated_low": bool(b <= 0.0),
        "saturated_high": bool(b >= 1.0),
        "margin_to_upper_bound": float(1.0 - b),
        "rule": (
            "在可行域内部 ⇔ 0.0 < baseline < 1.0（严格）。不在内部时，"
            "「优于 / 不劣于基线」类判据在该指标上**不可判**，只能标注，不得当作通过。"
        ),
    }


def baseline_criterion_table(baseline_macro: float, n_classes: int) -> Dict[str, Any]:
    """三条判据的现场判定表（含适用性与边界归属）。

    **W1 修复配套**：本函数只负责「给定**一个作用域**的基线 macro 与类数」算出该作用域
    的三条阈值；**作用域的合法性由调用方声明**（见 :func:`criterion_scope_declaration`
    与产物 ``criterion_tables_by_scope``）。跨作用域套用阈值是错配，不得发生。
    """
    app = macro_criterion_applicability(int(n_classes))
    sat = saturated_baseline_check(float(baseline_macro))
    b = float(baseline_macro)
    return {
        "scope": {
            "n_classes": int(n_classes),
            "baseline_macro": float(b),
            "baseline_in_feasible_interior": bool(sat["in_feasible_interior"]),
            "statement": (
                "本表的三条阈值**只**对「类数 = %d、基线 macro = %.6f」这一作用域成立；"
                "不得套用到其它作用域（分任务集 / 分侧）" % (int(n_classes), float(b))
            ),
        },
        "criterion_random_x10": {
            "expression": "macro >= 10/C",
            "threshold": float(app["threshold"]),
            "applicable": bool(app["applicable"]),
            "reason": str(app["reason"]),
            "boundary_rule": str(app["boundary_rule"]),
            "counted_as_pass": False if not bool(app["applicable"]) else None,
        },
        "criterion_majority_plus_10pp": {
            "expression": "macro >= majority_baseline + 0.10",
            "threshold": float(b + 0.10),
            "applicable": bool(sat["in_feasible_interior"]),
            "boundary_rule": ">= （闭区间下界）",
            "baseline_saturation": sat,
        },
        "criterion_majority_plus_5pp": {
            "expression": "macro >= majority_baseline + 0.05",
            "threshold": float(b + 0.05),
            "applicable": bool(sat["in_feasible_interior"]),
            "boundary_rule": ">= （闭区间下界）",
            "baseline_saturation": sat,
        },
        "rule": CRITERION_RULE,
    }


# ---------------------------------------------------------------------------
# 任务清单现场一致性校验（W3）
# ---------------------------------------------------------------------------


def validate_task_list(
    product_dir: str = DEFAULT_PRODUCT_DIR,
    *,
    declared: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """**W3 修复**：现场校验 ``ALL_TASKS`` 与冻结产物 / npz meta 的任务清单**一致**。

    为什么需要它（皋陶 warning 的根因）
    -----------------------------------
    ``ALL_TASKS`` 是硬编码元组，docstring 声称「现场来自 manifest ``tasks``」，
    但**全流程没有任何校验**；``load_row_tasks`` 明明算出了 ``tasks`` 与
    ``per_task_rows`` 却没人用。一旦上游新增 / 改名任务（例如 manifest 多了 ``math2``），
    本模块会**静默**地只统计旧清单里的任务 —— 而「静默漏掉一类样本」是最危险的一种错。

    校验口径（**双向**，缺失与新增都必须显式登记）
    ---------------------------------------------
    三处现场来源必须**逐项相同**：

    * ``manifest.json`` 的 ``tasks``（上游声明的任务清单）；
    * ``n3d_qa_all.npz`` 的 ``meta.task_of_row`` 去重（逐行真实任务，**保序**）；
    * ``n3d_qa_all_pairs.jsonl`` 按行回填后的任务去重（与上面二者逐项一致）。

    再与 ``declared``（默认 ``ALL_TASKS``）比集合。

    返回
    ----
    Dict[str, Any]
        ``declared`` / ``manifest_tasks`` / ``npz_tasks`` / ``pairs_tasks`` /
        ``missing_from_declared`` / ``extra_in_declared`` / ``consistent`` /
        ``allowed_extra`` / ``per_task_rows`` / ``rule``。

    异常
    ------
    AssertionError
        ① 三处来源不一致；② ``missing_from_declared`` 非空（现场出现未登记的任务）
        ⇒ **fail-closed**。``extra_in_declared`` 非空时**不抛**（允许的额外项 =
        清单里多登记的任务名），但必须在产物里**显式列出**。
    """
    tasks_declared = [str(t) for t in (declared if declared is not None else ALL_TASKS)]
    manifest = load_manifest(str(product_dir))
    manifest_tasks = [str(t) for t in manifest.get("tasks", [])]
    row_tasks = load_row_tasks(str(product_dir))
    npz_tasks = [str(t) for t in row_tasks["tasks"]]
    pairs = load_all_pairs(str(product_dir))
    if int(len(pairs)) != int(len(row_tasks["task_of_row"])):
        raise AssertionError(
            "任务清单校验失败：pairs 行数 "
            f"{len(pairs)} 与 npz task_of_row 长度 {len(row_tasks['task_of_row'])} 不一致"
        )
    pairs_tasks: List[str] = []
    for t in (str(row_tasks["task_of_row"][int(p.row_index)]) for p in pairs):
        if t not in pairs_tasks:
            pairs_tasks.append(str(t))
    if pairs_tasks != npz_tasks:
        raise AssertionError(
            "任务清单校验失败：pairs 回填后的任务去重序列 "
            f"{pairs_tasks} 与 npz meta.task_of_row 去重序列 {npz_tasks} 不一致"
        )
    if manifest_tasks != npz_tasks:
        raise AssertionError(
            "任务清单校验失败：manifest.tasks "
            f"{manifest_tasks} 与 npz meta.task_of_row 去重序列 {npz_tasks} 不一致；"
            "两处声明必须同源"
        )
    present = set(npz_tasks)
    declared_set = set(tasks_declared)
    missing = sorted(present - declared_set)
    extra = sorted(declared_set - present)
    if missing:
        raise AssertionError(
            "任务清单校验失败：冻结产物里出现了**未登记**的任务 "
            f"{missing}（现场 = {npz_tasks}，登记 = {tasks_declared}）；"
            "拒绝在会静默漏掉样本的任务清单上出数"
        )
    return {
        "declared": [str(t) for t in tasks_declared],
        "manifest_tasks": [str(t) for t in manifest_tasks],
        "npz_tasks": [str(t) for t in npz_tasks],
        "pairs_tasks": [str(t) for t in pairs_tasks],
        "missing_from_declared": missing,
        "extra_in_declared": extra,
        "consistent": True,
        "allowed_extra": (
            "允许的额外项 = `extra_in_declared`（清单里多登记、现场不存在的任务名），"
            "必须显式列出；`missing_from_declared` 非空即抛（fail-closed）。"
        ),
        "per_task_rows": dict(row_tasks["per_task_rows"]),
        "rule": (
            "三处来源（manifest.tasks / npz meta.task_of_row 去重 / pairs 回填后的任务去重）"
            "必须**逐项相同**；再与 ALL_TASKS 比集合：现场多出的任务名 ⇒ 抛 AssertionError；"
            "清单多出的任务名 ⇒ 显式登记。**反例（必须可复现）**：给 `declared` 传一个"
            "去掉某个真实任务的元组 ⇒ 必须抛 AssertionError。"
        ),
    }


def criterion_scope_declaration(
    *,
    all_classes: int,
    solve_macro: Optional[float] = None,
    match_macro: Optional[float] = None,
) -> Dict[str, Any]:
    """**W1 修复**：每条判据的**基线作用域**声明（禁止跨作用域套用）。

    为什么必须显式（皋陶 warning 的根因）
    ------------------------------------
    ``criterion_table_using_test_baseline`` 用的是**全量类口径**的
    ``constant_majority``（C=298 时 macro = 0.003610），而**分任务集**的 macro
    （如解题任务集只有 2 / 4 类，macro = 0.166667）与其**不是同一量纲** ——
    拿全量类的阈值（0.103610）去判解题任务集的分侧 macro 会得出完全错误的结论。
    故本函数与产物 ``criterion_tables_by_scope`` 一起把「作用域 → 基线 → 阈值」
    三者绑成一格，并**显式**写明不得跨格套用。

    参数
    ----
    all_classes : int
        全量类口径的类数（= ``corpus.n_classes``）。
    solve_macro / match_macro : Optional[float]
        解题任务集 / 匹配任务集在**同侧（测试侧）**的 ``constant_majority`` macro。

    返回
    ----
    Dict[str, Any]
        ``scopes``（各作用域的类数与基线说明）、``cross_scope_rule`` / ``why``。
    """
    scopes: Dict[str, Any] = {
        "all_classes": {
            "definition": "全部任务、全部类别（本轮口径的全量类）",
            "n_classes": int(all_classes),
            "baseline_scope": "全量类口径的 constant_majority（测试侧）",
        }
    }
    if solve_macro is not None:
        scopes["solve_tasks"] = {
            "definition": "解题任务集 " + str(list(SOLVE_TASKS)),
            "baseline_scope": "解题任务集内、按**本组范围内实际出现的类数**取多数类（测试侧）",
            "baseline_macro_test": float(solve_macro),
        }
    if match_macro is not None:
        scopes["match_tasks"] = {
            "definition": "匹配任务集 " + str(list(MATCH_TASKS)),
            "baseline_scope": "匹配任务集内、按**本组范围内实际出现的类数**取多数类（测试侧）",
            "baseline_macro_test": float(match_macro),
        }
    return {
        "scopes": scopes,
        "cross_scope_rule": (
            "每条判据的阈值只能与该判据**同作用域**的基线配对使用：`all_classes` 的阈值"
            "不得用于 `solve_tasks` / `match_tasks` 的分侧 macro，反之亦然。产物中每个作用域"
            "各有一格 `criterion_tables_by_scope[scope]`，含该作用域自己的类数、基线 macro "
            "与三条判据阈值。"
        ),
        "why": (
            "分任务集的类数与类分布与全量类口径**不同**，其 macro 的可达区间也不同："
            "解题任务集只有 2 / 4 类，macro 的可达格比 298 类口径粗得多；"
            "拿来自 298 类口径的阈值（如 0.103610）去判它是量纲错配。"
        ),
    }


# ---------------------------------------------------------------------------
# 确定性参照基线（**不做任何训练**）
# ---------------------------------------------------------------------------
#
# 口径声明（必须进产物）：本轮**不训练任何模型**（任务书：只改数据口径与任务分层，
# 不改模型结构、不改打分链路）。因此这里给出的是**纯确定性参照基线**，用于回答
# 「扩 C 之后数据规模是否够学」这个问题在**不涉及模型**时的下界/上界：
#   * ``constant_majority``：不看问题、一律回答「**本作用域（side + 任务过滤）内**
#     正样本行数最多的类」；
#   * ``row_argmax``：不看问题、一律回答「**本次统计的行集合**里正样本行数最多的类」。
# **W2 修复（皋陶 warning）**：命名与文档必须与实现对齐 —— 两者取的都是**本作用域**
# 的多数类，**不是**全局最高频类；作用域不同（全量 / 匹配任务集 / 解题任务集）时
# 取到的类**可以不同**。故 predictor 字面量统一带 ``scope_side_`` 前缀，并在返回块里
# 给出 ``scope_of_majority`` 说明它到底在哪个范围内取的多数类。
# 真正训练出来的 macro 必须**同时**超过这两条，本报告只负责把这两条钉死。


def majority_class_index(class_counts: Sequence[int]) -> int:
    """多数类下标（**平票按类下标升序**，确定性）。

    口径说明（W2 修复）：本函数本身只是「给定一组计数取 argmax」的纯工具，
    **不**隐含「全局」语义 —— 调用方决定传入**哪个作用域**的计数
    （本侧全量类 / 本侧某任务集 / 本次统计的行集合）。
    """
    counts = [int(x) for x in class_counts]
    if not counts:
        raise ValueError("class_counts 为空，无法取多数类")
    best = 0
    for i, c in enumerate(counts):
        if int(c) > int(counts[best]):
            best = i
    return int(best)


def deterministic_baseline_metrics(
    corpus: ScaleCorpus,
    split: ScaleSplit,
    *,
    side: str = "test",
    tasks: Optional[Sequence[str]] = None,
    unit_index: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """**确定性参照基线**的按类 macro 与 micro（**不训练、不用模型**）。

    参数
    ----
    corpus / split : 语料与切分。
    side : str
        统计哪一侧（``'train'`` / ``'test'``）。
    tasks : Optional[Sequence[str]]
        只统计这些任务（``None`` = 全部任务）。
    unit_index : Optional[Sequence[int]]
        直接给要统计的行下标（``None`` = 用 ``side`` 决定；给定时忽略 ``side`` 的行集，
        但 ``side`` 仍决定**类先验**的取值口径）。

    指标口径
    --------
    * 对**每一行**（pair）判定 ``pred == candidate_index``；行级 micro 准确率 =
      命中行数 / 行数；
    * 按**类别**分组求准确率后取算术平均 = macro（未出现的类**不计入**分母，
      分母 = 实际出现的类数，并在产物中登记 ``n_classes_in_scope``）；
    * ``constant_majority`` 与 ``row_argmax`` 两个预测器的阈值均为 0/1 边界外，
      故可行域核查在 :func:`baseline_criterion_table` 里单独做。
    """
    key = str(side)
    if key not in ("train", "test"):
        raise ValueError(f"side 只能是 'train' / 'test'，当前 {side!r}")
    rows = (
        [int(r) for r in unit_index]
        if unit_index is not None
        else (list(split.train_rows) if key == "train" else list(split.test_rows))
    )
    if tasks is not None:
        allowed = {str(t) for t in tasks}
        rows = [r for r in rows if str(corpus.pairs[int(r)].task) in allowed]
    if not rows:
        return {
            "n_rows": 0,
            "n_tasks": 0,
            "tasks": [] if tasks is None else [str(t) for t in tasks],
            "n_classes_in_scope": 0,
            "constant_majority": {
                "predictor": "scope_side_majority_class",
                "class_index": -1,
                "class_key": "",
                "micro": 0.0,
                "macro": 0.0,
                "n_hit": 0,
                "n_rows_in_scope": 0,
                "n_rows_out_of_scope": 0,
                "n_classes_in_scope": 0,
                "scope_of_majority": "no_samples",
                "scope_of_majority_detail": "本侧在给定任务过滤后没有样本，无法取多数类",
            },
            "row_argmax": {
                "predictor": "scope_rows_positive_row_count_argmax_class",
                "class_index": -1,
                "class_key": "",
                "micro": 0.0,
                "macro": 0.0,
                "n_hit": 0,
                "n_rows_in_scope": 0,
                "n_rows_out_of_scope": 0,
                "n_classes_in_scope": 0,
                "scope_of_majority": "no_samples",
                "scope_of_majority_detail": "本次统计的行集合为空，无法取多数类",
            },
            "rule": "本侧在给定任务过滤后没有样本；指标为 0 且明确标注「无样本」而非「测出来很差」",
        }
    # 类先验：优先用**本侧**的正样本行数（与 side 同侧），无样本时回退到全局表 count
    side_counts = side_class_counts(corpus, split, side=key, unit="row")
    prior = [int(x) for x in side_counts]
    prior_scope = "side_row_positive_counts"
    if sum(prior) == 0:
        prior = [int(x) for x in corpus.class_row_counts]
        prior_scope = "frozen_table_count_fallback"
    const_idx = majority_class_index(prior)
    rows_per_class = [0] * int(corpus.n_classes)
    for r in rows:
        pair = corpus.pairs[int(r)]
        if int(pair.label) == 1:
            pos = corpus.class_pos.get(int(pair.candidate_index))
            if pos is not None:
                rows_per_class[int(pos)] += 1
    argmax_idx = majority_class_index(rows_per_class) if sum(rows_per_class) else const_idx

    def _score(pred_index: int) -> Dict[str, Any]:
        hit = 0
        per_class_hit: Dict[int, int] = {}
        per_class_tot: Dict[int, int] = {}
        n_out_of_scope = 0
        for r in rows:
            pair = corpus.pairs[int(r)]
            pos = corpus.class_pos.get(int(pair.candidate_index))
            if pos is None:
                # 该行指向本轮口径**未保留**的类别（C=10 重建时必然出现）：
                # 既不是本轮的合法预测目标，也不进 micro/macro；**必须显式计数**。
                n_out_of_scope += 1
                continue
            cls = int(pos)
            per_class_tot[cls] = per_class_tot.get(cls, 0) + 1
            ok = int(cls) == int(pred_index)
            if ok:
                hit += 1
                per_class_hit[cls] = per_class_hit.get(cls, 0) + 1
        n_in_scope = int(sum(per_class_tot.values()))
        accs = [
            float(per_class_hit.get(c, 0)) / float(per_class_tot[c])
            for c in sorted(per_class_tot.keys())
        ]
        macro = float(sum(accs) / len(accs)) if accs else 0.0
        return {
            "class_index": int(pred_index),
            "class_key": str(corpus.answer_keys[int(pred_index)]),
            "n_hit": int(hit),
            "micro": float(hit) / float(n_in_scope) if n_in_scope else 0.0,
            "macro": float(macro),
            "n_rows_in_scope": int(n_in_scope),
            "n_rows_out_of_scope": int(n_out_of_scope),
            "n_classes_in_scope": int(len(accs)),
            "per_class_accuracy_min": float(min(accs)) if accs else 0.0,
            "per_class_accuracy_max": float(max(accs)) if accs else 0.0,
            "per_class_accuracy_frac_one": (
                float(sum(1 for a in accs if a >= 1.0 - 1e-12) / len(accs)) if accs else 0.0
            ),
        }

    const_scored = _score(int(const_idx))
    const_scored["predictor"] = "scope_side_majority_class"
    argmax_scored = _score(int(argmax_idx))
    argmax_scored["predictor"] = "scope_rows_positive_row_count_argmax_class"
    # **W2 修复**：显式登记「多数类是在哪个作用域取的」—— 实测取的是**本作用域**的多数类，
    # 不是全局最高频类；不同作用域（全量类 / 匹配任务集 / 解题任务集）可以取到**不同**的类。
    const_scored["scope_of_majority"] = str(prior_scope)
    const_scored["scope_of_majority_detail"] = (
        "本侧（side=%s）" % str(key)
        + ("、任务过滤 = " + str(sorted(str(t) for t in tasks)) if tasks is not None else "、无任务过滤（全量类）")
        + "；取自 `side_class_counts(side, unit='row')` 的 argmax（平票取最小下标）"
        + ("；本侧无正样本行 ⇒ 回退到冻结表 count" if prior_scope.endswith("fallback") else "")
    )
    argmax_scored["scope_of_majority"] = "scored_rows_positive_counts"
    argmax_scored["scope_of_majority_detail"] = (
        "本次统计的行集合（n_rows=%d）内、按 label==1 的逐类行数取 argmax（平票取最小下标）"
        % int(len(rows))
    )
    gold_counts = [0] * int(corpus.n_classes)
    cand_counts = [0] * int(corpus.n_classes)
    for r in rows:
        pair = corpus.pairs[int(r)]
        pos = corpus.class_pos.get(int(pair.candidate_index))
        if pos is None:
            continue
        cand_counts[int(pos)] += 1
        if int(pair.label) == 1:
            gold_counts[int(pos)] += 1
    return {
        "side": str(key),
        "tasks": sorted({str(corpus.pairs[int(r)].task) for r in rows}),
        "n_rows": int(len(rows)),
        "n_questions": int(
            len({str(corpus.pairs[int(r)].question_id) for r in rows})
        ),
        "n_positive_rows": int(
            sum(1 for r in rows if int(corpus.pairs[int(r)].label) == 1)
        ),
        "class_coverage": {
            "n_classes": int(corpus.n_classes),
            "n_classes_with_gold_rows": int(sum(1 for x in gold_counts if int(x) > 0)),
            "n_classes_with_candidate_rows": int(
                sum(1 for x in cand_counts if int(x) > 0)
            ),
            "classes_without_gold_rows_head": [
                int(i) for i, x in enumerate(gold_counts) if int(x) == 0
            ][:20],
            "rule": (
                "两个口径必须分开报：`with_gold_rows` = 该类在本范围内有 >= 1 条 label=1 的"
                "行（即出现在 macro 分母里）；`with_candidate_rows` = 该类作为候选出现过"
                "（即使一条都没标正）。**macro 的分母只由前者决定**"
                "（`n_classes_in_scope` 与之一致）。"
            ),
        },
        "constant_majority": const_scored,
        "row_argmax": argmax_scored,
        "out_of_scope_rule": (
            "指向本轮口径**未保留**类别（C=10 重建时被截断掉的那些类）的行：既不是合法预测"
            "目标、也不进 micro/macro 分母，但**逐行计数**在 n_rows_out_of_scope 里显式登记；"
            "C=298 全量口径下该计数恒为 0（现场核对）。"
        ),
        "rule": (
            "确定性参照基线（不训练、不用模型）：constant_majority = 一律回答本侧正样本"
            "行数最多的类；row_argmax = 同上但只按**本次统计的行的正样本行数**取多数类。"
            "macro = 逐类准确率的算术平均（分母 = 本范围内实际出现的类数）。"
            "两条基线都必须由真正训练出来的 macro 同时超过；本报告只把它们钉死。"
        ),
    }


__all__ = [
    "QA_SCALE_VERSION",
    "DEFAULT_PRODUCT_DIR",
    "ANSWER_TABLE_FILE",
    "ALL_PAIRS_FILE",
    "ALL_NPZ_FILE",
    "MANIFEST_FILE",
    "FULL_CLASSES",
    "BASELINE_MAX_CLASSES",
    "BASELINE_MIN_QUESTIONS",
    "BASELINE_TEST_EVERY",
    "BASELINE_TEST_PER_CLASS",
    "DEFAULT_HOLDOUT_PCT",
    "HOLDOUT_PCT_RANGE",
    "ALL_TASKS",
    "SOLVE_TASKS",
    "MATCH_TASKS",
    "TASK_STRATIFICATION_RULE",
    "HOLDOUT_RULE",
    "CRITERION_RULE",
    "CLASS_SIZE_BINS",
    "VERIFY_ROOT",
    "DEFAULT_OUT_DIR",
    "QAPair",
    "ScaleCorpus",
    "ScaleSplit",
    "answer_table_fields",
    "pair_record_fields",
    "baseline_criterion_table",
    "build_scale_corpus",
    "class_coverage",
    "class_size_distribution",
    "criterion_scope_declaration",
    "deterministic_baseline_metrics",
    "grouped_stats",
    "load_all_pairs",
    "load_answer_table",
    "load_manifest",
    "load_row_tasks",
    "macro_criterion_applicability",
    "majority_class_index",
    "make_scale_split",
    "proportional_holdout_evidence",
    "question_hash",
    "question_side",
    "question_side_with_hash",
    "saturated_baseline_check",
    "sha256_file",
    "sha256_text",
    "side_class_counts",
    "task_side_stats",
    "validate_task_list",
]
