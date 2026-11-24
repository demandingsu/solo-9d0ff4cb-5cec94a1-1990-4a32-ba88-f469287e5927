"""两区两阶段抽样的核心逻辑。

方案
----
1. 两个区各自独立、等概率无放回抽取 ``n_i`` 株（首轮）。
2. 首轮未入样的坡地苗木合并为补抽池，等概率无放回补抽 ``m`` 株，
   池内不足 ``m`` 株时全取。

入样概率（权重 = 入样概率的倒数）
----------------------------------
设第 i 区名册总量 ``N_i``、坡地数 ``a_i``、首轮名额 ``n_i``，
首轮抽样比 ``p_i = n_i / N_i``。

* 非坡地苗木只能通过首轮入样，``π = p_i``。
* 坡地苗木 j：若首轮未中（概率 ``1-p_i``），再进入补抽池。
  设首轮后补抽池大小为 S（随机变量），补抽 ``min(m, S)`` 株，
  给定 S 时一株池中苗木的补抽入样条件概率为 ``g(S)=min(m,S)/S``（S=0 时为 0）。

  在“j 首轮未中”条件下，j 必在池中。令
  ``Z_i ~ Hypergeom(N_i-1, a_i-1, n_i)`` 为其余 N_i-1 株（其中坡地
  a_i-1 株）中被首轮抽中的坡地数；
  ``X_k ~ Hypergeom(N_k, a_k, n_k)`` 为另一区首轮抽中坡地数，两者独立。
  补抽池大小（含 j）为
  ``S = 1 + (a_i-1-Z_i) + (a_k-X_k)``，恒有 S ≥ 1。

  故 ``q_i = E[g(S)] = Σ_z Σ_x P(Z_i=z)P(X_k=x) g(S(z,x))``，
  坡地入样概率 ``π = 1 - (1-p_i)(1-q_i)``。

  两个易错点：本区必须用“j 未中”条件下的超几何分布（总体 N_i-1、
  成功类 a_i-1），不能用首轮抽中坡地总数的无条件分布；池大小必须
  包含 j 自身（+1），否则全取场景会漏算。

计算只对坡地抽中数的取值范围做双重求和（规模 O(n_i·n_k)），
通过超几何概率质量函数汇总首轮所有可能结果，不逐份枚举首轮名单。
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

ROUND_FIRST = "首轮"
ROUND_SUPPLEMENT = "补抽"


class SamplingError(ValueError):
    """输入数据不满足方案约束。"""

    def __init__(self, errors: list[str], impossible_ids: list[str] | None = None):
        super().__init__("; ".join(errors))
        self.errors = errors
        self.impossible_ids = impossible_ids or []


@dataclass(frozen=True)
class Seedling:
    id: str
    is_slope: bool


@dataclass(frozen=True)
class District:
    name: str
    roster: list[Seedling]
    first_round_quota: int
    actual_drawn: list[str]


@dataclass(frozen=True)
class SampleEntry:
    id: str
    district: str
    round_name: str
    inclusion_probability: float
    weight: float


@dataclass(frozen=True)
class SamplingResult:
    entries: list[SampleEntry]
    supplement_pool_size: int
    supplement_drawn: int
    supplement_quota: int
    supplement_all_taken: bool
    # 每个区的 (非坡地入样概率, 坡地入样概率)
    inclusion_probabilities: dict[str, tuple[float, float]]


# --------------------------------------------------------------------------- #
# 超几何分布（对数空间计算，避免大名册下组合数溢出）
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return math.nan
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
    )


def hypergeom_pmf(k: int, population: int, successes: int, draws: int) -> float:
    """超几何分布 P(X=k)。

    从总量 ``population``、其中成功类 ``successes`` 个的总体中
    无放回抽取 ``draws`` 个，恰有 k 个成功类的概率。
    """
    if draws < 0 or draws > population or not (0 <= successes <= population):
        raise ValueError("超几何分布参数越界")
    lo = max(0, draws - (population - successes))
    hi = min(successes, draws)
    if k < lo or k > hi:
        return 0.0
    log_p = (
        _log_comb(successes, k)
        + _log_comb(population - successes, draws - k)
        - _log_comb(population, draws)
    )
    return math.exp(log_p)


def _hypergeom_pmf_array(
    population: int, successes: int, draws: int
) -> tuple[int, list[float]]:
    """返回支持集下界 lo 与 PMF 数组 [P(lo), ..., P(hi)]。

    起点用对数组合数计算一次，其余按超几何相邻项比值递推：
    ``P(k+1)/P(k) = (K-k)/(k+1) · (n-k)/(N-K-n+k+1)``，
    最后归一化以消除递推累积误差。复杂度 O(支持集长度)，
    避免双重求和中重复调用 lgamma。
    """
    if draws < 0 or draws > population or not (0 <= successes <= population):
        raise ValueError("超几何分布参数越界")
    lo = max(0, draws - (population - successes))
    hi = min(successes, draws)
    if lo == hi:
        return lo, [1.0]

    failures = population - successes
    probs = [0.0] * (hi - lo + 1)
    probs[0] = math.exp(
        _log_comb(successes, lo)
        + _log_comb(failures, draws - lo)
        - _log_comb(population, draws)
    )
    for idx, k in enumerate(range(lo, hi)):
        ratio = (
            (successes - k) / (k + 1)
            * (draws - k) / (failures - draws + k + 1)
        )
        probs[idx + 1] = probs[idx] * ratio
    total = sum(probs)
    return lo, [p / total for p in probs]


# --------------------------------------------------------------------------- #
# 入样概率
# --------------------------------------------------------------------------- #
def _expected_supplement_inclusion(
    own: District,
    a_own: int,
    other: District,
    a_other: int,
    supplement_quota: int,
) -> float:
    """本区某株首轮未中坡地苗木的补抽入样概率 q = E[g(S)]。

    给定该株首轮未中后：
      * 本区其余 N_i-1 株（坡地 a_i-1 株）中抽 n_i 株，抽中坡地数
        Z_i 服从条件超几何 Hypergeom(N_i-1, a_i-1, n_i)；
      * 另一区不受影响，首轮抽中坡地数 X_k 仍为无条件超几何。
    补抽池（含目标株自身）S = 1 + (a_i-1-Z_i) + (a_k-X_k)，恒有 S ≥ 1。
    """
    if supplement_quota == 0:
        return 0.0
    m = supplement_quota
    n_own, n_other = own.first_round_quota, other.first_round_quota
    n_total_own, n_total_other = len(own.roster), len(other.roster)

    lo_z, pz_array = _hypergeom_pmf_array(
        n_total_own - 1, a_own - 1, n_own
    )
    lo_x, px_array = _hypergeom_pmf_array(
        n_total_other, a_other, n_other
    )
    # g(S) 只与池大小有关，预算 1..(a_own+a_other) 各池大小下的取值
    max_pool = a_own + a_other
    g_by_pool = [0.0] * (max_pool + 1)
    for size in range(1, max_pool + 1):
        g_by_pool[size] = min(m, size) / size

    expectation = 0.0
    for iz, pz in enumerate(pz_array):
        z = lo_z + iz
        remaining_own = a_own - 1 - z
        for ix, px in enumerate(px_array):
            x = lo_x + ix
            pool_size = 1 + remaining_own + (a_other - x)
            expectation += pz * px * g_by_pool[pool_size]
    return expectation


def _compute_inclusion_probabilities(
    districts: list[District], supplement_quota: int
) -> dict[str, tuple[float, float]]:
    """返回 {区名: (非坡地 π, 坡地 π)}。"""
    result: dict[str, tuple[float, float]] = {}
    slope_counts = [
        sum(1 for s in d.roster if s.is_slope) for d in districts
    ]
    for i, district in enumerate(districts):
        other = districts[1 - i]
        n_i, n_i_total = district.first_round_quota, len(district.roster)
        a_i = slope_counts[i]
        p_first = n_i / n_i_total

        non_slope_pi = p_first

        # 边界短路：无坡地区不存在坡地入样概率；首轮全抽时每株必中，
        # 且条件超几何的总体 N_i-1 容纳不下 n_i=N_i 个抽样，必须直接判定。
        if a_i == 0 or n_i == n_i_total:
            slope_pi = 1.0 if n_i == n_i_total else 0.0
            result[district.name] = (non_slope_pi, slope_pi)
            continue

        q_supplement = _expected_supplement_inclusion(
            district, a_i, other, slope_counts[1 - i],
            supplement_quota,
        )
        # 吸附到边界：消除 PMF 求和/递推的浮点尾差（如补抽全取时理论值 1）
        eps = 1e-12
        if q_supplement < eps:
            q_supplement = 0.0
        elif q_supplement > 1.0 - eps:
            q_supplement = 1.0
        slope_pi = 1.0 - (1.0 - p_first) * (1.0 - q_supplement)
        if slope_pi < eps:
            slope_pi = 0.0
        elif slope_pi > 1.0 - eps:
            slope_pi = 1.0
        result[district.name] = (non_slope_pi, slope_pi)
    return result


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def _validate(districts: list[District], supplement_quota: int) -> None:
    errors: list[str] = []
    impossible: list[str] = []

    if len(districts) != 2:
        raise SamplingError(["必须恰好提供两个区的名册"])
    if supplement_quota < 0:
        errors.append("补抽名额不能为负")

    names = [d.name for d in districts]
    if len(set(names)) != 2:
        errors.append("两个区的名称必须不同")

    seen_ids: set[str] = set()
    district_ids: list[set[str]] = []
    for d in districts:
        if not d.roster:
            errors.append(f"区 {d.name!r} 的名册为空")
        local_ids = [s.id for s in d.roster]
        local_set: set[str] = set()
        # 线性查重，避免 list.count 的 O(N^2)
        counts = Counter(local_ids)
        dup_local = sorted(sid for sid, c in counts.items() if c > 1)
        if dup_local:
            errors.append(
                f"区 {d.name!r} 名册内编号重复: {dup_local}"
            )
        collide = sorted(set(local_ids) & seen_ids)
        if collide:
            errors.append(
                f"区 {d.name!r} 编号与其他区冲突（全局编号须唯一）: {collide}"
            )
        seen_ids.update(local_ids)
        local_set.update(local_ids)
        district_ids.append(local_set)

        if d.first_round_quota < 0:
            errors.append(f"区 {d.name!r} 首轮名额不能为负")
        elif d.first_round_quota > len(d.roster):
            errors.append(
                f"区 {d.name!r} 首轮名额 {d.first_round_quota} 超过名册量 "
                f"{len(d.roster)}"
            )
        if len(d.actual_drawn) != d.first_round_quota:
            errors.append(
                f"区 {d.name!r} 实抽编号数 {len(d.actual_drawn)} "
                f"与首轮名额 {d.first_round_quota} 不符"
            )
        if len(set(d.actual_drawn)) != len(d.actual_drawn):
            errors.append(f"区 {d.name!r} 实抽编号存在重复")
        foreign = sorted(set(d.actual_drawn) - local_set)
        if foreign:
            errors.append(
                f"区 {d.name!r} 实抽编号不属于本区名册: {foreign}"
            )

    if errors:
        raise SamplingError(errors)

    # 结构性不可能入样检查（输入本身合法但方案下永远无法入样）
    for d in districts:
        if d.first_round_quota == 0 and supplement_quota == 0:
            impossible.extend(s.id for s in d.roster)
        elif d.first_round_quota == 0:
            # 只有坡地能通过补抽入样，非坡地永远无法入样
            impossible.extend(
                s.id for s in d.roster if not s.is_slope
            )
    if impossible:
        raise SamplingError(
            ["以下苗木在整套方案下不可能入样，拒绝出单"],
            impossible_ids=sorted(impossible),
        )


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def build_sampling_plan(
    districts: list[District],
    supplement_quota: int,
    seed: int,
) -> SamplingResult:
    _validate(districts, supplement_quota)
    probs = _compute_inclusion_probabilities(districts, supplement_quota)

    # 首轮入样条目（按区、按名册顺序输出）
    entries: list[SampleEntry] = []
    drawn_by_district: list[set[str]] = []
    for d in districts:
        drawn_set = set(d.actual_drawn)
        drawn_by_district.append(drawn_set)
        non_slope_pi, slope_pi = probs[d.name]
        for s in d.roster:
            if s.id in drawn_set:
                pi = slope_pi if s.is_slope else non_slope_pi
                entries.append(
                    SampleEntry(
                        id=s.id,
                        district=d.name,
                        round_name=ROUND_FIRST,
                        inclusion_probability=pi,
                        weight=1.0 / pi,
                    )
                )

    # 补抽池：两区首轮均未抽中的坡地苗木，排序保证确定性
    pool: list[str] = []
    for d, drawn_set in zip(districts, drawn_by_district):
        pool.extend(
            s.id for s in d.roster if s.is_slope and s.id not in drawn_set
        )
    pool.sort()

    k = min(supplement_quota, len(pool))
    if k > 0:
        rng = random.Random(seed)
        selected = rng.sample(pool, k)
    else:
        selected = []

    id_to_district = {
        s.id: d.name for d in districts for s in d.roster
    }
    for sid in selected:
        slope_pi = probs[id_to_district[sid]][1]
        entries.append(
            SampleEntry(
                id=sid,
                district=id_to_district[sid],
                round_name=ROUND_SUPPLEMENT,
                inclusion_probability=slope_pi,
                weight=1.0 / slope_pi,
            )
        )

    return SamplingResult(
        entries=entries,
        supplement_pool_size=len(pool),
        supplement_drawn=len(selected),
        supplement_quota=supplement_quota,
        supplement_all_taken=len(selected) == len(pool) and len(pool) > 0,
        inclusion_probabilities=probs,
    )
