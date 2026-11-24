"""抽样方案自测。

- 超几何 PMF 与 math.comb 直接计数结果对照；
- 小名册下对首轮全部可能结果做全枚举，精确核对入样概率与权重；
- 中等名册用 Monte Carlo 仿真核对（含两区抽样比不同、补抽不足全取、
  首轮名额为 0 等场景）；
- 确定性重跑、补抽抽取结果、输入校验与不可能入样拒绝。
"""

from __future__ import annotations

import itertools
import math
import random

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.sampling import (
    District,
    SamplingError,
    Seedling,
    build_sampling_plan,
    hypergeom_pmf,
)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def make_district(name: str, n: int, slope_count: int, quota: int,
                  drawn: list[str]) -> District:
    roster = [
        Seedling(id=f"{name}-{i:02d}", is_slope=(i < slope_count))
        for i in range(n)
    ]
    return District(name=name, roster=roster, first_round_quota=quota,
                    actual_drawn=drawn)


def simulate_inclusion(districts: list[District], supplement_quota: int,
                       trials: int, seed: int = 12345) -> dict[str, float]:
    """独立 Monte Carlo 仿真整套方案，返回每株至少一次入样的频率。"""
    rng = random.Random(seed)
    counts = {s.id: 0 for d in districts for s in d.roster}
    members = {
        s.id: (d.name, s.is_slope) for d in districts for s in d.roster
    }
    for _ in range(trials):
        selected: set[str] = set()
        pool: list[str] = []
        for d in districts:
            ids = [s.id for s in d.roster]
            first = set(rng.sample(ids, d.first_round_quota))
            selected.update(first)
            pool.extend(
                s.id for s in d.roster
                if s.is_slope and s.id not in first
            )
        k = min(supplement_quota, len(pool))
        if k:
            selected.update(rng.sample(pool, k))
        for sid in selected:
            counts[sid] += 1
    return {sid: c / trials for sid, c in counts.items()}


def exhaustive_inclusion(districts: list[District],
                         supplement_quota: int) -> dict[str, float]:
    """对首轮所有等概率组合做全枚举，精确计算入样概率（仅用于小例子）。

    首轮入样概率恒为 n_i/N_i；分支枚举只负责计算首轮未中后经补抽
    入样的部分，补抽给定池时每株条件入样概率为 min(m,|池|)/|池|，
    无需枚举补抽结果。
    """
    probs: dict[str, float] = {}
    for d in districts:
        rate = d.first_round_quota / len(d.roster)
        for s in d.roster:
            probs[s.id] = rate  # 首轮入样部分（非坡地到此为止）

    combo_lists = []
    combo_weights = []
    for d in districts:
        ids = [s.id for s in d.roster]
        combos = list(itertools.combinations(ids, d.first_round_quota))
        combo_lists.append(combos)
        combo_weights.append(1.0 / len(combos))

    for c0, c1 in itertools.product(*combo_lists):
        first_sets = [set(c0), set(c1)]
        branch_prob = combo_weights[0] * combo_weights[1]
        pool = [
            s.id for di, d in enumerate(districts) for s in d.roster
            if s.is_slope and s.id not in first_sets[di]
        ]
        g = (min(supplement_quota, len(pool)) / len(pool)) if pool else 0.0
        for di, d in enumerate(districts):
            for s in d.roster:
                if s.is_slope and s.id not in first_sets[di]:
                    probs[s.id] += branch_prob * g
    return probs


# --------------------------------------------------------------------------- #
# 超几何分布
# --------------------------------------------------------------------------- #
def ref_hypergeom(k: int, n: int, k_success: int, draws: int) -> float:
    lo = max(0, draws - (n - k_success))
    hi = min(k_success, draws)
    if not lo <= k <= hi:
        return 0.0
    return (
        math.comb(k_success, k)
        * math.comb(n - k_success, draws - k)
        / math.comb(n, draws)
    )


@pytest.mark.parametrize("n,a,draws", [
    (10, 4, 3), (20, 7, 0), (20, 7, 20), (15, 15, 8), (15, 0, 5),
])
def test_hypergeom_pmf(n, a, draws):
    lo = max(0, draws - (n - a))
    hi = min(a, draws)
    for k in range(lo, hi + 1):
        assert hypergeom_pmf(k, n, a, draws) == pytest.approx(
            ref_hypergeom(k, n, a, draws)
        )
    assert hypergeom_pmf(lo - 1, n, a, draws) == 0.0
    assert hypergeom_pmf(hi + 1, n, a, draws) == 0.0


def test_hypergeom_normalizes_to_one():
    n, a, draws = 30, 12, 10
    total = sum(
        hypergeom_pmf(k, n, a, draws)
        for k in range(max(0, draws - (n - a)), min(a, draws) + 1)
    )
    assert total == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 入样概率：全枚举精确核对
# --------------------------------------------------------------------------- #
def test_inclusion_probability_exhaustive():
    d1 = make_district("甲", n=4, slope_count=3, quota=1, drawn=["甲-00"])
    d2 = make_district("乙", n=3, slope_count=2, quota=2,
                       drawn=["乙-00", "乙-02"])
    m = 2
    result = build_sampling_plan([d1, d2], supplement_quota=m, seed=7)
    exact = exhaustive_inclusion([d1, d2], m)

    for entry in result.entries:
        assert entry.weight == pytest.approx(
            1.0 / exact[entry.id], rel=1e-12
        )
        assert entry.inclusion_probability == pytest.approx(
            exact[entry.id], rel=1e-12
        )

    # 非坡地：只能走首轮，π 等于首轮抽样比（未入样株看汇总概率表）
    assert result.inclusion_probabilities["甲"][0] == pytest.approx(1 / 4)
    assert result.inclusion_probabilities["乙"][0] == pytest.approx(2 / 3)
    # 全枚举名册中每株的 π 都应与实现的分区/类型概率一致
    is_slope = {s.id: s.is_slope for d in (d1, d2) for s in d.roster}
    for sid, exact_pi in exact.items():
        district = sid.split("-")[0]
        p_non, p_slope = result.inclusion_probabilities[district]
        got = p_slope if is_slope[sid] else p_non
        assert got == pytest.approx(exact_pi, rel=1e-12)


def test_weights_equal_reciprocal_of_probability_always():
    d1 = make_district("甲", 6, 4, 2, ["甲-01", "甲-04"])
    d2 = make_district("乙", 8, 5, 3, ["乙-00", "乙-02", "乙-06"])
    result = build_sampling_plan([d1, d2], supplement_quota=3, seed=99)
    for e in result.entries:
        assert e.weight == pytest.approx(1 / e.inclusion_probability)


# --------------------------------------------------------------------------- #
# Monte Carlo：两区抽样比例不同
# --------------------------------------------------------------------------- #
def test_monte_carlo_different_sampling_rates():
    # 甲区抽样比 2/10=0.2，乙区抽样比 6/12=0.5，坡地数也不同
    d1 = make_district("甲", 10, 6, 2, ["甲-00", "甲-03"])
    d2 = make_district("乙", 12, 7, 6,
                       ["乙-00", "乙-01", "乙-03", "乙-05", "乙-08", "乙-10"])
    result = build_sampling_plan([d1, d2], supplement_quota=4, seed=2024)
    sim = simulate_inclusion([d1, d2], supplement_quota=4, trials=40000)

    is_slope = {s.id: s.is_slope for d in (d1, d2) for s in d.roster}

    def expected_pi(sid: str) -> float:
        district = sid.split("-")[0]
        p_non, p_slope = result.inclusion_probabilities[district]
        return p_slope if is_slope[sid] else p_non

    for sid, freq in sim.items():
        assert expected_pi(sid) == pytest.approx(freq, abs=0.02)

    # 同区同类型苗木的入样概率应相同（汇总概率不依赖首轮具体结果）
    slope_pi_甲 = {
        expected_pi(s.id) for s in d1.roster if s.is_slope
    }
    assert len(slope_pi_甲) == 1
    # 乙区首轮抽样比更高，其非坡地入样概率相应更大
    assert result.inclusion_probabilities["乙"][0] > \
           result.inclusion_probabilities["甲"][0]


# --------------------------------------------------------------------------- #
# 补抽池不足：全取，坡地 π=1
# --------------------------------------------------------------------------- #
def test_supplement_shortfall_all_taken():
    d1 = make_district("甲", 5, 2, 1, ["甲-00"])   # 剩 1 株坡地
    d2 = make_district("乙", 5, 1, 2, ["乙-00", "乙-03"])  # 剩 0 株坡地
    result = build_sampling_plan([d1, d2], supplement_quota=10, seed=3)

    assert result.supplement_pool_size == 1
    assert result.supplement_drawn == 1
    assert result.supplement_all_taken is True

    by_id = {e.id: e for e in result.entries}
    # 未中首轮的坡地必被补抽 -> 入样概率 1，权重 1
    assert by_id["甲-01"].round_name == "补抽"
    assert by_id["甲-01"].inclusion_probability == pytest.approx(1.0)
    assert by_id["甲-01"].weight == pytest.approx(1.0)

    sim = simulate_inclusion([d1, d2], supplement_quota=10, trials=5000)
    for e in result.entries:
        assert e.inclusion_probability == pytest.approx(sim[e.id], abs=0.02)


def test_zero_supplement_quota_slope_only_first_round():
    d1 = make_district("甲", 6, 3, 2, ["甲-00", "甲-02"])
    d2 = make_district("乙", 6, 2, 3, ["乙-01", "乙-03", "乙-04"])
    result = build_sampling_plan([d1, d2], supplement_quota=0, seed=4)
    assert result.supplement_drawn == 0
    assert all(e.round_name == "首轮" for e in result.entries)
    for e in result.entries:
        assert e.inclusion_probability == pytest.approx(
            2 / 6 if e.district == "甲" else 3 / 6
        )


def test_zero_first_round_quota_non_slope_rejected():
    d1 = make_district("甲", 5, 3, 0, [])
    d2 = make_district("乙", 5, 2, 1, ["乙-00"])
    with pytest.raises(SamplingError) as exc:
        build_sampling_plan([d1, d2], supplement_quota=2, seed=1)
    assert set(exc.value.impossible_ids) == {"甲-03", "甲-04"}


def test_both_quotas_zero_all_impossible():
    d1 = make_district("甲", 3, 2, 0, [])
    d2 = make_district("乙", 3, 3, 0, [])
    with pytest.raises(SamplingError) as exc:
        build_sampling_plan([d1, d2], supplement_quota=0, seed=1)
    assert len(exc.value.impossible_ids) == 6


def test_zero_first_round_slope_still_samplable():
    # 甲区首轮名额 0 但全是坡地：全部经补抽，无不可能入样株
    d1 = make_district("甲", 4, 4, 0, [])
    d2 = make_district("乙", 5, 2, 1, ["乙-00"])
    result = build_sampling_plan([d1, d2], supplement_quota=3, seed=8)
    sim = simulate_inclusion([d1, d2], supplement_quota=3, trials=20000)
    p_slope_甲 = result.inclusion_probabilities["甲"][1]
    p_slope_乙 = result.inclusion_probabilities["乙"][1]
    p_non_乙 = result.inclusion_probabilities["乙"][0]
    for sid, freq in sim.items():
        expected = p_slope_甲 if sid.startswith("甲") else (
            p_slope_乙 if int(sid.split("-")[1]) < 2 else p_non_乙
        )
        assert expected == pytest.approx(freq, abs=0.03)


# --------------------------------------------------------------------------- #
# 确定性与抽取行为
# --------------------------------------------------------------------------- #
def test_deterministic_same_seed():
    d1 = make_district("甲", 8, 5, 2, ["甲-02", "甲-05"])
    d2 = make_district("乙", 9, 6, 3, ["乙-01", "乙-04", "乙-07"])
    r1 = build_sampling_plan([d1, d2], supplement_quota=4, seed=42)
    r2 = build_sampling_plan([d1, d2], supplement_quota=4, seed=42)
    assert [(e.id, e.round_name, e.weight) for e in r1.entries] == \
           [(e.id, e.round_name, e.weight) for e in r2.entries]

    r3 = build_sampling_plan([d1, d2], supplement_quota=4, seed=43)
    # 不同种子首轮部分相同（首轮由实抽编号给定），仅补抽可能不同
    supp_42 = [e.id for e in r1.entries if e.round_name == "补抽"]
    supp_43 = [e.id for e in r3.entries if e.round_name == "补抽"]
    assert supp_42 != supp_43


def test_supplement_pool_and_no_overlap_with_first_round():
    d1 = make_district("甲", 7, 4, 2, ["甲-00", "甲-01"])
    d2 = make_district("乙", 6, 3, 1, ["乙-02"])
    result = build_sampling_plan([d1, d2], supplement_quota=5, seed=11)
    first_ids = {e.id for e in result.entries if e.round_name == "首轮"}
    supp_ids = [e.id for e in result.entries if e.round_name == "补抽"]
    assert not (first_ids & set(supp_ids))
    # 补抽池 = 未中首轮坡地：甲剩 2、乙剩 2
    assert result.supplement_pool_size == 4
    assert result.supplement_drawn == 4  # 名额 5 > 池 4，全取
    assert set(supp_ids) == {"甲-02", "甲-03", "乙-00", "乙-01"}
    # 编号全局唯一，无重复入样
    all_ids = [e.id for e in result.entries]
    assert len(all_ids) == len(set(all_ids))


# --------------------------------------------------------------------------- #
# 校验类
# --------------------------------------------------------------------------- #
def _build_simple(quota_a=2, drawn_a=None, quota_b=1, drawn_b=None, m=2):
    d1 = make_district("甲", 6, 3, quota_a, drawn_a or ["甲-00", "甲-01"])
    d2 = make_district("乙", 5, 2, quota_b, drawn_b or ["乙-00"])
    return build_sampling_plan([d1, d2], supplement_quota=m, seed=1)


def test_duplicate_actual_drawn_rejected():
    d1 = make_district("甲", 6, 3, 2, ["甲-00", "甲-00"])
    d2 = make_district("乙", 5, 2, 1, ["乙-00"])
    with pytest.raises(SamplingError, match="重复"):
        build_sampling_plan([d1, d2], supplement_quota=2, seed=1)


def test_foreign_drawn_id_rejected():
    d1 = make_district("甲", 6, 3, 2, ["甲-00", "乙-00"])
    d2 = make_district("乙", 5, 2, 1, ["乙-00"])
    with pytest.raises(SamplingError, match="不属于本区"):
        build_sampling_plan([d1, d2], supplement_quota=2, seed=1)


def test_drawn_count_mismatch_rejected():
    with pytest.raises(SamplingError, match="不符"):
        _build_simple(quota_a=3, drawn_a=["甲-00", "甲-01"])


def test_quota_exceeds_roster_rejected():
    with pytest.raises(SamplingError, match="超过名册量"):
        _build_simple(quota_a=99)


def test_full_quota_district_does_not_crash():
    # 甲区首轮名额等于名册量（全抽）：每株 π=1，条件超几何不应越界
    d1 = make_district("甲", 4, 2, 4, ["甲-00", "甲-01", "甲-02", "甲-03"])
    d2 = make_district("乙", 5, 3, 1, ["乙-04"])
    result = build_sampling_plan([d1, d2], supplement_quota=2, seed=1)
    p_non, p_slope = result.inclusion_probabilities["甲"]
    assert p_non == pytest.approx(1.0)
    assert p_slope == pytest.approx(1.0)
    assert all(e.weight == pytest.approx(1.0)
               for e in result.entries if e.district == "甲")
    # 甲区坡地全部首轮入样，补抽池只含乙区未中坡地
    assert result.supplement_pool_size == 3


def test_no_slope_district_does_not_crash():
    # 甲区没有坡地：坡地概率为 0，非坡地走首轮；不应触发超几何越界
    d1 = make_district("甲", 4, 0, 1, ["甲-00"])
    d2 = make_district("乙", 5, 3, 1, ["乙-04"])
    result = build_sampling_plan([d1, d2], supplement_quota=2, seed=1)
    p_non, p_slope = result.inclusion_probabilities["甲"]
    assert p_non == pytest.approx(1 / 4)
    assert p_slope == 0.0
    assert result.supplement_pool_size == 3  # 仅乙区有未中坡地
    sim = simulate_inclusion([d1, d2], supplement_quota=2, trials=10000)
    for sid, freq in sim.items():
        if sid.startswith("甲"):
            assert freq == pytest.approx(1 / 4, abs=0.03)


def test_duplicate_roster_id_rejected():
    roster = [Seedling(id="X", is_slope=True),
              Seedling(id="X", is_slope=False)]
    d1 = District("甲", roster, 1, ["X"])
    d2 = make_district("乙", 5, 2, 1, ["乙-00"])
    with pytest.raises(SamplingError, match="名册内编号重复"):
        build_sampling_plan([d1, d2], supplement_quota=1, seed=1)


def test_cross_district_id_collision_rejected():
    d1 = make_district("甲", 5, 2, 1, ["甲-00"])
    d2_roster = [Seedling(id="甲-00", is_slope=True),
                 Seedling(id="乙-01", is_slope=True)]
    d2 = District("乙", d2_roster, 1, ["甲-00"])
    with pytest.raises(SamplingError, match="冲突"):
        build_sampling_plan([d1, d2], supplement_quota=1, seed=1)


# --------------------------------------------------------------------------- #
# HTTP 层
# --------------------------------------------------------------------------- #
def _payload():
    return {
        "districts": [
            {
                "name": "甲",
                "roster": [
                    {"id": f"甲-{i:02d}", "is_slope": i < 3}
                    for i in range(6)
                ],
                "first_round_quota": 2,
                "actual_drawn": ["甲-00", "甲-02"],
            },
            {
                "name": "乙",
                "roster": [
                    {"id": f"乙-{i:02d}", "is_slope": i < 2}
                    for i in range(5)
                ],
                "first_round_quota": 1,
                "actual_drawn": ["乙-01"],
            },
        ],
        "supplement_quota": 2,
        "seed": 2024,
    }


def test_http_endpoint_and_determinism():
    client = TestClient(app)
    r1 = client.post("/sample", json=_payload())
    assert r1.status_code == 200
    body1 = r1.json()
    r2 = client.post("/sample", json=_payload())
    assert r2.json() == body1

    ids = [e["id"] for e in body1["entries"]]
    assert len(ids) == len(set(ids))
    assert {"首轮", "补抽"} >= {e["round_name"] for e in body1["entries"]}
    for e in body1["entries"]:
        assert e["weight"] == pytest.approx(1 / e["inclusion_probability"])
    assert "inclusion_probabilities" in body1


def test_http_impossible_ids_422():
    client = TestClient(app)
    payload = _payload()
    payload["districts"][0]["first_round_quota"] = 0
    payload["districts"][0]["actual_drawn"] = []
    resp = client.post("/sample", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert "impossible_ids" in body
    # 甲区非坡地为 甲-03..甲-05
    assert set(body["impossible_ids"]) == {"甲-03", "甲-04", "甲-05"}


def test_http_validation_duplicate_drawn_422():
    client = TestClient(app)
    payload = _payload()
    payload["districts"][1]["actual_drawn"] = ["乙-01", "乙-01"]
    payload["districts"][1]["first_round_quota"] = 2
    resp = client.post("/sample", json=payload)
    assert resp.status_code == 422
    assert "重复" in resp.json()["detail"][0]


def test_health():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
