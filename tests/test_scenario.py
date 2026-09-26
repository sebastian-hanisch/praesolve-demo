"""Instanzen: Bauart, Determinismus, Fixtures (Lehrbuch, Zentrum, entartete Ecke), Zulässigkeit der erzeugten Instanzen, Zentrum."""

import numpy as np
import pytest
from scipy.optimize import linprog

import pre_scenario as S


def _highs(inst, zero_objective=False):
    A, b, c = inst.arrays()
    if zero_objective:
        c = np.zeros_like(c)
    ub_a, ub_b, eq_a, eq_b = [], [], [], []
    for i, s in enumerate(inst.senses):
        if s == S.LE:
            ub_a.append(A[i]), ub_b.append(b[i])
        elif s == S.GE:
            ub_a.append(-A[i]), ub_b.append(-b[i])
        else:
            eq_a.append(A[i]), eq_b.append(b[i])
    return linprog(-c, A_ub=np.array(ub_a) if ub_a else None, b_ub=ub_b or None, A_eq=np.array(eq_a) if eq_a else None, b_eq=eq_b or None, bounds=(0, None), method="highs")


def reference_status(inst):
    """Status laut HiGHS; bei zulässigen, unbeschränkten LPs meldet HiGHS gelegentlich "unzulässig" (Präsolve), darum wird über ein Zulässigkeitsproblem ohne Zielfunktion abgesichert."""
    h = _highs(inst)
    if h.status == 0:
        return "optimal"
    return "infeasible" if _highs(inst, zero_objective=True).status == 2 else "unbounded"


def test_textbook_is_the_classic_two_service_example():
    inst = S.textbook_instance()
    assert (inst.m, inst.n) == (3, 2) and inst.senses == (S.LE,) * 3 and inst.b == (4.0, 12.0, 18.0) and inst.c == (3.0, 5.0)
    assert -_highs(inst).fun == pytest.approx(36.0)


def test_fixtures_have_the_status_they_claim():
    assert _highs(S.infeasible_instance()).status == 2
    assert _highs(S.unbounded_instance()).status == 3
    d = S.degenerate_instance()
    assert -_highs(d).fun == pytest.approx(36.0) and d.A[3] == (1.0, 1.0) and d.b[3] == 8.0 and d.kind == "degenerate"


def test_centre_instance_is_the_constructed_one():
    inst = S.centre_instance()
    assert (inst.m, inst.n) == (4, 5) and inst.senses == (S.LE,) * 4 and inst.b == (80.0, 200.0, 55.0, 400.0) and inst.c == (15.5, 15.0, 26.0, 20.0, 12.0)
    h = _highs(inst)
    assert -h.fun == pytest.approx(720.0) and list(h.x) == pytest.approx([20, 10, 10, 0, 0])


def test_generation_is_deterministic_and_seed_dependent_and_hashable():
    a = S.generate("random", 8, 6, 0.5, 3)
    assert a == S.generate("random", 8, 6, 0.5, 3) and a != S.generate("random", 8, 6, 0.5, 4) and hash(a) == hash(S.generate("random", 8, 6, 0.5, 3))


@pytest.mark.parametrize("kind", ["random", "mixed"])
def test_generated_instances_are_feasible_and_bounded(kind):
    for seed in range(40):
        for m, n in ((2, 2), (6, 4), (4, 9), (15, 15), (10, 80)):
            inst = S.generate(kind, m, n, 0.1, seed)
            assert (inst.m, inst.n) == (m, n) and len(inst.senses) == m and _highs(inst).status == 0, (kind, seed, m, n)


def test_every_row_and_column_is_used_and_row_zero_is_dense_even_at_low_density():
    for seed in range(20):
        inst = S.generate("random", 10, 40, 0.02, seed)
        A = np.array(inst.A)
        assert (A.sum(axis=1) > 0).all() and (A.sum(axis=0) > 0).all() and (A[0] > 0).all() and (A >= 0).all()


def test_mixed_instances_contain_all_three_senses_sometimes():
    seen = set()
    for seed in range(30):
        seen |= set(S.generate("mixed", 12, 8, 0.5, seed).senses)
    assert seen == {S.LE, S.GE, S.EQ}
    for seed in range(30):
        inst = S.generate("mixed", 10, 4, 0.5, seed)
        assert sum(s == S.EQ for s in inst.senses) <= 2 and inst.senses[0] == S.LE


def test_kinds_and_fixtures_are_consistent_and_unknown_kind_raises():
    assert set(S.FIXTURE_KINDS) <= set(S.KINDS) and set(S.KIND_LABELS) == set(S.KINDS)
    for kind in S.FIXTURE_KINDS:
        assert S.generate(kind, 9, 9, 0.5, 9) == S.generate(kind, 3, 3, 0.1, 1)
    with pytest.raises(ValueError):
        S.generate("nope", 3, 3, 0.5, 1)


def test_messy_instance_dirt_is_protocolled_and_the_instance_is_solvable():
    inst, dirt = S.messy_instance(6, 8, 35, 0, 0.5)
    assert (inst.m, inst.n) == (17, 9) and dirt == {"duplicate": 1, "dominated": 2, "singleton": 4, "fixed": 1, "empty_row": 1, "empty_col": 1, "redundant": 1, "core_rows": 6, "core_cols": 8}
    clean, none = S.messy_instance(6, 8, 35, 0, 0.0)
    assert (clean.m, clean.n) == (6, 8) and sum(v for k, v in none.items() if not k.startswith("core")) == 0
    for sd in range(20):
        for red in (0.25, 0.5, 1.0):
            assert _highs(S.messy_instance(6, 8, sd, 0, red)[0]).status == 0


def test_messy_spread_only_changes_the_units_not_the_optimum():
    base = _highs(S.messy_instance(6, 8, 35, 0, 0.5)[0]).fun
    for spread in (4, 8, 12):
        other = S.messy_instance(6, 8, 35, spread, 0.5)[0]
        assert _highs(other).fun == pytest.approx(base, rel=1e-6)
    import pre_scaling as SC
    a, b = S.messy_instance(6, 8, 35, 0, 0.5)[0].arrays()[0], S.messy_instance(6, 8, 35, 12, 0.5)[0].arrays()[0]
    assert np.count_nonzero(a) == np.count_nonzero(b) and SC.spread(b) > SC.spread(a) + 8.0


def test_dirty_textbook_is_the_textbook_with_four_dirt_types():
    inst = S.dirty_textbook_instance()
    assert (inst.m, inst.n) == (7, 2) and -_highs(inst).fun == pytest.approx(36.0) and inst.kind == "dirty" and "dirty" in S.FIXTURE_KINDS


def test_needle_and_ties_shapes():
    needle = S.needle_instance(10, 10, 6, 8)
    A_ = np.array(needle.A)
    assert needle.b[0] == 1.0 and A_[0, 0] == 1.0 and (A_[1:, 0] < 1e-2).all() and (A_[1:, 0] >= 1e-8).all() and all(b < a for a, b in zip(A_[1:, 0], needle.b[1:]))
    assert _highs(needle).status == 0
    ties = S.ties_instance(12, 12, 3)
    assert set(np.array(ties.A).ravel()) <= {0.0, 1.0} and set(ties.b) == {10.0} and _highs(ties).status == 0
    for kind in ("messy", "needle", "ties", "dirty"):
        assert kind in S.KINDS and kind in S.KIND_LABELS
