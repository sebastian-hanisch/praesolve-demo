"""Korrektheitskette: Präsolve erhält das Optimum, Postsolve-Zertifikat auf dem Original, jede Regel von Hand, Skalierung exakt rückrechenbar, Quotiententests, Störung, Sonderfälle."""

import math

import numpy as np
import pytest

import pre_algorithm as A
import pre_evaluation as ev
import pre_presolve as PS
import pre_scaling as SC
import pre_scenario as S
import pre_simplex as X
from tests.test_scenario import _highs, reference_status


def _custom(rows, b, c, senses):
    n = len(c)
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses), tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(len(b))), "custom")


def _roundtrip(inst, scaling="pow2"):
    """Präsolve -> Skalierung -> Simplex -> Rückrechnung -> Postsolve; Rückgabe (Status, x, y, obj)."""
    pipe = ev.pipeline(inst, True, scaling, "textbook", 1e-9, 0.0)
    return pipe


def _instances():
    for sd in range(40):
        for red in (0.25, 0.5, 1.0):
            yield S.messy_instance(6, 8, sd, 0, red)[0]
        yield S.messy_instance(8, 10, sd, 6, 0.5)[0]
        yield S.messy_instance(8, 10, sd, 10, 0.75)[0]
        yield S.generate("mixed", 7, 7, 0.5, sd)
        yield S.generate("random", 6, 6, 0.5, sd)
        yield S.generate("needle", 8, 8, 0.5, sd, spread=4)
        yield S.generate("ties", 8, 8, 0.5, sd)
    for kind in ("textbook", "dirty", "centre", "degenerate"):
        yield S.generate(kind, 6, 8, 0.5, 35)


# --- 1./2. Optimum erhalten, Postsolve-Zertifikat ---------------------------------------------------------------------------------------------------

def test_presolve_keeps_the_optimum_and_postsolve_returns_a_certificate_on_the_original():
    count = 0
    for inst in _instances():
        h = _highs(inst)
        assert h.status == 0
        pipe = _roundtrip(inst)
        assert pipe.status == "optimal", (inst.kind, pipe.status)
        assert pipe.obj == pytest.approx(-h.fun, rel=1e-6, abs=1e-6)
        x, y = np.array(pipe.x), np.array(pipe.y)
        assert A.primal_violation(inst, x) <= 1e-6 * (1.0 + np.abs(inst.b).max()) and A.dual_violation(inst, y) <= 1e-6 * (1.0 + np.abs(inst.c).max())
        assert abs(float(np.dot(inst.c, x)) - float(np.dot(inst.b, y))) <= 1e-6 * (1.0 + abs(pipe.obj))                # c·x = b·y auf dem Original
        count += 1
    assert count >= 300


def test_infeasible_and_unbounded_status_is_preserved():
    assert PS.presolve(S.infeasible_instance()).status == "infeasible"
    assert ev.pipeline(S.infeasible_instance(), True, "pow2", "textbook", 1e-9, 0.0).status == "infeasible"
    assert ev.pipeline(S.unbounded_instance(), True, "pow2", "textbook", 1e-9, 0.0).status == "unbounded"
    for sd in range(30):
        inst = S.messy_instance(6, 8, sd, 0, 0.5)[0]
        contradict = _custom([list(r) for r in inst.A] + [[1.0] + [0.0] * (inst.n - 1)], list(inst.b) + [1e9], list(inst.c), list(inst.senses) + [S.GE])
        assert reference_status(contradict) == "infeasible"
        assert ev.pipeline(contradict, True, "pow2", "textbook", 1e-9, 0.0).status == "infeasible"


# --- 3. Einzelregeln von Hand ---------------------------------------------------------------------------------------------------------------------

def _rules(inst):
    pre = PS.presolve(inst)
    return pre, [e["rule"] for e in pre.log]


def test_empty_row_rule():
    pre, rules = _rules(_custom([[0.0, 0.0], [1.0, 1.0]], [5.0, 4.0], [1.0, 1.0], [S.LE, S.LE]))
    assert rules[0] == "leere Zeile" and pre.reduced.m == 1 and pre.status == "reduced"
    assert PS.presolve(_custom([[0.0, 0.0], [1.0, 1.0]], [-1.0, 4.0], [1.0, 1.0], [S.LE, S.LE])).status == "infeasible"
    assert PS.presolve(_custom([[0.0, 0.0], [1.0, 1.0]], [1.0, 4.0], [1.0, 1.0], [S.GE, S.LE])).status == "infeasible"


def test_singleton_rows_become_bounds_with_the_right_sign():
    pre = PS.presolve(_custom([[2.0, 3.0], [2.0, 0.0], [0.0, -1.0]], [12.0, 6.0, -1.0], [1.0, 1.0], [S.LE, S.LE, S.LE]))     # x1 <= 3, x2 >= 1
    assert pre.u[0] == pytest.approx(3.0) and pre.l[1] == pytest.approx(1.0) and pre.src_u[0] == 1 and pre.src_l[1] == 2 and pre.status == "reduced"
    pre = PS.presolve(_custom([[1.0, 1.0], [-2.0, 0.0], [0.0, 4.0]], [9.0, -4.0, 8.0], [1.0, 1.0], [S.LE, S.GE, S.EQ]))       # -2 x1 >= -4: x1 <= 2, 4 x2 = 8: x2 = 2
    assert pre.u[0] == pytest.approx(2.0) and pre.xval[1] == pytest.approx(2.0) and [e["rule"] for e in pre.log if e["rule"] == "feste Variable"] == ["feste Variable"]
    assert PS.presolve(_custom([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]], [9.0, 3.0, 5.0], [1.0, 1.0], [S.LE, S.LE, S.GE])).status == "infeasible"


def test_redundant_rows_only_when_the_bounds_make_them_so():
    base = [[1.0, 0.0], [0.0, 1.0]]
    inst = _custom(base + [[1.0, 1.0]], [4.0, 6.0, 10.0], [1.0, 1.0], [S.LE] * 3)
    pre, rules = _rules(inst)
    assert "redundante Zeile" in rules and pre.status == "empty" and pre.objconst == pytest.approx(10.0)                    # beide Variablen liegen an ihren Schranken, nichts bleibt zu lösen
    tight = _custom(base + [[1.0, 1.0]], [4.0, 6.0, 9.999], [1.0, 1.0], [S.LE] * 3)
    pre, rules = _rules(tight)
    assert "redundante Zeile" not in rules and pre.reduced.m == 3
    ge = _custom(base + [[1.0, 1.0]], [4.0, 6.0, -1.0], [1.0, 1.0], [S.LE, S.LE, S.GE])
    assert "redundante Zeile" in _rules(ge)[1]
    assert PS.presolve(_custom(base + [[1.0, 1.0]], [4.0, 6.0, 11.0], [1.0, 1.0], [S.LE, S.LE, S.GE])).status == "infeasible"
    assert PS.presolve(_custom(base + [[1.0, 1.0]], [4.0, 6.0, 11.0], [1.0, 1.0], [S.LE, S.LE, S.EQ])).status == "infeasible"


def test_duplicate_rows_keep_the_strictest_and_detect_contradictions():
    inst = _custom([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [-1.0, -2.0, -1.0], [3.0, 1.0, 1.0]], [10.0, 16.0, -3.0, 9.0], [1.0, 1.0, 1.0], [S.LE, S.LE, S.LE, S.LE])
    pre, rules = _rules(inst)
    assert rules.count("doppelte Zeile") == 1 and pre.rows == [1, 2, 3]                     # Zeile 2 (2x, rhs 16 => x+2y+z <= 8) ist strenger als Zeile 1; Zeile 3 (-x >= -... => >= 3) ist die untere Grenze
    assert PS.presolve(_custom([[1.0, 2.0], [-1.0, -2.0]], [10.0, -30.0], [1.0, 1.0], [S.LE, S.LE])).status == "infeasible"


def test_fixed_variable_and_empty_column_rules():
    pre = PS.presolve(_custom([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]], [9.0, 2.0, 2.0], [1.0, 3.0], [S.LE, S.LE, S.GE]))
    assert pre.xval[0] == pytest.approx(2.0) and "feste Variable" in [e["rule"] for e in pre.log] and pre.xval[1] == pytest.approx(7.0) and pre.objconst == pytest.approx(2.0 + 21.0)
    empty = PS.presolve(_custom([[1.0, 0.0], [1.0, 0.0]], [5.0, 3.0], [1.0, 2.0], [S.LE, S.LE]))                       # x2 kommt nirgends vor, Kosten > 0, keine Schranke
    assert empty.unbounded_hint and "leere Spalte" in [e["rule"] for e in empty.log]
    capped = PS.presolve(_custom([[1.0, 0.0], [0.0, 1.0]], [5.0, 7.0], [1.0, 2.0], [S.LE, S.LE]))
    assert capped.status == "empty" and capped.objconst == pytest.approx(19.0) and not capped.unbounded_hint
    down = PS.presolve(_custom([[1.0, 0.0]], [5.0], [1.0, -2.0], [S.LE]))
    assert not down.unbounded_hint and down.xval[1] == 0.0


def test_reductions_trigger_each_other_and_stop_at_the_fixpoint():
    inst = _custom([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [3.0, 3.0, 5.0, 9.0], [1.0, 1.0, 1.0], [S.LE, S.GE, S.LE, S.LE])
    pre, rules = _rules(inst)
    assert "feste Variable" in rules and pre.xval[0] == pytest.approx(3.0) and pre.u[1] == pytest.approx(2.0) and pre.u[2] == pytest.approx(9.0)      # x1 = 3, dann Zeile 3 zu x2 <= 2
    assert pre.status == "empty" and pre.objconst == pytest.approx(3.0 + 2.0 + 9.0)


def test_dirty_textbook_log_matches_the_hand_calculation():
    pre = PS.presolve(S.dirty_textbook_instance())
    counts = ev.rules_count(pre)
    assert counts == {"leere Zeile": 1, "Singleton-Zeile": 3, "redundante Zeile": 1, "doppelte Zeile": 1, "feste Variable": 0, "leere Spalte": 0}
    assert (pre.reduced.m, pre.reduced.n) == (3, 2) and pre.removed_rows == 4 + 2 and pre.u[0] == pytest.approx(3.0) and pre.u[1] == pytest.approx(6.0)
    pipe = ev.pipeline(S.dirty_textbook_instance(), True, "none", "textbook", 1e-9, 0.0)
    assert pipe.obj == pytest.approx(36.0) and np.allclose(pipe.x, [2.0, 6.0]) and pipe.gap < 1e-12


def test_generated_dirt_is_found_by_the_presolve():
    for sd in range(20):
        inst, dirt = S.messy_instance(6, 8, sd, 0, 0.5)
        counts = ev.rules_count(PS.presolve(inst))
        assert counts["feste Variable"] >= dirt["fixed"] and counts["leere Spalte"] >= dirt["empty_col"] and counts["leere Zeile"] >= dirt["empty_row"]
        assert counts["doppelte Zeile"] >= dirt["duplicate"] and counts["Singleton-Zeile"] >= dirt["singleton"] + 2 * dirt["fixed"] + dirt["empty_col"] // 2
        dirt_rows = dirt["singleton"] + 2 * dirt["fixed"] + dirt["dominated"] + dirt["duplicate"] + dirt["empty_row"] + dirt["redundant"] + dirt["empty_col"] // 2
        assert PS.presolve(inst).removed_rows >= dirt_rows


# --- 4. Skalierung -------------------------------------------------------------------------------------------------------------------------------

def test_power_of_two_scaling_is_exact_and_round_trips_bit_for_bit():
    for sd in range(20):
        inst = S.messy_instance(6, 8, sd, 12, 0.5)[0]
        sc = SC.scale(inst, "pow2")
        assert all(math.frexp(v)[0] == 0.5 for v in list(sc.r) + list(sc.t))
        A0, b0, c0 = inst.arrays()
        As, bs, cs = sc.inst.arrays()
        assert np.array_equal(As / sc.r[:, None] / sc.t, A0) and np.array_equal(bs / sc.r, b0) and np.array_equal(cs / sc.t, c0)


def test_scaling_keeps_the_optimum_and_unscales_solution_and_duals():
    for meth in ("equilibrate", "geometric", "pow2"):
        for sd in range(12):
            inst = S.messy_instance(6, 8, sd, 6, 0.5)[0]
            ref = A.solve(SC.scale(inst, "pow2").inst)
            sc = SC.scale(inst, meth)
            sol = A.solve(sc.inst)
            assert sol.status == ref.status == "optimal" and sol.obj == pytest.approx(ref.obj, rel=1e-6)
            x, y = sc.unscale_x(sol.x), sc.unscale_y(sol.y)
            assert A.primal_violation(inst, x) <= 1e-6 * (1.0 + np.abs(inst.b).max()) and abs(float(np.dot(inst.c, x)) - float(np.dot(inst.b, y))) <= 1e-6 * (1.0 + abs(sol.obj))
    with pytest.raises(ValueError):
        SC.scale(S.textbook_instance(), "zufall")
    assert SC.scale(S.textbook_instance(), "none").inst is not None


def test_scaling_shrinks_the_spread_and_converges():
    for sd in range(10):
        inst = S.messy_instance(8, 10, sd, 12, 0.5)[0]
        before = SC.spread(inst.arrays()[0])
        spreads = {m: SC.spread(SC.scale(inst, m).inst.arrays()[0]) for m in ("equilibrate", "geometric", "pow2")}
        assert before > 15 and spreads["geometric"] < 2.0 and spreads["pow2"] < spreads["geometric"] + 1.0 and spreads["equilibrate"] < before - 3.0
        eq = SC.scale(inst, "equilibrate").inst.arrays()[0]
        colmax = np.abs(eq).max(axis=0)
        assert np.allclose(colmax[colmax > 0], 1.0, atol=1e-9) and np.abs(eq).max() <= 1.0 + 1e-9
    assert SC.spread(np.array([[1.0, 100.0], [0.0, 0.5]])) == pytest.approx(math.log10(200.0))


# --- 5. Quotiententests -------------------------------------------------------------------------------------------------------------------------------

def test_both_ratio_tests_reach_the_optimum_on_well_scaled_instances():
    for sd in range(40):
        for kind, m, n in (("random", 6, 8), ("mixed", 6, 6), ("ties", 8, 8)):
            inst = S.generate(kind, m, n, 0.5, sd)
            ref = A.solve(inst)
            for ratio in X.RATIOS:
                r = X.simplex(inst, ratio=ratio)
                assert r.status == ref.status == "optimal" and r.obj == pytest.approx(ref.obj, rel=1e-6)
    with pytest.raises(ValueError):
        X.simplex(S.textbook_instance(), ratio="zufall")


def test_needle_the_simple_test_returns_infeasible_solutions_and_harris_holds():
    insts = [S.needle_instance(10, 10, sd, 8) for sd in range(100000, 100010)]
    refs = [ev.reference(i) for i in insts]
    bad = {}
    for ratio in X.RATIOS:
        for tol in (1e-8, 1e-10):
            bad[(ratio, tol)] = sum(not ev.correct(ev.pipeline(i, False, "none", ratio, tol, 0.0), rs, ro, obj_tol=1e-6) for i, (rs, ro) in zip(insts, refs))
    assert bad[("textbook", 1e-8)] >= 3 and bad[("textbook", 1e-10)] >= 1 and bad[("harris", 1e-8)] == 0 and bad[("harris", 1e-10)] == 0
    rel = [ev.pipeline(i, False, "none", "harris", 1e-8, 0.0) for i in insts]
    assert all(r.rel_infeas <= 1e-6 for r in rel) and statistics_median([r.min_pivot_rel for r in rel]) > 0.1
    tx = [ev.pipeline(i, False, "none", "textbook", 1e-8, 0.0) for i in insts]
    assert statistics_median([r.min_pivot_rel for r in tx]) < 1e-3


def statistics_median(v):
    import statistics
    return statistics.median(v)


def test_harris_needs_a_tolerance_that_covers_the_spread_of_the_ratios():
    insts = [S.needle_instance(10, 10, sd, 8) for sd in range(100000, 100010)]
    refs = [ev.reference(i) for i in insts]
    bad = sum(not ev.correct(ev.pipeline(i, False, "none", "harris", 1e-12, 0.0), rs, ro, obj_tol=1e-6) for i, (rs, ro) in zip(insts, refs))
    assert bad >= 1


# --- 6. Störung -------------------------------------------------------------------------------------------------------------------------------------

def test_perturbation_removes_zero_steps_and_keeps_the_basis_and_the_value():
    ok_basis = zero_free = 0
    total = 0
    for sd in range(30):
        inst = S.ties_instance(24, 24, sd)
        ref = A.solve(inst)
        plain = X.simplex(inst)
        r = X.simplex(inst, perturb=1e-6, seed=sd)
        total += 1
        assert r.status == "optimal" and abs(r.obj - ref.obj) <= 1e-5 * (1.0 + abs(ref.obj)) and plain.zero_steps >= 0
        ok_basis += r.basis_ok
        zero_free += r.zero_steps == 0
        assert X.simplex(inst, perturb=1e-6, seed=sd).pivots == r.pivots                              # deterministisch je Seed
    assert ok_basis >= 0.9 * total and zero_free >= 0.9 * total
    assert sum(X.simplex(S.ties_instance(24, 24, sd)).zero_steps for sd in range(30)) >= 100
    p = X.perturbed_instance(S.textbook_instance(), 1e-6, 3)
    assert all(pb > b for pb, b in zip(p.b, S.textbook_instance().b)) and p.A == S.textbook_instance().A
    g = X.perturbed_instance(_custom([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], [4.0, 1.0, 2.0], [1.0, 1.0], [S.LE, S.GE, S.EQ]), 1e-6, 3)
    assert g.b[0] > 4.0 and g.b[1] < 1.0 and g.b[2] == 2.0


# --- 7. Sonderfälle, Buchführung, Zweige --------------------------------------------------------------------------------------------------------

def test_special_cases_and_determinism():
    single = _custom([[2.0, 3.0]], [12.0], [1.0, 1.0], [S.LE])
    pipe = ev.pipeline(single, True, "pow2", "harris", 1e-9, 0.0)
    assert pipe.status == "optimal" and pipe.obj == pytest.approx(6.0)
    allred = _custom([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [4.0, 6.0, 20.0], [0.0, 0.0], [S.LE] * 3)
    pre = PS.presolve(allred)
    assert pre.status in ("reduced", "empty")
    full = ev.pipeline(allred, True, "none", "textbook", 1e-9, 0.0)
    assert full.status == "optimal" and full.obj == pytest.approx(0.0)
    fixed = _custom([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], [2.0, 2.0, 3.0, 3.0], [1.0, 2.0], [S.LE, S.GE, S.LE, S.GE])
    pipe = ev.pipeline(fixed, True, "none", "textbook", 1e-9, 0.0)
    assert PS.presolve(fixed).status == "empty" and pipe.obj == pytest.approx(8.0) and pipe.gap < 1e-12
    a, b = ev.analyse(ev.Settings()), ev.analyse(ev.Settings())
    assert a is b
    x1, x2 = ev.pipeline(S.messy_instance(6, 8, 3, 4, 0.5)[0], True, "pow2", "textbook", 1e-9, 0.0), ev.pipeline(S.messy_instance(6, 8, 3, 4, 0.5)[0], True, "pow2", "textbook", 1e-9, 0.0)
    assert x1.x == x2.x and x1.pivots == x2.pivots


def test_every_rule_and_status_branch_is_executed():
    seen = set()
    for inst in _instances():
        pre = PS.presolve(inst)
        seen |= {e["rule"] for e in pre.log}
    assert seen >= set(PS.RULES)
    statuses = {X.simplex(S.infeasible_instance()).status, X.simplex(S.unbounded_instance()).status, X.simplex(S.textbook_instance()).status, X.simplex(S.centre_instance(), max_pivots=1).status}
    assert statuses == {"infeasible", "unbounded", "optimal", "limit"}
