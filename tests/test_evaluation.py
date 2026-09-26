"""Auswertung: Einstellungen, Pipeline gegen Referenz, Präsolve-Ausbeute, Skalierung über die Spanne, Quotiententest über die Toleranz, Stillstand und Störung."""

import math

import numpy as np
import pytest

import pre_constants as C
import pre_evaluation as ev
import pre_presolve as PS
import pre_scaling as SC
import pre_scenario as S
import pre_simplex as X
from pre_evaluation import Settings


def test_settings_clamp_the_index_controls():
    s = Settings(spread_i=99, red_i=-3, tol_i=99, perturb_i=99)
    assert s.spread == 16 and s.redundancy == 0.0 and s.tol == 1e-12 and s.perturb == 1e-6
    assert Settings().spread == 4 and Settings().redundancy == 0.5 and Settings().tol == 1e-9 and Settings().perturb == 0.0


def test_rel_infeasibility_is_scale_free_and_zero_for_feasible_points():
    inst = S.textbook_instance()
    assert ev.rel_infeasibility(inst, [2.0, 6.0]) == 0.0
    assert ev.rel_infeasibility(inst, [2.0, 7.0]) > 0.05
    small = S.Instance(((1e-9, 0.0),), (1e-9,), (1.0, 1.0), (S.LE,), ("a", "b"), ("r",), "custom")
    assert ev.rel_infeasibility(small, [2.0, 0.0]) == pytest.approx(1.0 / 3.0)
    assert ev.rel_infeasibility(inst, [-1.0, 0.0]) > 0.1


def test_reference_and_correct():
    status, obj = ev.reference(S.textbook_instance())
    assert status == "optimal" and obj == pytest.approx(36.0)
    good = ev.pipeline(S.textbook_instance(), False, "none", "textbook", 1e-9, 0.0)
    assert ev.correct(good, "optimal", 36.0) and not ev.correct(good, "infeasible", float("nan")) and not ev.correct(good, "optimal", 35.0)
    assert ev.reference(S.infeasible_instance())[0] == "infeasible" and ev.reference(S.unbounded_instance())[0] == "unbounded"


def test_analyse_compares_plain_and_chosen_pipeline_and_is_cached():
    a = ev.analyse(Settings())
    assert a is ev.analyse(Settings())
    assert a.ref_status == "optimal" and a.plain_ok and a.piped_ok and (a.inst.m, a.inst.n) == (17, 9) and a.dirt["singleton"] == 4
    assert a.piped.rows == 9 and a.piped.cols == 7 and a.piped.gap < 1e-9 and a.piped.pre.status == "reduced"
    bad = ev.analyse(Settings(spread_i=7, presolve_on=False, scaling="none"))
    assert bad.ref_status == "optimal" and not bad.plain_ok and not bad.piped_ok and bad.piped.rel_infeas > 0.05
    fixed = ev.analyse(Settings(spread_i=7, presolve_on=False, scaling="pow2"))
    assert fixed.piped_ok and not fixed.plain_ok and fixed.piped.rel_infeas < 1e-9
    inf = ev.analyse(Settings("infeasible"))
    assert inf.ref_status == "infeasible" and inf.piped.status == "infeasible" and inf.piped_ok and inf.piped.pre.status == "infeasible"
    unb = ev.analyse(Settings("unbounded"))
    assert unb.piped.status == "unbounded" and unb.piped_ok


def test_analyse_accepts_the_size_of_the_perturbation():
    a = ev.analyse(Settings("ties", 24, 24, 35, presolve_on=False, scaling="none", perturb_i=2))
    assert a.piped_ok and a.piped.zero_steps == 0 and a.piped.basis_ok and 1e-9 < a.piped.rel_infeas <= 2e-6


def test_presolve_sweep_shrinks_dirty_instances_and_keeps_correctness():
    rows = ev.presolve_sweep(Settings())
    assert [r["red"] for r in rows] == list(C.REDUNDANCIES) and all(r["ok"] == 5 for r in rows)
    assert rows[0]["rows_before"] == rows[0]["rows_after"] == 6 and rows[0]["pivots_before"] == rows[0]["pivots_after"]
    assert all(r["rows_after"] < r["rows_before"] for r in rows[1:]) and rows[-1]["rows_after"] <= 12 and rows[-1]["pivots_after"] < rows[-1]["pivots_before"]
    assert all(r["nnz_after"] <= r["nnz_before"] for r in rows) and [r["rows_before"] for r in rows] == sorted(r["rows_before"] for r in rows)


def test_scaling_sweep_geometric_and_powers_of_two_hold_the_others_fall_out():
    rows = {r["k"]: r for r in ev.scaling_sweep(Settings())}
    assert [r for r in rows] == list(C.SPREAD_EXPS)
    assert all(rows[k]["pow2"]["simplex"] == 5 and rows[k]["geometric"]["simplex"] == 5 for k in rows)
    assert all(rows[0][m]["simplex"] == 5 and rows[0][m]["ipm"] == 5 for m in SC.METHODS)
    assert rows[16]["none"]["simplex"] <= 4 and rows[16]["none"]["ipm"] <= 3 and rows[12]["equilibrate"]["ipm"] <= 3 and all(rows[k]["pow2"]["ipm"] >= 4 for k in rows)
    assert rows[16]["none"]["spread"] > 20 and rows[16]["pow2"]["spread"] < 2 and rows[16]["equilibrate"]["spread"] > 5


def test_ratio_sweep_shows_the_harris_trap_and_its_tolerance_limit():
    rows = {r["tol"]: r for r in ev.ratio_sweep(Settings())}
    assert list(rows) == list(C.TOLS)
    assert rows[1e-8]["textbook"]["bad"] >= 3 and rows[1e-9]["textbook"]["bad"] >= 1 and rows[1e-10]["textbook"]["bad"] >= 1
    assert all(rows[t]["harris"]["bad"] == 0 for t in (1e-6, 1e-8, 1e-9, 1e-10)) and rows[1e-12]["harris"]["bad"] >= 1
    assert rows[1e-8]["harris"]["min_pivot"] > 0.5 and rows[1e-8]["textbook"]["min_pivot"] < 1e-3 and rows[1e-8]["textbook"]["worst"] > 1e-3 and rows[1e-8]["harris"]["worst"] < 1e-6


def test_stall_sweep_perturbation_removes_zero_steps():
    rows = {r["n"]: r for r in ev.stall_sweep(Settings())}
    assert list(rows) == list(C.STALL_SIZES)
    assert all(rows[n][0.0]["zero"] >= 1 and rows[n][1e-7]["zero"] == 0 and rows[n][1e-6]["zero"] == 0 for n in rows) and rows[40][0.0]["zero"] > rows[8][0.0]["zero"]
    assert all(rows[n][1e-6]["basis_ok"] >= 4 and 1e-7 < rows[n][1e-6]["err"] < 1e-5 and rows[n][0.0]["err"] == 0.0 for n in rows)


def test_log_table_and_rules_count():
    pre = PS.presolve(S.dirty_textbook_instance())
    table = ev.log_table(pre)
    assert len(table) == 6 and table[0][0] == "Singleton-Zeile" and all(isinstance(t[1], str) and t[1] for t in table)
    assert sum(ev.rules_count(pre).values()) == 6 and ev.gap_ok(ev.pipeline(S.dirty_textbook_instance(), True, "none", "textbook", 1e-9, 0.0)) and not ev.gap_ok(ev.Pipe("optimal"))
    assert math.isnan(ev.Pipe("optimal").gap) and np.isinf(ev.Pipe("optimal").min_pivot_rel)
    assert X.RATIOS == ("textbook", "harris")
