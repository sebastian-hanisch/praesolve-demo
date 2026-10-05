"""Unabhängiges Orakel für Präsolve + Postsolve: zufällige LPs mit Singleton-, doppelten (auch negativ skalierten), leeren, redundanten Zeilen, festen Variablen und leeren Spalten,
gemischten Vorzeichen und allen Zeilenarten, gegen HiGHS (Status, Optimalwert) und gegen das Zertifikat auf dem ORIGINAL: primal zulässig, dual zulässig (A^T y >= c, Vorzeichen je
Zeilenart), c·x = b·y. Dazu die erste Pivotwahl der Nadel (kleinstes Pivotelement) aus den Instanzdaten von Hand."""

import random

import numpy as np
import pytest

import pre_evaluation as ev
import pre_scenario as S
import pre_simplex as X

pytest.importorskip("scipy")
from tests.test_scenario import _highs, reference_status  # noqa: E402


def _mk(rows, b, c, sen):
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(sen), tuple(f"x{j}" for j in range(len(c))),
                      tuple(f"r{i}" for i in range(len(b))), "custom")


def _random_lp(rng):
    n = rng.randint(1, 5)
    x0 = [rng.choice([0, 0, 1, 2, 3, 5]) for _ in range(n)]
    feasible = rng.random() < 0.9
    rows, b, sen = [], [], []

    def add(row, sense):
        act = sum(a * x for a, x in zip(row, x0))
        if not feasible and rng.random() < 0.5:
            act += rng.choice([-3, -1, 1, 3])
        elif sense == S.LE:
            act += rng.choice([0, 0, 1, 2, 5])
        elif sense == S.GE:
            act -= rng.choice([0, 0, 1, 2, 5])
        rows.append(row), b.append(act), sen.append(sense)

    for _ in range(rng.randint(1, 4)):
        add([rng.choice([-2, -1, 0, 0, 1, 2, 3]) for _ in range(n)], rng.choice([S.LE, S.LE, S.GE, S.EQ]))
    for _ in range(rng.randint(0, 6)):
        t = rng.random()
        if t < 0.35:                                                                          # Singleton-Zeile
            row = [0] * n
            row[rng.randrange(n)] = rng.choice([-2, -1, 1, 2, 4])
            add(row, rng.choice([S.LE, S.GE, S.EQ]))
        elif t < 0.7:                                                                         # parallele Zeile, auch negativ skaliert
            k = rng.randrange(len(rows))
            f = rng.choice([-3, -1, 0.5, 2, 3])
            add([f * v for v in rows[k]], rng.choice([S.LE, S.GE, S.EQ]))
        elif t < 0.8:
            add([0] * n, rng.choice([S.LE, S.GE, S.EQ]))                                      # leere Zeile
        else:
            add([rng.choice([0, 1, 2]) for _ in range(n)], S.LE)
    if rng.random() < 0.7:
        rows.append([1] * n), b.append(sum(x0) + rng.choice([0, 3, 10])), sen.append(S.LE)
    return _mk(rows, b, [rng.choice([-2, -1, 0, 1, 2, 3]) for _ in range(n)], sen)


def _certificate_errors(inst, pipe, val):
    x, y = np.array(pipe.x), np.array(pipe.y)
    A_, b, c = inst.arrays()
    errs = []
    if abs(pipe.obj - val) > 1e-6 * (1 + abs(val)):
        errs.append("Zielwert")
    act = A_ @ x
    pv = max(0.0, float(-x.min()))
    for i, s in enumerate(inst.senses):
        pv = max(pv, max(0.0, act[i] - b[i]) if s == S.LE else (max(0.0, b[i] - act[i]) if s == S.GE else abs(act[i] - b[i])))
    dv = float(max(0.0, (c - A_.T @ y).max()))
    for i, s in enumerate(inst.senses):
        dv = max(dv, -y[i] if s == S.LE else (y[i] if s == S.GE else 0.0))
    if pv > 1e-6 * (1 + np.abs(b).max()):
        errs.append(("primal", pv))
    if dv > 1e-6 * (1 + np.abs(c).max()):
        errs.append(("dual", dv))
    if abs(float(c @ x) - float(b @ y)) > 1e-6 * (1 + abs(val)):
        errs.append("c.x != b.y")
    return errs


def test_presolve_and_postsolve_on_random_structured_lps_agree_with_highs_and_certify_on_the_original():
    rng = random.Random(777)
    seen = set()
    for k in range(450):
        inst = _random_lp(rng)
        ref = reference_status(inst)
        seen.add(ref)
        for scaling, ratio in (("none", "textbook"), ("pow2", "harris")):
            pipe = ev.pipeline(inst, True, scaling, ratio, 1e-9, 0.0)
            assert pipe.status == ref, (k, scaling, pipe.status, ref)
            if ref == "optimal":
                assert not _certificate_errors(inst, pipe, -_highs(inst).fun), (k, scaling)
    assert seen == {"optimal", "infeasible", "unbounded"}


def test_needle_first_pivot_element_is_the_ratio_test_choice_computed_by_hand():
    for sd in range(100000, 100020):
        inst = S.needle_instance(10, 10, sd, 8)
        A_, b, _c = inst.arrays()
        col = A_[:, 0]                                    # x1 hat den größten Zielkoeffizienten und tritt zuerst ein
        ratios = b / col
        tied = [i for i in range(10) if ratios[i] <= ratios.min() + 1e-8 * max(1.0, abs(ratios.min()))]
        assert X.simplex(inst, tol=1e-8, ratio="textbook").min_pivot_rel == pytest.approx(col[min(tied)] / np.abs(col).max(), rel=1e-12)
