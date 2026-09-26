"""Jede Zahl aus README und App über die echten Auswertungsfunktionen (Zeilen-/Spaltenzahlen aus reiner Logik exakt, Pivots und Numerik-Grenzfälle mit Band: Windows und Linux können abweichen)."""

import pytest

import pre_constants as C
import pre_evaluation as ev
import pre_scaling as SC
from pre_evaluation import Settings


def near(x, want, rel=0.15, tol=1):
    return abs(x - want) <= max(rel * abs(want), tol)


def test_readme_presolve_yield_over_the_dirt_level():
    rows = {r["red"]: r for r in ev.presolve_sweep(Settings())}
    assert [(rows[r]["rows_before"], rows[r]["rows_after"]) for r in C.REDUNDANCIES] == [(6, 6), (13, 8), (17, 10), (24, 11), (31, 9)]
    assert [(rows[r]["cols_before"], rows[r]["cols_after"]) for r in C.REDUNDANCIES] == [(8, 8), (8, 7), (9, 7), (10, 6), (10, 5)]
    assert [(rows[r]["nnz_before"], rows[r]["nnz_after"]) for r in C.REDUNDANCIES] == [(29, 29), (42, 27), (53, 29), (61, 26), (80, 18)]
    assert [(round(rows[r]["pivots_before"]), round(rows[r]["pivots_after"])) for r in C.REDUNDANCIES] == pytest.approx([(5, 5), (5, 3), (6, 4), (10, 4), (12, 3)], abs=1.5)
    assert all(rows[r]["ok"] == 5 for r in rows)


def test_readme_scaling_over_the_spread():
    rows = {r["k"]: r for r in ev.scaling_sweep(Settings())}
    assert [rows[k]["pow2"]["simplex"] for k in rows] == [5] * 9 and [rows[k]["geometric"]["simplex"] for k in rows] == [5] * 9
    assert [rows[k]["none"]["simplex"] for k in (0, 4, 8, 10)] == [5, 5, 5, 5] and all(3 <= rows[k]["none"]["simplex"] <= 4 for k in (12, 14, 16)) and rows[16]["equilibrate"]["simplex"] <= 4
    # Mehrotra (Innere Punkte) löst bei großer Spanne schlecht konditionierte Normalgleichungen: wie viele der fünf Instanzen ohne Skalierung durchkommen, hängt an den Rundungen der Rechenumgebung
    # (Windows und Linux-Docker: 5 / 5 / 3 / 3 / 2 / 1 / 1 / 2 bei k = 0 / 4 / 6 / 8 / 10 / 12 / 14 / 16; die GitHub-CI erreichte bei k = 6 fünf von fünf). Geprüft wird deshalb nur, was robust ist:
    # bei kleiner Spanne alles richtig, bei großer Spanne höchstens die Hälfte, und die Skalierung (Zweierpotenzen, geometrisch) hilft deutlich.
    assert [rows[k]["none"]["ipm"] for k in (0, 4)] == [5, 5] and all(rows[k]["none"]["ipm"] <= 3 for k in (12, 14, 16)) and rows[10]["none"]["ipm"] <= 4
    assert all(rows[k]["pow2"]["ipm"] >= 4 for k in rows) and all(rows[k]["geometric"]["ipm"] >= 4 for k in rows)
    assert sum(rows[k]["pow2"]["ipm"] + rows[k]["geometric"]["ipm"] for k in (10, 12, 14, 16)) >= sum(2 * rows[k]["none"]["ipm"] for k in (10, 12, 14, 16)) + 8
    assert [round(rows[k]["none"]["spread"]) for k in (0, 8, 16)] == pytest.approx([1, 13, 25], abs=2) and all(rows[k]["geometric"]["spread"] < 1.0 for k in rows) and all(rows[k]["pow2"]["spread"] < 1.3 for k in rows)
    assert [round(rows[k]["equilibrate"]["spread"]) for k in (8, 16)] == pytest.approx([7, 12], abs=2)


def test_readme_harris_over_the_tolerance():
    rows = {r["tol"]: r for r in ev.ratio_sweep(Settings())}
    assert rows[1e-6]["textbook"]["bad"] == 0 and rows[1e-6]["harris"]["bad"] == 0
    assert 4 <= rows[1e-8]["textbook"]["bad"] <= 9 and 1 <= rows[1e-9]["textbook"]["bad"] <= 5 and 1 <= rows[1e-10]["textbook"]["bad"] <= 6 and 1 <= rows[1e-12]["textbook"]["bad"] <= 4
    assert [rows[t]["harris"]["bad"] for t in (1e-8, 1e-9, 1e-10)] == [0, 0, 0] and 1 <= rows[1e-12]["harris"]["bad"] <= 4
    assert rows[1e-8]["textbook"]["worst"] > 0.05 and rows[1e-8]["textbook"]["min_pivot"] < 1e-4 and rows[1e-8]["harris"]["min_pivot"] > 0.5 and rows[1e-10]["harris"]["worst"] < 1e-6


def test_readme_perturbation_over_the_size():
    rows = {r["n"]: r for r in ev.stall_sweep(Settings())}
    assert [rows[n][0.0]["zero"] for n in C.STALL_SIZES] == pytest.approx([1, 6, 11, 12, 12], abs=3)
    assert all(rows[n][1e-7]["zero"] == 0 and rows[n][1e-6]["zero"] == 0 for n in rows)
    assert [rows[n][0.0]["pivots"] for n in C.STALL_SIZES] == pytest.approx([4, 9, 20, 29, 43], abs=4)
    assert [rows[n][1e-6]["pivots"] for n in C.STALL_SIZES] == pytest.approx([4, 9, 18, 32, 34], abs=4)
    assert all(rows[n][1e-6]["basis_ok"] == 5 for n in rows) and all(1e-7 < rows[n][1e-6]["err"] < 1e-5 for n in rows)


def test_readme_hundred_dirty_instances_keep_optimum_and_certificate():
    ok = 0
    for sd in range(100):
        for red in (0.5,):
            inst = ev.S.messy_instance(6, 8, sd, 0, red)[0]
            status, obj = ev.reference(inst)
            pipe = ev.pipeline(inst, True, "pow2", "textbook", 1e-9, 0.0)
            ok += ev.correct(pipe, status, obj) and ev.gap_ok(pipe, 1e-9)
    assert ok == 100
    assert SC.METHODS == ("none", "equilibrate", "geometric", "pow2")
