"""Presets: gültige Werte und jede Zahl der Hilfetexte gegen die echten Auswertungsfunktionen (Numerik-Grenzfälle nur als Bänder: Windows und Linux können abweichen)."""

import pytest

import pre_constants as C
import pre_evaluation as ev
import pre_presolve as PS
import pre_scenario as S
import pre_simplex as X
from pre_evaluation import Settings
from pre_presets import PRESET_KEYS, SETTING_SPECS


def near(x, want, rel=0.1, tol=1):
    return abs(x - want) <= max(rel * abs(want), tol)


def _settings(name, **over):
    p = {**C.PRESETS[name], **over}
    return Settings(p["kind"], p["m"], p["n"], p["seed"], p["spread"], p["red"], p["pre"], p["scaling"], p["ratio"], p["tol"], p["perturb"])


def _has(name, *values):
    for v in values:
        assert v in C.PRESET_HELP[name], (name, v)


def test_every_preset_has_valid_values_and_a_help_text():
    assert list(C.PRESETS) == list(C.PRESET_HELP) and len(C.PRESETS) == 11
    for name, p in C.PRESETS.items():
        assert set(p) <= set(PRESET_KEYS) and {"kind", "step", "spread", "red", "pre", "scaling", "ratio", "tol", "perturb"} <= set(p), name
        for key, state_key in PRESET_KEYS.items():
            if key in p and state_key in SETTING_SPECS:
                spec = SETTING_SPECS[state_key]
                assert spec.caster(str(int(p[key])) if isinstance(p[key], bool) else p[key]) == p[key], (name, key)
                if spec.lo is not None:
                    assert spec.lo <= p[key] <= spec.hi, (name, key)
        assert C.PRESET_HELP[name].strip()


def test_help_presolve_presets():
    name = "Lehrbuch schmutzig: Präsolve von Hand"
    a = ev.analyse(_settings(name))
    pre = a.piped.pre
    assert ev.rules_count(pre) == {"leere Zeile": 1, "Singleton-Zeile": 3, "redundante Zeile": 1, "doppelte Zeile": 1, "feste Variable": 0, "leere Spalte": 0}
    assert (pre.reduced.m, pre.reduced.n) == (3, 2) and pre.removed_rows == 6 and a.piped.obj == pytest.approx(36.0) and a.piped.gap < 1e-9
    assert "größte Aktivität 10 ≤ 100" in " ".join(e["text"] for e in pre.log)
    _has(name, "6 Zeilen", "1 leere", "größter Aktivität 10 ≤ 100", "x1 ≤ 4, x2 ≤ 6 und x1 ≤ 3", "3 × 2", "(2, 6)", "36")
    name = "Schmutziges LP: Präsolve räumt auf"
    a = ev.analyse(_settings(name))
    assert ev.rules_count(a.piped.pre) == {"leere Zeile": 1, "Singleton-Zeile": 7, "redundante Zeile": 1, "doppelte Zeile": 3, "feste Variable": 1, "leere Spalte": 1}
    assert (a.inst.m, a.inst.n) == (17, 9) and (a.piped.rows, a.piped.cols) == (9, 7) and near(a.plain.pivots, 10, tol=1) and near(a.piped.pivots, 5, tol=1) and a.piped.gap < 1e-9 and a.piped_ok
    _has(name, "6 × 8", "17 × 9", "1 leere Zeile, 7 Singleton-Zeilen", "1 redundante und 3 doppelte", "1 feste Variable", "1 leere Spalte", "9 × 7", "5 statt 10 Pivots", "c·x = b·y")
    name = "Fast alles ist Schmutz"
    a = ev.analyse(_settings(name))
    assert ev.rules_count(a.piped.pre) == {"leere Zeile": 2, "Singleton-Zeile": 17, "redundante Zeile": 4, "doppelte Zeile": 6, "feste Variable": 3, "leere Spalte": 2}
    assert (a.inst.m, a.inst.n) == (31, 10) and (a.piped.rows, a.piped.cols) == (7, 5) and near(a.plain.pivots, 13, tol=1) and near(a.piped.pivots, 3, tol=1)
    _has(name, "31 × 10", "7 × 5", "17 Singleton-Zeilen, 6 doppelte, 4 redundante und 2 leere", "3 feste Variablen", "13 auf 3")
    name = "Präsolve beweist die Unzulässigkeit"
    a = ev.analyse(_settings(name))
    assert a.piped.status == "infeasible" and a.piped.pre.status == "infeasible" and "untere Schranke 6 liegt über der oberen 4" in a.piped.pre.note
    _has(name, "x1 ≤ 4 und x1 ≥ 6", "untere Schranke 6 liegt über der oberen 4")


def test_help_scaling_presets():
    name = "Spanne 10^14: ohne Skalierung falsch"
    a = ev.analyse(_settings(name))
    assert not a.plain_ok and not a.piped_ok and 0.1 < a.piped.rel_infeas < 1.0 and a.piped.min_pivot_rel < 1e-10
    _has(name, "10^14", "0.42", "4e-14")
    name = "Zweierpotenzen reparieren es"
    a = ev.analyse(_settings(name))
    assert a.piped_ok and near(a.piped.pivots, 6, tol=1) and 0.05 < a.piped.min_pivot_rel < 0.5 and a.piped.rel_infeas < 1e-12
    _has(name, "6 Pivots", "0.16", "7e-17")
    name = "Mehrotra jammt ohne Skalierung"
    a = ev.analyse(_settings(name))
    assert a.ipm_plain.status != "optimal" and a.ipm_scaled.status == "optimal" and near(a.ipm_scaled.iterations, 8, tol=3) and a.plain_ok and near(a.plain.pivots, 10, tol=1)
    _has(name, "10^10", "57 Iterationen", "8 Iterationen", "10 Pivots")


def test_help_ratio_and_stall_presets():
    name = "Nadel: der einfache Test liefert Unzulässiges"
    a = ev.analyse(_settings(name))
    assert not a.piped_ok and 1e-3 < a.piped.rel_infeas < 1.0 and a.piped.min_pivot_rel < 1e-4
    _has(name, "10 × 10 (Seed 6)", "1e-8", "1.6e-6", "1.6e-2")
    name = "Harris hält die Nadel"
    a = ev.analyse(_settings(name))
    assert a.piped_ok and a.piped.min_pivot_rel >= 0.5 and a.piped.rel_infeas <= 1e-7
    _has(name, "Harris", "1.0", "6e-9")
    name = "Harris braucht eine Lockerung"
    a = ev.analyse(_settings(name))
    assert not a.piped_ok and a.piped.zero_steps >= 3 and 1e-8 < a.piped.rel_infeas < 1e-2
    _has(name, "1e-12", "1e-8", "5.7e-4", "7 Nullschritte", "2.4e-6")
    name = "Entartete Familie: Störung beendet die Nullschritte"
    a = ev.analyse(_settings(name))
    plain = X.simplex(a.inst)
    assert near(plain.zero_steps, 8, tol=3) and a.piped.zero_steps == 0 and a.piped.basis_ok and a.piped_ok and near(plain.pivots, 13, tol=1) and near(a.piped.pivots, 13, tol=1)
    _has(name, "24 × 24", "8 Nullschritte bei 13 Pivots", "1e-6", "13 Pivots")


def test_the_helper_presolve_classes_are_consistent():
    assert set(PS.RULES) >= {"leere Zeile", "Singleton-Zeile", "redundante Zeile", "doppelte Zeile", "feste Variable", "leere Spalte"}
    assert S.dirty_textbook_instance().m == 7
