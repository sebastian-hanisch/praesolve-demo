"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz, Regler-Randwerte, Permalink-Grenzen, bedingte Regler, Berechnungen auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import pre_constants as C
import pre_scaling as SC
import pre_scenario as S
import pre_simplex as X

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    state.setdefault("pre_step", step)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m for m in at.metric if m.label == label)


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()


def test_default_run_shows_the_messy_lp():
    at = _run()
    _ok(at)
    assert {"Pivots ohne / mit", "Zeilen × Spalten (gelöst)", "Relative Unzulässigkeit", "Ergebnis"} <= {m.label for m in at.metric}
    assert _metric(at, "Ergebnis").value == "richtig" and _metric(at, "Zeilen × Spalten (gelöst)").value == "9 × 7"
    assert any("Ergebnis richtig" in s.value for s in at.success) and at.get("plotly_chart")


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run(kind_select="random")
    _click(at, f"preset_{name}")
    _ok(at)
    p = C.PRESETS[name]
    ss = at.session_state
    assert (ss["kind_select"], ss["spread_select"], ss["red_select"], ss["presolve_toggle"], ss["scaling_select"], ss["ratio_select"], ss["tol_select"], ss["perturb_select"], ss["pre_step"]) == (
        p["kind"], p["spread"], p["red"], p["pre"], p["scaling"], p["ratio"], p["tol"], p["perturb"], p["step"])


@pytest.mark.parametrize("step", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", list(S.KINDS))
def test_every_step_runs_for_every_kind(step, kind):
    at = _run(step=step, kind_select=kind, m_slider=5, n_slider=5)
    _ok(at)
    assert at.session_state["pre_step"] == step


@pytest.mark.parametrize("scaling", list(SC.METHODS))
@pytest.mark.parametrize("ratio", list(X.RATIOS))
def test_every_scaling_and_ratio_runs_on_every_step(scaling, ratio):
    for step in (1, 2, 3, 4):
        _ok(_run(step=step, scaling_select=scaling, ratio_select=ratio, presolve_toggle=True))
    _ok(_run(step=3, scaling_select=scaling, ratio_select=ratio, presolve_toggle=False, kind_select="needle", m_slider=10, n_slider=10))


def test_presolve_step_shows_the_log_and_the_infeasibility_proof():
    at = _run(kind_select="dirty", scaling_select="none")
    _ok(at)
    assert any(list(d.value.columns) == ["Regel", "Was geschah"] and len(d.value) == 6 for d in at.dataframe)
    assert {"Zeilen", "Spalten", "Nichtnullen"} == {m.label for m in at.metric if m.label in ("Zeilen", "Spalten", "Nichtnullen")}
    inf = _run(kind_select="infeasible")
    _ok(inf)
    assert any("unzulässig" in i.value and "ohne dass ein Simplex läuft" in i.value for i in inf.info)
    clean = _run(kind_select="needle", m_slider=10, n_slider=10)
    _ok(clean)
    assert any("keine Regel" in i.value for i in clean.info)


def test_scaling_and_ratio_messages_reflect_the_settings():
    bad = _run(step=2, spread_select=7, presolve_toggle=False, scaling_select="none")
    _ok(bad)
    assert any("Ergebnis falsch" in w.value for w in bad.warning) and _metric(bad, "Ergebnis").value == "falsch"
    good = _run(step=2, spread_select=7, presolve_toggle=False, scaling_select="pow2")
    _ok(good)
    assert _metric(good, "Ergebnis").value == "richtig"
    needle = _run(step=3, kind_select="needle", m_slider=10, n_slider=10, seed_input=6, spread_select=4, presolve_toggle=False, scaling_select="none", tol_select=1, ratio_select="harris")
    _ok(needle)
    assert _metric(needle, "Ergebnis").value == "richtig" and any(list(d.value.columns)[0] == "Quotiententest" for d in needle.dataframe)


def test_on_demand_experiments():
    at = _run(step=1)
    _click(at, "presolve_start")
    _ok(at)
    assert len(at.get("plotly_chart")) == 2
    two = _run(step=2)
    _click(two, "scaling_start")
    _ok(two)
    assert len(two.get("plotly_chart")) == 2 and any("Spanne" in d.value.columns for d in two.dataframe)
    three = _run(step=3)
    _click(three, "ratio_start")
    _ok(three)
    assert any("Einfach: falsch" in d.value.columns for d in three.dataframe)
    four = _run(step=4)
    _click(four, "stall_start")
    _ok(four)
    assert any("Nullschritte ohne Störung" in d.value.columns for d in four.dataframe)


@pytest.mark.parametrize("kw", [dict(kind_select="messy", m_slider=C.M_MIN, n_slider=C.N_MIN), dict(kind_select="messy", m_slider=C.M_MAX, n_slider=C.N_MAX), dict(kind_select="needle", m_slider=C.M_MIN, n_slider=C.N_MIN),
                                dict(kind_select="ties", m_slider=C.M_MAX, n_slider=C.N_MAX), dict(kind_select="mixed", m_slider=C.M_MIN, n_slider=C.N_MAX)])
def test_extreme_sizes_run_on_every_step(kw):
    for step in (1, 2, 3, 4):
        _ok(_run(step=step, **kw))


@pytest.mark.parametrize("kw", [dict(spread_select=0), dict(spread_select=len(C.SPREAD_EXPS) - 1), dict(red_select=0), dict(red_select=len(C.REDUNDANCIES) - 1), dict(tol_select=0), dict(tol_select=len(C.TOLS) - 1),
                                dict(perturb_select=len(C.PERTURBS) - 1), dict(spread_select=len(C.SPREAD_EXPS) - 1, scaling_select="none", presolve_toggle=False)])
def test_extreme_controls_on_every_kind(kw):
    for kind in ("dirty", "messy", "needle", "ties", "infeasible", "unbounded"):
        _ok(_run(step=1, kind_select=kind, m_slider=6, n_slider=6, **kw))


def test_dice_button_changes_the_seed():
    at = _run(kind_select="random")
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old and at.session_state["seed_widget"] == at.session_state["seed_input"]


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in dict(m="999", n="1", step="9", kind="nope", spread="99", red="-1", pre="ja", scaling="mittel", ratio="x", tol="99", perturb="-4", seed="-4").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["m_slider"], ss["n_slider"], ss["pre_step"], ss["kind_select"], ss["spread_select"], ss["red_select"], ss["presolve_toggle"], ss["scaling_select"], ss["ratio_select"], ss["tol_select"], ss["perturb_select"],
            ss["seed_input"]) == (C.M_MAX, C.N_MIN, 1, "messy", len(C.SPREAD_EXPS) - 1, 0, True, "pow2", "textbook", len(C.TOLS) - 1, 0, 0)


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in dict(kind="needle", m="10", n="9", seed="7", spread="3", red="1", pre="0", scaling="geometric", ratio="harris", tol="1", perturb="2", step="3").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["kind_select"], ss["m_slider"], ss["n_slider"], ss["seed_input"], ss["spread_select"], ss["red_select"], ss["presolve_toggle"], ss["scaling_select"], ss["ratio_select"], ss["tol_select"], ss["perturb_select"],
            ss["pre_step"]) == ("needle", 10, 9, 7, 3, 1, False, "geometric", "harris", 1, 2, 3)


def test_sidebar_shows_the_controls_that_belong_to_the_instance():
    fixed = _run(kind_select="dirty")
    assert not any(w.key in ("m_widget", "n_widget") for w in fixed.slider) and not any(s.key in ("spread_select", "red_select") for s in fixed.select_slider)
    messy = _run(kind_select="messy")
    assert any(s.key == "spread_select" for s in messy.select_slider) and any(s.key == "red_select" for s in messy.select_slider) and any(w.key == "m_widget" for w in messy.slider)
    needle = _run(kind_select="needle")
    assert any(s.key == "spread_select" for s in needle.select_slider) and not any(s.key == "red_select" for s in needle.select_slider)
    ties = _run(kind_select="ties")
    assert not any(s.key in ("spread_select", "red_select") for s in ties.select_slider) and any(w.key == "n_widget" for w in ties.slider)


def test_changing_kind_and_step_on_later_steps_does_not_crash():
    for step in (1, 2, 3, 4):
        at = _run(step=step)
        _ok(at)
        for kw in (dict(kind_select="mixed", m_slider=8, n_slider=8), dict(kind_select="infeasible"), dict(kind_select="unbounded"), dict(kind_select="needle", m_slider=10, n_slider=10), dict(kind_select="ties", m_slider=12, n_slider=12),
                   dict(kind_select="dirty"), dict(kind_select="random", m_slider=4, n_slider=3), dict(kind_select="messy", presolve_toggle=False)):
            for k, v in kw.items():
                at.session_state[k] = v
            at.run()
            _ok(at)


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Andersen" in m.value and "Harris" in m.value and "Gondzio" in m.value for e in at.expander for m in e.markdown)
