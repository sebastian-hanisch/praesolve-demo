"""Präsolve, Skalierung und Numerik – was echte Löser vor und beim Lösen tun - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Lineare-Programmierung-Reihe der "Konzepte"-Reihe: Die Stücke davor haben gemessen, dass schlecht skalierte Instanzen Simplex, Ellipsoid und Innere Punkte aus dem Tritt bringen. Echte Löser räumen vorher auf
(Präsolve mit Postsolve), skalieren, und der Simplex rechnet mit Toleranzen, Harris-Quotiententest und Störung. Die Demo zeigt jedes davon und misst, was es bringt.

Lauffähig mit: streamlit run app.py
"""

import math

import numpy as np
import pandas as pd
import streamlit as st

import pre_constants as C
import pre_evaluation as ev
import pre_presolve as PS
import pre_scaling as SC
import pre_scenario as S
import pre_simplex as X
from pre_evaluation import Settings, analyse
from pre_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    store_from_widget,
    sync_query_params,
)
from pre_visualization import (
    build_heat,
    build_presolve_bars,
    build_presolve_sweep,
    build_ratio_sweep,
    build_scaling_sweep,
    build_stall,
)

st.set_page_config(page_title="Präsolve und Numerik – Sebastian Hanisch", layout="wide")


def num(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{0.0 if abs(x) < 5e-13 else x:.{digits}f}"


def sci(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.1e}"


STATUS_TEXT = {"optimal": "Optimum", "infeasible": "unzulässig", "unbounded": "unbeschränkt", "limit": "Pivot-Grenze erreicht"}

st.title("🧹 Präsolve, Skalierung und Numerik – was echte Löser vor und beim Lösen tun")
st.markdown(
    """
**Neuntes Stück der Lineare-Programmierung-Reihe.** Die Stücke davor haben mehrfach gemessen: bei Spalten über 10¹⁰ liefert der Simplex still falsche Werte, das Ellipsoid bricht ab, Langschritt und Affine Scaling stehen still. Echte Modelle sind nicht sauber: sie enthalten
doppelte und redundante Zeilen, Schranken als Zeilen, feste Variablen und Einheiten von Stück bis Tonne. Löser bekämpfen das **vor** dem Lösen (**Präsolve** mit **Postsolve**, **Skalierung**) und rechnen im Simplex mit **Toleranzen**, dem **Harris-Quotiententest** und **Störung** gegen
Stillstand. Vier Fragen, alle gemessen: **(1) Präsolve** - wie viel schrumpft ein schmutziges LP, und gibt das Postsolve gültige Lösung und Duale zurück? **(2) Skalierung** - repariert sie die Ausfälle? **(3) Toleranzen und Harris** - wann liefert der einfache Quotiententest
Unzulässiges? **(4) Entartung und Störung** - hilft eine Störung gegen Nullschritte?
"""
)
st.caption("Kind des [Revised Simplex](https://github.com/sebastian-hanisch/revised-simplex-demo); greift die Befunde aus [Ellipsoid](https://github.com/sebastian-hanisch/ellipsoid-demo) und [Innere Punkte](https://github.com/sebastian-hanisch/innere-punkte-demo) auf. Folgestücke (PDLP, Crossover) sind [noch nicht gebaut].")

with st.expander("Was Präsolve, Skalierung und die Toleranzen tun", expanded=True):
    st.markdown(
        """
1. **Präsolve** wendet einfache, beweisbar sichere Regeln bis zum Fixpunkt an: **leere Zeile** streichen (oder Widerspruch), **Singleton-Zeile** (nur eine Variable) wird zur **Schranke**, **redundante Zeile** (nie bindend nach den Schranken) streichen, **doppelte oder parallele Zeile** (die strengere bleibt), **feste Variable** einsetzen, **leere Spalte** an ihre Schranke setzen. Ein Widerspruch beweist die Unzulässigkeit, ohne dass ein Simplex läuft.
2. **Postsolve** rechnet Lösung **und Duale** auf das Original zurück; die Probe ist c·x = b·y auf dem Original.
3. **Skalierung** multipliziert Zeilen mit r_i und Spalten mit t_j (x = t·x'), damit die Einträge der Matrix ähnlich groß werden. **Zweierpotenzen** rechnen exakt.
4. **Quotiententest:** der einfache Test wählt das kleinste Verhältnis rhs/Spalteneintrag; bei fast gleichen Verhältnissen kann das ein winziges Pivotelement sein (große Fehler). **Harris** lässt eine kleine Verletzung δ zu und wählt unter den fast gleichen Kandidaten das **größte** Pivotelement.
5. **Störung:** kleine zufällige Lockerung der rechten Seite beendet die Gleichstände (Nullschritte); danach prüft man, ob die Basis auch für das Original zulässig ist.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:8], preset_names[8:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.selectbox("Instanz", options=list(S.KINDS), format_func=lambda v: S.KIND_LABELS[v], key="kind_select",
                        help="Lehrbuch, Zentrum und entartete Ecke sind fest; Schmutzig, Nadel und Entartete Familie sind erzeugt; Zufall und Mischung sind sauber. Unzulässig zeigt den Widerspruch im Präsolve.")
    sized = kind not in S.FIXTURE_KINDS
    if sized:
        m = st.slider("Zeilen m (Kern)", *bounds("m_slider"), value=int(ss["m_slider"]), key="m_widget", on_change=store_from_widget, args=("m_slider",), help="Zahl der Bedingungen des sauberen Kerns; Schmutz kommt dazu.")
        n = st.slider("Dienste n (Kern)", *bounds("n_slider"), value=int(ss["n_slider"]), key="n_widget", on_change=store_from_widget, args=("n_slider",), help="Zahl der Variablen des sauberen Kerns.")
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        m, n, seed = C.DEFAULT_M, C.DEFAULT_N, C.DEFAULT_SEED
    if kind in ("messy", "needle"):
        spread_i = st.select_slider("Einheiten-Spanne" if kind == "messy" else "Nadelspanne", options=list(range(len(C.SPREAD_EXPS))), format_func=C.spread_label, key="spread_select",
                                    help="Schmutzig: Zeilen und Spalten mit Faktoren 10^(±k/2) umgerechnet (Stück, Tonne, Euro, Tausend Euro). Nadel: kleinste Koeffizienten der Spalte 1 bis 10^-k.")
    else:
        spread_i = int(ss["spread_select"])
    if kind == "messy":
        red_i = st.select_slider("Verschmutzungsgrad", options=list(range(len(C.REDUNDANCIES))), format_func=C.red_label, key="red_select",
                                 help="Anteil doppelter, dominierter, leerer und erzwungen redundanter Zeilen, Singleton-Zeilen, fester Variablen und leerer Spalten relativ zum Kern.")
    else:
        red_i = int(ss["red_select"])
    presolve_on = st.toggle("Präsolve", key="presolve_toggle", help="Vor dem Lösen aufräumen, danach Postsolve auf das Original.")
    scaling = st.radio("Skalierung", options=list(SC.METHODS), format_func=lambda v: SC.METHOD_LABELS[v], key="scaling_select")
    ratio = st.radio("Quotiententest", options=list(X.RATIOS), format_func=lambda v: X.RATIO_LABELS[v], key="ratio_select")
    tol_i = st.select_slider("Toleranz", options=list(range(len(C.TOLS))), format_func=C.tol_label, key="tol_select", help="Pivot-, Optimalitäts- und Verhältnis-Toleranz; bei Harris auch die zugelassene Verletzung δ.")
    perturb_i = st.select_slider("Störung von b", options=list(range(len(C.PERTURBS))), format_func=C.perturb_label, key="perturb_select", help="Relative Lockerung der rechten Seiten gegen Stillstand.")

sync_query_params({"kind_select": kind, "m_slider": int(ss["m_slider"]), "n_slider": int(ss["n_slider"]), "seed_input": int(ss["seed_input"]), "spread_select": int(spread_i), "red_select": int(red_i),
                   "presolve_toggle": int(bool(presolve_on)), "scaling_select": scaling, "ratio_select": ratio, "tol_select": int(tol_i), "perturb_select": int(perturb_i), "pre_step": int(ss["pre_step"])})

settings = Settings(kind, int(m), int(n), int(seed), int(spread_i), int(red_i), bool(presolve_on), scaling, ratio, int(tol_i), int(perturb_i))
with st.spinner("Rechne..."):
    a = analyse(settings)
inst, plain, piped = a.inst, a.plain, a.piped

# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Aufräumen, skalieren, vorsichtig rechnen")
step = st.select_slider("Schritt", options=list(C.STEPS), key="pre_step", format_func=lambda s: C.STEPS[s])

if a.piped_ok:
    st.success(f"✅ Ergebnis richtig: **{STATUS_TEXT.get(piped.status, piped.status)}**" + (f" {num(a.ref_obj)}" if piped.status == "optimal" else "") + f" (Referenz: Zweierpotenz-Skalierung; Instanz {inst.m} × {inst.n}).")
else:
    if piped.status == "optimal" and a.ref_status == "optimal":
        st.warning(f"⚠️ Ergebnis falsch: Wert {num(piped.obj)} gegen Referenz {num(a.ref_obj)}, relative Unzulässigkeit {sci(piped.rel_infeas)}. Mit diesen Einstellungen trägt die Numerik nicht.")
    else:
        st.warning(f"⚠️ Ergebnis falsch: {STATUS_TEXT.get(piped.status, piped.status)} gegen Referenz {STATUS_TEXT.get(a.ref_status, a.ref_status)}.")
if not a.plain_ok and piped.status != "":
    st.caption("Ohne alles (kein Präsolve, keine Skalierung, einfacher Test, Toleranz 1e-9) wäre das Ergebnis auf dieser Instanz falsch.")

if step == 1:
    pre = PS.presolve(inst)
    if pre.status == "infeasible":
        st.info(f"Der Präsolve beweist die Instanz als **unzulässig**, ohne dass ein Simplex läuft: {pre.note}.")
        st.dataframe(pd.DataFrame(ev.log_table(pre), columns=["Regel", "Was geschah"]), hide_index=True, width="stretch")
    else:
        red = pre.reduced
        nnz_before = int(np.count_nonzero(inst.arrays()[0]))
        nnz_after = int(np.count_nonzero(red.arrays()[0])) if red.m and red.n else 0
        before = {"rows": inst.m, "cols": inst.n, "nnz": nnz_before}
        after = {"rows": red.m, "cols": red.n, "nnz": nnz_after}
        c1, c2, c3 = st.columns(3)
        c1.metric("Zeilen", f"{inst.m} → {red.m}", delta=f"{pre.removed_rows} entfernt", delta_color="off")
        c2.metric("Spalten", f"{inst.n} → {red.n}", delta=f"{pre.removed_cols} entfernt", delta_color="off")
        c3.metric("Nichtnullen", f"{nnz_before} → {nnz_after}", delta="Matrix", delta_color="off")
        st.plotly_chart(build_presolve_bars(before, after), width="stretch", key="s1_bars")
        st.caption(f"Das reduzierte LP enthält endliche Variablenschranken als Zeilen ({len(pre.bound_cols)} Schranken-Zeilen), weil der Löser dieser Demo keine Schranken kennt; echte Löser behandeln sie direkt.")
        counts = ev.rules_count(pre)
        st.markdown("**Angewandte Regeln:** " + ", ".join(f"{r}: {k}" for r, k in counts.items() if k) + (f" (Verschmutzung erzeugt: {a.dirt})" if a.dirt else ""))
        if pre.log:
            st.dataframe(pd.DataFrame(ev.log_table(pre), columns=["Regel", "Was geschah"]), hide_index=True, width="stretch")
        else:
            st.info("Auf dieser Instanz greift keine Regel.")
        st.markdown(f"**Probe am Original:** Optimalwert nach Postsolve {num(piped.obj)} gegen Referenz {num(a.ref_obj)}; **c·x − b·y** = {sci(piped.gap * (1.0 + abs(piped.obj))) if piped.status == 'optimal' else '-'} (Duale zurückgerechnet); relative Unzulässigkeit {sci(piped.rel_infeas)}; "
                    f"Pivots: {plain.pivots} ohne, {piped.pivots} mit den gewählten Einstellungen.")
    st.markdown("**Über den Verschmutzungsgrad** (🔬 auf Abruf; Kern m × n aus dem Regler, 5 feste Instanzen):")
    tok = (settings.m, settings.n)
    if st.button("Ausbeute über den Verschmutzungsgrad berechnen", key="presolve_start"):
        ss["presolve_done"] = tok
    if ss.get("presolve_done") == tok:
        with st.spinner("Rechne..."):
            rows = ev.presolve_sweep(settings)
        st.plotly_chart(build_presolve_sweep(rows), width="stretch", key="s1_sweep")
        st.dataframe(pd.DataFrame([{"Verschmutzung": f"{r['red']:.0%}", "Zeilen": f"{r['rows_before']:.0f} → {r['rows_after']:.0f}", "Spalten": f"{r['cols_before']:.0f} → {r['cols_after']:.0f}",
                                     "Nichtnullen": f"{r['nnz_before']:.0f} → {r['nnz_after']:.0f}", "Pivots": f"{r['pivots_before']:.0f} → {r['pivots_after']:.0f}", "richtig": f"{r['ok']} von 5"} for r in rows]), hide_index=True, width="stretch")
        st.caption("Median über fünf feste Instanzen; 'richtig' heißt: Optimalwert und Zulässigkeit stimmen nach Postsolve mit der Referenz.")
elif step == 2:
    sc = SC.scale(inst, scaling)
    st.plotly_chart(build_heat(inst, sc.inst, SC.METHOD_LABELS[scaling]), width="stretch", key="s2_heat")
    st.caption("log10 der Beträge aller Nichtnullen (gleiche Farbskala); je einheitlicher die Farbe, desto besser skaliert.")
    st.dataframe(pd.DataFrame([{"Verfahren": SC.METHOD_LABELS[mt], "Spanne der Matrix (Zehnerpotenzen)": f"{SC.spread(SC.scale(inst, mt).inst.arrays()[0]):.1f}"} for mt in SC.METHODS]), hide_index=True, width="stretch")
    plain_ipm, scaled_ipm = a.ipm_plain, a.ipm_scaled
    st.markdown(f"**Simplex:** ohne alles {'richtig' if a.plain_ok else 'FALSCH'} (relative Unzulässigkeit {sci(plain.rel_infeas)}); mit den gewählten Einstellungen {'richtig' if a.piped_ok else 'FALSCH'}. "
                f"**Mehrotra** (Stück 8): unbearbeitet {STATUS_TEXT.get(plain_ipm.status, plain_ipm.status) if plain_ipm.status in STATUS_TEXT else 'Verdacht/Abbruch'} nach {plain_ipm.iterations} Iterationen, mit Zweierpotenz-Skalierung {STATUS_TEXT.get(scaled_ipm.status, 'Verdacht/Abbruch')} nach {scaled_ipm.iterations}.")
    st.markdown("**Über die Einheiten-Spanne** (🔬 auf Abruf; dieselben 5 schmutzigen Instanzen, Spanne 10^0 bis 10^16):")
    tok = (settings.m, settings.n, settings.red_i)
    if st.button("Skalierung über die Spanne testen", key="scaling_start"):
        ss["scaling_done"] = tok
    if ss.get("scaling_done") == tok:
        with st.spinner("Rechne..."):
            rows = ev.scaling_sweep(settings)
        st.plotly_chart(build_scaling_sweep(rows), width="stretch", key="s2_sweep")
        st.dataframe(pd.DataFrame([{"Spanne": f"10^{r['k']}", **{f"{C.METHOD_SHORT[mt]} (Simplex / Mehrotra)": f"{r[mt]['simplex']} / {r[mt]['ipm']} von 5" for mt in SC.METHODS}} for r in rows]), hide_index=True, width="stretch")
        st.caption("Richtig: Optimalwert wie die Referenz und relativ zulässig. Ohne Skalierung und mit Equilibrierung fallen Simplex und Mehrotra bei großer Spanne aus; die geometrische Skalierung und die Zweierpotenzen halten.")
elif step == 3:
    st.markdown("**Beide Quotiententests auf dieser Instanz** (mit den gewählten Einstellungen für Präsolve, Skalierung und Toleranz):")
    rows = []
    for rt in X.RATIOS:
        r = ev.pipeline(inst, settings.presolve_on, settings.scaling, rt, settings.tol, settings.perturb, settings.seed)
        ok = ev.correct(r, a.ref_status, a.ref_obj)
        rows.append({"Quotiententest": X.RATIO_LABELS[rt], "Pivots": r.pivots, "kleinstes Pivotelement (relativ zur Spalte)": sci(r.min_pivot_rel), "relative Unzulässigkeit": sci(r.rel_infeas), "Ergebnis": "richtig" if ok else "FALSCH"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Auf der Nadel-Instanz (Regler 'Instanz') findet der einfache Test bei Toleranzen um 1e-8 ein winziges Pivotelement und liefert eine unzulässige Lösung; Harris wählt das größte unter den fast gleichen Verhältnissen.")
    st.markdown("**Über die Toleranz** (🔬 auf Abruf; Nadel-Familie 10 × 10 mit Spanne 8, 10 feste Instanzen):")
    if st.button("Quotiententest über die Toleranz vergleichen", key="ratio_start"):
        ss["ratio_done"] = True
    if ss.get("ratio_done"):
        with st.spinner("Rechne..."):
            rs = ev.ratio_sweep(settings)
        st.plotly_chart(build_ratio_sweep(rs), width="stretch", key="s3_sweep")
        st.dataframe(pd.DataFrame([{"Toleranz": f"{r['tol']:.0e}", "Einfach: falsch": f"{r['textbook']['bad']} von 10", "Einfach: kleinstes Pivotelement": sci(r["textbook"]["min_pivot"]), "Harris: falsch": f"{r['harris']['bad']} von 10",
                                     "Harris: kleinstes Pivotelement": sci(r["harris"]["min_pivot"])} for r in rs]), hide_index=True, width="stretch")
        st.caption("Falsch: unzulässige Lösung (relative Verletzung über 1e-6) oder anderer Optimalwert. Harris braucht eine Toleranz, die die Streuung der Verhältnisse überdeckt; bei 1e-12 ist sie kleiner als die Streuung, und beide Tests scheitern.")
else:
    st.markdown("**Störung auf dieser Instanz** (mit den gewählten Einstellungen für Präsolve, Skalierung, Quotiententest und Toleranz):")
    rows = []
    for pi, pert in enumerate(C.PERTURBS):
        r = ev.pipeline(inst, settings.presolve_on, settings.scaling, settings.ratio, settings.tol, pert, settings.seed)
        slack = max(1.0, 10.0 * pert / 1e-6)
        ok = ev.correct(r, a.ref_status, a.ref_obj, obj_tol=1e-5 * slack, infeas_tol=1e-6 * slack)
        rows.append({"Störung": C.perturb_label(pi), "Pivots": r.pivots, "Nullschritte": r.zero_steps, "Endbasis auch für das Original zulässig": "ja" if r.basis_ok else "nein", "Ergebnis": "richtig (im Rahmen der Störung)" if ok else "FALSCH"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Nullschritte sind Pivots ohne Zielfortschritt (Entartung). Die Störung lockert die rechten Seiten um 1e-7 bzw. 1e-6 (relativ); die Lösung verletzt das Original dann um bis zu diese Größenordnung, die Endbasis ist meist auch für das Original zulässig.")
    st.markdown("**Über die Größe** (🔬 auf Abruf; entartete Familie m = n, 5 feste Instanzen):")
    if st.button("Stillstand über die Größe messen", key="stall_start"):
        ss["stall_done"] = True
    if ss.get("stall_done"):
        with st.spinner("Rechne..."):
            rows = ev.stall_sweep(settings)
        st.plotly_chart(build_stall(rows), width="stretch", key="s4_sweep")
        st.dataframe(pd.DataFrame([{"n": r["n"], "Nullschritte ohne Störung": f"{r[0.0]['zero']:.0f}", "mit 1e-7": f"{r[1e-7]['zero']:.0f}", "Pivots ohne / mit 1e-6": f"{r[0.0]['pivots']:.0f} / {r[1e-6]['pivots']:.0f}",
                                     "Basis für das Original zulässig": f"{r[1e-6]['basis_ok']} von 5", "größter Wertfehler (1e-6)": sci(r[1e-6]["err"])} for r in rows]), hide_index=True, width="stretch")
        st.caption("Die Bland-Notbremse gegen Zyklen ist eingeschaltet (in den Stücken davor auch); auf diesen Instanzen gibt es Nullschritte, aber keine langen Stillstände.")

st.markdown("---")
st.markdown("## ⚙️ Der gewählte Fall")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Pivots ohne / mit", f"{plain.pivots} / {piped.pivots}", delta="einfacher Simplex / gewählt", delta_color="off")
m2.metric("Zeilen × Spalten (gelöst)", f"{piped.rows} × {piped.cols}" if presolve_on else f"{inst.m} × {inst.n}", delta=f"Original {inst.m} × {inst.n}", delta_color="off")
m3.metric("Relative Unzulässigkeit", sci(piped.rel_infeas), delta="am Original", delta_color="off")
m4.metric("Ergebnis", "richtig" if a.piped_ok else "falsch", delta=STATUS_TEXT.get(piped.status, piped.status), delta_color="off")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Präsolve ist immer ein Gewinn.** | Auf sauberen Instanzen greift keine Regel; auf schmutzigen schrumpft das LP stark, das reduzierte LP enthält aber die Schranken als Zeilen, weil dieser Löser keine Schranken kennt. | Bound-Simplex |
| **Präsolve und Postsolve sind vollständig.** | Gebaut sind sechs Regeln; Doubleton-Aggregation, dominierte Spalten, Probing und Präsolve am Dual fehlen, ebenso ein Postsolve für die Basis. Erzwingende Zeilen (alle Variablen an einer Schranke) werden nicht behandelt. | Echte Löser |
| **Skalierung genügt.** | Sie repariert die Ausfälle bei großer Spanne, ändert aber nichts an schlecht bedingten Basen oder an Entartung. Die Equilibrierung mit drei Sweeps reicht bei sehr großer Spanne nicht. | Präsolve, dualer Simplex |
| **Harris löst das Toleranzproblem.** | Nur wenn die Toleranz die Streuung der fast gleichen Verhältnisse überdeckt; bei zu kleiner Toleranz scheitert es wie der einfache Test, und gegen schlechte Skalierung hilft es nicht. | Skalierung |
| **Störung beseitigt den Stillstand.** | Sie beendet die Nullschritte auf der entarteten Familie, verändert aber den Zielwert um ihre Größenordnung und verlangt eine Bereinigung; hier wird nur geprüft, ob die Basis für das Original zulässig bleibt. | Bereinigung |
| **Synthetischer Schmutz ist echter Schmutz.** | Die Instanzen sind erzeugt (bekannter Schmutz), keine echten Modelle aus Netlib oder MIPLIB. | Reale Sammlungen |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Singleton-Zeile** $a x_j \le b$: $x_j \le b/a$ ($a > 0$) bzw. $x_j \ge b/a$ ($a < 0$). **Redundant:** kleinste und größte Aktivität $\min/\max\ a^\top x$ über die Schranken; $\max \le b$ bei $\le$ heißt nie bindend, $\min > b$ heißt unzulässig.
**Postsolve der Duale:** für eine entfernte Schranken-Zeile $s$ zur Variable $j$ gilt $y_s = -r_j / a_{sj}$ mit $r_j = \sum_i a_{ij} y_i - c_j$ (nach den bereits bestimmten Dualen); alle übrigen entfernten Zeilen haben $y = 0$. Probe: $c^\top x = b^\top y$.
**Skalierung:** $A' = \mathrm{diag}(r) A\, \mathrm{diag}(t)$, $b' = r b$, $c' = t c$; Rückrechnung $x = t x'$, $y = r y'$. **Harris:** Schritt 1: $\theta_{\max} = \min_i (\beta_i + \delta)/\alpha_i$; Schritt 2: unter $\beta_i/\alpha_i \le \theta_{\max}$ das größte $\alpha_i$ (nur $\alpha_i > \text{tol}$).

**Literatur.** Andersen, E. D., & Andersen, K. D. (1995). *Presolving in linear programming.* Mathematical Programming 71, 221-245. Gondzio, J. (1997). *Presolve analysis of linear programs prior to applying an interior point method.* INFORMS Journal on Computing 9(1), 73-91.
Harris, P. M. J. (1973). *Pivot selection methods of the Devex LP code.* Mathematical Programming 5, 1-28. Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1989). *A practical anti-cycling procedure for linearly constrained optimization.* Mathematical Programming 45, 437-474.

Implementiert in `pre_presolve.py` (Regeln, Postsolve), `pre_scaling.py`, `pre_simplex.py` (Toleranzen, Harris, Störung), `pre_algorithm.py` (Tableau-Kern), `pre_ipm.py` (Mehrotra aus Stück 8), `pre_evaluation.py`, `pre_scenario.py`.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
