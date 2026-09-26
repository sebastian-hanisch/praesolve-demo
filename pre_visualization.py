"""Plotly-Abbildungen: Präsolve vorher/nachher, Ausbeute über den Verschmutzungsgrad, Heatmap der Koeffizientengrößen, Skalierung über die Spanne, Quotiententest über die Toleranz, Stillstand und Störung.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pre_constants as C
import pre_scaling as SC
import pre_simplex as X

TEAL, ORANGE, RED, BLUE, GREY, PURPLE = "#2F6B65", "#e8a13a", "#d62728", "#1f4e9c", "#8a8f98", "#7b3fbf"
METHOD_COLORS = {"none": GREY, "equilibrate": PURPLE, "geometric": BLUE, "pow2": ORANGE}
RATIO_COLORS = {"textbook": GREY, "harris": TEAL}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_presolve_bars(before, after):
    """before/after: dict mit rows, cols, nnz."""
    labels = ["Zeilen", "Spalten", "Nichtnullen"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[before["rows"], before["cols"], before["nnz"]], marker_color=GREY, name="Original", text=[str(before["rows"]), str(before["cols"]), str(before["nnz"])], textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=[after["rows"], after["cols"], after["nnz"]], marker_color=TEAL, name="nach Präsolve", text=[str(after["rows"]), str(after["cols"]), str(after["nnz"])], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Anzahl", rangemode="tozero")
    return _base(fig, 300, legend_y=-0.3)


def build_presolve_sweep(rows):
    xs = [f"{r['red']:.0%}" for r in rows]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=xs, y=[r["rows_before"] for r in rows], marker_color=GREY, name="Zeilen vorher"), secondary_y=False)
    fig.add_trace(go.Bar(x=xs, y=[r["rows_after"] for r in rows], marker_color=TEAL, name="Zeilen nachher (mit Schranken-Zeilen)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=xs, y=[r["pivots_before"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3, dash="dot"), name="Pivots vorher"), secondary_y=True)
    fig.add_trace(go.Scatter(x=xs, y=[r["pivots_after"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name="Pivots nachher"), secondary_y=True)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Verschmutzungsgrad", type="category")
    fig.update_yaxes(title_text="Zeilen", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="Pivots", secondary_y=True, rangemode="tozero", showgrid=False)
    return _base(fig, 340, legend_y=-0.4)


def build_heat(inst_before, inst_after, label_after):
    """log10 der Beträge der Nichtnullen vorher und nachher (gleiche Farbskala)."""
    def prep(inst):
        A = np.abs(inst.arrays()[0])
        return np.where(A > 0, np.log10(np.where(A > 0, A, 1.0)), np.nan)
    Z1, Z2 = prep(inst_before), prep(inst_after)
    lo = float(np.nanmin([np.nanmin(Z1) if np.isfinite(Z1).any() else 0.0, np.nanmin(Z2) if np.isfinite(Z2).any() else 0.0]))
    hi = float(np.nanmax([np.nanmax(Z1) if np.isfinite(Z1).any() else 0.0, np.nanmax(Z2) if np.isfinite(Z2).any() else 0.0]))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original", label_after), horizontal_spacing=0.08)
    fig.add_trace(go.Heatmap(z=Z1, zmin=lo, zmax=hi, colorscale="Viridis", showscale=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=Z2, zmin=lo, zmax=hi, colorscale="Viridis", colorbar=dict(title="log10 |a|", len=0.9), hoverinfo="skip"), row=1, col=2)
    fig.update_yaxes(autorange="reversed", title_text="Zeile", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="Spalte")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_scaling_sweep(rows):
    ks = [f"10^{r['k']}" for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Simplex", "Mehrotra"), shared_yaxes=True, horizontal_spacing=0.06)
    for meth in SC.METHODS:
        fig.add_trace(go.Bar(x=ks, y=[r[meth]["simplex"] for r in rows], marker_color=METHOD_COLORS[meth], name=C.METHOD_SHORT[meth], legendgroup=meth), row=1, col=1)
        fig.add_trace(go.Bar(x=ks, y=[r[meth]["ipm"] for r in rows], marker_color=METHOD_COLORS[meth], name=C.METHOD_SHORT[meth], legendgroup=meth, showlegend=False), row=1, col=2)
    fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title_text="richtig gelöst (von 5)", range=[0, 5.3], row=1, col=1)
    fig.update_xaxes(title_text="Einheiten-Spanne", type="category")
    return lock_axes(fig)


def build_ratio_sweep(rows):
    xs = [f"{r['tol']:.0e}".replace("e-0", "e-") for r in rows]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for ratio in X.RATIOS:
        fig.add_trace(go.Bar(x=xs, y=[r[ratio]["bad"] for r in rows], marker_color=RATIO_COLORS[ratio], name=("Einfach" if ratio == "textbook" else "Harris") + ": falsche Läufe (von 10)"), secondary_y=False)
    for ratio in X.RATIOS:
        fig.add_trace(go.Scatter(x=xs, y=[max(r[ratio]["worst"], 1e-17) for r in rows], mode="lines+markers", line=dict(color=RATIO_COLORS[ratio] if ratio == "textbook" else ORANGE, width=2, dash="dot"),
                                 name=("Einfach" if ratio == "textbook" else "Harris") + ": größte relative Verletzung"), secondary_y=True)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Toleranz", type="category")
    fig.update_yaxes(title_text="falsche Läufe", range=[0, 10.5], secondary_y=False)
    fig.update_yaxes(title_text="relative Verletzung", type="log", secondary_y=True, showgrid=False)
    return _base(fig, 360, legend_y=-0.45)


def build_stall(rows):
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    for pert, color, dash in ((0.0, GREY, "solid"), (1e-7, ORANGE, "dash"), (1e-6, TEAL, "dot")):
        label = "ohne Störung" if pert == 0 else f"Störung {pert:.0e}".replace("e-0", "e-")
        fig.add_trace(go.Scatter(x=ns, y=[r[pert]["zero"] for r in rows], mode="lines+markers", line=dict(color=color, width=3, dash=dash), name=f"Nullschritte, {label}"))
    fig.add_trace(go.Scatter(x=ns, y=[r[0.0]["pivots"] for r in rows], mode="lines", line=dict(color=BLUE, width=1.5), name="Pivots ohne Störung"))
    fig.add_trace(go.Scatter(x=ns, y=[r[1e-6]["pivots"] for r in rows], mode="lines", line=dict(color=BLUE, width=1.5, dash="dot"), name="Pivots mit Störung 1e-6"))
    fig.update_xaxes(title_text="Größe n (entartete Familie, m = n)", tickvals=ns)
    fig.update_yaxes(title_text="Schritte", rangemode="tozero")
    return _base(fig, 340, legend_y=-0.45)
