"""Präsolve für  max c·x,  A x (<=|>=|=) b,  x >= 0  mit Postsolve für Lösung UND Duale.

Regeln bis zum Fixpunkt (mit Variablenschranken l <= x <= u): leere Zeile, Singleton-Zeile (wird zur Schranke), redundante Zeile (über die Aktivitätsschranken; ein Widerspruch beweist die Unzulässigkeit), doppelte oder
parallele Zeile (die engste rechte Seite bleibt), feste Variable (l = u, wird eingesetzt), leere Spalte (Wert an der Schranke, Zielkoeffizient entscheidet). Das reduzierte LP verschiebt alle Unterschranken auf 0 und gibt endliche
Oberschranken als Zeilen zurück (der Löser kennt keine Schranken). Postsolve: x aus den festen Werten und der Verschiebung; Duale: reduzierte Zeilen wie gelöst, Schranken-Zeilen über die Quelle der Schranke (y_s = -r_j / a_sj),
alle anderen entfernten Zeilen y = 0. Zertifikat auf dem Original: primal zulässig, dual zulässig, c·x = b·y."""

import math
from dataclasses import dataclass, field

import numpy as np

import pre_scenario as S

LE, GE, EQ = S.LE, S.GE, S.EQ
RULES = ("leere Zeile", "Singleton-Zeile", "redundante Zeile", "doppelte Zeile", "feste Variable", "leere Spalte")


@dataclass
class Presolved:
    status: str                                       # "reduced" | "empty" (alles bestimmt) | "infeasible"
    reduced: object = None                            # Instance mit den lebenden Zeilen, danach den Schranken-Zeilen
    orig: object = None
    log: list = field(default_factory=list)           # je Reduktion: {"rule", "row", "col", "text"}
    ops: list = field(default_factory=list)           # Postsolve-Stapel: ("fix" | "empty_col", j)
    rows: list = field(default_factory=list)          # Indizes der lebenden Originalzeilen
    cols: list = field(default_factory=list)          # Indizes der lebenden Originalspalten
    l: object = None                                  # Unterschranken (verschoben)
    u: object = None
    xval: object = None                               # Werte der entfernten Variablen
    src_l: dict = field(default_factory=dict)         # Originalzeile, aus der die Unterschranke der Variable stammt
    src_u: dict = field(default_factory=dict)
    bound_cols: list = field(default_factory=list)    # Spalten mit Schranken-Zeile im reduzierten LP (in Zeilenreihenfolge)
    objconst: float = 0.0                             # Zielbeitrag der entfernten Variablen und der Verschiebung
    unbounded_hint: bool = False                      # leere Spalte mit positivem Ertrag und ohne Oberschranke
    note: str = ""

    @property
    def removed_rows(self):
        return self.orig.m - len(self.rows)

    @property
    def removed_cols(self):
        return self.orig.n - len(self.cols)


def _ftol(x):
    return 1e-9 * (1.0 + abs(x))


def presolve(inst, max_passes=50):
    A, b, c = inst.arrays()
    m, n = A.shape
    b_eff = b.copy()
    senses = list(inst.senses)
    l, u = np.zeros(n), np.full(n, math.inf)
    row_alive, col_alive = np.ones(m, bool), np.ones(n, bool)
    xval = np.full(n, np.nan)
    src_l, src_u = {}, {}
    log, ops = [], []
    objconst = 0.0
    unbounded_hint = False

    def emit(rule, row, col, text):
        log.append({"rule": rule, "row": row, "col": col, "text": text})

    def infeasible(text):
        return Presolved("infeasible", orig=inst, log=log + [{"rule": "Widerspruch", "row": -1, "col": -1, "text": text}], note=text)

    def nz(i):
        return [j for j in np.nonzero(A[i])[0] if col_alive[j]]

    for _ in range(max_passes):
        changed = False
        # --- Zeilen ---------------------------------------------------------------------------------------------------------------------------------
        for i in range(m):
            if not row_alive[i]:
                continue
            cols = nz(i)
            r, s = b_eff[i], senses[i]
            if not cols:
                ok = (r >= -_ftol(r)) if s == LE else ((r <= _ftol(r)) if s == GE else abs(r) <= _ftol(r))
                if not ok:
                    return infeasible(f"Zeile {i + 1} ist leer, verlangt aber 0 {s} {r:g}")
                row_alive[i] = False
                emit("leere Zeile", i, -1, f"Zeile {i + 1}: 0 {s} {r:.6g} gilt immer")
                changed = True
                continue
            if len(cols) == 1:
                j = cols[0]
                a = A[i, j]
                v = r / a
                up = (s == LE and a > 0) or (s == GE and a < 0) or s == EQ
                lo = (s == LE and a < 0) or (s == GE and a > 0) or s == EQ
                text = f"Zeile {i + 1}: Dienst {j + 1} " + ("=" if s == EQ else ("≤" if up else "≥")) + f" {v:.6g}"
                if up and v < u[j]:
                    u[j], src_u[j] = v, i
                if lo and v > l[j]:
                    l[j], src_l[j] = v, i
                if l[j] > u[j] + _ftol(u[j]):
                    return infeasible(f"Dienst {j + 1}: untere Schranke {l[j]:.6g} liegt über der oberen {u[j]:.6g}")
                row_alive[i] = False
                ops.append(("singleton", i, j))
                emit("Singleton-Zeile", i, j, text + " (Schranke)")
                changed = True
                continue
            lo_act = hi_act = 0.0
            for j in cols:
                a = A[i, j]
                lo_act += a * (l[j] if a > 0 else u[j])
                hi_act += a * (u[j] if a > 0 else l[j])
            tol = _ftol(r)
            if s == LE:
                if hi_act <= r + tol:
                    row_alive[i] = False
                    emit("redundante Zeile", i, -1, f"Zeile {i + 1}: größte Aktivität {hi_act:.6g} ≤ {r:.6g}: nie bindend")
                    changed = True
                elif lo_act > r + tol:
                    return infeasible(f"Zeile {i + 1}: kleinste Aktivität {lo_act:.6g} über {r:.6g}")
            elif s == GE:
                if lo_act >= r - tol:
                    row_alive[i] = False
                    emit("redundante Zeile", i, -1, f"Zeile {i + 1}: kleinste Aktivität {lo_act:.6g} ≥ {r:.6g}: nie bindend")
                    changed = True
                elif hi_act < r - tol:
                    return infeasible(f"Zeile {i + 1}: größte Aktivität {hi_act:.6g} unter {r:.6g}")
            else:
                if lo_act > r + tol or hi_act < r - tol:
                    return infeasible(f"Zeile {i + 1}: Gleichung {r:.6g} außerhalb der Aktivität [{lo_act:.6g}, {hi_act:.6g}]")
        # --- doppelte Zeilen -----------------------------------------------------------------------------------------------------------------------
        groups = {}
        for i in range(m):
            if not row_alive[i]:
                continue
            cols = nz(i)
            if len(cols) < 2:
                continue
            j0 = cols[0]
            sc = A[i, j0]
            key = tuple((int(j), float(f"{A[i, j] / sc:.9g}")) for j in cols)
            groups.setdefault(key, []).append((i, sc))
        for key, members in groups.items():
            if len(members) < 2:
                continue
            best_up, best_lo = (math.inf, None), (-math.inf, None)
            for i, sc in members:
                rhs = b_eff[i] / sc
                sense = senses[i] if sc > 0 else {LE: GE, GE: LE, EQ: EQ}[senses[i]]
                if sense in (LE, EQ) and rhs < best_up[0]:
                    best_up = (rhs, i)
                if sense in (GE, EQ) and rhs > best_lo[0]:
                    best_lo = (rhs, i)
            if best_lo[0] > best_up[0] + _ftol(best_up[0]) and best_up[1] is not None and best_lo[1] is not None:
                return infeasible(f"Zeilen {best_lo[1] + 1} und {best_up[1] + 1} sind parallel und widersprechen sich")
            keep = {best_up[1], best_lo[1]} - {None}
            for i, _sc in members:
                if i not in keep:
                    row_alive[i] = False
                    emit("doppelte Zeile", i, -1, f"Zeile {i + 1} ist parallel zu Zeile {min(keep) + 1} und nicht strenger")
                    changed = True
        # --- Spalten ---------------------------------------------------------------------------------------------------------------------------------
        for j in range(n):
            if not col_alive[j]:
                continue
            if u[j] - l[j] <= _ftol(l[j]) and math.isfinite(u[j]):
                val = 0.5 * (l[j] + u[j])
                for i in range(m):
                    if row_alive[i]:
                        b_eff[i] -= A[i, j] * val
                objconst += c[j] * val
                col_alive[j] = False
                xval[j] = val
                ops.append(("fix", j))
                emit("feste Variable", -1, j, f"Dienst {j + 1} = {val:.6g}")
                changed = True
                continue
            if not any(row_alive[i] and A[i, j] != 0 for i in range(m)):
                if c[j] > 0:
                    if math.isfinite(u[j]):
                        val = u[j]
                    else:
                        val = l[j]
                        unbounded_hint = True
                else:
                    val = l[j]
                objconst += c[j] * val
                col_alive[j] = False
                xval[j] = val
                ops.append(("empty_col", j))
                emit("leere Spalte", -1, j, f"Dienst {j + 1} kommt in keiner Zeile vor: " + ("unbeschränkt" if unbounded_hint and c[j] > 0 and not math.isfinite(u[j]) else f"= {val:.6g}"))
                changed = True
        if not changed:
            break
    rows = [i for i in range(m) if row_alive[i]]
    cols = [j for j in range(n) if col_alive[j]]
    pre = Presolved("reduced", orig=inst, log=log, ops=ops, rows=rows, cols=cols, l=l, u=u, xval=xval, src_l=src_l, src_u=src_u, unbounded_hint=unbounded_hint)
    Ared = A[np.ix_(rows, cols)] if rows and cols else np.zeros((len(rows), len(cols)))
    bred = np.array([b_eff[i] - sum(A[i, j] * l[j] for j in cols) for i in rows])
    cred = c[cols]
    total = objconst + float(sum(c[j] * l[j] for j in cols))
    rows_out, rhs_out, sen_out, names_out = [list(r) for r in Ared], list(bred), [senses[i] for i in rows], [inst.row_names[i] for i in rows]
    for k, j in enumerate(cols):
        if math.isfinite(u[j]):
            width = u[j] - l[j]
            if width < -_ftol(l[j]):
                return infeasible(f"Dienst {j + 1}: Schranken widersprechen sich")
            e = [0.0] * len(cols)
            e[k] = 1.0
            rows_out.append(e), rhs_out.append(max(width, 0.0)), sen_out.append(LE), names_out.append(f"Schranke Dienst {j + 1}")
            pre.bound_cols.append(j)
    pre.objconst = total
    if not cols and not rows_out:
        pre.status = "empty"
    pre.reduced = S.Instance(tuple(tuple(float(v) for v in r) for r in rows_out), tuple(float(v) for v in rhs_out), tuple(float(v) for v in cred), tuple(sen_out), tuple(inst.names[j] for j in cols),
                             tuple(names_out), "reduced")
    return pre


def postsolve(pre, x_red, y_red):
    """Lösung und Duale des Originals aus Lösung und Dualen des reduzierten LP."""
    inst = pre.orig
    A, b, c = inst.arrays()
    m, n = A.shape
    x = np.array(pre.xval, dtype=float)
    for k, j in enumerate(pre.cols):
        x[j] = pre.l[j] + (x_red[k] if len(x_red) else 0.0)
    y = np.zeros(m)
    for k, i in enumerate(pre.rows):
        y[i] = y_red[k] if len(y_red) else 0.0
    off = len(pre.rows)
    for k, j in enumerate(pre.bound_cols):
        s = pre.src_u.get(j)
        if s is not None and len(y_red) > off + k:
            y[s] = y_red[off + k] / A[s, j]

    def reduced_cost(j):
        return float(A[:, j] @ y - c[j])

    for k, j in enumerate(pre.cols):                                                        # verschobene Unterschranke: die Quelle trägt den Preis
        s = pre.src_l.get(j)
        if s is not None and pre.l[j] > 0:
            r = reduced_cost(j)
            if r > _ftol(r):
                y[s] = -r / A[s, j]
    for kind, j in reversed([(o[0], o[1] if o[0] != "singleton" else o[2]) for o in pre.ops if o[0] in ("fix", "empty_col")]):
        r = reduced_cost(j)
        if r > _ftol(r) and pre.src_l.get(j) is not None and pre.l[j] > 0:
            s = pre.src_l[j]
            y[s] = -r / A[s, j]
        elif r < -_ftol(r) and pre.src_u.get(j) is not None:
            s = pre.src_u[j]
            y[s] = -r / A[s, j]
    return x, y
