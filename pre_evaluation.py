"""Auswertung: Präsolve-Ausbeute, Skalierung gegen die Ausfälle, Quotiententest gegen Toleranzen, Stillstand und Störung."""

import math
import statistics
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

import pre_algorithm as A
import pre_constants as C
import pre_ipm as IP
import pre_presolve as PS
import pre_scaling as SC
import pre_scenario as S
import pre_simplex as X


@dataclass(frozen=True)
class Settings:
    kind: str = "messy"
    m: int = C.DEFAULT_M
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    spread_i: int = C.DEFAULT_SPREAD_I
    red_i: int = C.DEFAULT_RED_I
    presolve_on: bool = True
    scaling: str = "pow2"
    ratio: str = "textbook"
    tol_i: int = C.DEFAULT_TOL_I
    perturb_i: int = 0

    @property
    def spread(self):
        return C.SPREAD_EXPS[min(max(self.spread_i, 0), len(C.SPREAD_EXPS) - 1)]

    @property
    def redundancy(self):
        return C.REDUNDANCIES[min(max(self.red_i, 0), len(C.REDUNDANCIES) - 1)]

    @property
    def tol(self):
        return C.TOLS[min(max(self.tol_i, 0), len(C.TOLS) - 1)]

    @property
    def perturb(self):
        return C.PERTURBS[min(max(self.perturb_i, 0), len(C.PERTURBS) - 1)]


def instance_of(s):
    return S.generate(s.kind, s.m, s.n, C.DENSITY, s.seed, spread=s.spread, redundancy=s.redundancy)


def rel_infeasibility(inst, x):
    """Größte relative Verletzung der Zeilen (Verletzung / (|b_i| + |a_i|·|x|)) und der Nichtnegativität."""
    A_, b, _c = inst.arrays()
    x = np.asarray(x, dtype=float)
    act = A_ @ x
    scale = np.abs(b) + np.abs(A_) @ np.abs(x) + 1e-300
    viol = np.zeros(inst.m)
    for i, sense in enumerate(inst.senses):
        viol[i] = max(0.0, act[i] - b[i]) if sense == S.LE else (max(0.0, b[i] - act[i]) if sense == S.GE else abs(act[i] - b[i]))
    neg = float(np.max(np.maximum(-x, 0.0)) / (1.0 + np.max(np.abs(x)))) if len(x) else 0.0
    return float(max(np.max(viol / scale) if inst.m else 0.0, neg))


def reference(inst):
    """Verlässlicher Referenzwert: Zweierpotenz-Skalierung, dann der einfache Löser mit Toleranz 1e-9. Rückgabe (Status, Optimalwert)."""
    sc = SC.scale(inst, "pow2")
    r = X.simplex(sc.inst, tol=1e-9)
    return r.status, (r.obj if r.status == "optimal" else float("nan"))


@dataclass
class Pipe:
    status: str
    x: tuple = ()
    y: tuple = ()
    obj: float = float("nan")
    pivots: int = 0
    zero_steps: int = 0
    min_pivot_rel: float = float("inf")
    rel_infeas: float = float("nan")
    gap: float = float("nan")                         # |c·x - b·y| / (1 + |c·x|) auf dem Original
    pre: object = None
    rows: int = 0                                     # Zeilen des gelösten LP (mit Schranken-Zeilen)
    cols: int = 0
    spread_before: float = 0.0
    spread_after: float = 0.0
    basis_ok: bool = True


def pipeline(inst, presolve_on, scaling, ratio, tol, perturb, seed=0):
    """Original -> (Präsolve) -> (Skalierung) -> Simplex -> Rückrechnung -> (Postsolve) mit Kennzahlen auf dem Original."""
    A0 = inst.arrays()[0]
    out = Pipe("optimal", spread_before=SC.spread(A0))
    work, pre = inst, None
    if presolve_on:
        pre = PS.presolve(inst)
        out.pre = pre
        if pre.status == "infeasible":
            out.status = "infeasible"
            return out
        work = pre.reduced
    out.rows, out.cols = work.m, work.n
    if work.n == 0 and work.m == 0:
        x, y = PS.postsolve(pre, [], [])
        res = None
    else:
        sc = SC.scale(work, scaling)
        out.spread_after = SC.spread(sc.inst.arrays()[0]) if work.m and work.n else 0.0
        res = X.simplex(sc.inst, tol=tol, ratio=ratio, feas_tol=tol, perturb=perturb, seed=seed)
        out.pivots, out.zero_steps, out.min_pivot_rel, out.basis_ok = res.pivots, res.zero_steps, res.min_pivot_rel, res.basis_ok
        if res.status != "optimal":
            out.status = res.status
            return out
        x, y = sc.unscale_x(res.x), sc.unscale_y(res.y)
        if presolve_on:
            x, y = PS.postsolve(pre, x, y)
    if pre is not None and pre.unbounded_hint:
        out.status = "unbounded"
        return out
    out.x, out.y = tuple(float(v) for v in x), tuple(float(v) for v in y)
    c, b = np.array(inst.c), np.array(inst.b)
    out.obj = float(c @ x)
    out.rel_infeas = rel_infeasibility(inst, x)
    out.gap = abs(out.obj - float(b @ y)) / (1.0 + abs(out.obj))
    return out


def correct(pipe, ref_status, ref_obj, obj_tol=1e-5, infeas_tol=1e-6):
    """Ergebnis stimmt: gleicher Status, gleicher Optimalwert und relativ zulässig."""
    if pipe.status != ref_status:
        return False
    if ref_status != "optimal":
        return True
    return abs(pipe.obj - ref_obj) <= obj_tol * (1.0 + abs(ref_obj)) and pipe.rel_infeas <= infeas_tol


@dataclass
class Analysis:
    inst: object
    ref_status: str
    ref_obj: float
    plain: Pipe                                       # ohne Präsolve und Skalierung, einfacher Test, Toleranz 1e-9
    piped: Pipe                                       # gewählte Einstellungen
    plain_ok: bool = False
    piped_ok: bool = False
    dirt: dict = field(default_factory=dict)
    ipm_plain: object = None                          # Mehrotra auf dem unbearbeiteten LP
    ipm_scaled: object = None                         # Mehrotra auf dem skalierten LP


@lru_cache(maxsize=64)
def analyse(s):
    inst = instance_of(s)
    dirt = S.messy_instance(s.m, s.n, s.seed, s.spread, s.redundancy)[1] if s.kind == "messy" else {}
    ref_status, ref_obj = reference(inst)
    plain = pipeline(inst, False, "none", "textbook", 1e-9, 0.0)
    piped = pipeline(inst, s.presolve_on, s.scaling, s.ratio, s.tol, s.perturb, s.seed)
    slack = max(1.0, 10.0 * s.perturb / 1e-6)                                                         # gestörte Lösungen verletzen das Original um bis zu ~ die Störung
    ip = IP.ipm(inst, "mehrotra", eps=1e-8)
    isc = IP.ipm(SC.scale(inst, "pow2").inst, "mehrotra", eps=1e-8)
    return Analysis(inst, ref_status, ref_obj, plain, piped, correct(plain, ref_status, ref_obj), correct(piped, ref_status, ref_obj, obj_tol=1e-5 * slack, infeas_tol=1e-6 * slack), dirt, ip, isc)


def _med(v):
    return statistics.median(v) if v else float("nan")


@lru_cache(maxsize=32)
def presolve_sweep(s):
    """Präsolve-Ausbeute über den Verschmutzungsgrad (Kern m × n aus dem Regler, 5 feste Instanzen): Zeilen, Spalten, Nichtnullen vorher/nachher (das reduzierte LP enthält die Schranken als Zeilen), Pivots,
    Anteil richtiger Ergebnisse mit Präsolve."""
    rows = []
    for red in C.REDUNDANCIES:
        b_m, b_n, a_m, a_n, nz_b, nz_a, pv_b, pv_a, ok = [], [], [], [], [], [], [], [], 0
        for sd in C.SWEEP_SEEDS:
            inst = S.messy_instance(s.m, s.n, sd, 0, red)[0]
            ref_status, ref_obj = reference(inst)
            plain = pipeline(inst, False, "none", "textbook", 1e-9, 0.0)
            piped = pipeline(inst, True, "none", "textbook", 1e-9, 0.0)
            b_m.append(inst.m), b_n.append(inst.n), a_m.append(piped.rows), a_n.append(piped.cols)
            nz_b.append(int(np.count_nonzero(inst.arrays()[0])))
            red_inst = piped.pre.reduced if piped.pre is not None and piped.pre.status != "infeasible" else None
            nz_a.append(int(np.count_nonzero(red_inst.arrays()[0])) if red_inst is not None and red_inst.m and red_inst.n else 0)
            pv_b.append(plain.pivots), pv_a.append(piped.pivots)
            ok += correct(piped, ref_status, ref_obj)
        rows.append({"red": red, "rows_before": _med(b_m), "rows_after": _med(a_m), "cols_before": _med(b_n), "cols_after": _med(a_n), "nnz_before": _med(nz_b), "nnz_after": _med(nz_a),
                     "pivots_before": _med(pv_b), "pivots_after": _med(pv_a), "ok": ok})
    return rows


@lru_cache(maxsize=32)
def scaling_sweep(s):
    """Skalierung gegen die Einheiten-Spanne: dieselben fünf schmutzigen Instanzen (Kern m × n aus dem Regler, Verschmutzung aus dem Regler) mit Spanne 10^k; je Skalierungsverfahren Simplex und Mehrotra
    richtig gelöst (von 5) und Spanne der Matrix danach."""
    rows = []
    for k in C.SPREAD_EXPS:
        row = {"k": k}
        for meth in SC.METHODS:
            okS = okI = 0
            spr = []
            for sd in C.SWEEP_SEEDS:
                inst = S.messy_instance(s.m, s.n, sd, k, s.redundancy)[0]
                ref_status, ref_obj = reference(inst)
                piped = pipeline(inst, False, meth, "textbook", 1e-9, 0.0)
                okS += correct(piped, ref_status, ref_obj)
                sc = SC.scale(inst, meth)
                spr.append(SC.spread(sc.inst.arrays()[0]))
                r = IP.ipm(sc.inst, "mehrotra", eps=1e-8)
                if r.status == "optimal" and ref_status == "optimal":
                    x = sc.unscale_x(r.x)
                    okI += abs(float(np.dot(inst.c, x)) - ref_obj) <= 1e-4 * (1.0 + abs(ref_obj)) and rel_infeasibility(inst, x) <= 1e-5
            row[meth] = {"simplex": okS, "ipm": int(okI), "spread": _med(spr)}
        rows.append(row)
    return rows


@lru_cache(maxsize=32)
def ratio_sweep(s):
    """Nadel-Familie (Harris-Falle) über die Toleranz: einfacher Test gegen Harris; je 10 feste Instanzen: Läufe mit relativ unzulässiger Lösung (über 1e-6), größte relative Verletzung, kleinstes Pivotelement
    (relativ zur Spalte, Median), Pivots."""
    rows = []
    insts = [S.needle_instance(10, 10, sd, C.NEEDLE_SPREAD) for sd in C.NEEDLE_SEEDS]
    refs = [reference(i) for i in insts]
    for tol in C.TOLS:
        row = {"tol": tol}
        for ratio in X.RATIOS:
            bad, worst, mins, piv = 0, 0.0, [], []
            for inst, (rs, ro) in zip(insts, refs):
                r = pipeline(inst, False, "none", ratio, tol, 0.0)
                if not correct(r, rs, ro, obj_tol=1e-6, infeas_tol=1e-6):
                    bad += 1
                if r.status == "optimal":
                    worst = max(worst, r.rel_infeas)
                    mins.append(r.min_pivot_rel)
                    piv.append(r.pivots)
            row[ratio] = {"bad": bad, "worst": worst, "min_pivot": _med(mins), "pivots": _med(piv)}
        rows.append(row)
    return rows


@lru_cache(maxsize=32)
def stall_sweep(s):
    """Entartete Familie über die Größe: Pivots und Nullschritte ohne und mit Störung von b (1e-7, 1e-6), Anteil der Läufe, in denen die gestörte Endbasis auch für das Original zulässig ist, größter Fehler des Zielwerts."""
    rows = []
    for n in C.STALL_SIZES:
        row = {"n": n}
        for pert in C.PERTURBS:
            piv, zero, ok, err = [], [], 0, []
            for sd in C.SWEEP_SEEDS:
                inst = S.ties_instance(n, n, sd)
                ref = A.solve(inst)
                r = X.simplex(inst, perturb=pert, seed=sd)
                piv.append(r.pivots), zero.append(r.zero_steps)
                ok += r.basis_ok
                err.append(abs(r.obj - ref.obj) / (1.0 + abs(ref.obj)))
            row[pert] = {"pivots": _med(piv), "zero": _med(zero), "basis_ok": ok, "err": max(err)}
        rows.append(row)
    return rows


def log_table(pre):
    """Präsolve-Protokoll als Zeilen (Regel, Text)."""
    return [(e["rule"], e["text"]) for e in pre.log]


def rules_count(pre):
    counts = {r: 0 for r in PS.RULES}
    for e in pre.log:
        if e["rule"] in counts:
            counts[e["rule"]] += 1
    return counts


def gap_ok(pipe, tol=1e-6):
    return not math.isnan(pipe.gap) and pipe.gap <= tol
