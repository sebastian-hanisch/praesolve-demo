"""Dichtes Tableau-Simplex mit numerischen Stellschrauben: Toleranz, Quotiententest (einfach oder Harris), Störung der rechten Seite gegen Stillstand. Kern wie Stück 1-8 (Dantzig-Regel, Bland-Notbremse nach
STALL_LIMIT Nullschritten, Zwei-Phasen-Start); Ergebnis mit Kennzahlen: Pivotzahl, Nullschritte, kleinstes Pivotelement (relativ zur Spalte), Restunzulässigkeit der Lösung auf dem Original."""

import random
from dataclasses import dataclass

import numpy as np

import pre_algorithm as A
import pre_scenario as S

LE, GE, EQ = S.LE, S.GE, S.EQ
RATIOS = ("textbook", "harris")
RATIO_LABELS = {"textbook": "Einfach (kleinstes Verhältnis)", "harris": "Harris (größtes Pivotelement unter den fast gleichen)"}
STALL_LIMIT = A.STALL_LIMIT


@dataclass
class SResult:
    status: str
    x: tuple = ()
    obj: float = float("nan")                         # c·x auf dem Original (auch bei Störung)
    y: tuple = ()
    pivots: int = 0
    zero_steps: int = 0
    min_pivot_rel: float = float("inf")               # kleinstes |Pivotelement| / größter Betrag der Spalte über alle Pivots
    max_infeas: float = float("nan")                  # größte Verletzung der Originalbedingungen durch x
    basis_ok: bool = True                             # bei Störung: ist die Endbasis auch für das Original primal zulässig?
    perturbed: bool = False


def perturbed_instance(inst, size, seed):
    """b_i wird um eps_i (1 + |b_i|) gelockert, eps_i in [size, 2 size]: <= größer, >= kleiner, Gleichungen bleiben."""
    rng = random.Random(f"pre-perturb-{seed}")
    b = list(inst.b)
    for i, s in enumerate(inst.senses):
        e = rng.uniform(size, 2.0 * size) * (1.0 + abs(b[i]))
        if s == LE:
            b[i] += e
        elif s == GE:
            b[i] -= e
    return S.Instance(inst.A, tuple(b), inst.c, inst.senses, inst.names, inst.row_names, inst.kind)


def _basis_feasible(inst, basis):
    T, _basis, _info = A.standard_form(inst)
    try:
        xb = np.linalg.solve(T[:, list(basis)], T[:, -1])
    except np.linalg.LinAlgError:
        return False
    return bool(np.all(xb >= -1e-9 * (1.0 + np.abs(xb))))


def simplex(inst, tol=1e-9, ratio="textbook", feas_tol=1e-9, perturb=0.0, seed=0, max_pivots=20_000, stall_guard=True):
    if ratio not in RATIOS:
        raise ValueError(ratio)
    work = perturbed_instance(inst, perturb, seed) if perturb > 0 else inst
    T, basis, info = A.standard_form(work)
    m, n, ncols = info["m"], info["n"], info["ncols"]
    art = set(info["art_col"].values())
    init_basis = tuple(basis)
    full = np.zeros((m + 1, ncols + 1))
    full[:m] = T
    T = full
    stats = {"pivots": 0, "zero": 0, "min_rel": float("inf")}

    def run(cvec, allowed):
        cB = cvec[basis]
        T[m, :] = cB @ T[:m, :]
        T[m, :-1] -= cvec
        stall = 0
        while True:
            r = T[m, :-1]
            cand = [j for j in allowed if r[j] < -tol]
            if not cand:
                return "optimal"
            if stall_guard and stall >= STALL_LIMIT:
                enter = cand[0]
            else:
                best = min(r[j] for j in cand)
                enter = next(j for j in cand if r[j] <= best + 1e-9 * max(1.0, abs(best)))
            col = T[:m, enter]
            pos = [i for i in range(m) if col[i] > tol]
            if not pos:
                return "unbounded"
            rhs = {i: max(T[i, -1], 0.0) for i in pos}
            if ratio == "textbook":
                ratios = {i: T[i, -1] / col[i] for i in pos}
                rmin = min(ratios.values())
                tied = [i for i in pos if ratios[i] <= rmin + tol * max(1.0, abs(rmin))]
                leave = min(tied, key=lambda i: basis[i])
                step = rmin
            else:
                theta = min((rhs[i] + feas_tol) / col[i] for i in pos)
                cands = [i for i in pos if rhs[i] / col[i] <= theta]
                top = max(col[i] for i in cands)
                leave = min((i for i in cands if col[i] >= top * (1 - 1e-12)), key=lambda i: basis[i])
                step = rhs[leave] / col[leave]
                T[leave, -1] = rhs[leave]
            colmax = float(np.max(np.abs(col)))
            stats["min_rel"] = min(stats["min_rel"], abs(col[leave]) / colmax if colmax > 0 else 1.0)
            A._pivot(T, leave, enter)
            basis[leave] = enter
            stats["pivots"] += 1
            if max(step, 0.0) <= tol:
                stats["zero"] += 1
                stall += 1
            else:
                stall = 0
            if stats["pivots"] > max_pivots:
                return "limit"

    def finish(status, **kw):
        return SResult(status, pivots=stats["pivots"], zero_steps=stats["zero"], min_pivot_rel=stats["min_rel"], perturbed=perturb > 0, **kw)

    if art:
        c1 = np.zeros(ncols)
        for j in art:
            c1[j] = -1.0
        st = run(c1, list(range(ncols)))
        if st == "limit":
            return finish("limit")
        if T[m, -1] < -1e-7:
            return finish("infeasible")
        for a in sorted(art):
            if a not in basis:
                continue
            i = basis.index(a)
            cand = [j for j in range(ncols) if j not in art and abs(T[i, j]) > tol]
            if cand:
                best = max(abs(T[i, j]) for j in cand)
                enter = next(j for j in cand if abs(T[i, j]) >= best - 1e-9 * max(1.0, best))
                A._pivot(T, i, enter)
                basis[i] = enter
                stats["pivots"] += 1
    c2 = np.zeros(ncols)
    c2[:n] = info["c"]
    st = run(c2, [j for j in range(ncols) if j not in art])
    if st != "optimal":
        return finish(st)
    xf = np.zeros(ncols)
    for i, j in enumerate(basis):
        xf[j] = T[i, -1]
    x = xf[:n]
    y = tuple(float(T[m, init_basis[i]]) * info["sign"][i] for i in range(m))
    _A, _b, c = inst.arrays()
    return finish("optimal", x=tuple(float(v) for v in x), obj=float(c @ x), y=y, max_infeas=A.primal_violation(inst, x), basis_ok=(_basis_feasible(inst, basis) if perturb > 0 else True))

